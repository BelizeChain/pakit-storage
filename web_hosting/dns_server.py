"""
Authoritative DNS Server for BNS (.bz domains).

Provides real DNS resolution for blockchain-registered domains.
Queries BelizeChain for domain records and returns proper DNS responses.
"""

import logging
import asyncio
from typing import Optional, Dict, List
from datetime import datetime
import struct
import socket

from dnslib import DNSRecord, DNSHeader, DNSQuestion, RR, QTYPE, CLASS, A, AAAA, TXT, CNAME, MX
from dnslib.server import DNSServer, DNSHandler, BaseResolver

logger = logging.getLogger(__name__)


class BNSResolver(BaseResolver):
    """
    DNS resolver that queries BelizeChain blockchain for domain records.
    
    Supports:
    - A records (IPv4)
    - AAAA records (IPv6)
    - TXT records (metadata, verification)
    - CNAME records (aliases)
    - MX records (email routing)
    
    Queries Bns::DomainResolution storage for each domain.
    """
    
    def __init__(
        self,
        website_manager,
        default_ip: str = "127.0.0.1",
        default_ipv6: str = "::1",
        ttl: int = 300
    ):
        """
        Initialize DNS resolver.
        
        Args:
            website_manager: WebsiteManager instance for blockchain queries
            default_ip: Default IPv4 for hosted domains
            default_ipv6: Default IPv6 for hosted domains
            ttl: Time-to-live for DNS records (seconds)
        """
        self.manager = website_manager
        self.default_ip = default_ip
        self.default_ipv6 = default_ipv6
        self.ttl = ttl
        
        # Cache: domain -> records
        self.cache: Dict[str, Dict] = {}
        self.cache_timestamp: Dict[str, datetime] = {}
        
        logger.info(f"BNS DNS Resolver initialized (TTL: {ttl}s)")
    
    async def _query_blockchain_records(self, domain: str) -> Optional[Dict]:
        """
        Query blockchain for domain resolution records.
        
        Args:
            domain: Domain name (e.g., "example.bz")
        
        Returns:
            Dictionary of DNS records or None if not found
        """
        if not self.manager.substrate:
            logger.warning("No blockchain connection - using cached/default records")
            return None
        
        try:
            # Query DomainResolution storage
            resolution = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.manager.substrate.query(
                    module='Bns',
                    storage_function='DomainResolution',
                    params=[domain]
                )
            )
            
            if resolution.value is None:
                return None
            
            # Parse resolution records
            records_data = resolution.value
            
            # Expected format: Vec<ResolutionRecord>
            # ResolutionRecord { record_type: u8, value: Vec<u8> }
            records = {
                'a': [],      # IPv4 addresses
                'aaaa': [],   # IPv6 addresses
                'txt': [],    # TXT records
                'cname': [],  # CNAME aliases
                'mx': []      # MX records
            }
            
            # Parse based on record type (simplified for now)
            # In production, this would parse the actual Vec<ResolutionRecord>
            if isinstance(records_data, list):
                for record in records_data:
                    rec_type = record.get('record_type', 0)
                    value = record.get('value', b'').decode('utf-8') if isinstance(record.get('value'), bytes) else str(record.get('value', ''))
                    
                    if rec_type == 0:  # A record
                        records['a'].append(value)
                    elif rec_type == 1:  # AAAA record
                        records['aaaa'].append(value)
                    elif rec_type == 2:  # TXT record
                        records['txt'].append(value)
                    elif rec_type == 3:  # CNAME record
                        records['cname'].append(value)
                    elif rec_type == 4:  # MX record
                        records['mx'].append(value)
            
            return records
            
        except Exception as e:
            logger.error(f"Failed to query blockchain for {domain}: {e}")
            return None
    
    async def _get_domain_records(self, domain: str) -> Dict:
        """
        Get DNS records for domain (with caching).
        
        Args:
            domain: Domain name
        
        Returns:
            Dictionary of DNS records
        """
        # Check cache (5 minute TTL)
        if domain in self.cache:
            age = (datetime.now() - self.cache_timestamp[domain]).total_seconds()
            if age < 300:  # 5 minutes
                return self.cache[domain]
        
        # Query blockchain
        records = await self._query_blockchain_records(domain)
        
        if records is None:
            # Check if domain has hosting active
            ipfs_cid = await self.manager.get_domain_content(domain)
            
            if ipfs_cid:
                # Domain has hosting - return default IPs
                records = {
                    'a': [self.default_ip],
                    'aaaa': [self.default_ipv6],
                    'txt': [f"ipfs={ipfs_cid}"],
                    'cname': [],
                    'mx': []
                }
            else:
                # Domain not found
                return {}
        
        # Update cache
        self.cache[domain] = records
        self.cache_timestamp[domain] = datetime.now()
        
        return records
    
    def resolve(self, request, handler):
        """
        Resolve DNS query.
        
        Args:
            request: DNS request packet
            handler: DNS handler instance
        
        Returns:
            DNS response packet
        """
        reply = request.reply()
        qname = request.q.qname
        qtype = QTYPE[request.q.qtype]
        
        domain = str(qname).rstrip('.')
        
        logger.debug(f"DNS query: {domain} ({qtype})")
        
        # Only handle .bz domains
        if not domain.endswith('.bz'):
            logger.debug(f"Non-.bz domain: {domain}")
            reply.header.rcode = 3  # NXDOMAIN
            return reply
        
        # Get records from blockchain (synchronous wrapper for async)
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            records = loop.run_until_complete(self._get_domain_records(domain))
            loop.close()
        except Exception as e:
            logger.error(f"Error resolving {domain}: {e}")
            reply.header.rcode = 2  # SERVFAIL
            return reply
        
        if not records:
            logger.debug(f"Domain not found: {domain}")
            reply.header.rcode = 3  # NXDOMAIN
            return reply
        
        # Build response based on query type
        if qtype == 'A' and records.get('a'):
            for ip in records['a']:
                reply.add_answer(RR(
                    rname=qname,
                    rtype=QTYPE.A,
                    rclass=CLASS.IN,
                    ttl=self.ttl,
                    rdata=A(ip)
                ))
        
        elif qtype == 'AAAA' and records.get('aaaa'):
            for ip in records['aaaa']:
                reply.add_answer(RR(
                    rname=qname,
                    rtype=QTYPE.AAAA,
                    rclass=CLASS.IN,
                    ttl=self.ttl,
                    rdata=AAAA(ip)
                ))
        
        elif qtype == 'TXT' and records.get('txt'):
            for txt in records['txt']:
                reply.add_answer(RR(
                    rname=qname,
                    rtype=QTYPE.TXT,
                    rclass=CLASS.IN,
                    ttl=self.ttl,
                    rdata=TXT(txt)
                ))
        
        elif qtype == 'CNAME' and records.get('cname'):
            for cname in records['cname']:
                reply.add_answer(RR(
                    rname=qname,
                    rtype=QTYPE.CNAME,
                    rclass=CLASS.IN,
                    ttl=self.ttl,
                    rdata=CNAME(cname)
                ))
        
        elif qtype == 'MX' and records.get('mx'):
            for mx_record in records['mx']:
                # MX format: "priority hostname"
                parts = mx_record.split()
                if len(parts) == 2:
                    priority, hostname = int(parts[0]), parts[1]
                    reply.add_answer(RR(
                        rname=qname,
                        rtype=QTYPE.MX,
                        rclass=CLASS.IN,
                        ttl=self.ttl,
                        rdata=MX(hostname, preference=priority)
                    ))
        
        elif qtype == 'ANY':
            # Return all records
            for ip in records.get('a', []):
                reply.add_answer(RR(qname, QTYPE.A, CLASS.IN, self.ttl, A(ip)))
            for ip in records.get('aaaa', []):
                reply.add_answer(RR(qname, QTYPE.AAAA, CLASS.IN, self.ttl, AAAA(ip)))
            for txt in records.get('txt', []):
                reply.add_answer(RR(qname, QTYPE.TXT, CLASS.IN, self.ttl, TXT(txt)))
        
        # If no answers but domain exists, return SOA (Start of Authority)
        if len(reply.rr) == 0:
            reply.header.rcode = 0  # NOERROR (but no data)
        
        logger.info(f"Resolved {domain} ({qtype}): {len(reply.rr)} answer(s)")
        
        return reply


class BNSDNSServer:
    """
    Authoritative DNS server for .bz domains.
    
    Runs on UDP port 53 (requires sudo/root).
    Queries blockchain for domain resolution.
    """
    
    def __init__(
        self,
        website_manager,
        host: str = "127.0.0.1",
        port: int = 5353,  # Use 5353 for testing (non-root), 53 for production
        default_ip: str = "127.0.0.1"
    ):
        """
        Initialize DNS server.
        
        Args:
            website_manager: WebsiteManager instance
            host: Listen address (use 0.0.0.0 for Docker/cloud, set via PAKIT_DNS_HOST env var)
            port: Listen port (53 for production, 5353 for testing)
            default_ip: Default IP for hosted domains
        """
        self.manager = website_manager
        self.host = host
        self.port = port
        self.default_ip = default_ip
        
        # Create resolver
        self.resolver = BNSResolver(
            website_manager=website_manager,
            default_ip=default_ip
        )
        
        # Create DNS server instances (UDP and TCP)
        self.udp_server = None
        self.tcp_server = None
        
        logger.info(f"BNS DNS Server initialized on {host}:{port}")
    
    async def start(self):
        """Start DNS server (UDP and TCP)."""
        try:
            # UDP server (primary)
            self.udp_server = DNSServer(
                self.resolver,
                address=self.host,
                port=self.port
            )
            self.udp_server.start_thread()
            
            # TCP server (fallback for large responses)
            self.tcp_server = DNSServer(
                self.resolver,
                address=self.host,
                port=self.port,
                tcp=True
            )
            self.tcp_server.start_thread()
            
            logger.info(f"🌐 DNS Server running on {self.host}:{self.port} (UDP + TCP)")
            logger.info(f"   Authoritative for: *.bz")
            logger.info(f"   Default IP: {self.default_ip}")
            logger.info(f"\nTest with: dig @{self.host} -p {self.port} example.bz")
            
        except PermissionError:
            logger.error(
                f"Permission denied for port {self.port}. "
                f"Use port 5353 for testing or run with sudo for port 53."
            )
            raise
        except Exception as e:
            logger.error(f"Failed to start DNS server: {e}")
            raise
    
    async def stop(self):
        """Stop DNS server."""
        if self.udp_server:
            self.udp_server.stop()
        if self.tcp_server:
            self.tcp_server.stop()
        logger.info("DNS server stopped")
    
    def clear_cache(self):
        """Clear DNS cache."""
        self.resolver.cache.clear()
        self.resolver.cache_timestamp.clear()
        logger.info("DNS cache cleared")


async def run_dns_server(
    website_manager,
    host: str = "127.0.0.1",
    port: int = 5353,
    default_ip: str = "127.0.0.1"
):
    """
    Run DNS server (convenience function).
    
    Args:
        website_manager: WebsiteManager instance
        host: Listen address (use 0.0.0.0 for Docker/cloud, set via PAKIT_DNS_HOST env var)
        port: Listen port
        default_ip: Default IP for hosted domains
    """
    server = BNSDNSServer(website_manager, host, port, default_ip)
    await server.start()
    
    try:
        # Keep running until interrupted
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        logger.info("Shutting down DNS server...")
    finally:
        await server.stop()
