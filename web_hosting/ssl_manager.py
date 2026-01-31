"""
SSL/TLS Certificate Manager for BelizeChain BNS
Handles Let's Encrypt certificate issuance, renewal, and storage.
"""

import asyncio
import logging
import os
import json
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Tuple
from pathlib import Path
import hashlib

try:
    from acme import client as acme_client
    from acme import messages
    from acme import challenges
    from cryptography import x509
    from cryptography.x509.oid import NameOID
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.hazmat.backends import default_backend
    from OpenSSL import crypto
except ImportError:
    print("WARNING: SSL dependencies not installed. Run: pip install acme cryptography pyOpenSSL")

from pakit.web_hosting.website_manager import WebsiteManager

logger = logging.getLogger(__name__)


class SSLCertificateManager:
    """
    Manages SSL/TLS certificates for .bz domains using Let's Encrypt.
    
    Features:
    - Automatic certificate issuance via ACME protocol
    - HTTP-01 challenge verification
    - Certificate renewal (90 days → renew at 60 days)
    - On-chain certificate storage (hash only, full cert local)
    - Multi-domain certificates (SAN support)
    """
    
    # Let's Encrypt ACME endpoints
    STAGING_URL = "https://acme-staging-v02.api.letsencrypt.org/directory"
    PRODUCTION_URL = "https://acme-v02.api.letsencrypt.org/directory"
    
    def __init__(
        self,
        website_manager: WebsiteManager,
        cert_storage_dir: str = "/etc/belizechain/ssl",
        use_staging: bool = True,
        contact_email: str = "admin@belizechain.org"
    ):
        """
        Initialize SSL certificate manager.
        
        Args:
            website_manager: WebsiteManager instance for blockchain integration
            cert_storage_dir: Local directory for certificate storage
            use_staging: Use Let's Encrypt staging (testing) or production
            contact_email: Admin email for Let's Encrypt notifications
        """
        self.website_manager = website_manager
        self.cert_storage_dir = Path(cert_storage_dir)
        self.cert_storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.acme_url = self.STAGING_URL if use_staging else self.PRODUCTION_URL
        self.contact_email = contact_email
        
        # ACME client (initialized on first use)
        self.acme_client: Optional[acme_client.ClientV2] = None
        self.account_key: Optional[rsa.RSAPrivateKey] = None
        
        # Certificate cache (domain -> cert info)
        self.cert_cache: Dict[str, Dict] = {}
        
        logger.info(f"SSL Manager initialized (mode: {'STAGING' if use_staging else 'PRODUCTION'})")
    
    
    async def initialize_acme_client(self):
        """Initialize ACME client with account key."""
        try:
            # Load or create account private key
            account_key_path = self.cert_storage_dir / "account.key"
            
            if account_key_path.exists():
                logger.info("Loading existing ACME account key")
                with open(account_key_path, 'rb') as f:
                    self.account_key = serialization.load_pem_private_key(
                        f.read(),
                        password=None,
                        backend=default_backend()
                    )
            else:
                logger.info("Generating new ACME account key")
                self.account_key = rsa.generate_private_key(
                    public_exponent=65537,
                    key_size=2048,
                    backend=default_backend()
                )
                
                # Save account key
                with open(account_key_path, 'wb') as f:
                    f.write(self.account_key.private_bytes(
                        encoding=serialization.Encoding.PEM,
                        format=serialization.PrivateFormat.TraditionalOpenSSL,
                        encryption_algorithm=serialization.NoEncryption()
                    ))
                os.chmod(account_key_path, 0o600)
            
            # Create ACME client
            network = acme_client.ClientNetwork(self.account_key)
            directory = messages.Directory.from_json(
                network.get(self.acme_url).json()
            )
            self.acme_client = acme_client.ClientV2(directory, network)
            
            # Register account (idempotent)
            registration = messages.NewRegistration.from_data(
                email=self.contact_email,
                terms_of_service_agreed=True
            )
            await asyncio.to_thread(self.acme_client.new_account, registration)
            
            logger.info("ACME client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ACME client: {e}")
            raise
    
    
    async def issue_certificate(
        self,
        domain: str,
        subdomains: Optional[List[str]] = None,
        challenge_dir: str = "/var/www/html/.well-known/acme-challenge"
    ) -> Tuple[str, str, str]:
        """
        Issue a new SSL certificate for domain using Let's Encrypt.
        
        Args:
            domain: Primary domain (e.g., "example.bz")
            subdomains: Optional list of subdomains (e.g., ["www", "api"])
            challenge_dir: Directory for HTTP-01 challenge files
        
        Returns:
            Tuple of (cert_pem, privkey_pem, fullchain_pem)
        """
        try:
            # Initialize ACME client if not already
            if not self.acme_client:
                await self.initialize_acme_client()
            
            # Prepare domain list (primary + subdomains)
            domains = [domain]
            if subdomains:
                domains.extend([f"{sub}.{domain}" for sub in subdomains])
            
            logger.info(f"Issuing certificate for domains: {domains}")
            
            # Generate certificate private key
            cert_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048,
                backend=default_backend()
            )
            
            # Create Certificate Signing Request (CSR)
            csr = x509.CertificateSigningRequestBuilder().subject_name(
                x509.Name([
                    x509.NameAttribute(NameOID.COMMON_NAME, domains[0]),
                ])
            ).add_extension(
                x509.SubjectAlternativeName([
                    x509.DNSName(d) for d in domains
                ]),
                critical=False,
            ).sign(cert_key, hashes.SHA256(), default_backend())
            
            csr_pem = csr.public_bytes(serialization.Encoding.PEM)
            
            # Request certificate order
            order = await asyncio.to_thread(
                self.acme_client.new_order,
                csr_pem
            )
            
            # Complete challenges for all domains
            for authz in order.authorizations:
                await self._complete_http01_challenge(authz, challenge_dir)
            
            # Finalize order
            logger.info("Finalizing certificate order...")
            order = await asyncio.to_thread(
                self.acme_client.poll_and_finalize,
                order
            )
            
            # Get certificate
            cert_pem = order.fullchain_pem
            privkey_pem = cert_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption()
            ).decode('utf-8')
            
            # Save certificate locally
            await self._save_certificate(domain, cert_pem, privkey_pem)
            
            # Store certificate hash on blockchain
            await self._store_cert_hash_onchain(domain, cert_pem)
            
            logger.info(f"Certificate issued successfully for {domain}")
            return cert_pem, privkey_pem, cert_pem
            
        except Exception as e:
            logger.error(f"Certificate issuance failed for {domain}: {e}")
            raise
    
    
    async def _complete_http01_challenge(
        self,
        authz: messages.Authorization,
        challenge_dir: str
    ):
        """
        Complete HTTP-01 challenge for domain authorization.
        
        Args:
            authz: ACME authorization object
            challenge_dir: Directory to write challenge response
        """
        try:
            # Find HTTP-01 challenge
            challenge = None
            for chall in authz.body.challenges:
                if isinstance(chall.chall, challenges.HTTP01):
                    challenge = chall
                    break
            
            if not challenge:
                raise ValueError("No HTTP-01 challenge found")
            
            # Get challenge response
            response, validation = challenge.response_and_validation(
                self.acme_client.net.key
            )
            
            # Write challenge file
            challenge_path = Path(challenge_dir) / challenge.chall.encode("token")
            challenge_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(challenge_path, 'w') as f:
                f.write(validation)
            
            logger.info(f"Challenge file written: {challenge_path}")
            
            # Answer challenge
            await asyncio.to_thread(
                self.acme_client.answer_challenge,
                challenge,
                response
            )
            
            # Poll for validation
            logger.info("Waiting for challenge validation...")
            await asyncio.to_thread(
                self.acme_client.poll,
                authz
            )
            
            # Clean up challenge file
            challenge_path.unlink(missing_ok=True)
            logger.info("Challenge validated successfully")
            
        except Exception as e:
            logger.error(f"Challenge completion failed: {e}")
            raise
    
    
    async def _save_certificate(
        self,
        domain: str,
        cert_pem: str,
        privkey_pem: str
    ):
        """Save certificate and private key to local storage."""
        try:
            domain_dir = self.cert_storage_dir / domain
            domain_dir.mkdir(parents=True, exist_ok=True)
            
            # Save certificate
            cert_path = domain_dir / "fullchain.pem"
            with open(cert_path, 'w') as f:
                f.write(cert_pem)
            
            # Save private key
            key_path = domain_dir / "privkey.pem"
            with open(key_path, 'w') as f:
                f.write(privkey_pem)
            os.chmod(key_path, 0o600)
            
            # Save metadata
            cert_obj = x509.load_pem_x509_certificate(
                cert_pem.encode(),
                default_backend()
            )
            
            metadata = {
                "domain": domain,
                "issued_at": datetime.utcnow().isoformat(),
                "expires_at": cert_obj.not_valid_after.isoformat(),
                "serial_number": str(cert_obj.serial_number),
                "issuer": cert_obj.issuer.rfc4514_string(),
            }
            
            meta_path = domain_dir / "metadata.json"
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Update cache
            self.cert_cache[domain] = metadata
            
            logger.info(f"Certificate saved: {cert_path}")
            
        except Exception as e:
            logger.error(f"Failed to save certificate: {e}")
            raise
    
    
    async def _store_cert_hash_onchain(self, domain: str, cert_pem: str):
        """
        Store certificate hash on blockchain for verification.
        
        This allows anyone to verify the certificate is legitimate
        without storing the full cert on-chain.
        """
        try:
            # Calculate SHA-256 hash of certificate
            cert_hash = hashlib.sha256(cert_pem.encode()).digest()
            
            # Submit extrinsic to BNS pallet for SSL certificate verification
            if self.substrate and self.keypair:
                call = self.substrate.compose_call(
                    call_module='BNS',
                    call_function='update_ssl_certificate',
                    call_params={
                        'domain': domain,
                        'cert_hash': cert_hash.hex(),
                        'issued_at': int(datetime.utcnow().timestamp()),
                    }
                )
                
                extrinsic = self.substrate.create_signed_extrinsic(
                    call=call,
                    keypair=self.keypair
                )
                
                receipt = self.substrate.submit_extrinsic(
                    extrinsic,
                    wait_for_inclusion=True
                )
                
                if receipt.is_success:
                    logger.info(f"Certificate hash stored on-chain: {cert_hash.hex()[:16]}...")
                else:
                    logger.error(f"Failed to store certificate on-chain: {receipt.error_message}")
            else:
                logger.warning("No blockchain connection - skipping on-chain storage")
            
        except Exception as e:
            logger.error(f"Failed to store certificate hash on-chain: {e}")
            # Non-critical, don't raise
    
    
    async def renew_certificate(self, domain: str) -> bool:
        """
        Renew certificate if expiring within 30 days.
        
        Args:
            domain: Domain to renew certificate for
        
        Returns:
            True if renewed, False if not needed
        """
        try:
            # Check if certificate exists
            cert_path = self.cert_storage_dir / domain / "fullchain.pem"
            if not cert_path.exists():
                logger.warning(f"No certificate found for {domain}, issuing new one")
                await self.issue_certificate(domain)
                return True
            
            # Load certificate
            with open(cert_path, 'rb') as f:
                cert_pem = f.read()
            
            cert_obj = x509.load_pem_x509_certificate(cert_pem, default_backend())
            
            # Check expiration
            days_until_expiry = (cert_obj.not_valid_after - datetime.utcnow()).days
            
            if days_until_expiry > 30:
                logger.info(f"Certificate for {domain} valid for {days_until_expiry} days, no renewal needed")
                return False
            
            logger.info(f"Certificate for {domain} expires in {days_until_expiry} days, renewing...")
            
            # Issue new certificate
            await self.issue_certificate(domain)
            return True
            
        except Exception as e:
            logger.error(f"Certificate renewal failed for {domain}: {e}")
            raise
    
    
    async def auto_renew_all(self):
        """
        Auto-renew all certificates that are expiring soon.
        Run this as a cron job daily.
        """
        try:
            logger.info("Starting auto-renewal check for all certificates")
            
            renewed_count = 0
            failed_domains = []
            
            # Scan all domain directories
            for domain_dir in self.cert_storage_dir.iterdir():
                if not domain_dir.is_dir():
                    continue
                
                domain = domain_dir.name
                
                try:
                    renewed = await self.renew_certificate(domain)
                    if renewed:
                        renewed_count += 1
                except Exception as e:
                    logger.error(f"Failed to renew {domain}: {e}")
                    failed_domains.append(domain)
            
            logger.info(f"Auto-renewal complete: {renewed_count} renewed, {len(failed_domains)} failed")
            
            if failed_domains:
                logger.warning(f"Failed renewals: {failed_domains}")
            
        except Exception as e:
            logger.error(f"Auto-renewal process failed: {e}")
    
    
    def get_certificate_info(self, domain: str) -> Optional[Dict]:
        """
        Get certificate information for a domain.
        
        Args:
            domain: Domain to get info for
        
        Returns:
            Certificate metadata dict or None if not found
        """
        try:
            # Check cache first
            if domain in self.cert_cache:
                return self.cert_cache[domain]
            
            # Load from metadata file
            meta_path = self.cert_storage_dir / domain / "metadata.json"
            if not meta_path.exists():
                return None
            
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
            
            # Update cache
            self.cert_cache[domain] = metadata
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to get certificate info for {domain}: {e}")
            return None
    
    
    def get_certificate_paths(self, domain: str) -> Tuple[Path, Path]:
        """
        Get paths to certificate and private key files.
        
        Args:
            domain: Domain to get paths for
        
        Returns:
            Tuple of (cert_path, key_path)
        """
        domain_dir = self.cert_storage_dir / domain
        return (
            domain_dir / "fullchain.pem",
            domain_dir / "privkey.pem"
        )
    
    
    async def revoke_certificate(self, domain: str, reason: int = 0):
        """
        Revoke a certificate.
        
        Args:
            domain: Domain to revoke certificate for
            reason: Revocation reason code (0=unspecified, 1=key_compromise, etc.)
        """
        try:
            # Initialize ACME client if not already
            if not self.acme_client:
                await self.initialize_acme_client()
            
            # Load certificate
            cert_path = self.cert_storage_dir / domain / "fullchain.pem"
            with open(cert_path, 'rb') as f:
                cert_pem = f.read()
            
            cert_obj = crypto.load_certificate(crypto.FILETYPE_PEM, cert_pem)
            
            # Revoke certificate
            await asyncio.to_thread(
                self.acme_client.revoke,
                messages.Revocation(certificate=cert_obj, reason=reason)
            )
            
            logger.info(f"Certificate revoked for {domain}")
            
            # Remove local files
            (self.cert_storage_dir / domain).rmdir()
            
        except Exception as e:
            logger.error(f"Certificate revocation failed for {domain}: {e}")
            raise


# ===== CLI INTERFACE =====

async def main():
    """Command-line interface for SSL certificate management."""
    import argparse
    
    parser = argparse.ArgumentParser(description="BelizeChain SSL Certificate Manager")
    parser.add_argument("command", choices=["issue", "renew", "auto-renew", "info", "revoke"])
    parser.add_argument("--domain", help="Domain name (e.g., example.bz)")
    parser.add_argument("--subdomains", nargs="+", help="Subdomains to include in certificate")
    parser.add_argument("--staging", action="store_true", help="Use Let's Encrypt staging environment")
    parser.add_argument("--email", default="admin@belizechain.org", help="Contact email")
    parser.add_argument("--cert-dir", default="/etc/belizechain/ssl", help="Certificate storage directory")
    
    args = parser.parse_args()
    
    # Initialize manager with blockchain connection
    # In production, WebsiteManager should be initialized with SubstrateInterface
    from pakit.web_hosting.website_manager import WebsiteManager
    website_manager = WebsiteManager(
        node_url=os.getenv("BELIZECHAIN_NODE_URL", "ws://localhost:9944"),
        ipfs_api=os.getenv("IPFS_API", "http://localhost:5001"),
    )
    
    manager = SSLCertificateManager(
        website_manager=website_manager,
        cert_storage_dir=args.cert_dir,
        use_staging=args.staging,
        contact_email=args.email
    )
    
    # Execute command
    if args.command == "issue":
        if not args.domain:
            print("Error: --domain required for issue command")
            return
        
        cert_pem, privkey_pem, fullchain_pem = await manager.issue_certificate(
            args.domain,
            subdomains=args.subdomains
        )
        print(f"Certificate issued for {args.domain}")
        print(f"Saved to: {manager.cert_storage_dir / args.domain}")
    
    elif args.command == "renew":
        if not args.domain:
            print("Error: --domain required for renew command")
            return
        
        renewed = await manager.renew_certificate(args.domain)
        print(f"Certificate {'renewed' if renewed else 'does not need renewal'}")
    
    elif args.command == "auto-renew":
        await manager.auto_renew_all()
        print("Auto-renewal complete")
    
    elif args.command == "info":
        if not args.domain:
            print("Error: --domain required for info command")
            return
        
        info = manager.get_certificate_info(args.domain)
        if info:
            print(json.dumps(info, indent=2))
        else:
            print(f"No certificate found for {args.domain}")
    
    elif args.command == "revoke":
        if not args.domain:
            print("Error: --domain required for revoke command")
            return
        
        await manager.revoke_certificate(args.domain)
        print(f"Certificate revoked for {args.domain}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
