"""
DNS Verification for External Domains.

Verifies ownership of external domains via TXT records.
"""

import logging
import hashlib
import secrets
from typing import Optional, Tuple
from datetime import datetime, timedelta
import dns.resolver
import dns.exception

logger = logging.getLogger(__name__)


class DNSVerifier:
    """
    DNS-based verification for external domain linking.
    
    Verification Process:
    1. User requests to link external domain (e.g., example.com)
    2. System generates verification token
    3. User adds TXT record: _belizechain-verify.example.com = "token"
    4. System queries DNS to verify TXT record
    5. On success, domain is linked to BelizeID
    
    Security:
    - Tokens are cryptographically random (32 bytes)
    - Tokens expire after 48 hours
    - Verification can be re-attempted
    """
    
    def __init__(
        self,
        token_expiry_hours: int = 48,
        dns_timeout: int = 10
    ):
        """
        Initialize DNS verifier.
        
        Args:
            token_expiry_hours: Hours before verification token expires
            dns_timeout: Timeout for DNS queries (seconds)
        """
        self.token_expiry = timedelta(hours=token_expiry_hours)
        self.dns_timeout = dns_timeout
        
        # Pending verifications: domain -> (token, created_at, account_id)
        self.pending_verifications: dict[str, Tuple[str, datetime, str]] = {}
        
        logger.info("DNSVerifier initialized")
    
    def generate_verification_token(self, domain: str, account_id: str) -> str:
        """
        Generate verification token for external domain.
        
        Args:
            domain: External domain (e.g., "example.com")
            account_id: BelizeChain account ID
        
        Returns:
            Verification token to add to DNS TXT record
        """
        # Generate cryptographically secure random token
        random_bytes = secrets.token_bytes(32)
        
        # Include domain and account in hash for uniqueness
        data = f"{domain}:{account_id}:{random_bytes.hex()}".encode('utf-8')
        token = hashlib.sha256(data).hexdigest()
        
        # Store pending verification
        self.pending_verifications[domain] = (token, datetime.now(), account_id)
        
        logger.info(f"Generated verification token for {domain}")
        
        return token
    
    def get_verification_instructions(self, domain: str) -> Optional[str]:
        """
        Get DNS verification instructions for domain.
        
        Args:
            domain: Domain to verify
        
        Returns:
            Instructions string or None if no pending verification
        """
        if domain not in self.pending_verifications:
            return None
        
        token, created_at, account_id = self.pending_verifications[domain]
        
        # Check if expired
        if datetime.now() - created_at > self.token_expiry:
            logger.warning(f"Verification token for {domain} has expired")
            del self.pending_verifications[domain]
            return None
        
        return f"""
DNS Verification Instructions for {domain}
=============================================

To verify ownership of {domain}, add the following TXT record to your DNS:

Record Type: TXT
Name: _belizechain-verify.{domain}
Value: {token}

Example DNS configuration:
--------------------------
_belizechain-verify.{domain}.  IN  TXT  "{token}"

After adding the record:
1. Wait 5-10 minutes for DNS propagation
2. Click "Verify" in your wallet to complete verification
3. Token expires in {int(self.token_expiry.total_seconds() / 3600)} hours

Account: {account_id}
Generated: {created_at.strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    async def verify_domain(self, domain: str) -> Tuple[bool, str]:
        """
        Verify external domain via DNS TXT record.
        
        Args:
            domain: Domain to verify
        
        Returns:
            Tuple of (success: bool, message: str)
        """
        if domain not in self.pending_verifications:
            return False, "No pending verification for this domain"
        
        expected_token, created_at, account_id = self.pending_verifications[domain]
        
        # Check if expired
        if datetime.now() - created_at > self.token_expiry:
            logger.warning(f"Verification token for {domain} has expired")
            del self.pending_verifications[domain]
            return False, "Verification token has expired. Please generate a new one."
        
        # Query DNS for TXT record
        txt_record_name = f"_belizechain-verify.{domain}"
        
        try:
            logger.info(f"Querying DNS for {txt_record_name}")
            
            # Query TXT records
            resolver = dns.resolver.Resolver()
            resolver.timeout = self.dns_timeout
            resolver.lifetime = self.dns_timeout
            
            answers = resolver.resolve(txt_record_name, 'TXT')
            
            # Check if any TXT record matches our token
            for rdata in answers:
                # TXT records are quoted strings, strip quotes
                txt_value = rdata.to_text().strip('"')
                
                if txt_value == expected_token:
                    logger.info(f"✅ Successfully verified {domain}")
                    
                    # Remove from pending (verification complete)
                    del self.pending_verifications[domain]
                    
                    return True, f"Successfully verified ownership of {domain}"
            
            # Token found but didn't match
            logger.warning(f"TXT record found for {domain} but token doesn't match")
            return False, "TXT record found but verification token doesn't match. Please check your DNS configuration."
            
        except dns.resolver.NXDOMAIN:
            logger.warning(f"DNS record not found: {txt_record_name}")
            return False, f"DNS record not found. Please add TXT record: _belizechain-verify.{domain}"
            
        except dns.resolver.NoAnswer:
            logger.warning(f"No TXT records found for {txt_record_name}")
            return False, "No TXT records found. Please add the verification TXT record."
            
        except dns.resolver.Timeout:
            logger.error(f"DNS query timeout for {txt_record_name}")
            return False, "DNS query timed out. Please try again later."
            
        except dns.exception.DNSException as e:
            logger.error(f"DNS error for {txt_record_name}: {e}")
            return False, f"DNS error: {str(e)}"
            
        except Exception as e:
            logger.error(f"Unexpected error verifying {domain}: {e}")
            return False, f"Verification error: {str(e)}"
    
    def cancel_verification(self, domain: str) -> bool:
        """
        Cancel pending verification.
        
        Args:
            domain: Domain to cancel
        
        Returns:
            True if verification was cancelled
        """
        if domain in self.pending_verifications:
            del self.pending_verifications[domain]
            logger.info(f"Cancelled verification for {domain}")
            return True
        return False
    
    def cleanup_expired_verifications(self):
        """Remove expired verification tokens."""
        now = datetime.now()
        expired = [
            domain for domain, (token, created_at, account_id) in self.pending_verifications.items()
            if now - created_at > self.token_expiry
        ]
        
        for domain in expired:
            del self.pending_verifications[domain]
        
        if expired:
            logger.info(f"Cleaned up {len(expired)} expired verification(s)")
    
    def get_pending_verifications(self) -> list[dict]:
        """
        Get all pending verifications.
        
        Returns:
            List of verification info dictionaries
        """
        result = []
        for domain, (token, created_at, account_id) in self.pending_verifications.items():
            result.append({
                'domain': domain,
                'account_id': account_id,
                'created_at': created_at.isoformat(),
                'expires_at': (created_at + self.token_expiry).isoformat(),
                'expired': datetime.now() - created_at > self.token_expiry
            })
        return result
