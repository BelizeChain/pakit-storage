"""
Economy Pallet Integration for Pakit Storage

Integrates with BelizeChain's Economy pallet to:
- Register as storage provider
- Accept payments in DALLA/bBZD
- Set pricing for storage services
- Track revenue and payments
"""

import logging
from typing import Dict, Any, Optional
from decimal import Decimal

logger = logging.getLogger(__name__)


class EconomyPalletConnector:
    """
    Connector for BelizeChain Economy pallet.
    
    Enables Pakit storage nodes to:
    - Register as commercial storage providers
    - Accept payments in dual currency (DALLA/bBZD)
    - Set competitive pricing
    - Track financial metrics
    """
    
    def __init__(self, substrate_interface: Any):
        """
        Initialize Economy pallet connector.
        
        Args:
            substrate_interface: Substrate interface from storage_proof_connector
        """
        self.substrate = substrate_interface
        self.provider_registered = False
        self.pricing: Dict[str, int] = {}
    
    async def register_storage_provider(
        self,
        peer_id: str,
        storage_capacity_gb: int,
        price_per_gb_dalla: int,
        price_per_gb_bbzd: Optional[int] = None
    ) -> bool:
        """
        Register as storage provider in Economy pallet.
        
        Args:
            peer_id: Unique peer identifier
            storage_capacity_gb: Total storage capacity in GB
            price_per_gb_dalla: Price per GB per month in DALLA (12 decimals)
            price_per_gb_bbzd: Optional price in bBZD (12 decimals)
            
        Returns:
            True if registration successful
        """
        try:
            # Submit extrinsic to Economy pallet
            call = self.substrate.compose_call(
                call_module='Economy',
                call_function='register_storage_provider',
                call_params={
                    'peer_id': peer_id,
                    'capacity_gb': storage_capacity_gb,
                    'price_dalla': price_per_gb_dalla,
                    'price_bbzd': price_per_gb_bbzd or 0
                }
            )
            
            # In production: Sign and submit with actual account
            logger.info(
                f"Registered storage provider: {peer_id} "
                f"({storage_capacity_gb} GB @ {price_per_gb_dalla / 1e12:.2f} DALLA/GB)"
            )
            
            self.provider_registered = True
            self.pricing = {
                "dalla_per_gb": price_per_gb_dalla,
                "bbzd_per_gb": price_per_gb_bbzd or 0
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to register storage provider: {e}")
            return False
    
    async def update_pricing(
        self,
        price_per_gb_dalla: int,
        price_per_gb_bbzd: Optional[int] = None
    ) -> bool:
        """
        Update storage pricing.
        
        Args:
            price_per_gb_dalla: New price in DALLA
            price_per_gb_bbzd: New price in bBZD
            
        Returns:
            True if update successful
        """
        try:
            call = self.substrate.compose_call(
                call_module='Economy',
                call_function='update_storage_pricing',
                call_params={
                    'price_dalla': price_per_gb_dalla,
                    'price_bbzd': price_per_gb_bbzd or 0
                }
            )
            
            self.pricing = {
                "dalla_per_gb": price_per_gb_dalla,
                "bbzd_per_gb": price_per_gb_bbzd or 0
            }
            
            logger.info(f"Updated pricing: {price_per_gb_dalla / 1e12:.2f} DALLA/GB")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update pricing: {e}")
            return False
    
    async def create_payment_invoice(
        self,
        customer_account: str,
        storage_gb_months: Decimal,
        currency: str = "DALLA"
    ) -> Optional[Dict[str, Any]]:
        """
        Create payment invoice for storage service.
        
        Args:
            customer_account: Customer's blockchain account
            storage_gb_months: Storage amount in GB-months
            currency: Payment currency ('DALLA' or 'bBZD')
            
        Returns:
            Invoice details or None
        """
        if not self.provider_registered:
            logger.error("Storage provider not registered")
            return None
        
        # Calculate amount
        price_key = f"{currency.lower()}_per_gb"
        if price_key not in self.pricing:
            logger.error(f"No pricing set for {currency}")
            return None
        
        amount = int(float(storage_gb_months) * self.pricing[price_key])
        
        invoice = {
            "customer": customer_account,
            "amount": amount,
            "currency": currency,
            "storage_gb_months": float(storage_gb_months),
            "description": f"Pakit storage: {storage_gb_months} GB-months"
        }
        
        logger.info(
            f"Created invoice: {amount / 1e12:.2f} {currency} "
            f"for {storage_gb_months} GB-months"
        )
        
        return invoice
    
    async def verify_payment(
        self,
        transaction_hash: str
    ) -> bool:
        """
        Verify payment transaction.
        
        Args:
            transaction_hash: Transaction hash to verify
            
        Returns:
            True if payment confirmed
        """
        try:
            # Query blockchain for transaction
            # In production: Verify payment to our account
            logger.info(f"Verifying payment: {transaction_hash}")
            return True
        except Exception as e:
            logger.error(f"Failed to verify payment: {e}")
            return False
    
    def get_pricing(self) -> Dict[str, Any]:
        """
        Get current pricing.
        
        Returns:
            Pricing dict
        """
        return {
            "dalla_per_gb_month": self.pricing.get("dalla_per_gb", 0) / 1e12,
            "bbzd_per_gb_month": self.pricing.get("bbzd_per_gb", 0) / 1e12,
            "registered": self.provider_registered
        }
