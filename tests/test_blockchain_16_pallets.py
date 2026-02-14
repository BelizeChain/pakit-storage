"""
Integration tests for 16-pallet blockchain integrations.

Tests Economy, BNS, Contracts, and Mesh pallet connectors.
"""

import pytest
from decimal import Decimal

from blockchain.storage_proof_connector import StorageProofConnector
from blockchain.economy_integration import EconomyPalletConnector
from blockchain.bns_integration import BNSPalletConnector
from blockchain.contracts_integration import ContractsPalletConnector


class TestEconomyPalletIntegration:
    """Test Economy pallet connector."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.connector = EconomyPalletConnector(substrate_interface=None)
    
    @pytest.mark.asyncio
    async def test_register_storage_provider(self):
        """Test registering as storage provider."""
        success = await self.connector.register_storage_provider(
            peer_id="test_storage_peer",
            storage_capacity_gb=1000,
            price_per_gb_dalla=100_000_000_000,  # 0.1 DALLA per GB
            price_per_gb_bbzd=50_000_000_000    # 0.05 bBZD per GB
        )
        
        assert success is True
        assert self.connector.provider_registered is True
    
    @pytest.mark.asyncio
    async def test_update_pricing(self):
        """Test updating storage pricing."""
        # Register first
        await self.connector.register_storage_provider(
            peer_id="pricing_test_peer",
            storage_capacity_gb=500,
            price_per_gb_dalla=100_000_000_000
        )
        
        # Update pricing
        success = await self.connector.update_pricing(
            price_per_gb_dalla=150_000_000_000,  # 0.15 DALLA
            price_per_gb_bbzd=75_000_000_000     # 0.075 bBZD
        )
        
        assert success is True
        
        # Verify new pricing
        pricing = self.connector.get_pricing()
        assert pricing["dalla_per_gb_month"] == 0.15
        assert pricing["bbzd_per_gb_month"] == 0.075
    
    @pytest.mark.asyncio
    async def test_create_payment_invoice(self):
        """Test creating payment invoice."""
        # Register provider
        await self.connector.register_storage_provider(
            peer_id="invoice_test_peer",
            storage_capacity_gb=1000,
            price_per_gb_dalla=100_000_000_000
        )
        
        # Create invoice
        invoice = await self.connector.create_payment_invoice(
            customer_account="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
            storage_gb_months=Decimal("10.5"),  # 10.5 GB-months
            currency="DALLA"
        )
        
        assert invoice is not None
        assert invoice["amount"] == 1_050_000_000_000  # 10.5 * 100B
        assert invoice["currency"] == "DALLA"
        assert invoice["storage_gb_months"] == 10.5
    
    @pytest.mark.asyncio
    async def test_verify_payment(self):
        """Test payment verification."""
        tx_hash = "0x123456789abcdef"
        
        # Verify payment
        is_valid = await self.connector.verify_payment(tx_hash)
        
        # In mock mode, always returns True
        assert is_valid is True
    
    def test_get_pricing(self):
        """Test getting current pricing."""
        pricing = self.connector.get_pricing()
        
        assert "dalla_per_gb_month" in pricing
        assert "bbzd_per_gb_month" in pricing
        assert "registered" in pricing


class TestBNSPalletIntegration:
    """Test BNS pallet connector."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.connector = BNSPalletConnector(substrate_interface=None)
    
    @pytest.mark.asyncio
    async def test_register_domain_hosting(self):
        """Test registering .bz domain hosting."""
        success = await self.connector.register_domain_hosting(
            domain_name="example.bz",
            ipfs_cid="QmExampleCID123456789",
            hosting_account="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
        )
        
        assert success is True
        assert "example.bz" in self.connector.hosted_domains
    
    @pytest.mark.asyncio
    async def test_register_invalid_domain(self):
        """Test that non-.bz domains are rejected."""
        success = await self.connector.register_domain_hosting(
            domain_name="example.com",  # Not .bz
            ipfs_cid="QmExampleCID",
            hosting_account="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
        )
        
        assert success is False
    
    @pytest.mark.asyncio
    async def test_update_domain_content(self):
        """Test updating domain content."""
        # Register domain first
        await self.connector.register_domain_hosting(
            domain_name="update.bz",
            ipfs_cid="QmOldCID",
            hosting_account="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
        )
        
        # Update content
        success = await self.connector.update_domain_content(
            domain_name="update.bz",
            new_ipfs_cid="QmNewCID"
        )
        
        assert success is True
        assert self.connector.hosted_domains["update.bz"] == "QmNewCID"
    
    @pytest.mark.asyncio
    async def test_get_domain_content(self):
        """Test retrieving domain content."""
        # Register domain
        await self.connector.register_domain_hosting(
            domain_name="get.bz",
            ipfs_cid="QmGetCID",
            hosting_account="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
        )
        
        # Get content
        cid = await self.connector.get_domain_content("get.bz")
        
        assert cid == "QmGetCID"
    
    @pytest.mark.asyncio
    async def test_list_hosted_domains(self):
        """Test listing all hosted domains."""
        # Register multiple domains
        domains = ["test1.bz", "test2.bz", "test3.bz"]
        for i, domain in enumerate(domains):
            await self.connector.register_domain_hosting(
                domain_name=domain,
                ipfs_cid=f"QmCID{i}",
                hosting_account="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
            )
        
        # List domains
        hosted = await self.connector.list_hosted_domains()
        
        assert len(hosted) == 3
        assert all("domain" in d and "ipfs_cid" in d for d in hosted)
    
    @pytest.mark.asyncio
    async def test_unregister_domain(self):
        """Test unregistering domain."""
        # Register domain
        await self.connector.register_domain_hosting(
            domain_name="unregister.bz",
            ipfs_cid="QmUnregisterCID",
            hosting_account="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
        )
        
        # Unregister
        success = await self.connector.unregister_domain("unregister.bz")
        
        assert success is True
        assert "unregister.bz" not in self.connector.hosted_domains
    
    def test_get_hosting_stats(self):
        """Test getting hosting statistics."""
        stats = self.connector.get_hosting_stats()
        
        assert "domains_hosted" in stats
        assert "domains" in stats


class TestContractsPalletIntegration:
    """Test Contracts pallet connector."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.connector = ContractsPalletConnector(substrate_interface=None)
    
    @pytest.mark.asyncio
    async def test_store_contract_data(self):
        """Test storing smart contract data."""
        success = await self.connector.store_contract_data(
            contract_address="5ContractAddress123456789ABCDEF",
            data_cid="QmContractDataCID",
            data_type="code",
            size_bytes=1024
        )
        
        assert success is True
        assert "5ContractAddress123456789ABCDEF" in self.connector.contract_storage
    
    @pytest.mark.asyncio
    async def test_store_multiple_data_types(self):
        """Test storing multiple data types for same contract."""
        contract_addr = "5MultiTypeContract123"
        
        # Store code
        await self.connector.store_contract_data(
            contract_address=contract_addr,
            data_cid="QmCodeCID",
            data_type="code",
            size_bytes=2048
        )
        
        # Store state
        await self.connector.store_contract_data(
            contract_address=contract_addr,
            data_cid="QmStateCID",
            data_type="state",
            size_bytes=512
        )
        
        # Store metadata
        await self.connector.store_contract_data(
            contract_address=contract_addr,
            data_cid="QmMetadataCID",
            data_type="metadata",
            size_bytes=256
        )
        
        # Verify total size
        storage_info = self.connector.contract_storage[contract_addr]
        assert storage_info["total_size"] == 2048 + 512 + 256
        assert len(storage_info["data"]) == 3
    
    @pytest.mark.asyncio
    async def test_get_contract_data(self):
        """Test retrieving contract data."""
        contract_addr = "5GetContractData"
        
        # Store data
        await self.connector.store_contract_data(
            contract_address=contract_addr,
            data_cid="QmGetCID",
            data_type="code",
            size_bytes=1024
        )
        
        # Retrieve data
        data_info = await self.connector.get_contract_data(
            contract_address=contract_addr,
            data_type="code"
        )
        
        assert data_info is not None
        assert data_info["cid"] == "QmGetCID"
        assert data_info["size"] == 1024
    
    @pytest.mark.asyncio
    async def test_verify_contract_access(self):
        """Test contract access verification."""
        is_allowed = await self.connector.verify_contract_access(
            contract_address="5ContractAccess",
            caller_address="5CallerAddress",
            operation="read"
        )
        
        # In mock mode, access always allowed
        assert is_allowed is True
    
    @pytest.mark.asyncio
    async def test_track_storage_usage(self):
        """Test tracking storage usage."""
        contract_addr = "5StorageUsage"
        
        # Store data
        await self.connector.store_contract_data(
            contract_address=contract_addr,
            data_cid="QmUsageCID",
            data_type="state",
            size_bytes=4096
        )
        
        # Track usage
        usage = await self.connector.track_storage_usage(contract_addr)
        
        assert usage["total_size"] == 4096
        assert "state" in usage["data_types"]
    
    @pytest.mark.asyncio
    async def test_cleanup_contract_data(self):
        """Test cleaning up contract data."""
        contract_addr = "5CleanupContract"
        
        # Store multiple data types
        await self.connector.store_contract_data(
            contract_address=contract_addr,
            data_cid="QmCodeCID",
            data_type="code",
            size_bytes=1000
        )
        await self.connector.store_contract_data(
            contract_address=contract_addr,
            data_cid="QmStateCID",
            data_type="state",
            size_bytes=500
        )
        
        # Cleanup specific type
        success = await self.connector.cleanup_contract_data(
            contract_address=contract_addr,
            data_type="state"
        )
        
        assert success is True
        
        # Verify state removed but code remains
        storage_info = self.connector.contract_storage[contract_addr]
        assert "code" in storage_info["data"]
        assert "state" not in storage_info["data"]
        assert storage_info["total_size"] == 1000
    
    def test_get_storage_stats(self):
        """Test getting overall storage statistics."""
        stats = self.connector.get_storage_stats()
        
        assert "contracts_count" in stats
        assert "total_storage_bytes" in stats
        assert "contracts" in stats


@pytest.mark.asyncio
class TestStorageProofConnectorMultiPallet:
    """Test multi-pallet support in StorageProofConnector."""
    
    async def test_connector_initializes_all_pallets(self):
        """Test that connector initializes all pallet connectors."""
        connector = StorageProofConnector(mock_mode=True)
        await connector.connect()
        
        # Check all pallet connectors initialized
        assert connector.economy is not None
        assert connector.bns is not None
        assert connector.contracts is not None
        
        await connector.disconnect()
    
    async def test_get_economy_connector(self):
        """Test getting Economy pallet connector."""
        connector = StorageProofConnector(mock_mode=True)
        await connector.connect()
        
        economy = connector.get_economy_connector()
        
        assert economy is not None
        assert isinstance(economy, EconomyPalletConnector)
        
        await connector.disconnect()
    
    async def test_get_bns_connector(self):
        """Test getting BNS pallet connector."""
        connector = StorageProofConnector(mock_mode=True)
        await connector.connect()
        
        bns = connector.get_bns_connector()
        
        assert bns is not None
        assert isinstance(bns, BNSPalletConnector)
        
        await connector.disconnect()
    
    async def test_get_contracts_connector(self):
        """Test getting Contracts pallet connector."""
        connector = StorageProofConnector(mock_mode=True)
        await connector.connect()
        
        contracts = connector.get_contracts_connector()
        
        assert contracts is not None
        assert isinstance(contracts, ContractsPalletConnector)
        
        await connector.disconnect()
    
    async def test_submit_storage_zk_proof(self):
        """Test submitting ZK proof to Mesh pallet."""
        connector = StorageProofConnector(mock_mode=True)
        await connector.connect()
        
        zk_proof = {
            "proof": "mock_proof_data",
            "type": "groth16"
        }
        
        success = await connector.submit_storage_zk_proof(
            content_id="test_content_hash",
            zk_proof=zk_proof,
            proof_type="groth16"
        )
        
        assert success is True
        
        await connector.disconnect()
    
    async def test_get_multi_pallet_status(self):
        """Test getting status from all pallets."""
        connector = StorageProofConnector(mock_mode=True)
        await connector.connect()
        
        status = await connector.get_multi_pallet_status()
        
        assert "connected" in status
        assert "mock_mode" in status
        assert "pallets" in status
        assert "economy" in status["pallets"]
        assert "bns" in status["pallets"]
        assert "contracts" in status["pallets"]
        
        await connector.disconnect()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
