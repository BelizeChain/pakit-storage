"""
Version Manager

Manages model versions and deployment history.
"""

from typing import Dict, Any, List, Optional
import json
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class VersionManager:
    """
    Manages model versions.
    
    Tracks:
    - Version history
    - Deployment timestamps
    - Model artifacts
    - Rollback capability
    """
    
    def __init__(self, base_dir: str = "./models"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Version registry: {model_name: [version_info]}
        self.versions: Dict[str, List[Dict[str, Any]]] = {}
        
        # Load existing registry
        self._load_registry()
    
    def register_version(
        self,
        model_name: str,
        version: str,
        model_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Register new model version.
        
        Args:
            model_name: Model name
            version: Version string (e.g., '1.0.0')
            model_path: Path to model artifact
            metadata: Optional metadata
        """
        version_info = {
            'version': version,
            'path': model_path,
            'deployed_at': datetime.now().isoformat(),
            'metadata': metadata or {},
        }
        
        if model_name not in self.versions:
            self.versions[model_name] = []
        
        self.versions[model_name].append(version_info)
        
        # Save registry
        self._save_registry()
        
        logger.info(f"Registered {model_name} v{version}")
    
    def get_version(
        self,
        model_name: str,
        version: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get specific version info.
        
        Args:
            model_name: Model name
            version: Version string
            
        Returns:
            Version info or None
        """
        if model_name not in self.versions:
            return None
        
        for v in self.versions[model_name]:
            if v['version'] == version:
                return v
        
        return None
    
    def get_latest_version(
        self,
        model_name: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get latest version.
        
        Args:
            model_name: Model name
            
        Returns:
            Latest version info or None
        """
        if model_name not in self.versions:
            return None
        
        if not self.versions[model_name]:
            return None
        
        return self.versions[model_name][-1]
    
    def list_versions(
        self,
        model_name: str
    ) -> List[Dict[str, Any]]:
        """
        List all versions for model.
        
        Args:
            model_name: Model name
            
        Returns:
            List of version info (oldest to newest)
        """
        return self.versions.get(model_name, [])
    
    def delete_version(
        self,
        model_name: str,
        version: str
    ) -> bool:
        """
        Delete specific version.
        
        Args:
            model_name: Model name
            version: Version to delete
            
        Returns:
            True if deleted, False otherwise
        """
        if model_name not in self.versions:
            return False
        
        for i, v in enumerate(self.versions[model_name]):
            if v['version'] == version:
                # Delete artifact
                try:
                    Path(v['path']).unlink()
                except Exception as e:
                    logger.warning(f"Failed to delete artifact: {e}")
                
                # Remove from registry
                self.versions[model_name].pop(i)
                self._save_registry()
                
                logger.info(f"Deleted {model_name} v{version}")
                return True
        
        return False
    
    def cleanup_old_versions(
        self,
        model_name: str,
        keep_latest: int = 3
    ) -> int:
        """
        Cleanup old versions, keeping only recent ones.
        
        Args:
            model_name: Model name
            keep_latest: Number of versions to keep
            
        Returns:
            Number of versions deleted
        """
        if model_name not in self.versions:
            return 0
        
        versions = self.versions[model_name]
        
        if len(versions) <= keep_latest:
            return 0
        
        # Delete oldest versions
        to_delete = versions[:-keep_latest]
        deleted_count = 0
        
        for v in to_delete:
            if self.delete_version(model_name, v['version']):
                deleted_count += 1
        
        return deleted_count
    
    def compare_versions(
        self,
        model_name: str,
        version1: str,
        version2: str
    ) -> Dict[str, Any]:
        """
        Compare two versions.
        
        Args:
            model_name: Model name
            version1: First version
            version2: Second version
            
        Returns:
            Comparison info
        """
        v1 = self.get_version(model_name, version1)
        v2 = self.get_version(model_name, version2)
        
        if not v1 or not v2:
            return {}
        
        return {
            'model': model_name,
            'version1': {
                'version': v1['version'],
                'deployed_at': v1['deployed_at'],
                'metadata': v1['metadata'],
            },
            'version2': {
                'version': v2['version'],
                'deployed_at': v2['deployed_at'],
                'metadata': v2['metadata'],
            },
        }
    
    def _load_registry(self) -> None:
        """Load version registry from disk."""
        registry_path = self.base_dir / "versions.json"
        
        if not registry_path.exists():
            return
        
        try:
            with open(registry_path) as f:
                self.versions = json.load(f)
            
            logger.info(f"Loaded version registry ({len(self.versions)} models)")
            
        except Exception as e:
            logger.error(f"Failed to load registry: {e}")
    
    def _save_registry(self) -> None:
        """Save version registry to disk."""
        registry_path = self.base_dir / "versions.json"
        
        try:
            with open(registry_path, 'w') as f:
                json.dump(self.versions, f, indent=2)
            
        except Exception as e:
            logger.error(f"Failed to save registry: {e}")
