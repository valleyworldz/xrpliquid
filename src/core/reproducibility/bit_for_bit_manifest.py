"""
Bit-for-Bit Reproducibility Manifest
Generates comprehensive hash manifests for complete reproducibility verification.
"""

import json
import hashlib
import subprocess
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class HashManifest:
    """Represents a complete hash manifest for reproducibility."""
    timestamp: str
    signer: str
    code_commit: str
    config_hashes: Dict[str, str]
    data_hashes: Dict[str, str]
    artifact_hashes: Dict[str, str]
    environment_hash: str
    total_manifest_hash: str


class BitForBitReproducibility:
    """Manages bit-for-bit reproducibility verification."""
    
    def __init__(self, reports_dir: str = "reports"):
        self.reports_dir = Path(reports_dir)
        self.manifest_dir = self.reports_dir / "reproducibility"
        self.manifest_dir.mkdir(parents=True, exist_ok=True)
        
        # Hash manifest file
        self.manifest_file = self.reports_dir / "hash_manifest.json"
    
    def generate_complete_manifest(self, signer: str = "HatManifestoSystem") -> HashManifest:
        """Generate complete bit-for-bit reproducibility manifest."""
        
        timestamp = datetime.now(timezone.utc).isoformat()
        
        # Get code commit hash
        code_commit = self._get_git_commit_hash()
        
        # Calculate config hashes
        config_hashes = self._calculate_config_hashes()
        
        # Calculate data hashes
        data_hashes = self._calculate_data_hashes()
        
        # Calculate artifact hashes
        artifact_hashes = self._calculate_artifact_hashes()
        
        # Calculate environment hash
        environment_hash = self._calculate_environment_hash()
        
        # Create manifest
        manifest = HashManifest(
            timestamp=timestamp,
            signer=signer,
            code_commit=code_commit,
            config_hashes=config_hashes,
            data_hashes=data_hashes,
            artifact_hashes=artifact_hashes,
            environment_hash=environment_hash,
            total_manifest_hash=""  # Will be calculated after creation
        )
        
        # Calculate total manifest hash
        manifest.total_manifest_hash = self._calculate_manifest_hash(manifest)
        
        return manifest
    
    def _get_git_commit_hash(self) -> str:
        """Get current git commit hash."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return "unknown"
    
    def _calculate_config_hashes(self) -> Dict[str, str]:
        """Calculate hashes for all configuration files."""
        
        config_hashes = {}
        
        # Configuration files to hash
        config_files = [
            "config/",
            "requirements.txt",
            "requirements-dev.txt",
            ".pre-commit-config.yaml",
            "pyproject.toml",
            "setup.py"
        ]
        
        for config_path in config_files:
            path = Path(config_path)
            if path.exists():
                if path.is_file():
                    config_hashes[config_path] = self._calculate_file_hash(path)
                elif path.is_dir():
                    config_hashes[config_path] = self._calculate_directory_hash(path)
        
        return config_hashes
    
    def _calculate_data_hashes(self) -> Dict[str, str]:
        """Calculate hashes for all data files."""
        
        data_hashes = {}
        
        # Data directories to hash
        data_dirs = [
            "data/",
            "data/warehouse/",
            "data/raw/",
            "data/processed/"
        ]
        
        for data_dir in data_dirs:
            path = Path(data_dir)
            if path.exists() and path.is_dir():
                data_hashes[data_dir] = self._calculate_directory_hash(path)
        
        return data_hashes
    
    def _calculate_artifact_hashes(self) -> Dict[str, str]:
        """Calculate hashes for all generated artifacts."""
        
        artifact_hashes = {}
        
        # Artifact files to hash
        artifact_files = [
            "reports/ledgers/trades.csv",
            "reports/ledgers/trades.parquet",
            "reports/tearsheets/comprehensive_tearsheet.html",
            "reports/executive_dashboard.html",
            "reports/equity_curve.csv",
            "reports/drawdown_curve.csv",
            "reports/backtest_results.json",
            "reports/risk_events/risk_events.json",
            "reports/latency/latency_analysis.json",
            "reports/regime/regime_analysis.json",
            "reports/maker_taker/routing_analysis.json"
        ]
        
        for artifact_file in artifact_files:
            path = Path(artifact_file)
            if path.exists():
                artifact_hashes[artifact_file] = self._calculate_file_hash(path)
        
        return artifact_hashes
    
    def _calculate_environment_hash(self) -> str:
        """Calculate hash of environment configuration."""
        
        env_info = {
            "python_version": self._get_python_version(),
            "platform": os.name,
            "working_directory": str(Path.cwd()),
            "environment_variables": self._get_relevant_env_vars()
        }
        
        env_str = json.dumps(env_info, sort_keys=True)
        return hashlib.sha256(env_str.encode()).hexdigest()
    
    def _get_python_version(self) -> str:
        """Get Python version."""
        import sys
        return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    
    def _get_relevant_env_vars(self) -> Dict[str, str]:
        """Get relevant environment variables."""
        
        relevant_vars = [
            "PATH", "PYTHONPATH", "PYTHON_VERSION", "VIRTUAL_ENV",
            "CONDA_DEFAULT_ENV", "PIPENV_ACTIVE", "POETRY_ACTIVE"
        ]
        
        env_vars = {}
        for var in relevant_vars:
            if var in os.environ:
                env_vars[var] = os.environ[var]
        
        return env_vars
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of a file."""
        
        sha256_hash = hashlib.sha256()
        
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            return sha256_hash.hexdigest()
        except Exception as e:
            return f"error: {str(e)}"
    
    def _calculate_directory_hash(self, dir_path: Path) -> str:
        """Calculate SHA-256 hash of a directory."""
        
        sha256_hash = hashlib.sha256()
        
        try:
            # Get all files in directory recursively
            for file_path in sorted(dir_path.rglob("*")):
                if file_path.is_file():
                    # Add file path and content to hash
                    sha256_hash.update(str(file_path).encode())
                    with open(file_path, "rb") as f:
                        sha256_hash.update(f.read())
            
            return sha256_hash.hexdigest()
        except Exception as e:
            return f"error: {str(e)}"
    
    def _calculate_manifest_hash(self, manifest: HashManifest) -> str:
        """Calculate hash of the manifest itself."""
        
        # Create a copy without the total_manifest_hash field
        manifest_dict = {
            "timestamp": manifest.timestamp,
            "signer": manifest.signer,
            "code_commit": manifest.code_commit,
            "config_hashes": manifest.config_hashes,
            "data_hashes": manifest.data_hashes,
            "artifact_hashes": manifest.artifact_hashes,
            "environment_hash": manifest.environment_hash
        }
        
        manifest_str = json.dumps(manifest_dict, sort_keys=True)
        return hashlib.sha256(manifest_str.encode()).hexdigest()
    
    def save_manifest(self, manifest: HashManifest) -> str:
        """Save manifest to file."""
        
        manifest_dict = {
            "timestamp": manifest.timestamp,
            "signer": manifest.signer,
            "code_commit": manifest.code_commit,
            "config_hashes": manifest.config_hashes,
            "data_hashes": manifest.data_hashes,
            "artifact_hashes": manifest.artifact_hashes,
            "environment_hash": manifest.environment_hash,
            "total_manifest_hash": manifest.total_manifest_hash,
            "verification_instructions": {
                "step_1": "Verify code commit matches: git rev-parse HEAD",
                "step_2": "Recalculate config hashes and compare",
                "step_3": "Recalculate data hashes and compare",
                "step_4": "Recalculate artifact hashes and compare",
                "step_5": "Recalculate environment hash and compare",
                "step_6": "Recalculate total manifest hash and compare"
            }
        }
        
        with open(self.manifest_file, 'w') as f:
            json.dump(manifest_dict, f, indent=2)
        
        return str(self.manifest_file)
    
    def verify_manifest(self, manifest_file: str = None) -> Dict[str, Any]:
        """Verify a manifest against current state."""
        
        if manifest_file is None:
            manifest_file = self.manifest_file
        
        if not Path(manifest_file).exists():
            return {
                "status": "error",
                "message": f"Manifest file not found: {manifest_file}"
            }
        
        # Load existing manifest
        with open(manifest_file, 'r') as f:
            existing_manifest = json.load(f)
        
        # Generate current manifest
        current_manifest = self.generate_complete_manifest()
        
        # Compare manifests
        verification_result = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "existing_manifest": existing_manifest,
            "current_manifest": {
                "timestamp": current_manifest.timestamp,
                "signer": current_manifest.signer,
                "code_commit": current_manifest.code_commit,
                "config_hashes": current_manifest.config_hashes,
                "data_hashes": current_manifest.data_hashes,
                "artifact_hashes": current_manifest.artifact_hashes,
                "environment_hash": current_manifest.environment_hash,
                "total_manifest_hash": current_manifest.total_manifest_hash
            },
            "verification_results": {
                "code_commit_match": existing_manifest["code_commit"] == current_manifest.code_commit,
                "config_hashes_match": existing_manifest["config_hashes"] == current_manifest.config_hashes,
                "data_hashes_match": existing_manifest["data_hashes"] == current_manifest.data_hashes,
                "artifact_hashes_match": existing_manifest["artifact_hashes"] == current_manifest.artifact_hashes,
                "environment_hash_match": existing_manifest["environment_hash"] == current_manifest.environment_hash,
                "total_manifest_hash_match": existing_manifest["total_manifest_hash"] == current_manifest.total_manifest_hash
            }
        }
        
        # Determine overall status
        all_match = all(verification_result["verification_results"].values())
        verification_result["overall_status"] = "verified" if all_match else "mismatch"
        
        return verification_result


def main():
    """Test bit-for-bit reproducibility functionality."""
    reproducibility = BitForBitReproducibility()
    
    # Generate complete manifest
    manifest = reproducibility.generate_complete_manifest("TestSigner")
    print(f"✅ Generated manifest with {len(manifest.config_hashes)} config hashes")
    print(f"✅ Generated manifest with {len(manifest.data_hashes)} data hashes")
    print(f"✅ Generated manifest with {len(manifest.artifact_hashes)} artifact hashes")
    print(f"✅ Total manifest hash: {manifest.total_manifest_hash[:16]}...")
    
    # Save manifest
    manifest_file = reproducibility.save_manifest(manifest)
    print(f"✅ Manifest saved to: {manifest_file}")
    
    # Verify manifest
    verification = reproducibility.verify_manifest()
    print(f"✅ Manifest verification: {verification['overall_status']}")
    
    print("✅ Bit-for-bit reproducibility testing completed")


if __name__ == "__main__":
    main()
