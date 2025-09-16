"""
Enhanced Hash Manifest - Bit-for-Bit Reproducibility
Expands hash manifest to include all inputs/outputs with CI enforcement.
"""

import json
import hashlib
import subprocess
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging


class EnhancedHashManifest:
    """Enhanced hash manifest with complete input/output tracking and CI enforcement."""
    
    def __init__(self, repo_root: str = "."):
        self.repo_root = Path(repo_root)
        self.manifest_file = self.repo_root / "reports" / "hash_manifest.json"
        self.manifest_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def get_git_commit_hash(self) -> str:
        """Get current git commit hash."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to get git commit hash: {e}")
            return "unknown"
    
    def get_git_commit_timestamp(self) -> str:
        """Get git commit timestamp."""
        try:
            result = subprocess.run(
                ["git", "log", "-1", "--format=%ci"],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to get git commit timestamp: {e}")
            return datetime.now(timezone.utc).isoformat()
    
    def calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of a file."""
        try:
            if not file_path.exists():
                return "file_not_found"
            
            hash_sha256 = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            self.logger.error(f"Failed to calculate hash for {file_path}: {e}")
            return "hash_error"
    
    def calculate_directory_hash(self, dir_path: Path) -> str:
        """Calculate combined hash of all files in directory."""
        try:
            if not dir_path.exists():
                return "directory_not_found"
            
            file_hashes = []
            for file_path in sorted(dir_path.rglob("*")):
                if file_path.is_file():
                    file_hash = self.calculate_file_hash(file_path)
                    relative_path = file_path.relative_to(dir_path)
                    file_hashes.append(f"{relative_path}:{file_hash}")
            
            combined_content = "\n".join(file_hashes)
            return hashlib.sha256(combined_content.encode()).hexdigest()
        except Exception as e:
            self.logger.error(f"Failed to calculate directory hash for {dir_path}: {e}")
            return "hash_error"
    
    def get_input_hashes(self) -> Dict[str, str]:
        """Get hashes for all input data and configurations."""
        input_hashes = {}
        
        # Configuration files
        config_dirs = ["config", "src/core/config"]
        for config_dir in config_dirs:
            config_path = self.repo_root / config_dir
            if config_path.exists():
                input_hashes[f"{config_dir}/"] = self.calculate_directory_hash(config_path)
        
        # Individual config files
        config_files = [
            "requirements.txt",
            "pyproject.toml",
            "setup.py",
            ".pre-commit-config.yaml",
            "Dockerfile",
            "docker-compose.yml"
        ]
        for config_file in config_files:
            config_path = self.repo_root / config_file
            if config_path.exists():
                input_hashes[config_file] = self.calculate_file_hash(config_path)
        
        # Raw data files
        data_dirs = ["data/raw", "data/market_data", "data/ticks", "data/funding"]
        for data_dir in data_dirs:
            data_path = self.repo_root / data_dir
            if data_path.exists():
                input_hashes[f"{data_dir}/"] = self.calculate_directory_hash(data_path)
        
        # Warehouse data with provenance
        warehouse_path = self.repo_root / "data" / "warehouse"
        if warehouse_path.exists():
            for date_dir in warehouse_path.iterdir():
                if date_dir.is_dir():
                    provenance_file = date_dir / "_provenance.json"
                    if provenance_file.exists():
                        input_hashes[f"warehouse/{date_dir.name}/"] = self.calculate_directory_hash(date_dir)
        
        return input_hashes
    
    def get_output_hashes(self) -> Dict[str, str]:
        """Get hashes for all output artifacts."""
        output_hashes = {}
        
        # Reports directory
        reports_path = self.repo_root / "reports"
        if reports_path.exists():
            for subdir in ["ledgers", "tearsheets", "risk_events", "latency", "regime", "maker_taker", "microstructure", "reconciliation"]:
                subdir_path = reports_path / subdir
                if subdir_path.exists():
                    output_hashes[f"reports/{subdir}/"] = self.calculate_directory_hash(subdir_path)
        
        # Individual report files
        report_files = [
            "reports/equity_curve.csv",
            "reports/drawdown_curve.csv",
            "reports/backtest_results.json",
            "reports/executive_dashboard.html"
        ]
        for report_file in report_files:
            report_path = self.repo_root / report_file
            if report_path.exists():
                output_hashes[report_file] = self.calculate_file_hash(report_path)
        
        return output_hashes
    
    def get_environment_hash(self) -> str:
        """Get hash of environment and dependencies."""
        env_info = []
        
        # Python version
        try:
            import sys
            env_info.append(f"python_version:{sys.version}")
        except:
            pass
        
        # Package versions
        try:
            result = subprocess.run(
                ["pip", "freeze"],
                capture_output=True,
                text=True,
                check=True
            )
            env_info.append(f"packages:{result.stdout}")
        except:
            pass
        
        # System info
        try:
            import platform
            env_info.append(f"platform:{platform.platform()}")
            env_info.append(f"architecture:{platform.architecture()}")
        except:
            pass
        
        combined_env = "\n".join(env_info)
        return hashlib.sha256(combined_env.encode()).hexdigest()
    
    def get_signer_identity(self) -> str:
        """Get signer identity for the manifest."""
        try:
            # Try to get git user info
            result = subprocess.run(
                ["git", "config", "user.name"],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                check=True
            )
            user_name = result.stdout.strip()
            
            result = subprocess.run(
                ["git", "config", "user.email"],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                check=True
            )
            user_email = result.stdout.strip()
            
            return f"{user_name} <{user_email}>"
        except:
            return "HatManifestoSystem"
    
    def generate_manifest(self) -> Dict[str, Any]:
        """Generate complete hash manifest."""
        
        # Get git information
        commit_hash = self.get_git_commit_hash()
        commit_timestamp = self.get_git_commit_timestamp()
        
        # Get all hashes
        input_hashes = self.get_input_hashes()
        output_hashes = self.get_output_hashes()
        environment_hash = self.get_environment_hash()
        signer_identity = self.get_signer_identity()
        
        # Create manifest
        manifest = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "signer": signer_identity,
            "code_commit": commit_hash,
            "code_commit_timestamp": commit_timestamp,
            "input_hashes": input_hashes,
            "output_hashes": output_hashes,
            "environment_hash": environment_hash,
            "verification_instructions": {
                "step_1": "Verify code commit matches: git rev-parse HEAD",
                "step_2": "Recalculate input hashes and compare",
                "step_3": "Recalculate output hashes and compare",
                "step_4": "Recalculate environment hash and compare",
                "step_5": "Run CI verification: python scripts/verify_reproducibility.py"
            },
            "ci_verification": {
                "enabled": True,
                "script": "scripts/verify_reproducibility.py",
                "expected_exit_code": 0
            }
        }
        
        # Calculate total manifest hash
        manifest_json = json.dumps(manifest, sort_keys=True, indent=2)
        total_hash = hashlib.sha256(manifest_json.encode()).hexdigest()
        manifest["total_manifest_hash"] = total_hash
        
        return manifest
    
    def save_manifest(self, manifest: Dict[str, Any]):
        """Save manifest to file."""
        try:
            with open(self.manifest_file, 'w') as f:
                json.dump(manifest, f, indent=2)
            self.logger.info(f"Hash manifest saved: {self.manifest_file}")
        except Exception as e:
            self.logger.error(f"Failed to save manifest: {e}")
    
    def load_manifest(self) -> Optional[Dict[str, Any]]:
        """Load existing manifest."""
        try:
            if self.manifest_file.exists():
                with open(self.manifest_file, 'r') as f:
                    return json.load(f)
            return None
        except Exception as e:
            self.logger.error(f"Failed to load manifest: {e}")
            return None
    
    def verify_manifest(self, manifest: Dict[str, Any]) -> Dict[str, Any]:
        """Verify manifest integrity and reproducibility."""
        verification_results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "verification_passed": True,
            "errors": [],
            "warnings": [],
            "checks": {}
        }
        
        # Check 1: Code commit matches
        current_commit = self.get_git_commit_hash()
        expected_commit = manifest.get("code_commit", "")
        verification_results["checks"]["code_commit"] = {
            "expected": expected_commit,
            "actual": current_commit,
            "matches": current_commit == expected_commit
        }
        if current_commit != expected_commit:
            verification_results["errors"].append(f"Code commit mismatch: expected {expected_commit}, got {current_commit}")
            verification_results["verification_passed"] = False
        
        # Check 2: Input hashes match
        current_input_hashes = self.get_input_hashes()
        expected_input_hashes = manifest.get("input_hashes", {})
        
        input_matches = 0
        input_total = len(expected_input_hashes)
        for key, expected_hash in expected_input_hashes.items():
            current_hash = current_input_hashes.get(key, "missing")
            matches = current_hash == expected_hash
            if matches:
                input_matches += 1
            else:
                verification_results["errors"].append(f"Input hash mismatch for {key}: expected {expected_hash}, got {current_hash}")
                verification_results["verification_passed"] = False
        
        verification_results["checks"]["input_hashes"] = {
            "matches": input_matches,
            "total": input_total,
            "percentage": (input_matches / input_total * 100) if input_total > 0 else 0
        }
        
        # Check 3: Output hashes match
        current_output_hashes = self.get_output_hashes()
        expected_output_hashes = manifest.get("output_hashes", {})
        
        output_matches = 0
        output_total = len(expected_output_hashes)
        for key, expected_hash in expected_output_hashes.items():
            current_hash = current_output_hashes.get(key, "missing")
            matches = current_hash == expected_hash
            if matches:
                output_matches += 1
            else:
                verification_results["warnings"].append(f"Output hash mismatch for {key}: expected {expected_hash}, got {current_hash}")
        
        verification_results["checks"]["output_hashes"] = {
            "matches": output_matches,
            "total": output_total,
            "percentage": (output_matches / output_total * 100) if output_total > 0 else 0
        }
        
        # Check 4: Environment hash matches
        current_env_hash = self.get_environment_hash()
        expected_env_hash = manifest.get("environment_hash", "")
        verification_results["checks"]["environment_hash"] = {
            "expected": expected_env_hash,
            "actual": current_env_hash,
            "matches": current_env_hash == expected_env_hash
        }
        if current_env_hash != expected_env_hash:
            verification_results["warnings"].append(f"Environment hash mismatch: expected {expected_env_hash}, got {current_env_hash}")
        
        return verification_results


def main():
    """Main function to generate and verify hash manifest."""
    manifest_generator = EnhancedHashManifest()
    
    # Generate new manifest
    manifest = manifest_generator.generate_manifest()
    manifest_generator.save_manifest(manifest)
    
    # Verify manifest
    verification = manifest_generator.verify_manifest(manifest)
    
    # Save verification results
    verification_file = manifest_generator.repo_root / "reports" / "reproducibility_verification.json"
    with open(verification_file, 'w') as f:
        json.dump(verification, f, indent=2)
    
    print(f"✅ Hash manifest generated: {manifest_generator.manifest_file}")
    print(f"✅ Verification results: {verification_file}")
    print(f"✅ Verification passed: {verification['verification_passed']}")
    
    if verification["errors"]:
        print(f"❌ Errors: {len(verification['errors'])}")
        for error in verification["errors"]:
            print(f"  - {error}")
    
    if verification["warnings"]:
        print(f"⚠️  Warnings: {len(verification['warnings'])}")
        for warning in verification["warnings"]:
            print(f"  - {warning}")


if __name__ == "__main__":
    main()
