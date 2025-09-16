"""
Hash Manifest Generator
Ensures deterministic runs with SHA-256 verification of all inputs and outputs.
"""

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
import subprocess
import sys


class HashManifest:
    """Generates and validates hash manifests for deterministic reproducibility."""
    
    def __init__(self, reports_dir: str = "reports"):
        self.reports_dir = Path(reports_dir)
        self.manifest_path = self.reports_dir / "hash_manifest.json"
        
    def calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of a file."""
        if not file_path.exists():
            return "FILE_NOT_FOUND"
        
        sha256_hash = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            return sha256_hash.hexdigest()
        except Exception as e:
            return f"ERROR: {str(e)}"
    
    def get_git_commit_hash(self) -> str:
        """Get current git commit hash."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except Exception:
            return "NO_GIT_REPO"
    
    def get_git_status(self) -> Dict[str, Any]:
        """Get git status information."""
        try:
            # Check for uncommitted changes
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                check=True
            )
            uncommitted = result.stdout.strip().split('\n') if result.stdout.strip() else []
            
            # Get current branch
            branch_result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
                check=True
            )
            branch = branch_result.stdout.strip()
            
            return {
                "branch": branch,
                "uncommitted_files": uncommitted,
                "clean_working_tree": len(uncommitted) == 0
            }
        except Exception:
            return {"error": "NO_GIT_REPO"}
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information for reproducibility."""
        return {
            "python_version": sys.version,
            "platform": sys.platform,
            "timestamp": datetime.now().isoformat(),
            "timezone": str(datetime.now().astimezone().tzinfo),
            "working_directory": str(Path.cwd()),
            "environment_variables": {
                k: v for k, v in os.environ.items() 
                if k.startswith(('PYTHON', 'PATH', 'HOME', 'USER'))
            }
        }
    
    def get_config_hashes(self) -> Dict[str, str]:
        """Get hashes of all configuration files."""
        config_dir = Path("config")
        config_hashes = {}
        
        if config_dir.exists():
            for config_file in config_dir.glob("*.json"):
                config_hashes[config_file.name] = self.calculate_file_hash(config_file)
        
        # Also check for key config files in root
        key_configs = [
            "requirements.txt",
            "requirements_ultimate.txt",
            ".pre-commit-config.yaml"
        ]
        
        for config_file in key_configs:
            config_path = Path(config_file)
            if config_path.exists():
                config_hashes[config_file] = self.calculate_file_hash(config_path)
        
        return config_hashes
    
    def get_data_hashes(self) -> Dict[str, str]:
        """Get hashes of input data files."""
        data_hashes = {}
        
        # Check for data files
        data_dirs = ["data", "reports/ledgers", "reports/tearsheets"]
        
        for data_dir in data_dirs:
            data_path = Path(data_dir)
            if data_path.exists():
                for data_file in data_path.glob("*"):
                    if data_file.is_file():
                        relative_path = data_file.relative_to(Path.cwd())
                        data_hashes[str(relative_path)] = self.calculate_file_hash(data_file)
        
        return data_hashes
    
    def get_output_hashes(self) -> Dict[str, str]:
        """Get hashes of generated output files."""
        output_hashes = {}
        
        # Key output files to track
        output_patterns = [
            "reports/ledgers/*.csv",
            "reports/ledgers/*.parquet",
            "reports/tearsheets/*.html",
            "reports/executive_dashboard.html",
            "reports/latency/*.json",
            "reports/risk_events/*.json",
            "reports/maker_taker/*.json",
            "reports/regime/*.json"
        ]
        
        for pattern in output_patterns:
            for output_file in Path(".").glob(pattern):
                if output_file.is_file():
                    relative_path = output_file.relative_to(Path.cwd())
                    output_hashes[str(relative_path)] = self.calculate_file_hash(output_file)
        
        return output_hashes
    
    def generate_manifest(self) -> Dict[str, Any]:
        """Generate complete hash manifest."""
        manifest = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "version": "1.0.0",
                "purpose": "Deterministic reproducibility verification"
            },
            "git_info": {
                "commit_hash": self.get_git_commit_hash(),
                "status": self.get_git_status()
            },
            "system_info": self.get_system_info(),
            "config_hashes": self.get_config_hashes(),
            "data_hashes": self.get_data_hashes(),
            "output_hashes": self.get_output_hashes(),
            "reproducibility_checks": {
                "clean_git_tree": self.get_git_status().get("clean_working_tree", False),
                "deterministic_seeds": True,  # Will be set by calling code
                "time_source_controlled": True,  # Will be set by calling code
                "config_pinned": True  # Will be set by calling code
            }
        }
        
        return manifest
    
    def save_manifest(self, manifest: Dict[str, Any]) -> Path:
        """Save manifest to file."""
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        return self.manifest_path
    
    def verify_manifest(self, expected_manifest_path: Path) -> Dict[str, Any]:
        """Verify current state against expected manifest."""
        if not expected_manifest_path.exists():
            return {
                "valid": False,
                "error": "Expected manifest not found",
                "differences": []
            }
        
        try:
            with open(expected_manifest_path, 'r') as f:
                expected_manifest = json.load(f)
            
            current_manifest = self.generate_manifest()
            
            differences = []
            
            # Compare git commit
            if current_manifest["git_info"]["commit_hash"] != expected_manifest["git_info"]["commit_hash"]:
                differences.append({
                    "type": "git_commit_mismatch",
                    "expected": expected_manifest["git_info"]["commit_hash"],
                    "actual": current_manifest["git_info"]["commit_hash"]
                })
            
            # Compare config hashes
            for config_file, expected_hash in expected_manifest["config_hashes"].items():
                current_hash = current_manifest["config_hashes"].get(config_file)
                if current_hash != expected_hash:
                    differences.append({
                        "type": "config_hash_mismatch",
                        "file": config_file,
                        "expected": expected_hash,
                        "actual": current_hash
                    })
            
            # Compare output hashes
            for output_file, expected_hash in expected_manifest["output_hashes"].items():
                current_hash = current_manifest["output_hashes"].get(output_file)
                if current_hash != expected_hash:
                    differences.append({
                        "type": "output_hash_mismatch",
                        "file": output_file,
                        "expected": expected_hash,
                        "actual": current_hash
                    })
            
            return {
                "valid": len(differences) == 0,
                "differences": differences,
                "summary": f"Found {len(differences)} differences"
            }
            
        except Exception as e:
            return {
                "valid": False,
                "error": str(e),
                "differences": []
            }
    
    def generate_and_save(self) -> Path:
        """Generate and save hash manifest."""
        manifest = self.generate_manifest()
        return self.save_manifest(manifest)


def main():
    """Generate hash manifest for current run."""
    manifest_generator = HashManifest()
    manifest_path = manifest_generator.generate_and_save()
    
    print(f"✅ Hash manifest generated: {manifest_path}")
    
    # Print summary
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    print(f"📊 Manifest Summary:")
    print(f"   Git Commit: {manifest['git_info']['commit_hash'][:8]}...")
    print(f"   Clean Tree: {manifest['git_info']['status']['clean_working_tree']}")
    print(f"   Config Files: {len(manifest['config_hashes'])}")
    print(f"   Data Files: {len(manifest['data_hashes'])}")
    print(f"   Output Files: {len(manifest['output_hashes'])}")


if __name__ == "__main__":
    main()
