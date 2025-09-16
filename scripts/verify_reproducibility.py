"""
Reproducibility Verification Script
CI script to verify bit-for-bit reproducibility of results.
"""

import json
import sys
import subprocess
from pathlib import Path
from src.core.reproducibility.enhanced_hash_manifest import EnhancedHashManifest


def main():
    """Main verification function."""
    
    print("🔍 Verifying bit-for-bit reproducibility...")
    
    # Load existing manifest
    manifest_generator = EnhancedHashManifest()
    manifest = manifest_generator.load_manifest()
    
    if not manifest:
        print("❌ No hash manifest found. Run enhanced_hash_manifest.py first.")
        sys.exit(1)
    
    # Verify manifest
    verification = manifest_generator.verify_manifest(manifest)
    
    # Save verification results
    verification_file = manifest_generator.repo_root / "reports" / "reproducibility_verification.json"
    with open(verification_file, 'w') as f:
        json.dump(verification, f, indent=2)
    
    # Check if verification passed
    if verification["verification_passed"]:
        print("✅ Reproducibility verification PASSED")
        print(f"📊 Input hashes: {verification['checks']['input_hashes']['matches']}/{verification['checks']['input_hashes']['total']} match")
        print(f"📊 Output hashes: {verification['checks']['output_hashes']['matches']}/{verification['checks']['output_hashes']['total']} match")
        print(f"📊 Code commit: {'✅' if verification['checks']['code_commit']['matches'] else '❌'}")
        print(f"📊 Environment: {'✅' if verification['checks']['environment_hash']['matches'] else '⚠️'}")
        
        if verification["warnings"]:
            print(f"⚠️  {len(verification['warnings'])} warnings (non-critical)")
        
        sys.exit(0)
    else:
        print("❌ Reproducibility verification FAILED")
        print(f"❌ {len(verification['errors'])} errors found:")
        for error in verification["errors"]:
            print(f"  - {error}")
        
        if verification["warnings"]:
            print(f"⚠️  {len(verification['warnings'])} warnings:")
            for warning in verification["warnings"]:
                print(f"  - {warning}")
        
        sys.exit(1)


if __name__ == "__main__":
    main()
