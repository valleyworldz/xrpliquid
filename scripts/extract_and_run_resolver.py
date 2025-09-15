#!/usr/bin/env python3
"""
Extract credentials and run the ultimate position resolver in the same process.
"""
import os
import sys
import importlib.util

# Step 1: Extract credentials and set environment variables
print("🔐 Extracting credentials and setting environment variables...")

# Import and run extract_credentials.py
spec = importlib.util.spec_from_file_location("extract_credentials", "extract_credentials.py")
extract_mod = importlib.util.module_from_spec(spec)
sys.modules["extract_credentials"] = extract_mod
spec.loader.exec_module(extract_mod)

success = extract_mod.extract_credentials()
if not success:
    print("❌ Failed to extract credentials. Exiting.")
    sys.exit(1)

# Step 2: Run the ultimate position resolver in the same process
def run_resolver():
    print("\n🚀 Running Ultimate Position Resolver...")
    print("=" * 50)
    # Import and run the resolver's main logic
    spec2 = importlib.util.spec_from_file_location("ultimate_position_resolver", "ultimate_position_resolver.py")
    resolver_mod = importlib.util.module_from_spec(spec2)
    sys.modules["ultimate_position_resolver"] = resolver_mod
    spec2.loader.exec_module(resolver_mod)
    if hasattr(resolver_mod, "main"):
        resolver_mod.main()
    elif hasattr(resolver_mod, "UltimatePositionResolver"):
        resolver = resolver_mod.UltimatePositionResolver()
        if hasattr(resolver, "main"):
            resolver.main()
        else:
            print("No main() found in UltimatePositionResolver.")
    else:
        print("No main entry point found in ultimate_position_resolver.py.")

run_resolver() 