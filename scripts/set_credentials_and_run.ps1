# PowerShell script to load credentials and run ultimate position resolver
Write-Host "🔐 Loading credentials and setting environment variables..." -ForegroundColor Green

# Import the credential manager module
$pythonScript = @"
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath('.')))

from core.utils.credential_manager import CredentialManager

try:
    cm = CredentialManager()
    private_key = cm.get_credential("HYPERLIQUID_PRIVATE_KEY")
    address = cm.get_credential("HYPERLIQUID_API_KEY")
    
    if private_key and address:
        print(f"PRIVATE_KEY:{private_key}")
        print(f"ADDRESS:{address}")
        print("SUCCESS")
    else:
        print("FAILED: No credentials found")
except Exception as e:
    print(f"ERROR:{str(e)}")
"@

# Run the Python script to extract credentials
$pythonOutput = python -c $pythonScript

if ($pythonOutput -match "SUCCESS") {
    # Extract the credentials from the output
    $privateKey = ($pythonOutput | Select-String "PRIVATE_KEY:").ToString().Replace("PRIVATE_KEY:", "").Trim()
    $address = ($pythonOutput | Select-String "ADDRESS:").ToString().Replace("ADDRESS:", "").Trim()
    
    # Set environment variables
    $env:HYPERLIQUID_PRIVATE_KEY = $privateKey
    $env:HYPERLIQUID_API_KEY = $address
    
    Write-Host "✅ Credentials loaded and environment variables set" -ForegroundColor Green
    Write-Host "📍 Address: $($address.Substring(0,10))..." -ForegroundColor Cyan
    Write-Host "🔐 Private key: [LOADED]" -ForegroundColor Cyan
    
    # Run the ultimate position resolver
    Write-Host "`n🚀 Running Ultimate Position Resolver..." -ForegroundColor Yellow
    python ultimate_position_resolver.py
    
} else {
    Write-Host "❌ Failed to load credentials" -ForegroundColor Red
    Write-Host "Output: $pythonOutput" -ForegroundColor Red
} 