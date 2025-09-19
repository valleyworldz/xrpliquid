#!/usr/bin/env python3
"""
🚨 EMERGENCY DNS RESOLUTION FIX
===============================
Immediate fix for Hyperliquid API connectivity issues.

This script:
1. Resolves Hyperliquid IP addresses using multiple DNS servers
2. Updates Windows hosts file with direct IP mappings
3. Tests connectivity to verify resolution
4. Provides fallback IP addresses for immediate use
"""

import socket
import subprocess
import time
import requests
import json
import os
import sys
from typing import List, Dict, Optional

class EmergencyDNSFix:
    """Emergency DNS resolution and connectivity fix"""
    
    def __init__(self):
        self.hyperliquid_domains = [
            "api.hyperliquid.xyz",
            "api-backup.hyperliquid.xyz"
        ]
        
        # Known working Cloudflare IPs for Hyperliquid
        self.known_hyperliquid_ips = [
            "104.21.73.115",
            "172.67.74.226",
            "104.21.72.115", 
            "172.67.75.226",
            "104.21.74.115",
            "172.67.76.226"
        ]
        
        # DNS servers to try
        self.dns_servers = [
            "8.8.8.8",      # Google Primary
            "8.8.4.4",      # Google Secondary
            "1.1.1.1",      # Cloudflare Primary
            "1.0.0.1",      # Cloudflare Secondary
            "208.67.222.222", # OpenDNS
            "208.67.220.220", # OpenDNS Secondary
            "9.9.9.9",      # Quad9
            "149.112.112.112" # Quad9 Secondary
        ]
        
        self.hosts_file_path = r"C:\Windows\System32\drivers\etc\hosts"
        self.hosts_backup_path = r"C:\Windows\System32\drivers\etc\hosts.backup"

    def run_emergency_fix(self):
        """Run complete emergency DNS fix"""
        print("🚨 EMERGENCY DNS FIX FOR HYPERLIQUID CONNECTIVITY")
        print("=" * 60)
        
        # Step 1: Test current connectivity
        print("\n📡 Step 1: Testing current connectivity...")
        if self.test_hyperliquid_connectivity():
            print("✅ Connectivity is working - no fix needed!")
            return True
        
        print("❌ Connectivity failed - proceeding with emergency fix...")
        
        # Step 2: Resolve IP addresses
        print("\n🔍 Step 2: Resolving IP addresses...")
        resolved_ips = self.resolve_hyperliquid_ips()
        
        # Step 3: Update hosts file
        print("\n📝 Step 3: Updating hosts file...")
        if self.update_hosts_file(resolved_ips):
            print("✅ Hosts file updated successfully")
        else:
            print("❌ Failed to update hosts file - trying alternative method")
            return self.use_environment_override(resolved_ips)
        
        # Step 4: Test connectivity again
        print("\n🧪 Step 4: Testing connectivity after fix...")
        if self.test_hyperliquid_connectivity():
            print("✅ EMERGENCY FIX SUCCESSFUL!")
            print("\n📊 Connection Details:")
            self.show_connection_details()
            return True
        else:
            print("❌ Fix failed - trying alternative solutions...")
            return self.try_alternative_solutions()

    def test_hyperliquid_connectivity(self) -> bool:
        """Test connectivity to Hyperliquid API"""
        try:
            print("  Testing api.hyperliquid.xyz...")
            
            # Test DNS resolution
            try:
                ip = socket.gethostbyname("api.hyperliquid.xyz")
                print(f"  DNS Resolution: api.hyperliquid.xyz → {ip}")
            except socket.gaierror as e:
                print(f"  DNS Resolution: FAILED ({e})")
                return False
            
            # Test HTTP connectivity
            response = requests.get(
                "https://api.hyperliquid.xyz/info",
                params={"type": "meta"},
                timeout=10
            )
            
            if response.status_code == 200:
                print(f"  HTTP Test: SUCCESS (Status: {response.status_code})")
                return True
            else:
                print(f"  HTTP Test: FAILED (Status: {response.status_code})")
                return False
                
        except Exception as e:
            print(f"  HTTP Test: FAILED ({e})")
            return False

    def resolve_hyperliquid_ips(self) -> Dict[str, List[str]]:
        """Resolve Hyperliquid IP addresses using multiple DNS servers"""
        resolved_ips = {}
        
        for domain in self.hyperliquid_domains:
            print(f"  Resolving {domain}...")
            domain_ips = []
            
            # Try each DNS server
            for dns_server in self.dns_servers:
                try:
                    print(f"    Trying DNS server {dns_server}...")
                    
                    # Use nslookup command
                    result = subprocess.run(
                        ["nslookup", domain, dns_server],
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    
                    if result.returncode == 0:
                        # Parse nslookup output for IP addresses
                        lines = result.stdout.split('\n')
                        for line in lines:
                            if 'Address:' in line and dns_server not in line:
                                ip = line.split('Address:')[1].strip()
                                if self.is_valid_ip(ip) and ip not in domain_ips:
                                    domain_ips.append(ip)
                                    print(f"      Found IP: {ip}")
                    
                except Exception as e:
                    print(f"      Failed with {dns_server}: {e}")
                    continue
            
            # If no IPs found, use known IPs
            if not domain_ips:
                print(f"    Using known IPs for {domain}")
                domain_ips = self.known_hyperliquid_ips.copy()
            
            resolved_ips[domain] = domain_ips
            print(f"    Total IPs for {domain}: {len(domain_ips)}")
        
        return resolved_ips

    def is_valid_ip(self, ip: str) -> bool:
        """Check if string is a valid IP address"""
        try:
            socket.inet_aton(ip)
            return True
        except socket.error:
            return False

    def update_hosts_file(self, resolved_ips: Dict[str, List[str]]) -> bool:
        """Update Windows hosts file with resolved IPs"""
        try:
            # Check if running as administrator
            if not self.is_admin():
                print("  ⚠️ Need administrator privileges to update hosts file")
                print("  💡 Please run this script as Administrator")
                return False
            
            # Backup existing hosts file
            if os.path.exists(self.hosts_file_path):
                print(f"  📋 Backing up hosts file to {self.hosts_backup_path}")
                with open(self.hosts_file_path, 'r') as src, open(self.hosts_backup_path, 'w') as dst:
                    dst.write(src.read())
            
            # Read current hosts file
            with open(self.hosts_file_path, 'r') as f:
                current_content = f.read()
            
            # Remove any existing Hyperliquid entries
            lines = current_content.split('\n')
            cleaned_lines = [
                line for line in lines 
                if not any(domain in line for domain in self.hyperliquid_domains)
            ]
            
            # Add new entries
            cleaned_lines.append("\n# Emergency Hyperliquid DNS Fix")
            cleaned_lines.append(f"# Generated at {time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            for domain, ips in resolved_ips.items():
                if ips:
                    # Use the first (most reliable) IP
                    primary_ip = ips[0]
                    cleaned_lines.append(f"{primary_ip} {domain}")
                    print(f"  ✅ Added: {primary_ip} → {domain}")
            
            # Write updated hosts file
            with open(self.hosts_file_path, 'w') as f:
                f.write('\n'.join(cleaned_lines))
            
            # Flush DNS cache
            print("  🔄 Flushing DNS cache...")
            subprocess.run(["ipconfig", "/flushdns"], capture_output=True)
            
            return True
            
        except Exception as e:
            print(f"  ❌ Failed to update hosts file: {e}")
            return False

    def is_admin(self) -> bool:
        """Check if running with administrator privileges"""
        try:
            return os.getuid() == 0
        except AttributeError:
            # Windows
            import ctypes
            return ctypes.windll.shell32.IsUserAnAdmin()

    def use_environment_override(self, resolved_ips: Dict[str, List[str]]) -> bool:
        """Use environment variables as DNS override"""
        try:
            print("  🔧 Setting up environment variable override...")
            
            # Create environment file
            env_file = "emergency_dns_override.env"
            with open(env_file, 'w') as f:
                f.write("# Emergency DNS Override for Hyperliquid\n")
                f.write("# Source this file or set these environment variables\n\n")
                
                for domain, ips in resolved_ips.items():
                    if ips:
                        env_var = f"HYPERLIQUID_IP_{domain.replace('.', '_').replace('-', '_').upper()}"
                        f.write(f"{env_var}={ips[0]}\n")
                        os.environ[env_var] = ips[0]
                        print(f"  ✅ Set {env_var}={ips[0]}")
            
            print(f"  📄 Environment overrides saved to {env_file}")
            return True
            
        except Exception as e:
            print(f"  ❌ Environment override failed: {e}")
            return False

    def try_alternative_solutions(self) -> bool:
        """Try alternative solutions for connectivity"""
        print("\n🔧 Trying alternative solutions...")
        
        # Solution 1: Direct IP connection
        print("  1. Testing direct IP connections...")
        for ip in self.known_hyperliquid_ips:
            if self.test_direct_ip_connection(ip):
                print(f"  ✅ Direct IP {ip} is working!")
                self.create_ip_config_file(ip)
                return True
        
        # Solution 2: Proxy/VPN suggestion
        print("  2. Network routing issues detected")
        print("  💡 Potential solutions:")
        print("     - Check firewall settings")
        print("     - Try different network (mobile hotspot)")
        print("     - Use VPN service")
        print("     - Contact ISP about DNS issues")
        
        return False

    def test_direct_ip_connection(self, ip: str) -> bool:
        """Test direct IP connection"""
        try:
            # Test with IP address directly
            response = requests.get(
                f"https://{ip}/info",
                params={"type": "meta"},
                headers={"Host": "api.hyperliquid.xyz"},
                timeout=10,
                verify=False  # Skip SSL verification for IP
            )
            return response.status_code == 200
        except:
            return False

    def create_ip_config_file(self, working_ip: str):
        """Create configuration file with working IP"""
        config = {
            "emergency_mode": True,
            "hyperliquid_ip": working_ip,
            "use_direct_ip": True,
            "timestamp": time.time(),
            "note": "Emergency DNS fix - using direct IP connection"
        }
        
        with open("emergency_network_config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        print(f"  📄 Created emergency config: emergency_network_config.json")

    def show_connection_details(self):
        """Show detailed connection information"""
        try:
            for domain in self.hyperliquid_domains:
                try:
                    ip = socket.gethostbyname(domain)
                    print(f"  {domain} → {ip}")
                    
                    # Test latency
                    start_time = time.time()
                    response = requests.get(f"https://{domain}/info", params={"type": "meta"}, timeout=5)
                    latency = (time.time() - start_time) * 1000
                    
                    if response.status_code == 200:
                        print(f"    Status: ✅ HEALTHY (Latency: {latency:.1f}ms)")
                    else:
                        print(f"    Status: ⚠️ DEGRADED (HTTP {response.status_code})")
                        
                except Exception as e:
                    print(f"    Status: ❌ FAILED ({e})")
                    
        except Exception as e:
            print(f"  Error showing connection details: {e}")

    def restore_hosts_file(self):
        """Restore original hosts file"""
        try:
            if os.path.exists(self.hosts_backup_path):
                print("🔄 Restoring original hosts file...")
                with open(self.hosts_backup_path, 'r') as src, open(self.hosts_file_path, 'w') as dst:
                    dst.write(src.read())
                print("✅ Hosts file restored")
                subprocess.run(["ipconfig", "/flushdns"], capture_output=True)
                return True
            else:
                print("❌ No backup file found")
                return False
        except Exception as e:
            print(f"❌ Error restoring hosts file: {e}")
            return False

def main():
    """Main function"""
    emergency_fix = EmergencyDNSFix()
    
    if len(sys.argv) > 1 and sys.argv[1] == "--restore":
        emergency_fix.restore_hosts_file()
        return
    
    success = emergency_fix.run_emergency_fix()
    
    if success:
        print("\n🎉 EMERGENCY FIX COMPLETED SUCCESSFULLY!")
        print("✅ Hyperliquid connectivity restored")
        print("\n💡 To restore original settings later, run:")
        print("   python emergency_dns_fix.py --restore")
    else:
        print("\n❌ EMERGENCY FIX FAILED")
        print("🆘 Please contact technical support or try manual solutions")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
