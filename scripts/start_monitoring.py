#!/usr/bin/env python3
"""
📊 MONITORING STACK STARTER
Script to start Prometheus, Grafana, and trading system with metrics
"""

import os
import sys
import subprocess
import time
import signal
import threading
from pathlib import Path
import json
import requests
from typing import Dict, Any, Optional

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.monitoring.prometheus_metrics import get_metrics_collector
from core.utils.logger import Logger

class MonitoringStack:
    """Manages the complete monitoring stack"""
    
    def __init__(self, logger: Optional[Logger] = None):
        self.logger = logger or Logger()
        self.processes = {}
        self.metrics_collector = None
        self.running = False
        
        # Ports
        self.prometheus_port = 9090
        self.grafana_port = 3000
        self.metrics_port = 8000
        
        # Paths
        self.project_root = Path(__file__).parent.parent
        self.prometheus_config = self.project_root / "monitoring" / "prometheus" / "prometheus.yml"
        self.grafana_dashboard = self.project_root / "monitoring" / "grafana" / "dashboards" / "trading_dashboard.json"
        
        self.logger.info("📊 [MONITORING] Monitoring stack initialized")
    
    def check_dependencies(self) -> bool:
        """Check if required dependencies are available"""
        self.logger.info("🔍 [MONITORING] Checking dependencies...")
        
        # Check if Prometheus is available
        try:
            result = subprocess.run(['prometheus', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                self.logger.info("✅ [MONITORING] Prometheus found")
            else:
                self.logger.warning("⚠️ [MONITORING] Prometheus not found - will use Docker")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            self.logger.warning("⚠️ [MONITORING] Prometheus not found - will use Docker")
        
        # Check if Grafana is available
        try:
            result = subprocess.run(['grafana-server', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                self.logger.info("✅ [MONITORING] Grafana found")
            else:
                self.logger.warning("⚠️ [MONITORING] Grafana not found - will use Docker")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            self.logger.warning("⚠️ [MONITORING] Grafana not found - will use Docker")
        
        # Check if Docker is available
        try:
            result = subprocess.run(['docker', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                self.logger.info("✅ [MONITORING] Docker found")
                return True
            else:
                self.logger.error("❌ [MONITORING] Docker not found")
                return False
        except (subprocess.TimeoutExpired, FileNotFoundError):
            self.logger.error("❌ [MONITORING] Docker not found")
            return False
    
    def start_metrics_collector(self):
        """Start the Prometheus metrics collector"""
        self.logger.info("📊 [MONITORING] Starting metrics collector...")
        
        try:
            self.metrics_collector = get_metrics_collector(port=self.metrics_port, logger=self.logger)
            self.logger.info(f"✅ [MONITORING] Metrics collector started on port {self.metrics_port}")
            return True
        except Exception as e:
            self.logger.error(f"❌ [MONITORING] Failed to start metrics collector: {e}")
            return False
    
    def start_prometheus(self):
        """Start Prometheus server"""
        self.logger.info("📊 [MONITORING] Starting Prometheus...")
        
        try:
            # Create data directory
            data_dir = self.project_root / "monitoring" / "prometheus" / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            
            # Start Prometheus with Docker
            cmd = [
                'docker', 'run', '-d',
                '--name', 'prometheus',
                '-p', f'{self.prometheus_port}:9090',
                '-v', f'{self.prometheus_config.parent.absolute()}:/etc/prometheus',
                '-v', f'{data_dir.absolute()}:/prometheus',
                'prom/prometheus:latest',
                '--config.file=/etc/prometheus/prometheus.yml',
                '--storage.tsdb.path=/prometheus',
                '--web.console.libraries=/etc/prometheus/console_libraries',
                '--web.console.templates=/etc/prometheus/consoles',
                '--web.enable-lifecycle'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                self.logger.info(f"✅ [MONITORING] Prometheus started on port {self.prometheus_port}")
                return True
            else:
                self.logger.error(f"❌ [MONITORING] Failed to start Prometheus: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ [MONITORING] Failed to start Prometheus: {e}")
            return False
    
    def start_grafana(self):
        """Start Grafana server"""
        self.logger.info("📊 [MONITORING] Starting Grafana...")
        
        try:
            # Create data directory
            data_dir = self.project_root / "monitoring" / "grafana" / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            
            # Start Grafana with Docker
            cmd = [
                'docker', 'run', '-d',
                '--name', 'grafana',
                '-p', f'{self.grafana_port}:3000',
                '-v', f'{data_dir.absolute()}:/var/lib/grafana',
                '-e', 'GF_SECURITY_ADMIN_PASSWORD=admin',
                'grafana/grafana:latest'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                self.logger.info(f"✅ [MONITORING] Grafana started on port {self.grafana_port}")
                return True
            else:
                self.logger.error(f"❌ [MONITORING] Failed to start Grafana: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ [MONITORING] Failed to start Grafana: {e}")
            return False
    
    def setup_grafana_datasource(self):
        """Setup Prometheus datasource in Grafana"""
        self.logger.info("📊 [MONITORING] Setting up Grafana datasource...")
        
        # Wait for Grafana to be ready
        max_retries = 30
        for i in range(max_retries):
            try:
                response = requests.get(f'http://localhost:{self.grafana_port}/api/health', timeout=5)
                if response.status_code == 200:
                    break
            except requests.RequestException:
                pass
            
            if i == max_retries - 1:
                self.logger.error("❌ [MONITORING] Grafana not ready after 30 retries")
                return False
            
            time.sleep(2)
        
        # Create Prometheus datasource
        datasource_config = {
            "name": "Prometheus",
            "type": "prometheus",
            "url": f"http://host.docker.internal:{self.prometheus_port}",
            "access": "proxy",
            "isDefault": True
        }
        
        try:
            response = requests.post(
                f'http://localhost:{self.grafana_port}/api/datasources',
                json=datasource_config,
                auth=('admin', 'admin'),
                headers={'Content-Type': 'application/json'},
                timeout=10
            )
            
            if response.status_code in [200, 409]:  # 409 means already exists
                self.logger.info("✅ [MONITORING] Prometheus datasource configured")
                return True
            else:
                self.logger.error(f"❌ [MONITORING] Failed to configure datasource: {response.text}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ [MONITORING] Failed to setup datasource: {e}")
            return False
    
    def import_grafana_dashboard(self):
        """Import the trading dashboard into Grafana"""
        self.logger.info("📊 [MONITORING] Importing Grafana dashboard...")
        
        try:
            # Read dashboard JSON
            with open(self.grafana_dashboard, 'r') as f:
                dashboard_config = json.load(f)
            
            # Import dashboard
            import_config = {
                "dashboard": dashboard_config,
                "overwrite": True
            }
            
            response = requests.post(
                f'http://localhost:{self.grafana_port}/api/dashboards/db',
                json=import_config,
                auth=('admin', 'admin'),
                headers={'Content-Type': 'application/json'},
                timeout=10
            )
            
            if response.status_code == 200:
                self.logger.info("✅ [MONITORING] Trading dashboard imported")
                return True
            else:
                self.logger.error(f"❌ [MONITORING] Failed to import dashboard: {response.text}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ [MONITORING] Failed to import dashboard: {e}")
            return False
    
    def start_monitoring_stack(self):
        """Start the complete monitoring stack"""
        self.logger.info("🚀 [MONITORING] Starting monitoring stack...")
        
        # Check dependencies
        if not self.check_dependencies():
            self.logger.error("❌ [MONITORING] Dependencies not met")
            return False
        
        # Start metrics collector
        if not self.start_metrics_collector():
            return False
        
        # Start Prometheus
        if not self.start_prometheus():
            return False
        
        # Start Grafana
        if not self.start_grafana():
            return False
        
        # Wait for services to be ready
        self.logger.info("⏳ [MONITORING] Waiting for services to be ready...")
        time.sleep(10)
        
        # Setup Grafana
        if not self.setup_grafana_datasource():
            return False
        
        if not self.import_grafana_dashboard():
            return False
        
        self.running = True
        self.logger.info("🎉 [MONITORING] Monitoring stack started successfully!")
        self.logger.info(f"📊 [MONITORING] Prometheus: http://localhost:{self.prometheus_port}")
        self.logger.info(f"📊 [MONITORING] Grafana: http://localhost:{self.grafana_port} (admin/admin)")
        self.logger.info(f"📊 [MONITORING] Metrics: http://localhost:{self.metrics_port}/metrics")
        
        return True
    
    def stop_monitoring_stack(self):
        """Stop the monitoring stack"""
        self.logger.info("🛑 [MONITORING] Stopping monitoring stack...")
        
        self.running = False
        
        # Stop Docker containers
        containers = ['grafana', 'prometheus']
        for container in containers:
            try:
                subprocess.run(['docker', 'stop', container], 
                             capture_output=True, text=True, timeout=10)
                subprocess.run(['docker', 'rm', container], 
                             capture_output=True, text=True, timeout=10)
                self.logger.info(f"✅ [MONITORING] Stopped {container}")
            except Exception as e:
                self.logger.warning(f"⚠️ [MONITORING] Failed to stop {container}: {e}")
        
        self.logger.info("✅ [MONITORING] Monitoring stack stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of monitoring stack"""
        status = {
            'running': self.running,
            'services': {}
        }
        
        # Check Prometheus
        try:
            response = requests.get(f'http://localhost:{self.prometheus_port}/api/v1/status/config', timeout=5)
            status['services']['prometheus'] = 'running' if response.status_code == 200 else 'stopped'
        except:
            status['services']['prometheus'] = 'stopped'
        
        # Check Grafana
        try:
            response = requests.get(f'http://localhost:{self.grafana_port}/api/health', timeout=5)
            status['services']['grafana'] = 'running' if response.status_code == 200 else 'stopped'
        except:
            status['services']['grafana'] = 'stopped'
        
        # Check metrics collector
        if self.metrics_collector:
            metrics_summary = self.metrics_collector.get_metrics_summary()
            status['services']['metrics'] = metrics_summary.get('status', 'unknown')
        else:
            status['services']['metrics'] = 'stopped'
        
        return status

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Start/Stop XRP Trading System Monitoring Stack')
    parser.add_argument('action', choices=['start', 'stop', 'status'], 
                       help='Action to perform')
    parser.add_argument('--prometheus-port', type=int, default=9090,
                       help='Prometheus port (default: 9090)')
    parser.add_argument('--grafana-port', type=int, default=3000,
                       help='Grafana port (default: 3000)')
    parser.add_argument('--metrics-port', type=int, default=8000,
                       help='Metrics port (default: 8000)')
    
    args = parser.parse_args()
    
    logger = Logger()
    monitoring_stack = MonitoringStack(logger)
    
    # Set ports
    monitoring_stack.prometheus_port = args.prometheus_port
    monitoring_stack.grafana_port = args.grafana_port
    monitoring_stack.metrics_port = args.metrics_port
    
    if args.action == 'start':
        success = monitoring_stack.start_monitoring_stack()
        if success:
            print("\n🎉 Monitoring stack started successfully!")
            print(f"📊 Prometheus: http://localhost:{args.prometheus_port}")
            print(f"📊 Grafana: http://localhost:{args.grafana_port} (admin/admin)")
            print(f"📊 Metrics: http://localhost:{args.metrics_port}/metrics")
            print("\nPress Ctrl+C to stop...")
            
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n🛑 Stopping monitoring stack...")
                monitoring_stack.stop_monitoring_stack()
        else:
            print("❌ Failed to start monitoring stack")
            sys.exit(1)
    
    elif args.action == 'stop':
        monitoring_stack.stop_monitoring_stack()
        print("✅ Monitoring stack stopped")
    
    elif args.action == 'status':
        status = monitoring_stack.get_status()
        print("📊 Monitoring Stack Status:")
        print(f"   Overall: {'Running' if status['running'] else 'Stopped'}")
        for service, state in status['services'].items():
            print(f"   {service.capitalize()}: {state}")

if __name__ == "__main__":
    main()
