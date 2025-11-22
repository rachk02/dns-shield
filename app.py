#!/usr/bin/env python3
"""
DNS Shield Service Orchestrator
Launches all microservices sequentially using 'uv run' with health checks
"""

import subprocess
import time
import sys
import os
import requests
from typing import List, Dict, Optional

# Service configuration
SERVICES = [
    {
        "name": "DGA Detector",
        "module": "src.dga_detector",
        "port": 8001,
        "startup_time": 3
    },
    {
        "name": "BERT Service",
        "module": "src.bert_service",
        "port": 8002,
        "startup_time": 5  # BERT needs more time to load model
    },
    {
        "name": "Ensemble ML",
        "module": "src.ensemble_ml",
        "port": 8003,
        "startup_time": 4
    },
    {
        "name": "API Gateway",
        "module": "src.api_gateway",
        "port": 9000,
        "startup_time": 2
    },
]

class ServiceOrchestrator:
    """Manages lifecycle of all DNS Shield services"""
    
    def __init__(self):
        self.processes: List[subprocess.Popen] = []
        self.services: List[Dict] = SERVICES
        
    def check_health(self, port: int, timeout: int = 2) -> bool:
        """Check if service is responding on health endpoint"""
        try:
            response = requests.get(
                f"http://localhost:{port}/health",
                timeout=timeout
            )
            return response.status_code == 200
        except:
            return False
    
    def wait_for_service(self, service: Dict, max_wait: int = 30) -> bool:
        """Wait for service to become healthy"""
        print(f"   Waiting for {service['name']} to be ready...", end="", flush=True)
        
        start_time = time.time()
        while time.time() - start_time < max_wait:
            if self.check_health(service['port']):
                print(" ✅ Ready!")
                return True
            print(".", end="", flush=True)
            time.sleep(0.5)
        
        print(" ❌ Timeout!")
        return False
    
    def start_service(self, service: Dict) -> Optional[subprocess.Popen]:
        """Start a single service using uv run"""
        print(f"\n{'='*60}")
        print(f"Starting: {service['name']}")
        print(f"Module:   {service['module']}")
        print(f"Port:     {service['port']}")
        print(f"{'='*60}")
        
        try:
            # Build command: uv run python -m <module>
            cmd = ["uv", "run", "python", "-m", service['module']]
            
            print(f"Command: {' '.join(cmd)}")
            
            # Start process
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,  # Line buffered
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == 'win32' else 0
            )
            
            # Initial wait
            time.sleep(service['startup_time'])
            
            # Check if process died immediately
            if process.poll() is not None:
                print(f"❌ ERROR: Process exited with code {process.returncode}")
                return None
            
            print(f"   Process started with PID: {process.pid}")
            
            # Wait for health check
            if self.wait_for_service(service):
                return process
            else:
                print(f"⚠️  Warning: Service started but health check failed")
                return process  # Return anyway, might still work
                
        except FileNotFoundError:
            print(f"❌ ERROR: 'uv' command not found!")
            print("   Install uv or make sure it's in your PATH")
            return None
        except Exception as e:
            print(f"❌ ERROR: {e}")
            return None
    
    def start_all(self) -> bool:
        """Start all services sequentially"""
        print("\n" + "="*60)
        print(" DNS SHIELD SERVICE ORCHESTRATOR")
        print("="*60)
        print(f"Python: {sys.executable}")
        print(f"Working Directory: {os.getcwd()}")
        print("="*60)
        
        for service in self.services:
            process = self.start_service(service)
            
            if process:
                self.processes.append({
                    'name': service['name'],
                    'process': process,
                    'port': service['port']
                })
            else:
                print(f"\n⚠️  Failed to start {service['name']}")
                print("   Continuing with other services...")
        
        # Summary
        print("\n" + "="*60)
        print(" STARTUP SUMMARY")
        print("="*60)
        
        if not self.processes:
            print("❌ No services started successfully!")
            return False
        
        print(f"✅ Started {len(self.processes)}/{len(self.services)} services:\n")
        for proc_info in self.processes:
            status = "🟢" if self.check_health(proc_info['port']) else "🔴"
            print(f"   {status} {proc_info['name']:20s} (PID: {proc_info['process'].pid}, Port: {proc_info['port']})")
        
        print("\n" + "="*60)
        print("🌐 Access Points:")
        print("="*60)
        print("   DGA Detector:  http://localhost:8001")
        print("   BERT Service:  http://localhost:8002")
        print("   Ensemble ML:   http://localhost:8003")
        print("   API Gateway:   http://localhost:9000")
        print("\n   Prometheus:    http://localhost:9090  (if running)")
        print("   Grafana:       http://localhost:3000  (if running)")
        print("="*60)
        print("\n💡 Test with:")
        print('   curl -X POST http://localhost:9000/analyze \\')
        print('        -H "Content-Type: application/json" \\')
        print('        -d \'{"domain": "google.com"}\'')
        print("\n🛑 Press CTRL+C to stop all services")
        print("="*60)
        
        return True
    
    def stop_all(self):
        """Stop all running services"""
        print("\n\n" + "="*60)
        print("Stopping services...")
        print("="*60)
        
        for proc_info in reversed(self.processes):
            print(f"Stopping {proc_info['name']}...", end=" ", flush=True)
            try:
                proc_info['process'].terminate()
                proc_info['process'].wait(timeout=5)
                print("✅")
            except subprocess.TimeoutExpired:
                print("⚠️  Force killing...")
                proc_info['process'].kill()
                proc_info['process'].wait()
            except Exception as e:
                print(f"❌ Error: {e}")
        
        print("="*60)
        print("All services stopped. Goodbye! 👋")
        print("="*60)
    
    def run(self):
        """Main run loop"""
        if not self.start_all():
            print("\n❌ Startup failed. Exiting.")
            return
        
        try:
            # Keep alive and monitor
            while True:
                time.sleep(5)
                
                # Check if any process died
                for proc_info in self.processes:
                    if proc_info['process'].poll() is not None:
                        print(f"\n⚠️  WARNING: {proc_info['name']} has stopped!")
                        
        except KeyboardInterrupt:
            self.stop_all()


def main():
    """Entry point"""
    orchestrator = ServiceOrchestrator()
    orchestrator.run()


if __name__ == "__main__":
    main()
