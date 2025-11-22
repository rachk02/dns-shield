#!/usr/bin/env python3
"""
Traffic Simulator - Generate realistic DNS traffic for testing
Sends queries to API Gateway to populate Prometheus/Grafana dashboards
"""

import requests
import time
import random
import argparse
from datetime import datetime
from typing import List

# API Gateway endpoint
API_URL = "http://localhost:9000/analyze"

# Legitimate domains (should be accepted)
LEGITIMATE_DOMAINS = [
    'google.com', 'amazon.com', 'facebook.com', 'microsoft.com',
    'apple.com', 'netflix.com', 'youtube.com', 'twitter.com',
    'linkedin.com', 'github.com', 'stackoverflow.com', 'reddit.com',
    'wikipedia.org', 'bbc.com', 'cnn.com', 'nytimes.com',
    'instagram.com', 'tiktok.com', 'zoom.us', 'slack.com'
]

# Suspicious/DGA-like domains (should be blocked)
SUSPICIOUS_DOMAINS = [
    'xkjhqwerty.com', 'bcdfghjklmnpqrstvwxyz.com', 'aaaabbbbccccdddd.com',
    'qwertyuiop123.com', 'zxcvbnmasdfgh.com', 'mnbvcxzlkjhgfd.com',
    'abcdefgh12345.com', 'randomqwerty789.com', 'dgafakesite.com',
    'malicioustest.com', 'phishingsite123.com', 'trojandomain.com',
    'ransomware-test.com', 'botnet-c2.com', 'cryptominer.com'
]

# Mix of patterns
MIXED_DOMAINS = [
    'test-domain.com', 'my-app.com', 'user123.com',
    'temp-site.net', 'demo-website.org', 'example-test.com'
]

class TrafficSimulator:
    """Simulate realistic DNS traffic"""
    
    def __init__(self, rate: float = 2.0, legitimate_ratio: float = 0.7):
        """
        Initialize simulator
        
        Args:
            rate: Requests per second
            legitimate_ratio: Ratio of legitimate vs suspicious (0.0-1.0)
        """
        self.rate = rate
        self.legitimate_ratio = legitimate_ratio
        self.stats = {
            'total': 0,
            'success': 0,
            'errors': 0,
            'blocked': 0,
            'accepted': 0
        }
        
    def get_random_domain(self) -> str:
        """Get random domain based on legitimate ratio"""
        if random.random() < self.legitimate_ratio:
            # 70% legitimate, 20% mixed, 10% suspicious
            roll = random.random()
            if roll < 0.7:
                return random.choice(LEGITIMATE_DOMAINS)
            elif roll < 0.9:
                return random.choice(MIXED_DOMAINS)
            else:
                return random.choice(SUSPICIOUS_DOMAINS)
        else:
            # More suspicious traffic
            return random.choice(SUSPICIOUS_DOMAINS)
    
    def send_query(self, domain: str) -> dict:
        """Send single query to API Gateway"""
        try:
            response = requests.post(
                API_URL,
                json={'domain': domain},
                timeout=10  # Increased timeout for slower services
            )
            
            if response.status_code == 200:
                data = response.json()
                self.stats['success'] += 1
                
                if data['decision'] == 'BLOCK':
                    self.stats['blocked'] += 1
                else:
                    self.stats['accepted'] += 1
                
                return data
            else:
                self.stats['errors'] += 1
                return {'error': f'HTTP {response.status_code}'}
                
        except Exception as e:
            self.stats['errors'] += 1
            return {'error': str(e)}
    
    def run_continuous(self, duration: int = None):
        """
        Run continuous traffic simulation
        
        Args:
            duration: Duration in seconds (None = infinite)
        """
        print("=" * 60)
        print("DNS Shield Traffic Simulator")
        print("=" * 60)
        print(f"API Gateway: {API_URL}")
        print(f"Rate: {self.rate} requests/sec")
        print(f"Legitimate ratio: {self.legitimate_ratio * 100:.0f}%")
        if duration:
            print(f"Duration: {duration} seconds")
        else:
            print("Duration: Continuous (Ctrl+C to stop)")
        print("=" * 60)
        print()
        
        start_time = time.time()
        last_stats_time = start_time
        
        try:
            while True:
                # Check duration
                if duration and (time.time() - start_time) >= duration:
                    break
                
                # Get random domain
                domain = self.get_random_domain()
                
                # Send query
                result = self.send_query(domain)
                self.stats['total'] += 1
                
                # Print result
                timestamp = datetime.now().strftime('%H:%M:%S')
                if 'error' in result:
                    print(f"[{timestamp}] ❌ {domain:30s} ERROR: {result['error']}")
                else:
                    decision = result['decision']
                    confidence = result['confidence']
                    latency = result['latency_ms']
                    stage = result['stage_resolved']
                    
                    emoji = "🚫" if decision == "BLOCK" else "✅"
                    print(f"[{timestamp}] {emoji} {domain:30s} {decision:6s} "
                          f"(conf: {confidence:.3f}, stage: {stage}, {latency:.1f}ms)")
                
                # Print stats every 10 seconds
                if time.time() - last_stats_time >= 10:
                    self._print_stats(time.time() - start_time)
                    last_stats_time = time.time()
                
                # Wait for next request
                time.sleep(1.0 / self.rate)
                
        except KeyboardInterrupt:
            print("\n\n🛑 Stopped by user")
        
        # Final stats
        print("\n" + "=" * 60)
        print("SIMULATION COMPLETE")
        print("=" * 60)
        self._print_stats(time.time() - start_time)
    
    def run_batch(self, count: int):
        """Run batch of requests"""
        print("=" * 60)
        print(f"Running batch of {count} requests...")
        print("=" * 60)
        
        start_time = time.time()
        
        for i in range(count):
            domain = self.get_random_domain()
            result = self.send_query(domain)
            self.stats['total'] += 1
            
            if (i + 1) % 10 == 0:
                print(f"Progress: {i + 1}/{count}")
        
        duration = time.time() - start_time
        print("\n" + "=" * 60)
        print("BATCH COMPLETE")
        print("=" * 60)
        self._print_stats(duration)
    
    def _print_stats(self, duration: float):
        """Print statistics"""
        print()
        print(f"📊 Statistics (duration: {duration:.1f}s)")
        print(f"  Total requests:  {self.stats['total']}")
        print(f"  Success:         {self.stats['success']} ({self.stats['success']/max(1, self.stats['total'])*100:.1f}%)")
        print(f"  Errors:          {self.stats['errors']}")
        print(f"  Blocked:         {self.stats['blocked']} ({self.stats['blocked']/max(1, self.stats['success'])*100:.1f}%)")
        print(f"  Accepted:        {self.stats['accepted']} ({self.stats['accepted']/max(1, self.stats['success'])*100:.1f}%)")
        
        if duration > 0:
            throughput = self.stats['total'] / duration
            print(f"  Throughput:      {throughput:.2f} req/s")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Simulate DNS traffic for Prometheus/Grafana dashboards"
    )
    
    parser.add_argument(
        '-r', '--rate',
        type=float,
        default=2.0,
        help='Requests per second (default: 2.0)'
    )
    
    parser.add_argument(
        '-l', '--legitimate-ratio',
        type=float,
        default=0.7,
        help='Ratio of legitimate domains 0.0-1.0 (default: 0.7)'
    )
    
    parser.add_argument(
        '-d', '--duration',
        type=int,
        default=None,
        help='Duration in seconds (default: infinite)'
    )
    
    parser.add_argument(
        '-b', '--batch',
        type=int,
        default=None,
        help='Run batch mode with N requests'
    )
    
    args = parser.parse_args()
    
    # Validate
    if args.legitimate_ratio < 0 or args.legitimate_ratio > 1:
        print("Error: legitimate-ratio must be between 0.0 and 1.0")
        return
    
    # Create simulator
    simulator = TrafficSimulator(
        rate=args.rate,
        legitimate_ratio=args.legitimate_ratio
    )
    
    # Check API Gateway
    try:
        response = requests.get("http://localhost:9000/health", timeout=10)
        if response.status_code != 200:
            print("⚠️  Warning: API Gateway not responding correctly")
            print("   Make sure all services are running:")
            print("   1. python src/dga_detector.py")
            print("   2. python src/bert_service.py")
            print("   3. python src/ensemble_ml.py")
            print("   4. python src/api_gateway.py")
            return
    except Exception as e:
        print(f"❌ Error: Cannot connect to API Gateway at {API_URL}")
        print(f"   {e}")
        print("\n   Make sure the API Gateway is running:")
        print("   python src/api_gateway.py")
        return
    
    # Run simulation
    if args.batch:
        simulator.run_batch(args.batch)
    else:
        simulator.run_continuous(args.duration)


if __name__ == '__main__':
    main()
