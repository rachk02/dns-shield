#!/usr/bin/env python3
"""
Generate Test Data - Create synthetic datasets
"""

import csv
import random
import string
from pathlib import Path

def generate_legitimate_domains(count: int = 1000) -> list:
    """Generate legitimate-looking domain names"""
    domains = []
    prefixes = ['www', 'mail', 'ftp', 'api', 'dev', 'test', 'blog', 'shop', 'admin']
    tlds = ['.com', '.net', '.org', '.io', '.co', '.us', '.fr', '.de']
    
    common_names = [
        'google', 'amazon', 'microsoft', 'apple', 'facebook', 'twitter',
        'github', 'stackexchange', 'wikipedia', 'reddit', 'youtube',
        'linkedin', 'instagram', 'pinterest', 'slack', 'discord'
    ]
    
    # Add real-like domains
    for name in common_names:
        for tld in tlds:
            domains.append(f"{name}{tld}")
    
    # Generate random legitimate-looking
    for _ in range(count - len(domains)):
        name = ''.join(random.choices(string.ascii_lowercase, k=random.randint(3, 10)))
        tld = random.choice(tlds)
        domains.append(f"{name}{tld}")
    
    return domains[:count]

def generate_dga_domains(count: int = 500) -> list:
    """Generate DGA-like domain names"""
    domains = []
    tlds = ['.com', '.net', '.org', '.info', '.biz']
    
    # Random character sequences (high entropy)
    for _ in range(count // 2):
        length = random.randint(15, 25)
        name = ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))
        tld = random.choice(tlds)
        domains.append(f"{name}{tld}")
    
    # Dictionary but suspicious (no vowels or high consonant ratio)
    consonants = 'bcdfghjklmnpqrstvwxyz'
    for _ in range(count // 2):
        length = random.randint(12, 18)
        name = ''.join(random.choices(consonants, k=length))
        tld = random.choice(tlds)
        domains.append(f"{name}{tld}")
    
    return domains[:count]

def create_test_dataset(
    output_file: str = r"C:\dns-shield\data\test\test_domains.csv",
    legitimate_count: int = 1000,
    malicious_count: int = 500
):
    """Create balanced test dataset"""
    
    print(f"Generating {legitimate_count} legitimate domains...")
    legitimate = generate_legitimate_domains(legitimate_count)
    
    print(f"Generating {malicious_count} DGA domains...")
    malicious = generate_dga_domains(malicious_count)
    
    # Create dataset
    data = []
    for d in legitimate:
        data.append([d, 0])  # 0 = legitimate
    
    for d in malicious:
        data.append([d, 1])  # 1 = malicious
    
    # Shuffle
    random.shuffle(data)
    
    # Save
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['domain', 'label'])
        writer.writerows(data)
    
    print(f"✓ Dataset saved: {output_file}")
    print(f"  Total: {len(data)} domains")
    print(f"  Legitimate: {legitimate_count}")
    print(f"  Malicious: {malicious_count}")

def create_whitelist(output_file: str = r"C:\dns-shield\data\test\whitelist.txt"):
    """Create whitelist of known legitimate domains"""
    whitelist = [
        'google.com', 'microsoft.com', 'apple.com', 'amazon.com',
        'github.com', 'stackoverflow.com', 'wikipedia.org', 'reddit.com',
        'youtube.com', 'twitter.com', 'facebook.com', 'instagram.com',
        'linkedin.com', 'spotify.com', 'netflix.com', 'slack.com',
        'discord.com', 'telegram.org', 'whatsapp.com', 'skype.com'
    ]
    
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for domain in whitelist:
            f.write(f"{domain}\n")
    
    print(f"✓ Whitelist saved: {output_file}")

def create_blacklist(output_file: str = r"C:\dns-shield\data\test\blacklist.txt"):
    """Create blacklist of known malicious domains"""
    blacklist = [
        'malicious-dga-1.com', 'phishing-site.net', 'c2-command.info',
        'ransomware-payment.com', 'botnet-control.xyz', 'trojan-download.ru',
        'xkjhqwerty.com', 'zxcvbnmasdfgh.net', 'qwertyuiopasdf.org',
        'mnbvcxzasdfgh.com', 'infected-system.top', 'malware-distribution.pw'
    ]
    
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for domain in blacklist:
            f.write(f"{domain}\n")
    
    print(f"✓ Blacklist saved: {output_file}")

if __name__ == '__main__':
    print("=" * 60)
    print("DNS Shield - Test Data Generation")
    print("=" * 60)
    
    create_test_dataset()
    create_whitelist()
    create_blacklist()
    
    print("\n✓ Test data generation complete!")