#!/usr/bin/env python3
"""
Generate Test Data - Create synthetic datasets
"""

import csv
import random
import string
from pathlib import Path
from typing import Iterable, List

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
DEFAULT_TEST_FILE = DATA_ROOT / "test" / "test_domains.csv"
DEFAULT_TRAIN_FILE = DATA_ROOT / "train" / "train_domains.csv"


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

def _shuffle_and_trim(domains: Iterable[str], count: int) -> List[str]:
    domains_list = list(domains)
    random.shuffle(domains_list)
    return domains_list[:count]


def _random_entropy_family(count: int, tlds) -> List[str]:
    return [
        ''.join(random.choices(string.ascii_lowercase + string.digits, k=random.randint(15, 26))) + random.choice(tlds)
        for _ in range(count)
    ]


def _consonant_cluster_family(count: int, tlds) -> List[str]:
    consonants = 'bcdfghjklmnpqrstvwxyz'
    return [
        ''.join(random.choices(consonants, k=random.randint(12, 20))) + random.choice(tlds)
        for _ in range(count)
    ]


def _numeric_timestamp_family(count: int, tlds) -> List[str]:
    domains = []
    for _ in range(count):
        year = random.randint(2015, 2035)
        day = random.randint(1, 365)
        suffix = ''.join(random.choices(string.ascii_lowercase, k=5))
        domains.append(f"{year}{day:03d}{suffix}{random.choice(tlds)}")
    return domains


def _keyboard_walk_family(count: int, tlds) -> List[str]:
    patterns = ["qwerty", "asdfgh", "zxcvbn", "poiuyt", "lkjhgf", "mnbvcx"]
    domains = []
    for _ in range(count):
        base = random.choice(patterns)
        extension = ''.join(random.choices(string.ascii_lowercase, k=random.randint(4, 8)))
        domains.append(f"{base}{extension}{random.choice(tlds)}")
    return domains


def _word_mutation_family(count: int, tlds) -> List[str]:
    dictionary = [
        'update', 'service', 'client', 'network', 'secure', 'backup', 'storage',
        'gateway', 'monitor', 'system', 'analytics', 'cloud', 'device', 'router',
    ]
    domains = []
    for _ in range(count):
        word = random.choice(dictionary)
        mutated = ''.join(chr(((ord(c) - 96 + random.randint(1, 5)) % 26) + 97) for c in word)
        suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=random.randint(2, 4)))
        domains.append(f"{mutated}{suffix}{random.choice(tlds)}")
    return domains


def generate_dga_domains(count: int = 500) -> list:
    """Generate DGA-like domain names across multiple synthetic families"""
    tlds = ['.com', '.net', '.org', '.info', '.biz']
    families = [
        _random_entropy_family,
        _consonant_cluster_family,
        _numeric_timestamp_family,
        _keyboard_walk_family,
        _word_mutation_family,
    ]

    per_family = max(1, count // len(families))
    domains: List[str] = []
    for family in families:
        domains.extend(family(per_family, tlds))

    # Adjust rounding differences by topping up with high-entropy domains
    if len(domains) < count:
        domains.extend(_random_entropy_family(count - len(domains), tlds))

    return _shuffle_and_trim(domains, count)


def generate_dga_family_map(count: int = 500) -> List[tuple[str, str]]:
    """Return tuples of (domain, family_name) for analysis/training."""
    tlds = ['.com', '.net', '.org', '.info', '.biz']
    families = {
        'entropy': _random_entropy_family,
        'consonant': _consonant_cluster_family,
        'timestamp': _numeric_timestamp_family,
        'keyboard': _keyboard_walk_family,
        'mutation': _word_mutation_family,
    }

    per_family = max(1, count // len(families))
    labelled: List[tuple[str, str]] = []
    for name, generator in families.items():
        for domain in generator(per_family, tlds):
            labelled.append((domain, name))

    if len(labelled) < count:
        labelled.extend((domain, 'entropy') for domain in _random_entropy_family(count - len(labelled), tlds))

    random.shuffle(labelled)
    return labelled[:count]


def load_domains_from_csv(
    path: Path,
    domain_column: int | str = 0,
    skip_header: bool = False
) -> list:
    """Load domains from arbitrary CSV file"""
    if not path or not path.exists():
        return []

    domains: list[str] = []
    with path.open('r', encoding='utf-8') as handle:
        reader = csv.reader(handle)
        header = None

        if skip_header:
            header = next(reader, None)
        elif isinstance(domain_column, str):
            header = next(reader, None)

        if isinstance(domain_column, str):
            if not header:
                raise ValueError('CSV header required to access column by name')
            try:
                domain_index = header.index(domain_column)
            except ValueError as exc:
                raise ValueError(f"Column '{domain_column}' not found in {path}") from exc
        else:
            domain_index = domain_column

        for row in reader:
            if not row:
                continue
            try:
                domains.append(row[domain_index].strip())
            except IndexError:
                continue

    return [d for d in domains if d]


def create_test_dataset(
    output_file: str | Path = DEFAULT_TEST_FILE,
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
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with output_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['domain', 'label'])
        writer.writerows(data)
    
    print(f"✓ Dataset saved: {output_path}")
    print(f"  Total: {len(data)} domains")
    print(f"  Legitimate: {legitimate_count}")
    print(f"  Malicious: {malicious_count}")


def create_whitelist(output_file: str | Path = DATA_ROOT / 'test' / 'whitelist.txt'):
    """Create whitelist of known legitimate domains"""
    whitelist = [
        'google.com', 'microsoft.com', 'apple.com', 'amazon.com',
        'github.com', 'stackoverflow.com', 'wikipedia.org', 'reddit.com',
        'youtube.com', 'twitter.com', 'facebook.com', 'instagram.com',
        'linkedin.com', 'spotify.com', 'netflix.com', 'slack.com',
        'discord.com', 'telegram.org', 'whatsapp.com', 'skype.com'
    ]
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with output_path.open('w', encoding='utf-8') as f:
        for domain in whitelist:
            f.write(f"{domain}\n")
    
    print(f"✓ Whitelist saved: {output_path}")


def create_blacklist(output_file: str | Path = DATA_ROOT / 'test' / 'blacklist.txt'):
    """Create blacklist of known malicious domains"""
    blacklist = [
        'malicious-dga-1.com', 'phishing-site.net', 'c2-command.info',
        'ransomware-payment.com', 'botnet-control.xyz', 'trojan-download.ru',
        'xkjhqwerty.com', 'zxcvbnmasdfgh.net', 'qwertyuiopasdf.org',
        'mnbvcxzasdfgh.com', 'infected-system.top', 'malware-distribution.pw'
    ]
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with output_path.open('w', encoding='utf-8') as f:
        for domain in blacklist:
            f.write(f"{domain}\n")
    
    print(f"✓ Blacklist saved: {output_path}")


def create_training_dataset(
    output_file: str | Path = DEFAULT_TRAIN_FILE,
    top_domains_csv: str | Path | None = None,
    malicious_csv: str | Path | None = None,
    synthetic_legitimate: int = 5000,
    synthetic_malicious_per_family: int = 800
):
    """Create enriched training dataset from available CSVs and synthetic data"""

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    legitimate: list[str] = []
    if top_domains_csv:
        legitimate_path = Path(top_domains_csv)
        legitimate = load_domains_from_csv(legitimate_path, domain_column=1, skip_header=False)
        print(f"Loaded {len(legitimate)} legitimate domains from {legitimate_path}")

    if len(legitimate) < synthetic_legitimate:
        needed = synthetic_legitimate - len(legitimate)
        print(f"Generating {needed} additional legitimate domains (synthetic)")
        legitimate.extend(generate_legitimate_domains(needed))

    malicious: list[str] = []
    if malicious_csv:
        malicious_path = Path(malicious_csv)
        malicious = load_domains_from_csv(malicious_path, domain_column='domain', skip_header=True)
        print(f"Loaded {len(malicious)} malicious domains from {malicious_path}")

    synthetic_malicious = generate_dga_domains(len(malicious) + synthetic_malicious_per_family)
    print(f"Generated {len(synthetic_malicious)} synthetic DGA domains")
    malicious.extend(synthetic_malicious)

    # Build dataset
    data: list[list[str | int]] = [[domain, 0] for domain in legitimate]
    data.extend([domain, 1] for domain in malicious)
    random.shuffle(data)

    with output_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['domain', 'label'])
        writer.writerows(data)

    print(f"✓ Training dataset saved: {output_path}")
    print(f"  Total: {len(data)} domains")
    print(f"  Legitimate: {len(legitimate)}")
    print(f"  Malicious: {len(malicious)}")


if __name__ == '__main__':
    print("=" * 60)
    print("DNS Shield - Test Data Generation")
    print("=" * 60)
    
    create_test_dataset()
    create_training_dataset(
        top_domains_csv=DATA_ROOT / 'top-1m.csv',
        malicious_csv=DATA_ROOT / 'tif_domains.csv'
    )
    create_whitelist()
    create_blacklist()
    
    print("\n✓ Test data generation complete!")