"""Mapping for ML dataset attack types to numerical labels."""

# Mapping of attack types to numerical labels
attack_type_to_label = {
    'Normal': 0,
    'DDoS_UDP': 1,
    'DDoS_ICMP': 2,
    'Ransomware': 3,
    'DDoS_HTTP': 4,
    'SQL_injection': 5,
    'Uploading': 6,
    'DDoS_TCP': 7,
    'Backdoor': 8,
    'Vulnerability_scanner': 9,
    'Port_Scanning': 10,
    'XSS': 11,
    'Password': 12,
    'MITM': 13,
    'Fingerprinting': 14
}

# Reverse mapping for converting numerical predictions back to labels
label_to_attack_type = {v: k for k, v in attack_type_to_label.items()}

def map_attack_type(attack_type):
    """Map attack type string to numerical label."""
    return attack_type_to_label.get(attack_type, 0)  # Default to Normal (0) if unknown

def map_label_to_attack(label):
    """Map numerical label back to attack type string."""
    return label_to_attack_type.get(label, 'Normal')  # Default to Normal if unknown
