import hashlib

def hash_string(text: str) -> str:
    """
    Generate SHA-256 hash of a string.
    
    Args:
        text: Input string to hash
        
    Returns:
        Hexadecimal hash string
    """
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def hash_file(file_path: str) -> str:
    """
    Generate SHA-256 hash of a file's contents.
    
    Args:
        file_path: Path to file
        
    Returns:
        Hexadecimal hash string
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()
