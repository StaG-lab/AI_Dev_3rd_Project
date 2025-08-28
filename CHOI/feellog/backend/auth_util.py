import hashlib

def hash_email(email: str) -> str:
    """이메일을 SHA512로 해싱합니다."""
    return hashlib.sha512(email.encode('utf-8')).hexdigest()

def hash_password(password: str, salt: str) -> bytes:
    """1차 해싱된 비밀번호를 솔트(1차 해싱된 이메일)와 함께 2차 해싱합니다."""
    # 1. 비밀번호 1차 해싱
    primary_hashed_password = hashlib.sha512(password.encode('utf-8')).hexdigest()
    
    # 2. 솔트를 이용해 2차 해싱
    salted_password = primary_hashed_password + salt
    secondary_hashed_password_hex = hashlib.sha512(salted_password.encode('utf-8')).hexdigest()
    
    # 3. 16진수 문자열을 바이트로 변환하여 반환
    return bytes.fromhex(secondary_hashed_password_hex)

def verify_password(provided_password: str, salt: str, stored_hash: bytes) -> bool:
    """입력된 비밀번호가 저장된 해시와 일치하는지 확인합니다."""
    new_hash = hash_password(provided_password, salt)
    return new_hash == stored_hash
