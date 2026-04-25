from datetime import datetime, timedelta
from jose import jwt
# Ganti passlib dengan pwdlib
from pwdlib import PasswordHash

SECRET_KEY = "RAHASIA_NEGARA_TIDAK_BOLEH_BOCOR_123!"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 

# Konfigurasi PwdLib menggunakan algoritma bcrypt
pwd_context = PasswordHash.recommended()

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt