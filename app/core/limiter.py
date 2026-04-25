from slowapi import Limiter
from slowapi.util import get_remote_address

# Membuat mesin pembatas berdasarkan alamat IP (remote address) pengguna
limiter = Limiter(key_func=get_remote_address)