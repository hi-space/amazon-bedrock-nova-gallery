import secrets
import string


def seed():
    return secrets.randbelow(2147483647)

def random_id(length=12):
    characters = string.ascii_letters + string.digits
    return ''.join(secrets.choice(characters) for _ in range(length))
