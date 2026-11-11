"""Optional Supabase login. Disabled unless SUPABASE_JWT_SECRET is set (local development stays open).

The browser signs in with Supabase (magic link or Google) and sends the access token as `Authorization: Bearer`.
The server checks the HS256 signature with the project's JWT secret (Supabase -> Project Settings -> API -> JWT
secret), the expiry and the audience, and uses the token's `sub` as the owner id of projects.
"""
import base64
import hashlib
import hmac
import json
import os
import time

SECRET = os.environ.get("SUPABASE_JWT_SECRET", "")
URL = os.environ.get("SUPABASE_URL", "")
ANON_KEY = os.environ.get("SUPABASE_ANON_KEY", "")
ENABLED = bool(SECRET)
MAX_QUEUED_PER_USER = int(os.environ.get("KAGU_MAX_QUEUED_PER_USER", "3"))


class AuthError(Exception):
    pass


def _b64(s):
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def verify(token):
    """Returns {"id", "email"} or raises AuthError."""
    try:
        h, p, s = token.split(".")
        if json.loads(_b64(h)).get("alg") != "HS256":
            raise AuthError("Unsupported token type")
        good = hmac.new(SECRET.encode(), f"{h}.{p}".encode(), hashlib.sha256).digest()
        if not hmac.compare_digest(good, _b64(s)):
            raise AuthError("Bad token signature")
        c = json.loads(_b64(p))
    except AuthError:
        raise
    except Exception:
        raise AuthError("Malformed token")
    if c.get("exp", 0) < time.time():
        raise AuthError("Session expired, sign in again")
    if c.get("aud") not in ("authenticated", None):
        raise AuthError("Wrong token audience")
    if not c.get("sub"):
        raise AuthError("Token has no user")
    return {"id": c["sub"], "email": c.get("email", "")}


def user_of(request):
    """None when auth is disabled; the user dict when signed in; raises AuthError otherwise."""
    if not ENABLED:
        return None
    a = request.headers.get("authorization", "")
    if not a.lower().startswith("bearer "):
        raise AuthError("Sign in to use Kagu")
    return verify(a[7:].strip())
