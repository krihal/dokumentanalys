"""Encrypted per-user document vault.

Accounts, keys and documents live in VAULT_DIR (default ./vault):

    vault.db                    users and document keys (SQLite)
    blobs/<user_id>/<doc>.idx   encrypted search index for one document

Original files are not kept: only what search and answers need (text
passages, metadata, embeddings), so answers can name the source document.

Cryptography (libsodium via PyNaCl):

  * Login password: Argon2id hash (nacl.pwhash.argon2id.str).
  * Every user has an X25519 key pair. The private key is encrypted with
    XChaCha20-Poly1305 under a key derived from the user's password with
    Argon2id (own salt and a higher cost than the login hash, so the stored
    hash does not unlock it). The password is never stored; without it the
    private key, and therefore every document, is unreadable. An admin
    password reset therefore deletes the user's documents and keys.
    (Accounts from before used a separate passphrase, key_mode "passphrase";
    they are moved to the password at their next login.)
  * Every document gets a random 256-bit data key. Its blobs are encrypted
    with XChaCha20-Poly1305 under that key, with the user id, document id and
    blob kind as associated data, so blobs cannot be swapped between
    documents or users. The data key is sealed to the user's public key
    (crypto_box_seal), so adding documents needs only the public key.
  * Deleting a document deletes its sealed data key (with SQLite
    secure_delete), which makes the blobs permanently undecryptable even if
    the files survive on disk.

Nothing in this module logs or prints document content, passwords or keys.
"""

import os
import re
import sqlite3
import time
import uuid
from contextlib import closing
from pathlib import Path

import nacl.bindings as sodium
import nacl.exceptions
import nacl.pwhash
import nacl.utils
from nacl.public import PrivateKey, PublicKey, SealedBox

VAULT_DIR = Path(os.environ.get("VAULT_DIR", Path(__file__).parent / "vault"))
DB_PATH = VAULT_DIR / "vault.db"
BLOB_DIR = VAULT_DIR / "blobs"

# Argon2id cost for the passphrase KDF: 256 MiB, ~1-2 s. Stored per user so
# it can be raised later without breaking existing keys.
KDF_OPS = nacl.pwhash.argon2id.OPSLIMIT_MODERATE
KDF_MEM = nacl.pwhash.argon2id.MEMLIMIT_MODERATE
# Login password hash: 64 MiB. Brute force is also throttled in the app.
PW_OPS = nacl.pwhash.argon2id.OPSLIMIT_INTERACTIVE
PW_MEM = nacl.pwhash.argon2id.MEMLIMIT_INTERACTIVE

MIN_PASSWORD_LEN = 12
USERNAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{1,31}$")

_NONCE = sodium.crypto_aead_xchacha20poly1305_ietf_NPUBBYTES
# Verified against when the username does not exist, so a login attempt costs
# the same whether or not the account exists.
_DUMMY_HASH = nacl.pwhash.argon2id.str(b"dummy-password", opslimit=PW_OPS, memlimit=PW_MEM)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY,
    username TEXT UNIQUE NOT NULL,
    pw_hash BLOB NOT NULL,
    must_change_pw INTEGER NOT NULL DEFAULT 1,
    public_key BLOB,
    wrapped_private_key BLOB,
    kdf_salt BLOB,
    kdf_ops INTEGER,
    kdf_mem INTEGER,
    key_mode TEXT NOT NULL DEFAULT 'passphrase',
    created REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    sealed_key BLOB NOT NULL,
    created REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS documents_user ON documents(user_id);
"""


class VaultError(Exception):
    """A user-facing error; the message never contains secret data."""


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def _connect() -> sqlite3.Connection:
    VAULT_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(VAULT_DIR, 0o700)
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA secure_delete = ON")  # overwrite deleted rows (sealed keys)
    return conn


def init() -> int:
    """Create the vault if needed. Returns the number of stored original files
    removed (earlier versions kept them)."""
    old = os.umask(0o077)
    try:
        with closing(_connect()) as conn, conn:
            conn.executescript(_SCHEMA)
        os.chmod(DB_PATH, 0o600)
        BLOB_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    finally:
        os.umask(old)
    with closing(_connect()) as conn, conn:
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(users)")}
        if "key_mode" not in columns:  # vaults created before the password became the key
            conn.execute("ALTER TABLE users ADD COLUMN key_mode TEXT NOT NULL DEFAULT 'passphrase'")
    removed = 0
    for orig in BLOB_DIR.glob("*/*.orig"):
        orig.unlink(missing_ok=True)
        removed += 1
    return removed


def _encrypt(key: bytes, plaintext: bytes, aad: bytes) -> bytes:
    nonce = nacl.utils.random(_NONCE)
    return nonce + sodium.crypto_aead_xchacha20poly1305_ietf_encrypt(plaintext, aad, nonce, key)


def _decrypt(key: bytes, blob: bytes, aad: bytes) -> bytes:
    nonce, ct = blob[:_NONCE], blob[_NONCE:]
    return sodium.crypto_aead_xchacha20poly1305_ietf_decrypt(ct, aad, nonce, key)


def _write_private(path: Path, data: bytes) -> None:
    """Write atomically with mode 0600."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def _user_blob_dir(user_id: int) -> Path:
    d = BLOB_DIR / str(int(user_id))
    d.mkdir(mode=0o700, parents=True, exist_ok=True)
    return d


def _blob_path(user_id: int, doc_id: str, kind: str) -> Path:
    uuid.UUID(doc_id)  # rejects anything that is not a plain UUID (path traversal)
    return _user_blob_dir(user_id) / f"{doc_id}.{kind}"


def _doc_aad(user_id: int, doc_id: str, kind: str) -> bytes:
    return f"vr-doc|{int(user_id)}|{doc_id}|{kind}".encode()


def _sk_aad(user_id: int) -> bytes:
    return f"vr-sk|{int(user_id)}".encode()


def _derive_kek(passphrase: str, salt: bytes, ops: int, mem: int) -> bytes:
    return nacl.pwhash.argon2id.kdf(
        sodium.crypto_aead_xchacha20poly1305_ietf_KEYBYTES,
        passphrase.encode(),
        salt,
        opslimit=ops,
        memlimit=mem,
    )


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------


def check_password_policy(password: str) -> None:
    if len(password) < MIN_PASSWORD_LEN:
        raise VaultError(f"Lösenordet måste vara minst {MIN_PASSWORD_LEN} tecken.")


def create_user(username: str, password: str) -> int:
    username = username.strip().lower()
    if not USERNAME_RE.match(username):
        raise VaultError("Ogiltigt användarnamn (a-z, 0-9, . _ -, 2-32 tecken).")
    check_password_policy(password)
    pw_hash = nacl.pwhash.argon2id.str(password.encode(), opslimit=PW_OPS, memlimit=PW_MEM)
    with closing(_connect()) as conn, conn:
        try:
            cur = conn.execute(
                "INSERT INTO users (username, pw_hash, must_change_pw, created) VALUES (?, ?, 1, ?)",
                (username, pw_hash, time.time()),
            )
        except sqlite3.IntegrityError:
            raise VaultError("Användarnamnet finns redan.") from None
        return cur.lastrowid


def get_user(user_id: int) -> sqlite3.Row | None:
    with closing(_connect()) as conn:
        return conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()


def get_user_by_name(username: str) -> sqlite3.Row | None:
    with closing(_connect()) as conn:
        return conn.execute("SELECT * FROM users WHERE username = ?", (username.strip().lower(),)).fetchone()


def list_users() -> list[dict]:
    with closing(_connect()) as conn:
        rows = conn.execute(
            "SELECT u.id, u.username, u.must_change_pw, u.public_key IS NOT NULL AS has_keys, u.created, "
            "COUNT(d.id) AS documents FROM users u LEFT JOIN documents d ON d.user_id = u.id "
            "GROUP BY u.id ORDER BY u.username"
        ).fetchall()
    return [dict(r) for r in rows]


def verify_login(username: str, password: str) -> sqlite3.Row | None:
    """The user row if username and password match, else None. Same cost either way."""
    row = get_user_by_name(username)
    stored = row["pw_hash"] if row else _DUMMY_HASH
    try:
        nacl.pwhash.verify(stored, password.encode())
    except nacl.exceptions.InvalidkeyError:
        return None
    return row


def set_password(user_id: int, password: str, must_change: bool = False) -> None:
    """Set the login password of an account whose key does not depend on it.
    (For an account with a password-protected key use change_password.)"""
    user = get_user(user_id)
    if user and has_keys(user) and user["key_mode"] == "password":
        raise VaultError("Lösenordet skyddar kontots nyckel; använd change_password.")
    check_password_policy(password)
    pw_hash = nacl.pwhash.argon2id.str(password.encode(), opslimit=PW_OPS, memlimit=PW_MEM)
    with closing(_connect()) as conn, conn:
        conn.execute(
            "UPDATE users SET pw_hash = ?, must_change_pw = ? WHERE id = ?",
            (pw_hash, int(must_change), user_id),
        )


def _vacuum() -> None:
    """Rewrite the database file so no freed pages remain. secure_delete already
    zeroes deleted rows; on SSDs and copy-on-write file systems (APFS) old blocks
    can still survive physically, which is why full-disk encryption matters."""
    with closing(_connect()) as conn:
        conn.execute("VACUUM")


def _remove_blob_dir(user_id: int) -> None:
    d = BLOB_DIR / str(int(user_id))
    if d.exists():
        for p in d.iterdir():
            p.unlink(missing_ok=True)
        d.rmdir()


def delete_user(user_id: int) -> None:
    """Delete the account, its key pair and every document."""
    with closing(_connect()) as conn, conn:
        conn.execute("DELETE FROM documents WHERE user_id = ?", (user_id,))
        conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
    _remove_blob_dir(user_id)
    _vacuum()


def delete_all_documents(user_id: int) -> int:
    """Delete every document of the user (keys first: crypto-shredding). Returns the count."""
    with closing(_connect()) as conn, conn:
        n = conn.execute("DELETE FROM documents WHERE user_id = ?", (user_id,)).rowcount
    _remove_blob_dir(user_id)
    _vacuum()
    return n


# ---------------------------------------------------------------------------
# Key pair
# ---------------------------------------------------------------------------


def has_keys(user: sqlite3.Row) -> bool:
    return user["public_key"] is not None


def password_is_key(user: sqlite3.Row) -> bool:
    """True when the private key is protected by the login password (not a legacy passphrase)."""
    return has_keys(user) and user["key_mode"] == "password"


def _wrap(user_id: int, sk: PrivateKey, secret: str) -> tuple[bytes, bytes]:
    salt = nacl.utils.random(nacl.pwhash.argon2id.SALTBYTES)
    return _encrypt(_derive_kek(secret, salt, KDF_OPS, KDF_MEM), bytes(sk), _sk_aad(user_id)), salt


def create_keys(user_id: int, password: str) -> PrivateKey:
    """Generate the user's key pair and protect the private key with the password.

    Refuses to overwrite an existing key pair: that would orphan every document.
    """
    sk = PrivateKey.generate()
    wrapped, salt = _wrap(user_id, sk, password)
    with closing(_connect()) as conn, conn:
        cur = conn.execute(
            "UPDATE users SET public_key = ?, wrapped_private_key = ?, kdf_salt = ?, kdf_ops = ?, kdf_mem = ?, "
            "key_mode = 'password' WHERE id = ? AND public_key IS NULL",
            (bytes(sk.public_key), wrapped, salt, KDF_OPS, KDF_MEM, user_id),
        )
        if cur.rowcount != 1:
            raise VaultError("Nycklar finns redan för kontot.")
    return sk


def unlock(user_id: int, secret: str) -> PrivateKey | None:
    """The user's private key, or None if the password (or legacy passphrase) is wrong."""
    user = get_user(user_id)
    if not user or not has_keys(user):
        return None
    kek = _derive_kek(secret, user["kdf_salt"], user["kdf_ops"], user["kdf_mem"])
    try:
        raw = _decrypt(kek, user["wrapped_private_key"], _sk_aad(user_id))
    except nacl.exceptions.CryptoError:
        return None
    sk = PrivateKey(raw)
    if bytes(sk.public_key) != user["public_key"]:
        return None
    return sk


def password_kek(password: str) -> tuple[bytes, bytes]:
    """(key-encryption key, salt) for a password, to re-wrap a key later
    without keeping the password itself."""
    salt = nacl.utils.random(nacl.pwhash.argon2id.SALTBYTES)
    return _derive_kek(password, salt, KDF_OPS, KDF_MEM), salt


def protect_key_with_kek(user_id: int, sk: PrivateKey, kek: bytes, salt: bytes) -> None:
    """Move a legacy account to the password: re-wrap the key under the
    password's key-encryption key (from password_kek). Documents are untouched."""
    wrapped = _encrypt(kek, bytes(sk), _sk_aad(user_id))
    with closing(_connect()) as conn, conn:
        conn.execute(
            "UPDATE users SET wrapped_private_key = ?, kdf_salt = ?, kdf_ops = ?, kdf_mem = ?, key_mode = 'password' "
            "WHERE id = ?",
            (wrapped, salt, KDF_OPS, KDF_MEM, user_id),
        )


def change_password(user_id: int, sk: PrivateKey, new_password: str) -> None:
    """New login password and the key re-wrapped under it, in one transaction.
    Needs the unlocked key; documents stay readable."""
    check_password_policy(new_password)
    pw_hash = nacl.pwhash.argon2id.str(new_password.encode(), opslimit=PW_OPS, memlimit=PW_MEM)
    wrapped, salt = _wrap(user_id, sk, new_password)
    with closing(_connect()) as conn, conn:
        conn.execute(
            "UPDATE users SET pw_hash = ?, must_change_pw = 0, wrapped_private_key = ?, kdf_salt = ?, kdf_ops = ?, "
            "kdf_mem = ?, key_mode = 'password' WHERE id = ?",
            (pw_hash, wrapped, salt, KDF_OPS, KDF_MEM, user_id),
        )


def reset_password(user_id: int, temporary_password: str) -> int:
    """Admin reset. The key is protected by the old password, so it and every
    document are deleted; the user starts over. Returns the documents deleted."""
    check_password_policy(temporary_password)
    pw_hash = nacl.pwhash.argon2id.str(temporary_password.encode(), opslimit=PW_OPS, memlimit=PW_MEM)
    with closing(_connect()) as conn, conn:
        n = conn.execute("DELETE FROM documents WHERE user_id = ?", (user_id,)).rowcount
        conn.execute(
            "UPDATE users SET pw_hash = ?, must_change_pw = 1, public_key = NULL, wrapped_private_key = NULL, "
            "kdf_salt = NULL, kdf_ops = NULL, kdf_mem = NULL, key_mode = 'password' WHERE id = ?",
            (pw_hash, user_id),
        )
    _remove_blob_dir(user_id)
    _vacuum()
    return n


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------


def add_document(user_id: int, index_payload: bytes) -> str:
    """Encrypt and store one document's search index. Needs only the user's public key."""
    user = get_user(user_id)
    if not user or not has_keys(user):
        raise VaultError("Kontot saknar nycklar.")
    doc_id = str(uuid.uuid4())
    dek = nacl.utils.random(sodium.crypto_aead_xchacha20poly1305_ietf_KEYBYTES)
    sealed = SealedBox(PublicKey(user["public_key"])).encrypt(dek)
    old = os.umask(0o077)
    try:
        _write_private(_blob_path(user_id, doc_id, "idx"), _encrypt(dek, index_payload, _doc_aad(user_id, doc_id, "idx")))
    finally:
        os.umask(old)
    with closing(_connect()) as conn, conn:
        conn.execute(
            "INSERT INTO documents (id, user_id, sealed_key, created) VALUES (?, ?, ?, ?)",
            (doc_id, user_id, sealed, time.time()),
        )
    return doc_id


def list_documents(user_id: int) -> list[dict]:
    """Document ids and upload times; everything else is inside the encrypted blobs."""
    with closing(_connect()) as conn:
        rows = conn.execute(
            "SELECT id, created FROM documents WHERE user_id = ? ORDER BY created", (user_id,)
        ).fetchall()
    return [dict(r) for r in rows]


def read_document(user_id: int, doc_id: str, sk: PrivateKey) -> bytes:
    """Decrypt the search index of a document owned by user_id."""
    kind = "idx"
    with closing(_connect()) as conn:
        row = conn.execute(
            "SELECT sealed_key FROM documents WHERE id = ? AND user_id = ?", (doc_id, user_id)
        ).fetchone()
    if not row:
        raise VaultError("Dokumentet finns inte.")
    try:
        dek = SealedBox(sk).decrypt(row["sealed_key"])
        return _decrypt(dek, _blob_path(user_id, doc_id, kind).read_bytes(), _doc_aad(user_id, doc_id, kind))
    except (nacl.exceptions.CryptoError, FileNotFoundError):
        raise VaultError("Dokumentet kunde inte dekrypteras.") from None


def delete_document(user_id: int, doc_id: str) -> None:
    with closing(_connect()) as conn, conn:
        cur = conn.execute("DELETE FROM documents WHERE id = ? AND user_id = ?", (doc_id, user_id))
        if cur.rowcount != 1:
            raise VaultError("Dokumentet finns inte.")
    for kind in ("idx", "orig"):
        _blob_path(user_id, doc_id, kind).unlink(missing_ok=True)
