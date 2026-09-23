"""Manage accounts in the encrypted vault.

    uv run users.py add <användare>             skapa konto med tillfälligt lösenord
    uv run users.py list
    uv run users.py reset-password <användare>  nytt tillfälligt lösenord; raderar användarens nyckel och dokument
    uv run users.py delete <användare>          ta bort kontot och alla dess dokument
    uv run users.py import <användare> <mapp> [--ocr]
                                                kryptera in alla PDF/DOCX/HTML i en mapp i användarens bibliotek

The temporary password is printed once; the user must change it at first
login. Their own password then also protects the key that encrypts their
documents, so a forgotten password cannot be reset without losing them:
reset-password deletes the key and every document.

`import` needs only the user's public key, so it works without the
password, but the user must have logged in once (key pair created). It
loads the embedding model locally, like the worker.
"""

import argparse
import getpass
import json
import secrets
import sys
import time
from pathlib import Path

import vault


def _require(username: str):
    user = vault.get_user_by_name(username)
    if not user:
        sys.exit(f"Användaren {username} finns inte.")
    return user


def _temp_password() -> str:
    return secrets.token_urlsafe(15)


def cmd_add(args):
    password = getpass.getpass("Tillfälligt lösenord (tomt = generera): ") if args.prompt else ""
    password = password or _temp_password()
    try:
        vault.create_user(args.username, password)
    except vault.VaultError as ex:
        sys.exit(str(ex))
    print(f"Skapade {args.username.lower()}.")
    if not args.prompt:
        print(f"Tillfälligt lösenord (visas bara nu): {password}")
    print("Användaren väljer nytt lösenord och en lösenfras vid första inloggningen.")


def cmd_list(_args):
    rows = vault.list_users()
    if not rows:
        print("Inga användare.")
        return
    print(f"{'användare':<24} {'nycklar':<8} {'dokument':>8}  status")
    for r in rows:
        status = "måste byta lösenord" if r["must_change_pw"] else ""
        print(f"{r['username']:<24} {'ja' if r['has_keys'] else 'nej':<8} {r['documents']:>8}  {status}")


def cmd_reset(args):
    user = _require(args.username)
    if vault.has_keys(user):
        n = len(vault.list_documents(user["id"]))
        print(f"Lösenordet skyddar {user['username']}s krypteringsnyckel. Ett nytt lösenord kan inte öppna den,")
        print(f"så nyckeln och alla {n} dokument raderas permanent. Användaren börjar om med ett tomt bibliotek.")
        answer = input("Skriv användarnamnet för att fortsätta: ")
        if answer.strip().lower() != user["username"]:
            sys.exit("Avbrutet.")
    password = _temp_password()
    n = vault.reset_password(user["id"], password)
    print(f"Tillfälligt lösenord för {user['username']} (visas bara nu): {password}")
    if n:
        print(f"{n} dokument raderades.")


def cmd_delete(args):
    user = _require(args.username)
    answer = input(f"Ta bort {user['username']} och alla dokument permanent? Skriv användarnamnet: ")
    if answer.strip().lower() != user["username"]:
        sys.exit("Avbrutet.")
    vault.delete_user(user["id"])
    print("Borttagen.")


def cmd_import(args):
    user = _require(args.username)
    if not vault.has_keys(user):
        sys.exit("Användaren har inga nycklar ännu. Låt hen logga in och välja lösenord först.")
    folder = Path(args.folder).expanduser()
    files = sorted(p for p in folder.glob("**/*") if p.suffix.lower() in (".pdf", ".docx", ".htm", ".html") and p.is_file())
    if not files:
        sys.exit(f"Inga PDF- eller DOCX-filer i {folder}.")

    from ingest import DocumentError, load_embed_model, process_document

    print(f"{len(files)} filer. Laddar inbäddningsmodell...")
    model = load_embed_model()
    ok = failed = 0
    started = time.time()
    for i, path in enumerate(files, 1):
        try:
            data = path.read_bytes()
            record = process_document(data, path.name, model, ocr=args.ocr)
            vault.add_document(user["id"], json.dumps(record).encode())
            ok += 1
        except DocumentError:
            failed += 1
        print(f"\r{i}/{len(files)}  ({ok} inlästa, {failed} misslyckade)", end="", flush=True)
    print(f"\nKlart på {(time.time() - started) / 60:.1f} min. Dubbletter kontrolleras inte vid import.")


def main():
    parser = argparse.ArgumentParser(description="Hantera konton i valvet.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("add", help="Skapa konto")
    p.add_argument("username")
    p.add_argument("--prompt", action="store_true", help="Ange tillfälligt lösenord själv i stället för att generera")
    p.set_defaults(fn=cmd_add)
    sub.add_parser("list", help="Lista konton").set_defaults(fn=cmd_list)
    p = sub.add_parser("reset-password", help="Nytt tillfälligt lösenord (raderar nyckel och dokument)")
    p.add_argument("username")
    p.set_defaults(fn=cmd_reset)
    p = sub.add_parser("delete", help="Ta bort konto och dokument")
    p.add_argument("username")
    p.set_defaults(fn=cmd_delete)
    p = sub.add_parser("import", help="Kryptera in en mapp med PDF/DOCX")
    p.add_argument("username")
    p.add_argument("folder")
    p.add_argument("--ocr", action="store_true", help="OCR:a sidor utan textlager (kräver tesseract)")
    p.set_defaults(fn=cmd_import)
    args = parser.parse_args()
    vault.init()
    args.fn(args)


if __name__ == "__main__":
    main()
