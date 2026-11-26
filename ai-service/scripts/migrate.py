"""Apply database/migrations/*.sql in order, once each (tracked in schema_migrations).

    python scripts/migrate.py            # apply pending migrations
    python scripts/migrate.py --status   # list applied / pending
    python scripts/migrate.py --rollback 003_integrity_and_index_status.sql   # undo one migration (asks first)

Back up first (scripts/backup-db.ps1). Rollback files live in database/migrations/down/ and only
remove what the migration added - they never delete products, images or embeddings.
"""
from __future__ import annotations

import argparse

import psycopg

from _common import PROJECT_ROOT
from app.config import get_settings

MIGRATIONS_DIR = PROJECT_ROOT / "database" / "migrations"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--rollback", metavar="MIGRATION_FILE")
    ap.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    args = ap.parse_args()
    if args.rollback:
        return rollback(args.rollback, args.yes)
    files = sorted(MIGRATIONS_DIR.glob("*.sql"))
    with psycopg.connect(get_settings().database_url, autocommit=False) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS schema_migrations (version TEXT PRIMARY KEY, "
                     "applied_at TIMESTAMPTZ NOT NULL DEFAULT now())")
        applied = {r[0] for r in conn.execute("SELECT version FROM schema_migrations")}
        conn.commit()
        for f in files:
            if f.name in applied:
                print(f"  applied  {f.name}")
                continue
            if args.status:
                print(f"  PENDING  {f.name}")
                continue
            print(f"  applying {f.name} ...", end=" ")
            with conn.transaction():
                conn.execute(f.read_text(encoding="utf-8"))
                conn.execute("INSERT INTO schema_migrations(version) VALUES (%s)", (f.name,))
            print("ok")
        ver = conn.execute("SELECT extversion FROM pg_extension WHERE extname='vector'").fetchone()
        print(f"pgvector version: {ver[0] if ver else 'NOT INSTALLED'}")


def rollback(name: str, yes: bool) -> None:
    down = MIGRATIONS_DIR / "down" / name.replace(".sql", ".down.sql")
    if not down.exists():
        raise SystemExit(f"No rollback script {down.relative_to(PROJECT_ROOT)}")
    with psycopg.connect(get_settings().database_url) as conn:
        applied = conn.execute("SELECT 1 FROM schema_migrations WHERE version = %s", (name,)).fetchone()
        if not applied:
            raise SystemExit(f"{name} is not applied")
        if not yes and input(f"Roll back {name}? Back up first. Type 'yes': ").strip().lower() != "yes":
            raise SystemExit("cancelled")
        with conn.transaction():
            conn.execute(down.read_text(encoding="utf-8"))
            conn.execute("DELETE FROM schema_migrations WHERE version = %s", (name,))
    print(f"rolled back {name}")


if __name__ == "__main__":
    main()
