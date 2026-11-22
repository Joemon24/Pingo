from pathlib import Path


def parse(path: Path) -> list[tuple[str, str, str]]:
    """Split Markdown into one database row per level-two heading."""
    entries = []
    for block in path.read_text(encoding="utf-8").split("\n## ")[1:]:
        title, _, content = block.partition("\n")
        if not title.strip() or not content.strip():
            raise ValueError(f"Empty knowledge entry in {path.name}")
        entries.append((path.stem, title.strip(), content.strip()))
    return entries


def main() -> None:
    from app import db

    root = Path(__file__).resolve().parents[1] / "knowledge_base"
    rows = [row for path in sorted(root.glob("*.md")) for row in parse(path)]
    if not rows:
        raise ValueError("Knowledge base has no entries")
    with db.connect() as conn, conn.transaction(), conn.cursor() as cursor:
        cursor.executemany(
            """
            insert into knowledge_base (section, title, content) values (%s, %s, %s)
            on conflict (section, title) do update
              set content = excluded.content, updated_at = now()
            """,
            rows,
        )
    print(f"Upserted {len(rows)} knowledge entries")


if __name__ == "__main__":
    main()
