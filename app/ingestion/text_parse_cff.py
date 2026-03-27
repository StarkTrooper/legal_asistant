from __future__ import annotations

from pathlib import Path

from normative_parser import parse_cff_articles


CFF_PATH = Path("C:/asistente_juridico/app/ingestion/normas/cff.txt")


def main() -> None:
    raw_text = CFF_PATH.read_text(encoding="utf-8", errors="ignore")
    articles = parse_cff_articles(raw_text)

    print(f"Total artículos detectados: {len(articles)}")

    targets = ["1o", "4o-A", "17-A", "17-H Bis", "20-Bis", "69-B", "69-B Bis"]
    detected = {a.articulo: a for a in articles}

    for key in targets:
        art = detected.get(key)
        print("=" * 60)
        print(f"Buscando: {key}")
        if not art:
            print("NO DETECTADO")
            continue

        print(f"Título: {art.titulo}")
        print(f"Units: {len(art.units)}")
        for unit in art.units[:8]:
            print(f"- {unit.unit_type} | {unit.unit_id} | {unit.path}")
            print(unit.text[:180].replace("\n", " "))


if __name__ == "__main__":
    main()
