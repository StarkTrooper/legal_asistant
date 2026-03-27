from __future__ import annotations

import re
import logging
from pathlib import Path
from urllib.parse import quote
from typing import Any, Dict, List, Optional

from app.ingestion.normative_parser import ParsedArticle, NormUnit, parse_cff_articles
from app.services.ingest_service import upsert_document_and_reindex


logger = logging.getLogger(__name__)

CFF_CANONICAL_URL = (
    "https://www.diputados.gob.mx/LeyesBiblio/pdf/CFF.pdf"
)


def _build_article_canonical_url(article: ParsedArticle) -> str:
    article_ref = quote(article.articulo, safe="")
    return f"{CFF_CANONICAL_URL}#articulo={article_ref}"


def _deduplicate_articles(articles: List[ParsedArticle]) -> List[ParsedArticle]:
    seen_articles = set()
    unique_articles: List[ParsedArticle] = []

    for art in articles:
        if art.articulo in seen_articles:
            logger.warning(
                "Artículo duplicado detectado en CFF: %s. Se conservará la primera ocurrencia.",
                art.articulo,
            )
            continue

        seen_articles.add(art.articulo)
        unique_articles.append(art)

    return unique_articles


def _resolve_cff_path(explicit_path: Optional[str] = None) -> Path:
    candidates: List[Path] = []

    if explicit_path:
        candidates.append(Path(explicit_path))

    here = Path(__file__).resolve()

    candidates.extend(
        [
            Path.cwd() / "cff.txt",
            here.parent / "normas" / "cff.txt",
            here.parent.parent / "cff.txt",
            here.parent.parent.parent / "cff.txt",
        ]
    )

    for path in candidates:
        if path.exists() and path.is_file():
            return path

    tried = "\n".join(f"- {str(p)}" for p in candidates)
    raise FileNotFoundError(
        "No se encontró cff.txt. Revisé estas rutas:\n"
        f"{tried}"
    )


def _render_chunk_text(article: ParsedArticle, unit: NormUnit) -> str:
    lines: List[str] = [
        article.ordenamiento,
        f"Artículo {article.articulo}",
    ]

    if unit.fraccion_id:
        lines.append(f"Fracción {unit.fraccion_id}")

    if unit.inciso_id:
        lines.append(f"Inciso {unit.inciso_id})")

    if unit.unit_type == "paragraph" and unit.unit_id == "pre":
        if unit.fraccion_id:
            lines.append(f"Preámbulo de la fracción {unit.fraccion_id}")
        else:
            lines.append("Preámbulo del artículo")
    elif unit.label not in {
        f"Fracción {unit.fraccion_id}" if unit.fraccion_id else "",
        f"Inciso {unit.inciso_id})" if unit.inciso_id else "",
    }:
        lines.append(unit.label)

    lines.append("")
    lines.append(unit.text.strip())

    return "\n".join(lines).strip()


def _build_chunk_meta(article: ParsedArticle, unit: NormUnit) -> Dict[str, Any]:
    return {
        "source_type": "normativa",
        "doc_kind": "articulo_normativo",
        "ordenamiento": article.ordenamiento,
        "abreviatura": article.abreviatura,
        "articulo": article.articulo,
        "titulo": article.titulo,
        "unit_type": unit.unit_type,
        "unit_id": unit.unit_id,
        "label": unit.label,
        "path": unit.path,
        "parent_path": unit.parent_path,
        "order_index": unit.order_index,
        "source_order": unit.source_order,
        "fraccion": unit.fraccion_id,
        "inciso": unit.inciso_id,
        "cited_as": unit.cited_as,
    }

def _truncate_cff_to_main_body(raw_text: str) -> str:
    """
    Corta el CFF al final del cuerpo normativo principal.

    Caso detectado:
    después del artículo 263 aparecen TRANSITORIOS históricos y
    ARTÍCULOS TRANSITORIOS DE DECRETOS DE REFORMA, que no deben
    ingerirse como parte del artículo 263.
    """
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")

    # Buscar el artículo 263 como último artículo del cuerpo principal
    art_263_match = re.search(
        r"(?im)^\s*art[ií]culo\s+263(?:\.\-|\.|\s)",
        text
    )
    if not art_263_match:
        return text

    tail = text[art_263_match.start():]

    # Buscar el primer marcador fuerte de cola editorial / histórica
    cut_markers = [
        r"(?im)^\s*TRANSITORIOS\s*$",
        r"(?im)^\s*ART[ÍI]CULOS\s+TRANSITORIOS\s+DE\s+DECRETOS\s+DE\s+REFORMA\s*$",
    ]

    cut_positions = []
    for pattern in cut_markers:
        m = re.search(pattern, tail)
        if m:
            cut_positions.append(art_263_match.start() + m.start())

    if not cut_positions:
        return text

    cut_at = min(cut_positions)
    return text[:cut_at].strip()


def _build_prebuilt_chunks(article: ParsedArticle) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []

    for unit in article.units:
        rendered_text = _render_chunk_text(article, unit)
        chunk_meta = _build_chunk_meta(article, unit)

        chunks.append(
            {
                "chunk_text": rendered_text,
                "chunk_meta": chunk_meta,
            }
        )

    return chunks



def ingest_cff(txt_path: Optional[str] = None) -> int:
    path = _resolve_cff_path(txt_path)
    raw_text = path.read_text(encoding="utf-8", errors="ignore")
    raw_text = _truncate_cff_to_main_body(raw_text)

    articles = parse_cff_articles(raw_text)
    articles = _deduplicate_articles(articles)

    if not articles:
        raise RuntimeError(
            "El parser no detectó artículos válidos en cff.txt. "
            "No conviene continuar con la ingesta."
        )

    total = 0

    for article in articles:
        identifiers = {
            "source_type": "normativa",
            "doc_kind": "articulo_normativo",
            "ordenamiento": article.ordenamiento,
            "abreviatura": article.abreviatura,
            "articulo": article.articulo,
            "titulo": article.titulo,
            "collection": "normativa_mexicana",
            "source_format": "txt",
            "parser_name": "parse_cff_articles",
            "parser_version": "2.0.0",
            "unit_types_present": sorted({u.unit_type for u in article.units}),
            "is_current": True,
        }

        prebuilt_chunks = _build_prebuilt_chunks(article)

        upsert_document_and_reindex(
            title=f"{article.abreviatura} {article.articulo}",
            raw_text=article.full_text,
            source="Cámara de Diputados / DOF",
            authority="Congreso de la Unión",
            jurisdiction="MX",
            collection="normativa_mexicana",
            canonical_url=_build_article_canonical_url(article),
            identifiers=identifiers,
            prebuilt_chunks=prebuilt_chunks,
            publication_date=None,
            effective_date=None,
            chunk_max_chars=1800,
            chunk_overlap=100,
            doc_type="normativa",
            source_format="txt",
            mime_type="text/plain",
        )

        total += 1

    return total


if __name__ == "__main__":
    inserted = ingest_cff()
    print(f"Ingesta CFF completada. Artículos procesados: {inserted}")