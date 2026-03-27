from __future__ import annotations

from typing import List, Dict, Any
import re


def _safe_get(obj, attr: str, default=None):
    return getattr(obj, attr, default)


def _extract_article(chunk_text: str) -> str | None:
    if not chunk_text:
        return None

    m = re.search(r"\bart[ií]culo\s+(\d+[A-Z\-]*)\b", chunk_text, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()

    m = re.search(r"\b(\d{1,3}-[A-Z])\b", chunk_text, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()

    return None


def _build_reference(chunk) -> str:
    chunk_text = _safe_get(chunk, "chunk_text", "") or ""
    identifiers = _safe_get(chunk, "identifiers", {}) or {}

    expediente = identifiers.get("expediente")
    registro = identifiers.get("registro_digital")
    tesis = identifiers.get("tesis")
    tribunal = identifiers.get("tribunal")

    doc_type = (_safe_get(chunk, "doc_type", "") or "").lower()
    authority = _safe_get(chunk, "authority", None)

    if expediente and tribunal:
        return f"{tribunal}. Amparo en revisión {expediente}"
    if expediente:
        return f"Amparo en revisión {expediente}"
    if registro:
        return f"Tesis aislada. Registro digital {registro}"
    if tesis:
        return f"Tesis {tesis}"

    article = _extract_article(chunk_text)
    if article and authority:
        return f"{authority}, artículo {article}"
    if article:
        return f"Artículo {article}"

    if doc_type and authority:
        return f"{doc_type.title()} de {authority}"
    if authority:
        return f"Documento de {authority}"

    return f"Documento jurídico (chunk {chunk.chunk_id})"


def _infer_authority(chunk, reference: str) -> str:
    authority = _safe_get(chunk, "authority", None)
    chunk_text = (_safe_get(chunk, "chunk_text", "") or "").lower()

    invalid_authorities = {"user", "unknown", "upload", "n/a", "none"}
    if authority and authority.lower().strip() in invalid_authorities:
        authority = None

    if not authority:
        ref_lower = reference.lower()

        if "amparo" in ref_lower or "tesis" in ref_lower or "jurisprudencia" in ref_lower:
            return "Poder Judicial de la Federación (PJF)"

        if (
            "código fiscal de la federación" in chunk_text
            or "ley del impuesto" in chunk_text
            or "diario oficial de la federación" in chunk_text
        ):
            return "Cámara de Diputados del H. Congreso de la Unión"

        return "Documento jurídico"

    return authority


def _build_apa(chunk, reference: str, authority: str) -> str:
    publication_date = _safe_get(chunk, "publication_date", None)
    year = None

    if publication_date:
        try:
            year = publication_date.year
        except Exception:
            year = None

    if not year:
        year = "s. f."

    return f"{authority}. ({year}). {reference}."


def build_citations(chunks) -> List[Dict[str, Any]]:
    citations: List[Dict[str, Any]] = []
    seen = set()

    for c in chunks:
        reference = _build_reference(c)
        inferred_authority = _infer_authority(c, reference)
        apa = _build_apa(c, reference, inferred_authority)

        url = _safe_get(c, "canonical_url", None)
        if not url:
            identifiers = _safe_get(c, "identifiers", {}) or {}
            filename = identifiers.get("filename")
            doc_id = _safe_get(c, "document_id")

            if filename:
                url = f"local://{filename}"
            elif doc_id:
                url = f"doc://{doc_id}"

        key = (reference, url)
        if key in seen:
            continue
        seen.add(key)

        citations.append({
            "reference": reference,
            "format_apa": apa,
            "url": url,
            "chunk_id": c.chunk_id,
            "document_id": _safe_get(c, "document_id", None),
            "authority": inferred_authority,
            "doc_type": _safe_get(c, "doc_type", None),
        })

    return citations