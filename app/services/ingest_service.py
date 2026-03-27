from __future__ import annotations

import hashlib
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import psycopg
from openai import OpenAI
from psycopg.types.json import Json


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


DATABASE_URL = os.getenv("DATABASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "64"))
OPENAI_MAX_RETRIES = int(os.getenv("OPENAI_MAX_RETRIES", "3"))

if not DATABASE_URL:
    raise RuntimeError("Falta la variable de entorno DATABASE_URL.")

if not OPENAI_API_KEY:
    raise RuntimeError("Falta la variable de entorno OPENAI_API_KEY.")

_client = OpenAI(api_key=OPENAI_API_KEY)


@dataclass
class IngestMetadata:
    title: str
    raw_text: str
    source: str
    authority: str
    jurisdiction: str = "MX"
    collection: str = "general"
    canonical_url: Optional[str] = None
    identifiers: Optional[Dict[str, Any]] = None
    publication_date: Optional[str] = None
    effective_date: Optional[str] = None
    doc_type: str = "document"
    source_format: str = "txt"
    mime_type: str = "text/plain"


def _normalize_psycopg_dsn(dsn: str) -> str:
    dsn = dsn.strip()
    if dsn.startswith("postgresql+psycopg://"):
        return dsn.replace("postgresql+psycopg://", "postgresql://", 1)
    if dsn.startswith("postgres+psycopg://"):
        return dsn.replace("postgres+psycopg://", "postgresql://", 1)
    return dsn


def _normalize_text_for_hash(text: str) -> str:
    text = text.replace("\ufeff", "")
    text = text.replace("\x00", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _vector_literal(values: List[float]) -> str:
    return "[" + ",".join(f"{v:.8f}" for v in values) + "]"


def _batched(items: List[str], batch_size: int) -> Iterable[List[str]]:
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def simple_chunker(
    text: str,
    max_chars: int = 1800,
    overlap: int = 100,
) -> List[str]:
    text = _normalize_text_for_hash(text)
    if not text:
        return []

    if max_chars <= 0:
        raise ValueError("max_chars debe ser mayor a 0.")

    if overlap < 0:
        raise ValueError("overlap no puede ser negativo.")

    if overlap >= max_chars:
        raise ValueError("overlap debe ser menor que max_chars.")

    if len(text) <= max_chars:
        return [text]

    chunks: List[str] = []
    start = 0
    step = max_chars - overlap

    while start < len(text):
        end = min(start + max_chars, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start += step

    return chunks


def embed_texts(texts: List[str]) -> List[List[float]]:
    for i, text in enumerate(texts):
        if not text or not text.strip():
            raise ValueError(f"Texto vacío para embeddings en índice {i}")

        if len(text) > 32000:
            raise ValueError(
                f"Chunk demasiado grande antes de embeddings en índice {i}: "
                f"{len(text)} caracteres. Debe fragmentarse antes."
            )

    embeddings: List[List[float]] = []

    for batch in _batched(texts, EMBEDDING_BATCH_SIZE):
        last_error: Optional[Exception] = None

        for attempt in range(1, OPENAI_MAX_RETRIES + 1):
            try:
                response = _client.embeddings.create(
                    model=EMBEDDING_MODEL,
                    input=batch,
                )
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Error generando embeddings (intento %s/%s): %s",
                    attempt,
                    OPENAI_MAX_RETRIES,
                    exc,
                )
                if attempt < OPENAI_MAX_RETRIES:
                    time.sleep(1.5 * attempt)

        if last_error is not None:
            raise last_error

    return embeddings


def normalize_input_document(
    title: str,
    raw_text: str,
    source: str,
    authority: str,
    jurisdiction: str = "MX",
    collection: str = "general",
    canonical_url: Optional[str] = None,
    identifiers: Optional[Dict[str, Any]] = None,
    publication_date: Optional[str] = None,
    effective_date: Optional[str] = None,
    doc_type: str = "document",
    source_format: str = "txt",
    mime_type: str = "text/plain",
) -> IngestMetadata:
    base_identifiers = dict(identifiers or {})
    base_identifiers.setdefault("title", title.strip())
    base_identifiers.setdefault("jurisdiction", jurisdiction.strip())
    base_identifiers.setdefault("collection", collection.strip())

    if effective_date is not None:
        base_identifiers.setdefault("effective_date", effective_date)

    return IngestMetadata(
        title=title.strip(),
        raw_text=_normalize_text_for_hash(raw_text),
        source=source.strip(),
        authority=authority.strip(),
        jurisdiction=jurisdiction.strip(),
        collection=collection.strip(),
        canonical_url=canonical_url.strip() if canonical_url else None,
        identifiers=base_identifiers,
        publication_date=publication_date,
        effective_date=effective_date,
        doc_type=doc_type.strip(),
        source_format=source_format.strip(),
        mime_type=mime_type.strip(),
    )

def json_to_raw_text(data: Dict[str, Any]) -> str:
    """
    Convierte un JSON ya estructurado a texto plano utilizable para ingesta.
    """
    if not isinstance(data, dict):
        raise TypeError("json_to_raw_text espera un dict.")

    preferred_fields = [
        "raw_text",
        "text",
        "content",
        "body",
        "document_text",
    ]

    for field in preferred_fields:
        value = data.get(field)
        if isinstance(value, str) and value.strip():
            return _normalize_text_for_hash(value)
        if isinstance(value, list):
            joined = "\n\n".join(str(x) for x in value if str(x).strip())
            if joined.strip():
                return _normalize_text_for_hash(joined)

    # Fallback: serializar el JSON de forma legible
    return _normalize_text_for_hash(
        "\n".join(
            f"{k}: {v}"
            for k, v in data.items()
            if isinstance(v, (str, int, float, bool)) and str(v).strip()
        )
    )


def extract_identifiers_from_text(
    raw_text: str,
    filename: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Heurísticas ligeras para archivos TXT/PDF subidos manualmente.
    No sustituye parsers especializados.
    """
    text = _normalize_text_for_hash(raw_text)
    identifiers: Dict[str, Any] = {}

    if filename:
        identifiers["filename"] = filename

    article_match = re.search(
        r"""(?imx)
        \bart[ií]culo\s+
        (
            \d+(?:o\.)?
            (?:
                \s*-\s*[A-Z]
                |
                -[A-Z]
            )?
            (?:
                \s+|-
            )?
            (?:Bis|Ter|Qu[aá]ter|Quater)?
        )
        \b
        """,
        text,
    )

    if article_match:
        identifiers["articulo"] = article_match.group(1).strip()

    expediente_match = re.search(
        r"\b\d{1,6}/\d{4}\b",
        text,
    )
    if expediente_match:
        identifiers["expediente"] = expediente_match.group(0)

    if "código fiscal de la federación" in text.lower():
        identifiers.setdefault("ordenamiento", "Código Fiscal de la Federación")
        identifiers.setdefault("abreviatura", "CFF")

    if "ley de amparo" in text.lower():
        identifiers.setdefault("ordenamiento", "Ley de Amparo")
        identifiers.setdefault("abreviatura", "LA")

    if "constitución política de los estados unidos mexicanos" in text.lower():
        identifiers.setdefault("ordenamiento", "Constitución Política de los Estados Unidos Mexicanos")
        identifiers.setdefault("abreviatura", "CPEUM")

    return identifiers


def extract_identifiers_from_json(
    data: Dict[str, Any],
    filename: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Extrae metadata útil desde JSON estructurado.
    """
    identifiers: Dict[str, Any] = {}

    if filename:
        identifiers["filename"] = filename

    candidate_fields = [
        "identifiers",
        "metadata",
    ]

    for field in candidate_fields:
        value = data.get(field)
        if isinstance(value, dict):
            identifiers.update(value)

    passthrough_fields = [
        "articulo",
        "expediente",
        "ordenamiento",
        "abreviatura",
        "doc_kind",
        "source_type",
        "collection",
    ]

    for field in passthrough_fields:
        value = data.get(field)
        if value is not None:
            identifiers[field] = value

    return identifiers

def _normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def _split_text_by_paragraphs(text: str, max_chars: int, overlap: int = 120) -> List[str]:
    text = _normalize_whitespace(text)
    if not text:
        return []

    if len(text) <= max_chars:
        return [text]

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return _split_text_hard(text, max_chars=max_chars, overlap=overlap)

    chunks: List[str] = []
    current = ""

    for p in paragraphs:
        candidate = p if not current else f"{current}\n\n{p}"
        if len(candidate) <= max_chars:
            current = candidate
            continue

        if current:
            chunks.append(current.strip())

        if len(p) <= max_chars:
            current = p
        else:
            hard_parts = _split_text_hard(p, max_chars=max_chars, overlap=overlap)
            chunks.extend(hard_parts[:-1])
            current = hard_parts[-1]

    if current:
        chunks.append(current.strip())

    return [c for c in chunks if c]

def _split_text_hard(text: str, max_chars: int, overlap: int = 120) -> List[str]:
    text = _normalize_whitespace(text)
    if not text:
        return []

    chunks: List[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + max_chars, text_len)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= text_len:
            break

        start = max(end - overlap, start + 1)

    return chunks

def _deduplicate_chunk_payloads(
    chunk_payloads: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Elimina chunks duplicados dentro del mismo documento usando chunk_hash.
    Conserva la primera ocurrencia y descarta las siguientes.
    """
    deduped: List[Dict[str, Any]] = []
    seen_hashes: set[str] = set()

    for item in chunk_payloads:
        chunk_hash = item.get("chunk_hash")
        chunk_text = item.get("chunk_text", "")

        if not chunk_hash:
            chunk_hash = _sha256_text(_normalize_text_for_hash(str(chunk_text)))
            item["chunk_hash"] = chunk_hash

        if chunk_hash in seen_hashes:
            logger.warning(
                "Chunk duplicado detectado y omitido. hash=%s text_preview=%r",
                chunk_hash,
                str(chunk_text)[:120],
            )
            continue

        seen_hashes.add(chunk_hash)
        deduped.append(item)

    return deduped

def _expand_oversized_prebuilt_chunks(
    prebuilt_chunks: List[Dict[str, Any]],
    max_chars: int,
    overlap: int = 120,
) -> List[Dict[str, Any]]:
    expanded: List[Dict[str, Any]] = []

    for idx, item in enumerate(prebuilt_chunks):
        raw_chunk_text = str(item.get("chunk_text", ""))
        chunk_text = _normalize_text_for_hash(raw_chunk_text)
        chunk_meta = dict(item.get("chunk_meta", {}) or {})

        if not chunk_text:
            continue

        if len(chunk_text) <= max_chars:
            expanded.append(
                {
                    "chunk_text": chunk_text,
                    "chunk_meta": chunk_meta,
                    "chunk_hash": _sha256_text(chunk_text),
                }
            )
            continue

        parts = _split_text_by_paragraphs(
            chunk_text,
            max_chars=max_chars,
            overlap=overlap,
        )

        total_parts = len(parts)
        for part_idx, part_text in enumerate(parts, start=1):
            normalized_part = _normalize_text_for_hash(part_text)
            if not normalized_part:
                continue

            part_meta = dict(chunk_meta)
            part_meta["is_split_from_prebuilt"] = True
            part_meta["split_part"] = part_idx
            part_meta["split_total"] = total_parts
            part_meta["original_prebuilt_index"] = idx

            expanded.append(
                {
                    "chunk_text": normalized_part,
                    "chunk_meta": part_meta,
                    "chunk_hash": _sha256_text(normalized_part),
                }
            )

    return expanded

def upsert_document_and_reindex(
    title: str,
    raw_text: str,
    source: str,
    authority: str,
    jurisdiction: str = "MX",
    collection: str = "general",
    canonical_url: Optional[str] = None,
    identifiers: Optional[Dict[str, Any]] = None,
    publication_date: Optional[str] = None,
    effective_date: Optional[str] = None,
    prebuilt_chunks: Optional[List[Dict[str, Any]]] = None,
    chunk_max_chars: int = 1800,
    chunk_overlap: int = 100,
    doc_type: str = "document",
    source_format: str = "txt",
    mime_type: str = "text/plain",
) -> int:
    """
    Inserta o actualiza un documento y reindexa sus chunks.

    Tu schema real:
    - documents.id
    - chunks.chunk_id
    - chunk_vectors.chunk_id

    Si prebuilt_chunks viene informado:
    [
        {
            "chunk_text": "...",
            "chunk_meta": {...}
        }
    ]
    """
    doc = normalize_input_document(
        title=title,
        raw_text=raw_text,
        source=source,
        authority=authority,
        jurisdiction=jurisdiction,
        collection=collection,
        canonical_url=canonical_url,
        identifiers=identifiers,
        publication_date=publication_date,
        effective_date=effective_date,
        doc_type=doc_type,
        source_format=source_format,
        mime_type=mime_type,
    )

    if not doc.raw_text:
        raise ValueError("raw_text está vacío. No se puede ingerir un documento vacío.")

    document_hash = _sha256_text(doc.raw_text)

    chunk_payloads: List[Dict[str, Any]] = []

    if prebuilt_chunks is not None:
        expanded_prebuilt = _expand_oversized_prebuilt_chunks(
            prebuilt_chunks=prebuilt_chunks,
            max_chars=chunk_max_chars,
            overlap=chunk_overlap,
        )

        for item in expanded_prebuilt:
            if not isinstance(item, dict):
                raise TypeError("Cada prebuilt_chunk debe ser un dict.")

            if "chunk_text" not in item:
                raise ValueError("Cada prebuilt_chunk debe incluir 'chunk_text'.")

            chunk_text = _normalize_text_for_hash(str(item["chunk_text"]))
            if not chunk_text:
                continue

            chunk_meta = item.get("chunk_meta") or {}
            if not isinstance(chunk_meta, dict):
                raise TypeError("'chunk_meta' debe ser un dict si viene informado.")

            chunk_payloads.append(
                {
                    "chunk_text": chunk_text,
                    "chunk_meta": chunk_meta,
                    "chunk_hash": item.get("chunk_hash") or _sha256_text(chunk_text),
                    "start_offset": item.get("start_offset"),
                    "end_offset": item.get("end_offset"),
                }
            )
    else:
        normalized_text = _normalize_text_for_hash(doc.raw_text)
        running_offset = 0

        for chunk_text in simple_chunker(
            text=normalized_text,
            max_chars=chunk_max_chars,
            overlap=chunk_overlap,
        ):
            start_offset = normalized_text.find(chunk_text, running_offset)
            if start_offset < 0:
                start_offset = running_offset
            end_offset = start_offset + len(chunk_text)
            running_offset = end_offset

            chunk_payloads.append(
                {
                    "chunk_text": chunk_text,
                    "chunk_meta": {},
                    "chunk_hash": _sha256_text(chunk_text),
                    "start_offset": start_offset,
                    "end_offset": end_offset,
                }
            )

    if not chunk_payloads:
        raise ValueError("No se generaron chunks válidos para indexar.")
    
    chunk_payloads = _deduplicate_chunk_payloads(chunk_payloads)

    embeddings = embed_texts([item["chunk_text"] for item in chunk_payloads])

    if len(embeddings) != len(chunk_payloads):
        raise RuntimeError("La cantidad de embeddings no coincide con la cantidad de chunks.")

    dsn = _normalize_psycopg_dsn(DATABASE_URL)

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            row = None

            if doc.canonical_url:
                cur.execute(
                    """
                    SELECT id, version
                    FROM documents
                    WHERE canonical_url = %s
                    LIMIT 1
                    """,
                    (doc.canonical_url,),
                )
                row = cur.fetchone()

            if row is None:
                cur.execute(
                    """
                    SELECT id, version
                    FROM documents
                    WHERE document_hash = %s
                    LIMIT 1
                    """,
                    (document_hash,),
                )
                row = cur.fetchone()

            if row:
                document_id, current_version = row
                new_version = int(current_version or 1) + 1

                cur.execute(
                    """
                    UPDATE documents
                    SET
                        source = %s,
                        doc_type = %s,
                        authority = %s,
                        publication_date = %s,
                        identifiers = %s,
                        canonical_url = %s,
                        raw_text = %s,
                        ingestion_date = NOW(),
                        version = %s,
                        source_format = %s,
                        mime_type = %s
                    WHERE id = %s
                    """,
                    (
                        doc.source,
                        doc.doc_type,
                        doc.authority,
                        doc.publication_date,
                        Json(doc.identifiers or {}),
                        doc.canonical_url,
                        doc.raw_text,
                        new_version,
                        doc.source_format,
                        doc.mime_type,
                        document_id,
                    ),
                )

                logger.info(
                    "Documento existente encontrado. Reindexando document_id=%s version=%s",
                    document_id,
                    new_version,
                )
            else:
                cur.execute(
                    """
                    INSERT INTO documents (
                        source,
                        doc_type,
                        authority,
                        publication_date,
                        identifiers,
                        canonical_url,
                        raw_text,
                        document_hash,
                        ingestion_date,
                        version,
                        source_format,
                        mime_type
                    )
                    VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, NOW(), %s, %s, %s
                    )
                    RETURNING id
                    """,
                    (
                        doc.source,
                        doc.doc_type,
                        doc.authority,
                        doc.publication_date,
                        Json(doc.identifiers or {}),
                        doc.canonical_url,
                        doc.raw_text,
                        document_hash,
                        1,
                        doc.source_format,
                        doc.mime_type,
                    ),
                )
                document_id = cur.fetchone()[0]

                logger.info("Documento insertado document_id=%s", document_id)

            cur.execute(
                """
                DELETE FROM chunk_vectors
                WHERE chunk_id IN (
                    SELECT chunk_id
                    FROM chunks
                    WHERE document_id = %s
                )
                """,
                (document_id,),
            )

            cur.execute(
                """
                DELETE FROM chunks
                WHERE document_id = %s
                """,
                (document_id,),
            )

            normalized_doc_text = _normalize_text_for_hash(doc.raw_text)
            rolling_find_start = 0

            for idx, (payload, embedding) in enumerate(zip(chunk_payloads, embeddings), start=1):
                chunk_text = payload["chunk_text"]
                chunk_meta = payload.get("chunk_meta") or {}

                start_offset = payload.get("start_offset")
                end_offset = payload.get("end_offset")

                if start_offset is None or end_offset is None:
                    found_at = normalized_doc_text.find(chunk_text, rolling_find_start)
                    if found_at < 0:
                        found_at = normalized_doc_text.find(chunk_text)
                    if found_at < 0:
                        start_offset = 0
                        end_offset = len(chunk_text)
                    else:
                        start_offset = found_at
                        end_offset = found_at + len(chunk_text)
                        rolling_find_start = end_offset

                cur.execute(
                    """
                    INSERT INTO chunks (
                        document_id,
                        chunk_text,
                        chunk_hash,
                        start_offset,
                        end_offset,
                        chunk_index,
                        chunk_meta
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING chunk_id
                    """,
                    (
                        document_id,
                        chunk_text,
                        payload["chunk_hash"],
                        start_offset,
                        end_offset,
                        idx,
                        Json(chunk_meta),
                    ),
                )
                chunk_id = cur.fetchone()[0]

                cur.execute(
                    """
                    INSERT INTO chunk_vectors (
                        chunk_id,
                        embedding
                    )
                    VALUES (%s, %s::vector)
                    """,
                    (
                        chunk_id,
                        _vector_literal(embedding),
                    ),
                )

            conn.commit()

    logger.info(
        "Reindexación completada. document_id=%s chunks=%s",
        document_id,
        len(chunk_payloads),
    )
    return document_id