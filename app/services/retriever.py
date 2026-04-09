from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import psycopg
import logging
import re
from openai import OpenAI
from app.core.config import settings

client = OpenAI(api_key=settings.OPENAI_API_KEY)

STOPWORDS = {"el","la","los","las","un","una","y","o","de","del","al","en","por","para","con","sin"}

logger = logging.getLogger(__name__)

@dataclass
class RetrievedChunk:
    chunk_id: int
    chunk_text: str
    score: float
    vec_sim: float = 0.0
    fts_rank: float = 0.0
    fts_norm: float = 0.0
    header_penalty: float = 1.0
    identifiers: Dict[str, Any] = field(default_factory=dict)
    chunk_meta: Dict[str, Any] = field(default_factory=dict)
    canonical_url: Optional[str] = None
    document_id: Optional[int] = None
    authority: Optional[str] = None
    doc_type: Optional[str] = None
    publication_date: Optional[Any] = None


def _db_url_psycopg() -> str:
    return settings.DATABASE_URL.replace("postgresql+psycopg://", "postgresql://")


def _embedding_to_pgvector_str(emb: list[float]) -> str:
    return "[" + ",".join(str(x) for x in emb) + "]"


def _extract_expediente(query: str) -> str | None:
    m = re.search(r"\b(\d{1,4}/\d{4})\b", query)
    return m.group(1) if m else None


def _normalize_norm_ref(value: str) -> str:
    s = (value or "").strip()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("–", "-").replace("—", "-")
    s = re.sub(r"\s*-\s*", "-", s)
    return s


def _parse_norm_reference(query: str) -> dict[str, Optional[str]]:
    q = query.strip()

    articulo = None
    fraccion = None
    apartado = None
    inciso = None
    abreviatura = None

    art_match = re.search(
        r"""(?ix)
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
        """,
        q,
    )
    if art_match:
        articulo = _normalize_norm_ref(art_match.group(1))

    frac_match = re.search(r"(?i)\bfracci[oó]n\s+([IVXLCDM]+)\b", q)
    if frac_match:
        fraccion = frac_match.group(1).upper()

    ap_match = re.search(r"(?i)\bapartado\s+([A-Z])\b", q)
    if ap_match:
        apartado = ap_match.group(1).upper()

    inc_match = re.search(r"(?i)\binciso\s+([a-z])\b", q)
    if inc_match:
        inciso = inc_match.group(1).lower()

    if re.search(r"(?i)\bCFF\b|c[oó]digo fiscal de la federaci[oó]n|codigo fiscal de la federacion", q):
        abreviatura = "CFF"
    elif re.search(r"(?i)\bCPEUM\b|constituci[oó]n pol[ií]tica de los estados unidos mexicanos", q):
        abreviatura = "CPEUM"
    elif re.search(r"(?i)\bconstituci[oó]n\b|\bconstitucional\b", q):
        abreviatura = "CPEUM"
    elif re.search(r"(?i)\bLey de Amparo\b", q):
        abreviatura = "LA"
    elif re.search(
        r"(?i)\bLFPCA\b|ley federal de procedimiento contencioso administrativo",
        q,
    ):
        abreviatura = "LFPCA"

    return {
        "articulo": articulo,
        "fraccion": fraccion,
        "apartado": apartado,
        "inciso": inciso,
        "abreviatura": abreviatura,
    }


def _resolve_document_ids(conn: psycopg.Connection, expediente: str | None, limit: int = 10) -> list[int] | None:
    if not expediente:
        return None

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id
            FROM documents
            WHERE identifiers->>'expediente' = %s
            ORDER BY version DESC, id DESC
            LIMIT %s;
            """,
            (expediente, limit),
        )
        rows = [r[0] for r in cur.fetchall()]
        if rows:
            return rows

        cur.execute(
            """
            SELECT id
            FROM documents
            WHERE canonical_url ILIKE ('%%' || %s || '%%')
               OR raw_text ILIKE ('%%' || %s || '%%')
            ORDER BY version DESC, id DESC
            LIMIT %s;
            """,
            (expediente, expediente, limit),
        )
        rows = [r[0] for r in cur.fetchall()]
        return rows or None


def _is_structured_exact_query(norm_ref: dict[str, Optional[str]]) -> bool:
    if norm_ref["articulo"] and norm_ref["abreviatura"]:
        return True

    return bool(norm_ref["articulo"]) and bool(
        norm_ref["apartado"] or norm_ref["fraccion"] or norm_ref["inciso"]
    )


def _extract_apartado_from_meta(meta: dict) -> str | None:
    if not meta:
        return None

    apartado = meta.get("apartado")
    if apartado:
        return str(apartado).upper()

    path = meta.get("path") or ""
    m = re.search(r"/ap:([A-Z])(?:/|$)", path)
    if m:
        return m.group(1).upper()

    return None


def _extract_fraccion_from_meta(meta: dict) -> str | None:
    if not meta:
        return None

    fraccion = meta.get("fraccion")
    if fraccion:
        return str(fraccion).upper()

    path = meta.get("path") or ""
    m = re.search(r"/fr:([IVXLCDM]+)(?:/|$)", path)
    if m:
        return m.group(1).upper()

    return None


def _extract_inciso_from_meta(meta: dict) -> str | None:
    if not meta:
        return None

    inciso = meta.get("inciso")
    if inciso:
        return str(inciso).lower()

    path = meta.get("path") or ""
    m = re.search(r"/inc:([a-z])(?:/|$)", path)
    if m:
        return m.group(1).lower()

    return None


# ─────────────────────────────────────────────────────────────────────────────
# MODIFICACIÓN PRINCIPAL: query_embedding se recibe como parámetro.
# Ya NO se llama a client.embeddings aquí dentro.
# ─────────────────────────────────────────────────────────────────────────────
def _run_search_query(
    *,
    query: str,
    query_embedding: list[float],   # ← recibe el vector pre-calculado
    top_k: int,
    w_vec: float,
    w_fts: float,
    article_ref: Optional[str],
    apartado_ref: Optional[str],
    fraccion_ref: Optional[str],
    inciso_ref: Optional[str],
    abreviatura_ref: Optional[str],
) -> List[RetrievedChunk]:
    expediente = _extract_expediente(query)

    q_norm = query.lower()
    q_norm = re.sub(r"\b(\d+)\s*-\s*([a-z])\b", r"\1 \2", q_norm, flags=re.IGNORECASE)
    s_clean = re.sub(r"[^\w\sáéíóúüñ-]", " ", q_norm, flags=re.UNICODE)
    words = [w for w in s_clean.split() if w not in STOPWORDS and len(w) >= 2]

    safe_words = [w for w in words if re.fullmatch(r"[a-záéíóúüñ0-9]+", w)]
    fts_strict = " ".join(safe_words[:8]) if safe_words else query
    fts_loose = " | ".join(safe_words) if safe_words else ""

    # Usar el embedding recibido — sin llamar a la API de OpenAI
    emb_str = _embedding_to_pgvector_str(query_embedding)

    db_url = _db_url_psycopg()
    TSV_COL = "chunk_tsv"

    # Reutilizar una sola conexión para resolver expediente y ejecutar la query
    with psycopg.connect(db_url) as conn:
        document_ids = _resolve_document_ids(conn, expediente)

        sql = f"""
        WITH
        q AS (
            SELECT
                (%s)::vector AS qvec,
                websearch_to_tsquery('spanish', %s) AS tsq_strict,
                to_tsquery('spanish', NULLIF(%s, '')) AS tsq_loose,
                (%s)::text AS expediente,
                %s::int[] AS doc_ids,
                %s::text AS articulo_ref,
                %s::text AS fraccion_ref,
                %s::text AS apartado_ref,
                %s::text AS inciso_ref,
                %s::text AS abreviatura_ref
        ),
        base AS (
            SELECT
                c.chunk_id,
                c.document_id,
                c.chunk_text,
                c.chunk_meta,
                d.identifiers,
                d.canonical_url,
                d.authority,
                d.doc_type,
                d.publication_date,
                (v.embedding <=> q.qvec) AS vec_dist,
                COALESCE(ts_rank_cd(c.{TSV_COL}, q.tsq_loose), 0) +
                (COALESCE(ts_rank_cd(c.{TSV_COL}, q.tsq_strict), 0) * 2.5) AS fts_rank,
                CASE
                    WHEN c.chunk_text ~* '^(\\s*AMPARO\\s+EN\\s+REVISI[ÓO]N|\\s*PONENTE:|\\s*SECRETARIO:|\\s*RECURRENTE|\\s*QUEJOSA)'
                        THEN 0.85
                    ELSE 1.00
                END AS header_penalty,
                (
                    CASE
                        WHEN q.abreviatura_ref IS NOT NULL
                             AND d.identifiers->>'abreviatura' = q.abreviatura_ref
                        THEN 0.45 ELSE 0.0
                    END
                    +
                    CASE
                        WHEN q.articulo_ref IS NOT NULL
                             AND d.identifiers->>'articulo' = q.articulo_ref
                        THEN 0.55 ELSE 0.0
                    END
                    +
                    CASE
                        WHEN q.fraccion_ref IS NOT NULL
                             AND c.chunk_meta->>'fraccion' = q.fraccion_ref
                        THEN 0.32 ELSE 0.0
                    END
                    +
                    CASE
                        WHEN q.inciso_ref IS NOT NULL
                             AND c.chunk_meta->>'inciso' = q.inciso_ref
                        THEN 0.18 ELSE 0.0
                    END
                    +
                    CASE
                        WHEN q.apartado_ref IS NOT NULL
                             AND (c.chunk_meta->>'path') LIKE ('%%/ap:' || q.apartado_ref || '%%')
                        THEN 0.28 ELSE 0.0
                    END
                ) AS metadata_bonus
            FROM chunk_vectors v
            JOIN chunks c ON c.chunk_id = v.chunk_id
            JOIN documents d ON c.document_id = d.id
            CROSS JOIN q
            WHERE
                (
                    (q.doc_ids IS NOT NULL AND c.document_id = ANY(q.doc_ids))
                    OR
                    (q.doc_ids IS NULL AND (q.expediente IS NULL OR c.chunk_text ILIKE ('%%' || q.expediente || '%%')))
                )
                AND (q.abreviatura_ref IS NULL OR d.identifiers->>'abreviatura' = q.abreviatura_ref)
                AND (q.articulo_ref IS NULL OR d.identifiers->>'articulo' = q.articulo_ref)
                AND (q.fraccion_ref IS NULL OR c.chunk_meta->>'fraccion' = q.fraccion_ref)
                AND (q.inciso_ref IS NULL OR c.chunk_meta->>'inciso' = q.inciso_ref)
                AND (
                    q.apartado_ref IS NULL
                    OR (c.chunk_meta->>'path') LIKE ('%%/ap:' || q.apartado_ref || '%%')
                )
        ),
        stats AS (
            SELECT GREATEST(MAX(fts_rank), 1e-9) AS max_fts FROM base
        ),
        scored AS (
            SELECT
                b.*,
                GREATEST(0.0, LEAST(1.0, 1.0 - (b.vec_dist / 2.0))) AS vec_sim,
                (b.fts_rank / s.max_fts) AS fts_norm
            FROM base b
            CROSS JOIN stats s
        )
        SELECT
            s.chunk_id,
            s.chunk_text,
            s.vec_sim,
            s.fts_rank,
            s.fts_norm,
            s.header_penalty,
            ((%s * s.vec_sim) + (%s * s.fts_norm) + s.metadata_bonus) * s.header_penalty AS score,
            s.document_id,
            s.identifiers,
            s.chunk_meta,
            s.canonical_url,
            s.authority,
            s.doc_type,
            s.publication_date
        FROM scored s
        ORDER BY score DESC
        LIMIT %s;
        """

        params = (
            emb_str,
            fts_strict,
            fts_loose,
            expediente,
            document_ids,
            article_ref,
            fraccion_ref,
            apartado_ref,
            inciso_ref,
            abreviatura_ref,
            w_vec,
            w_fts,
            top_k,
        )

        out: List[RetrievedChunk] = []
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

            for (
                chunk_id,
                chunk_text,
                vec_sim,
                fts_rank,
                fts_norm,
                header_penalty,
                score,
                document_id,
                identifiers,
                chunk_meta,
                canonical_url,
                authority,
                doc_type,
                publication_date,
            ) in rows:
                out.append(
                    RetrievedChunk(
                        chunk_id=int(chunk_id),
                        chunk_text=chunk_text,
                        score=float(score),
                        vec_sim=float(vec_sim),
                        fts_rank=float(fts_rank),
                        fts_norm=float(fts_norm),
                        header_penalty=float(header_penalty),
                        identifiers=identifiers if identifiers else {},
                        chunk_meta=chunk_meta if chunk_meta else {},
                        canonical_url=canonical_url,
                        document_id=int(document_id) if document_id is not None else None,
                        authority=authority,
                        doc_type=doc_type,
                        publication_date=publication_date,
                    )
                )

    return out


def _merge_unique_chunks(chunks: List[RetrievedChunk], top_k: int) -> List[RetrievedChunk]:
    seen: set[int] = set()
    merged: List[RetrievedChunk] = []

    for c in chunks:
        if c.chunk_id in seen:
            continue
        seen.add(c.chunk_id)
        merged.append(c)
        if len(merged) >= top_k:
            break

    return merged


def _sort_by_source_order(chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
    def _key(c: RetrievedChunk):
        meta = c.chunk_meta or {}
        return (
            int(meta.get("source_order", 10**9)),
            int(meta.get("order_index", 10**9)),
            c.chunk_id,
        )
    return sorted(chunks, key=_key)


def _inject_pre_context(
    primary_chunks: List[RetrievedChunk],
    candidate_chunks: List[RetrievedChunk],
    norm_ref: dict[str, Optional[str]],
    top_k: int,
) -> List[RetrievedChunk]:
    if not primary_chunks:
        return primary_chunks

    articulo_ref = norm_ref.get("articulo")
    apartado_ref = norm_ref.get("apartado")
    fraccion_ref = norm_ref.get("fraccion")

    extras: List[RetrievedChunk] = []

    for c in candidate_chunks:
        meta = c.chunk_meta or {}
        ids = c.identifiers or {}

        if (ids.get("articulo") or meta.get("articulo")) != articulo_ref:
            continue

        meta_apartado = _extract_apartado_from_meta(meta)
        meta_fraccion = _extract_fraccion_from_meta(meta)

        if apartado_ref and meta_apartado != apartado_ref:
            continue

        if fraccion_ref:
            if meta_fraccion != fraccion_ref:
                continue
            if meta.get("unit_id") == "pre":
                extras.append(c)
        else:
            if apartado_ref and meta.get("unit_id") == "pre" and meta_apartado == apartado_ref:
                extras.append(c)

    merged = _merge_unique_chunks(extras + primary_chunks, top_k)
    return merged


def _count_exact_structural_matches(
    chunks: List[RetrievedChunk],
    norm_ref: dict[str, Optional[str]],
) -> int:
    count = 0
    for c in chunks:
        meta = c.chunk_meta or {}
        ids = c.identifiers or {}

        articulo = ids.get("articulo") or meta.get("articulo")
        apartado = _extract_apartado_from_meta(meta)
        fraccion = _extract_fraccion_from_meta(meta)
        inciso = _extract_inciso_from_meta(meta)

        if norm_ref.get("articulo") and articulo != norm_ref["articulo"]:
            continue
        if norm_ref.get("apartado") and apartado != norm_ref["apartado"]:
            continue
        if norm_ref.get("fraccion") and fraccion != norm_ref["fraccion"]:
            continue
        if norm_ref.get("inciso") and inciso != norm_ref["inciso"]:
            continue

        count += 1
    return count


def _keep_exact_subtree_if_available(
    chunks: List[RetrievedChunk],
    norm_ref: dict[str, Optional[str]],
    top_k: int,
) -> List[RetrievedChunk]:
    exact: List[RetrievedChunk] = []

    for c in chunks:
        meta = c.chunk_meta or {}
        ids = c.identifiers or {}

        articulo = ids.get("articulo") or meta.get("articulo")
        apartado = _extract_apartado_from_meta(meta)
        fraccion = _extract_fraccion_from_meta(meta)
        inciso = _extract_inciso_from_meta(meta)

        if norm_ref.get("articulo") and articulo != norm_ref["articulo"]:
            continue
        if norm_ref.get("apartado") and apartado != norm_ref["apartado"]:
            continue
        if norm_ref.get("fraccion") and fraccion != norm_ref["fraccion"]:
            continue
        if norm_ref.get("inciso") and inciso != norm_ref["inciso"]:
            continue

        exact.append(c)

    if exact:
        exact = _sort_by_source_order(_merge_unique_chunks(exact, top_k))
        return exact

    return _merge_unique_chunks(chunks, top_k)


def search_hybrid(query: str, top_k: int = 8, w_vec: float = 0.55, w_fts: float = 0.25) -> List[RetrievedChunk]:
    if top_k < 1:
        top_k = 1

    query_embedding = client.embeddings.create(
        model=settings.EMBEDDINGS_MODEL,
        input=query
    ).data[0].embedding

    norm_ref = _parse_norm_reference(query)
    
    articulo_ref = norm_ref["articulo"]
    fraccion_ref = norm_ref["fraccion"]
    apartado_ref = norm_ref["apartado"]
    inciso_ref = norm_ref["inciso"]
    abreviatura_ref = norm_ref["abreviatura"]

    is_exact = _is_structured_exact_query(norm_ref)

    logger.debug("norm_ref=%s", norm_ref)
    logger.debug(
        "strict_search=%s",
        {
            "articulo_ref": articulo_ref,
            "apartado_ref": apartado_ref,
            "fraccion_ref": fraccion_ref,
            "inciso_ref": inciso_ref,
            "abreviatura_ref": abreviatura_ref,
            "is_exact": is_exact,
        },
    )

    strict_results = _run_search_query(
        query=query,
        query_embedding=query_embedding,
        top_k=top_k,
        w_vec=w_vec,
        w_fts=w_fts,
        article_ref=articulo_ref,
        apartado_ref=apartado_ref,
        fraccion_ref=fraccion_ref,
        inciso_ref=inciso_ref,
        abreviatura_ref=abreviatura_ref,
    )

    strict_results = _inject_pre_context(
        primary_chunks=strict_results,
        candidate_chunks=strict_results,
        norm_ref=norm_ref,
        top_k=top_k,
    )

    if not is_exact:
        return strict_results[:top_k]

    exact_count = _count_exact_structural_matches(strict_results, norm_ref)
    if exact_count >= 1:
        return _keep_exact_subtree_if_available(strict_results, norm_ref, top_k)

    fallback_results: List[RetrievedChunk] = list(strict_results)

    if inciso_ref:
        fallback_results.extend(
            _run_search_query(
                query=query,
                query_embedding=query_embedding,   # ← mismo vector
                top_k=top_k,
                w_vec=w_vec,
                w_fts=w_fts,
                article_ref=articulo_ref,
                apartado_ref=apartado_ref,
                fraccion_ref=fraccion_ref,
                inciso_ref=None,
                abreviatura_ref=abreviatura_ref,
            )
        )

    merged = _merge_unique_chunks(fallback_results, top_k)
    if len(merged) >= min(top_k, 3):
        return _keep_exact_subtree_if_available(merged, norm_ref, top_k)

    if fraccion_ref:
        fallback_results.extend(
            _run_search_query(
                query=query,
                query_embedding=query_embedding,   # ← mismo vector
                top_k=top_k,
                w_vec=w_vec,
                w_fts=w_fts,
                article_ref=articulo_ref,
                apartado_ref=apartado_ref,
                fraccion_ref=None,
                inciso_ref=None,
                abreviatura_ref=abreviatura_ref,
            )
        )

    merged = _merge_unique_chunks(fallback_results, top_k)
    if len(merged) >= min(top_k, 3):
        return _keep_exact_subtree_if_available(merged, norm_ref, top_k)

    if apartado_ref:
        fallback_results.extend(
            _run_search_query(
                query=query,
                query_embedding=query_embedding,   # ← mismo vector
                top_k=top_k,
                w_vec=w_vec,
                w_fts=w_fts,
                article_ref=articulo_ref,
                apartado_ref=None,
                fraccion_ref=None,
                inciso_ref=None,
                abreviatura_ref=abreviatura_ref,
            )
        )

    merged = _merge_unique_chunks(fallback_results, top_k)
    return _keep_exact_subtree_if_available(merged, norm_ref, top_k)
