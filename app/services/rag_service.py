from __future__ import annotations

from typing import Dict, Any, List, Set
import re
import logging

from openai import OpenAI

from app.core.config import settings
from app.services.retriever import search_hybrid, RetrievedChunk
from app.services.citation_service import build_citations
from app.services.legal_reranker import legal_rerank


logger = logging.getLogger(__name__)
client = OpenAI(api_key=settings.OPENAI_API_KEY)

SYSTEM = """Eres un asistente jurídico mexicano.

Reglas:
- Responde SOLO con base en la evidencia proporcionada.
- Si la evidencia no alcanza, dilo explícitamente y pide el dato faltante.
- No inventes artículos, tesis, criterios, plazos o recursos.
- Devuelve respuesta clara, práctica y honesta.
- La lista de referencias será generada automáticamente por el sistema.
- No generes citas externas.
- No cites documentos que no estén en los fragmentos proporcionados.
- Cuando un fragmento tenga metadata estructural (artículo, apartado, fracción, inciso), debes tomar esa metadata como la ubicación jurídica correcta del fragmento.
- No confundas remisiones internas del texto con la ubicación estructural real del fragmento.
- Si el chunk pertenece al apartado D, fracción II, debes tratarlo como apartado D, fracción II, aunque el texto remita a otra parte de la norma.

Formato de respuesta:
- No muestres chunk_id, path, metadata interna ni trazas de depuración.
- No enumeres la “ubicación estructural de los fragmentos aportados”.
- Redacta como respuesta jurídica limpia para usuario final.
- Si la pregunta pide el contenido de una norma, inicia indicando directamente qué prevé la disposición consultada.
- Si hay incisos, preséntalos como incisos a), b), c), etc., sin repetir innecesariamente el preámbulo dos veces.
- Cierra, cuando corresponda, con una referencia breve del tipo: “Referencia: CFF, artículo 27, apartado D, fracción II.”
"""

STOPWORDS_CHECK = {
    "el","la","los","las","un","una","unos","unas","de","del","al","a","ante","bajo","con","contra","desde",
    "durante","en","entre","hacia","hasta","para","por","segun","según","sin","sobre","tras","y","o","u","e",
    "que","qué","cual","cuál","cuales","cuáles","quien","quién","quienes","quiénes","como","cómo",
    "donde","dónde","cuando","cuándo","este","esta","estos","estas","ese","esa","esos","esas",
    "mi","mis","tu","tus","su","sus","se","lo","le","les","ya","no","sí","si","muy","más","menos",
    "es","son","ser","fue","fueron","haber","ha","han","qué","pasa","según"
}

LEGAL_NOISE_CHECK = {
    "amparo","revisión","revision","expediente", "ley","federal",
    "procedimiento","contencioso","administrativo","artículo","articulo", "argumenta",
    "dice", "señala", "establece", "dispone", "explica", "explicq"
}

BASE_CRITICAL_TERMS = {
    "revocación", "revocacion",
    "relatividad",
    "prescripción", "prescripcion",
    "caducidad",
    "definitividad",
    "suspensión", "suspension",
    "competencia",
    "improcedencia",
    "sobreseimiento",
}

CRITICAL_BY_QUESTION_TYPE = {
    "deadline": {
        "plazo", "término", "termino", "días", "dias",
        "interponer", "vencimiento",
    },
    "remedy": {
        "recurso", "revocación", "revocacion",
        "apelación", "apelacion",
        "queja",
        "revisión", "revision",
        "impugnación", "impugnacion",
    },
    "definition": {
        "principio", "concepto", "definición", "definicion",
        "significa", "alcance",
    },
    "norm_reference": {
        "artículo", "articulo",
        "fracción", "fraccion",
        "párrafo", "parrafo",
        "inciso",
    },
    "holding": {
        "resolvió", "resolvio",
        "decidió", "decidio",
        "sentido",
        "fallo",
        "determinó", "determino",
    },
}


def _classify_question_type(question: str) -> str:
    q = question.lower()

    if any(x in q for x in [
        "qué establece", "que establece",
        "qué dice", "que dice",
        "artículo", "articulo",
        "fracción", "fraccion",
        "párrafo", "parrafo",
        "inciso",
        "69-c", "69-b"
    ]):
        return "norm_reference"

    if any(x in q for x in [
        "qué es", "que es",
        "define", "concepto de", "principio de"
    ]):
        return "definition"

    if any(x in q for x in [
        "cuál es el plazo", "cual es el plazo",
        "término", "termino",
        "cuántos días", "cuantos dias",
        "interponer"
    ]):
        return "deadline"

    if any(x in q for x in [
        "qué resolvió", "que resolvió",
        "qué decidió", "que decidió",
        "sentido del fallo", "resolutivo"
    ]):
        return "holding"

    if any(x in q for x in [
        "recurso", "revocación", "revocacion",
        "apelación", "apelacion",
        "queja", "revisión", "revision"
    ]):
        return "remedy"

    return "general"


def _get_dynamic_critical_terms(question: str, coverage: Dict[str, Any]) -> set[str]:
    qtype = _classify_question_type(question)
    question_terms = set(coverage.get("question_terms", []))

    dynamic_critical = set(BASE_CRITICAL_TERMS)
    dynamic_critical |= CRITICAL_BY_QUESTION_TYPE.get(qtype, set())

    return dynamic_critical.intersection(question_terms)


def _extract_question_terms(question: str) -> Set[str]:
    qtype = _classify_question_type(question)
    q = question.lower()

    q = re.sub(r"\b(\d+)\s*-\s*([a-z])\b", r"\1 \2", q, flags=re.IGNORECASE)
    q = re.sub(r"\b(\d+)([a-z])\b", r"\1 \2", q, flags=re.IGNORECASE)
    q = re.sub(r"[^\w\sáéíóúüñ/]", " ", q, flags=re.UNICODE)

    tokens = [t.strip() for t in q.split() if t.strip()]

    dynamic_legal_noise = set(LEGAL_NOISE_CHECK)

    if qtype == "remedy":
        dynamic_legal_noise.discard("recurso")
    if qtype == "definition":
        dynamic_legal_noise.discard("principio")
    if qtype == "norm_reference":
        dynamic_legal_noise.discard("artículo")
        dynamic_legal_noise.discard("articulo")

    terms: Set[str] = set()
    for t in tokens:
        if t in STOPWORDS_CHECK:
            continue
        if t in dynamic_legal_noise:
            continue

        if re.fullmatch(r"\d{1,4}/\d{4}", t):
            terms.add(t)
            continue

        if re.fullmatch(r"\d+", t):
            if len(t) >= 2:
                terms.add(t)
            continue

        if len(t) < 3:
            continue

        terms.add(t)

    return terms


def _normalize_text_for_match(text: str) -> str:
    s = text.lower()
    s = re.sub(r"\b(\d+)\s*-\s*([a-z])\b", r"\1 \2", s, flags=re.IGNORECASE)
    s = re.sub(r"\b(\d+)([a-z])\b", r"\1 \2", s, flags=re.IGNORECASE)
    return s


def _looks_like_full_article_request(question: str) -> bool:
    q = question.lower()
    asks_article = bool(re.search(r"\bart[ií]culo\b", q))
    asks_subunit = any(x in q for x in ["fracción", "fraccion", "apartado", "inciso", "párrafo", "parrafo"])
    return asks_article and not asks_subunit


def _compute_coverage(question: str, chunks: List[RetrievedChunk]) -> Dict[str, Any]:
    if not chunks:
        return {
            "question_terms": [],
            "matched_terms": [],
            "coverage_ratio": 0.0,
            "matches_any_strong_term": False,
            "core_terms": [],
            "matched_core_terms": [],
            "core_coverage_ratio": 0.0,
        }

    q_terms = _extract_question_terms(question)
    corpus = "\n".join(_normalize_text_for_match(c.chunk_text) for c in chunks)

    matched = []
    strong_match = False

    context_only = {"significa", "significq", "explica", "argumenta", "dice", "señala", "establece", "dispone", "plazo", "amparo", "ley", "articulo", "artículo", "recurso", "código", "fiscal", "federación", "codigo", "federacion"}

    core_terms = []
    matched_core = []

    qtype = _classify_question_type(question)

    if qtype == "deadline":
        context_only.discard("plazo")
    if qtype == "remedy":
        context_only.discard("recurso")
    if qtype == "norm_reference":
        context_only.difference_update({"articulo", "artículo"})

    for t in q_terms:
        is_expediente = bool(re.fullmatch(r"\d{1,4}/\d{4}", t))
        is_core = (t not in context_only) and not is_expediente

        if is_core:
            core_terms.append(t)

        if re.search(rf"\b{re.escape(t)}\b", corpus):
            matched.append(t)

            if is_expediente or re.fullmatch(r"\d+", t) or len(t) >= 6:
                strong_match = True

            if is_core:
                matched_core.append(t)

    coverage_ratio = (len(matched) / len(q_terms)) if q_terms else 0.0
    core_coverage_ratio = (len(matched_core) / len(core_terms)) if core_terms else 0.0

    return {
        "question_terms": sorted(q_terms),
        "matched_terms": sorted(set(matched)),
        "coverage_ratio": coverage_ratio,
        "matches_any_strong_term": strong_match,
        "core_terms": sorted(set(core_terms)),
        "matched_core_terms": sorted(set(matched_core)),
        "core_coverage_ratio": core_coverage_ratio,
    }


def _should_abstain(question: str, chunks: List[RetrievedChunk], coverage: Dict[str, Any]) -> bool:
    if not chunks:
        logger.warning(f"Guardrail activado para pregunta: '{question}'. Motivo: sin chunks")
        return True

    top_score = chunks[0].score if chunks else 0.0
    top_fts = chunks[0].fts_rank if chunks else 0.0
    coverage_ratio = coverage.get("coverage_ratio", 0.0)
    core_coverage_ratio = coverage.get("core_coverage_ratio", 0.0)
    strong = coverage.get("matches_any_strong_term", False)

    matched_core_terms = set(coverage.get("matched_core_terms", []))
    dynamic_critical = _get_dynamic_critical_terms(question, coverage)
    critical_missing = dynamic_critical - matched_core_terms

    if top_score < 0.35:
        logger.warning(f"Guardrail activado para pregunta: '{question}'. Motivo: top_score bajo ({top_score:.3f})")
        return True

    if core_coverage_ratio < 0.34:
        logger.warning(f"Guardrail activado para pregunta: '{question}'. Motivo: core_coverage_ratio bajo ({core_coverage_ratio:.2f})")
        return True

    if top_fts == 0.0 and coverage_ratio < 0.45:
        logger.warning(f"Guardrail activado para pregunta: '{question}'. Motivo: top_fts=0 y coverage_ratio bajo ({coverage_ratio:.2f})")
        return True

    if strong and len(matched_core_terms) == 0:
        logger.warning(f"Guardrail activado para pregunta: '{question}'. Motivo: strong=True pero sin matched_core_terms")
        return True

    if critical_missing:
        logger.warning(
            f"Guardrail activado para pregunta: '{question}'. "
            f"Motivo: critical terms faltantes {sorted(critical_missing)}. "
            f"Coverage: {core_coverage_ratio:.2f}"
        )
        return True

    return False


def _build_abstention_answer(question: str, chunks: List[RetrievedChunk], coverage: Dict[str, Any]) -> str:
    matched = ", ".join(coverage["matched_terms"]) if coverage["matched_terms"] else "ninguno"
    chunk_ids = ", ".join(str(c.chunk_id) for c in chunks[:10]) if chunks else "ninguno"

    return (
        "No es posible responder de forma confiable esa pregunta con la evidencia actualmente cargada.\n\n"
        f"Se localizaron fragmentos jurídicamente cercanos, pero no cobertura suficiente de los términos clave de la pregunta. "
        f"Términos detectados en evidencia: {matched}.\n\n"
        "Para responder con seguridad, necesito que cargues un documento o fragmento donde aparezca expresamente el tema consultado, "
        "o ampliar el corpus con la norma, tesis, jurisprudencia o sentencia correspondiente.\n\n"
        f"Chunk_id revisados: {chunk_ids}"
    )


def _parse_structured_ref(question: str) -> dict[str, str | None]:
    articulo = None
    apartado = None
    fraccion = None
    inciso = None

    art_match = re.search(r"(?i)art[ií]culo\s+([0-9]+(?:-[A-Z])?(?:\s+(?:Bis|Ter|Qu[aá]ter|Quater))?)", question)
    if art_match:
        articulo = re.sub(r"\s+", " ", art_match.group(1)).strip()

    ap_match = re.search(r"(?i)apartado\s+([A-Z])", question)
    if ap_match:
        apartado = ap_match.group(1).upper()

    frac_match = re.search(r"(?i)fracci[oó]n\s+([IVXLCDM]+)", question)
    if frac_match:
        fraccion = frac_match.group(1).upper()

    inc_match = re.search(r"(?i)inciso\s+([a-z])", question)
    if inc_match:
        inciso = inc_match.group(1).lower()

    return {
        "articulo": articulo,
        "apartado": apartado,
        "fraccion": fraccion,
        "inciso": inciso,
    }


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


def _has_exact_structural_match(question: str, chunks: list[RetrievedChunk]) -> bool:
    ref = _parse_structured_ref(question)
    if not ref["articulo"]:
        return False

    for c in chunks:
        meta = c.chunk_meta or {}
        ids = c.identifiers or {}

        articulo = ids.get("articulo") or meta.get("articulo")
        apartado = _extract_apartado_from_meta(meta)
        fraccion = _extract_fraccion_from_meta(meta)
        inciso = _extract_inciso_from_meta(meta)

        if articulo != ref["articulo"]:
            continue
        if ref["apartado"] and apartado != ref["apartado"]:
            continue
        if ref["fraccion"] and fraccion != ref["fraccion"]:
            continue
        if ref["inciso"] and inciso != ref["inciso"]:
            continue

        return True

    return False


def _filter_to_exact_subtree(
    question: str,
    chunks: list[RetrievedChunk],
    top_k: int,
) -> list[RetrievedChunk]:
    ref = _parse_structured_ref(question)
    if not ref["articulo"]:
        return chunks[:top_k]

    exact = []
    for c in chunks:
        meta = c.chunk_meta or {}
        ids = c.identifiers or {}

        articulo = ids.get("articulo") or meta.get("articulo")
        apartado = _extract_apartado_from_meta(meta)
        fraccion = _extract_fraccion_from_meta(meta)
        inciso = _extract_inciso_from_meta(meta)

        if articulo != ref["articulo"]:
            continue
        if ref["apartado"] and apartado != ref["apartado"]:
            continue
        if ref["fraccion"] and fraccion != ref["fraccion"]:
            continue
        if ref["inciso"] and inciso != ref["inciso"]:
            continue

        exact.append(c)

    return exact[:top_k] if exact else chunks[:top_k]


def _sort_chunks_by_source_order(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    def _key(c: RetrievedChunk):
        meta = c.chunk_meta or {}
        return (
            int(meta.get("source_order", 10**9)),
            int(meta.get("order_index", 10**9)),
            c.chunk_id,
        )
    return sorted(chunks, key=_key)


def ask_rag(question: str, top_k: int = 8) -> Dict[str, Any]:
    initial_k = max(top_k * 3, 20)

    chunks = search_hybrid(question, top_k=initial_k)
    chunks = legal_rerank(question, chunks, top_k=max(top_k * 3, 12))

    exact_match = _has_exact_structural_match(question, chunks)

    if exact_match:
        chunks = _filter_to_exact_subtree(question, chunks, top_k=max(top_k * 3, 12))
        chunks = _sort_chunks_by_source_order(chunks)

    if _looks_like_full_article_request(question) and not exact_match and chunks:
        top_doc_id = chunks[0].document_id
        expanded_chunks = search_hybrid(question, top_k=top_k * 6)
        sibling_chunks = [c for c in expanded_chunks if c.document_id == top_doc_id]
        if sibling_chunks:
            chunks = legal_rerank(question, sibling_chunks, top_k=max(top_k * 3, 12))

    final_chunks = chunks[:top_k]

    coverage = _compute_coverage(question, final_chunks)
    citations = build_citations(final_chunks)
    used_chunk_ids = [c.chunk_id for c in final_chunks]
    exact_match = _has_exact_structural_match(question, final_chunks)

    if _should_abstain(question, final_chunks, coverage) and not exact_match:
        logger.warning(
            f"Guardrail activado para pregunta: {question}. "
            f"Coverage: {coverage['core_coverage_ratio']}"
        )
        return {
            "question": question,
            "answer": _build_abstention_answer(question, final_chunks, coverage),
            "citations": [],
            "chunks": [c.__dict__ for c in final_chunks],
            "diagnostics": coverage,
            "abstained": True,
            "confidence": {
                "top_score": final_chunks[0].score if final_chunks else 0.0,
                "top_fts": final_chunks[0].fts_rank if final_chunks else 0.0,
                "coverage_ratio": coverage.get("coverage_ratio", 0.0),
                "core_coverage_ratio": coverage.get("core_coverage_ratio", 0.0),
            },
            "question_type": _classify_question_type(question),
            "used_chunk_ids": used_chunk_ids,
        }

    evidence = "\n\n".join(
        [
            (
                f"UBICACION_JURIDICA: "
                f"articulo={c.identifiers.get('articulo', c.chunk_meta.get('articulo', 'N/D'))}; "
                f"apartado={_extract_apartado_from_meta(c.chunk_meta) or 'N/D'}; "
                f"fraccion={_extract_fraccion_from_meta(c.chunk_meta) or 'N/D'}; "
                f"inciso={_extract_inciso_from_meta(c.chunk_meta) or 'N/D'}; "
                f"path={c.chunk_meta.get('path', 'N/D')}\n"
                f"TEXTO:\n{c.chunk_text}"
            )
            for c in final_chunks
        ]
    )

    resp = client.chat.completions.create(
        model=settings.LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Pregunta:\n{question}\n\n"
                    f"Evidencia:\n{evidence}\n\n"
                    "Instrucción: Responde SOLO con base en la evidencia. "
                    "Primero identifica internamente la ubicación jurídica correcta de los fragmentos usando articulo/apartado/fraccion/inciso/path. "
                    "No confundas remisiones internas del texto con la ubicación del fragmento. "
                    "No muestres esa validación interna en la respuesta final. "
                    "Entrega una respuesta limpia para usuario final, sin metadata técnica ni trazas. "
                    "Si la pregunta pide el contenido de una disposición, empieza directamente con lo que prevé esa disposición. "

                ),
            },
        ],
    )

    return {
        "question": question,
        "answer": resp.choices[0].message.content,
        "citations": citations,
        "chunks": [c.__dict__ for c in final_chunks],
        "diagnostics": coverage,
        "abstained": False,
        "confidence": {
            "top_score": final_chunks[0].score if final_chunks else 0.0,
            "top_fts": final_chunks[0].fts_rank if final_chunks else 0.0,
            "coverage_ratio": coverage.get("coverage_ratio", 0.0),
            "core_coverage_ratio": coverage.get("core_coverage_ratio", 0.0),
        },
        "question_type": _classify_question_type(question),
        "used_chunk_ids": used_chunk_ids,
    }