from __future__ import annotations

import re
from typing import List

from app.services.retriever import RetrievedChunk


HOLDING_PATTERNS = [
    r"\bla respuesta .* sentido\b",
    r"\bresultan infundados\b",
    r"\bresultan fundados\b",
    r"\bse confirma\b",
    r"\bse concede\b",
    r"\bse niega\b",
    r"\bdebe determinarse\b",
    r"\besta (primera|segunda) sala considera\b",
    r"\besta sala observa\b",
]

LEGAL_KEY_TERMS = {
    "revocación", "revocacion",
    "caducidad",
    "prescripción", "prescripcion",
    "suspensión", "suspension",
    "competencia",
    "improcedencia",
    "sobreseimiento",
    "progresividad",
    "definitividad",
    "relatividad",
}

NORM_ALIASES = {
    "código fiscal de la federación": [
        "código fiscal de la federación",
        "codigo fiscal de la federacion",
        "cff",
    ],
    "ley de amparo": [
        "ley de amparo",
    ],
    "constitución": [
        "constitución",
        "constitucion",
        "constitución política de los estados unidos mexicanos",
        "constitucion politica de los estados unidos mexicanos",
        "cpeum",
    ],
    "ley federal de procedimiento contencioso administrativo": [
        "ley federal de procedimiento contencioso administrativo",
        "lfpca",
    ],
}

GENERIC_HEADER_PATTERNS = [
    r"^\s*amparo en revisión",
    r"^\s*ponente:",
    r"^\s*secretario:",
    r"^\s*recurrente",
    r"^\s*quejosa",
]

NORMATIVE_PATTERNS = [
    r"\bart[ií]culo\b",
    r"\bfracci[oó]n\b",
    r"\bp[aá]rrafo\b",
    r"\binciso\b",
    r"\btexto\b",
    r"\bcontenido textual\b",
]

DEADLINE_PATTERNS = [
    r"\bplazo\b",
    r"\bt[ée]rmino\b",
    r"\bd[ií]as\b",
    r"\bveinte d[ií]as\b",
    r"\bdoce meses\b",
    r"\bdentro de\b",
    r"\bhasta\b",
]

DEFINITION_PATTERNS = [
    r"\bse entiende por\b",
    r"\bconsiste en\b",
    r"\bsignifica\b",
    r"\bdefine\b",
    r"\bconcepto\b",
]

REMEDY_PATTERNS = [
    r"\brecurso\b",
    r"\binterponer\b",
    r"\bprocede\b",
    r"\bprocedencia\b",
    r"\bimpugnaci[oó]n\b",
]

LOW_SIGNAL_PATTERNS = [
    r"\boportunidad y legitimaci[oó]n\b",
    r"\bcompetencia\b",
    r"\bpublicaci[oó]n\b",
    r"\bavocamiento\b",
    r"\btr[aá]mite ante la suprema corte\b",
]

LITERAL_NORM_TEXT_PATTERNS = [
    r"\bcontenido textual es el siguiente\b",
    r"“art[ií]culo\s+\d+[a-z\-]*\.",
    r"\"art[ií]culo\s+\d+[a-z\-]*\.",
    r"\bart[ií]culo\s+69-c\.",
]

HISTORY_OR_CONTEXT_PATTERNS = [
    r"\bexposici[oó]n de motivos\b",
    r"\blegislador ordinario\b",
    r"\bcolegisladora\b",
    r"\bcomisiones unidas\b",
    r"\bminuta\b",
]

INTENT_NORMATIVE_PATTERNS = [
    r"\bqué establece\b",
    r"\bque establece\b",
    r"\bqué dice\b",
    r"\bque dice\b",
    r"\bcuál es el plazo\b",
    r"\bcual es el plazo\b",
]

ARTICLE_NEIGHBOR_PATTERNS = [
    r"\b69-f\b",
    r"\b46-a\b",
    r"\b50\b",
]


def _normalize(s: str) -> str:
    return (s or "").lower()


def _extract_expediente(question: str) -> str | None:
    m = re.search(r"\b(\d{1,4}/\d{4})\b", question)
    return m.group(1) if m else None


def _extract_article_refs(question: str) -> set[str]:
    refs = set()

    for m in re.finditer(r"\b(\d{1,3}-[A-Z])\b", question, flags=re.IGNORECASE):
        refs.add(m.group(1).upper())

    for m in re.finditer(r"\bart[ií]culo\s+(\d+[A-Z\-]*)\b", question, flags=re.IGNORECASE):
        refs.add(m.group(1).upper())

    return refs


def _classify_question_type(question: str) -> str:
    q = question.lower()

    if any(x in q for x in ["artículo", "articulo", "fracción", "fraccion", "párrafo", "parrafo", "inciso", "qué establece", "que establece", "qué dice", "que dice"]):
        return "norm_reference"

    if any(x in q for x in ["qué es", "que es", "define", "concepto de", "principio de"]):
        return "definition"

    if any(x in q for x in ["cuál es el plazo", "cual es el plazo", "término", "termino", "cuántos días", "cuantos dias", "interponer"]):
        return "deadline"

    if any(x in q for x in ["qué resolvió", "que resolvió", "qué decidió", "que decidió", "fallo", "sentido del fallo", "resolutivo"]):
        return "holding"

    if any(x in q for x in ["recurso", "revocación", "revocacion", "apelación", "apelacion", "queja", "revisión", "revision"]):
        return "remedy"

    return "general"


def _detect_norms(question: str) -> set[str]:
    q = _normalize(question)
    found = set()

    for canon, aliases in NORM_ALIASES.items():
        for alias in aliases:
            if alias in q:
                found.add(canon)
                break

    return found


def _parse_structured_ref(question: str) -> dict[str, str | None]:
    q = question.strip()

    articulo = None
    fraccion = None
    apartado = None
    inciso = None

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
        articulo = re.sub(r"\s+", " ", art_match.group(1)).strip()
        articulo = articulo.replace("–", "-").replace("—", "-")
        articulo = re.sub(r"\s*-\s*", "-", articulo)

    frac_match = re.search(r"(?i)\bfracci[oó]n\s+([IVXLCDM]+)\b", q)
    if frac_match:
        fraccion = frac_match.group(1).upper()

    ap_match = re.search(r"(?i)\bapartado\s+([A-Z])\b", q)
    if ap_match:
        apartado = ap_match.group(1).upper()

    inc_match = re.search(r"(?i)\binciso\s+([a-z])\b", q)
    if inc_match:
        inciso = inc_match.group(1).lower()

    return {
        "articulo": articulo,
        "fraccion": fraccion,
        "apartado": apartado,
        "inciso": inciso,
    }


def _contains_any_pattern(text: str, patterns: list[str]) -> bool:
    for pat in patterns:
        if re.search(pat, text, flags=re.IGNORECASE):
            return True
    return False


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


def legal_rerank(question: str, chunks: List[RetrievedChunk], top_k: int | None = None) -> List[RetrievedChunk]:
    expediente = _extract_expediente(question)
    article_refs = _extract_article_refs(question)
    qtype = _classify_question_type(question)
    q_lower = _normalize(question)
    norms_in_question = _detect_norms(question)
    asks_normative = _contains_any_pattern(q_lower, INTENT_NORMATIVE_PATTERNS)

    rescored = []
    structured_ref = _parse_structured_ref(question)

    for c in chunks:
        text = c.chunk_text or ""
        text_lower = _normalize(text)
        identifiers = c.identifiers or {}
        chunk_meta = getattr(c, "chunk_meta", {}) or {}

        meta_articulo = identifiers.get("articulo") or chunk_meta.get("articulo")
        meta_fraccion = _extract_fraccion_from_meta(chunk_meta)
        meta_apartado = _extract_apartado_from_meta(chunk_meta)
        meta_inciso = _extract_inciso_from_meta(chunk_meta)

        bonus = 0.0

        if expediente:
            if identifiers.get("expediente") == expediente or expediente in text:
                bonus += 0.35

        if structured_ref["articulo"]:
            if meta_articulo == structured_ref["articulo"]:
                bonus += 0.75
            else:
                bonus -= 0.45

        if structured_ref["apartado"]:
            if meta_apartado == structured_ref["apartado"]:
                bonus += 0.45
            elif meta_apartado:
                bonus -= 0.30
            else:
                bonus -= 0.10

        if structured_ref["fraccion"]:
            if meta_fraccion == structured_ref["fraccion"]:
                bonus += 0.50
            elif meta_fraccion:
                bonus -= 0.35
            else:
                bonus -= 0.12

        if structured_ref["inciso"]:
            if meta_inciso == structured_ref["inciso"]:
                bonus += 0.22
            elif meta_inciso:
                bonus -= 0.15
            else:
                bonus -= 0.06

        for ref in article_refs:
            if re.search(rf"\b{re.escape(ref.lower())}\b", text_lower):
                if meta_articulo == ref:
                    bonus += 0.30
                elif not meta_articulo:
                    bonus += 0.10
                else:
                    bonus += 0.02

        if norms_in_question:
            for canon in norms_in_question:
                for alias in NORM_ALIASES[canon]:
                    if alias in text_lower:
                        bonus += 0.18
                        break

        if _contains_any_pattern(text_lower, NORMATIVE_PATTERNS):
            bonus += 0.08

        if _contains_any_pattern(text_lower, LITERAL_NORM_TEXT_PATTERNS):
            bonus += 0.25

        for term in LEGAL_KEY_TERMS:
            if term in q_lower and term in text_lower:
                bonus += 0.10

        if qtype == "holding":
            if _contains_any_pattern(text_lower, HOLDING_PATTERNS):
                bonus += 0.24

        elif qtype == "deadline":
            if _contains_any_pattern(text_lower, DEADLINE_PATTERNS):
                bonus += 0.24
            if any(x in text_lower for x in ["veinte días", "veinte dias", "doce meses"]):
                bonus += 0.12

        elif qtype == "definition":
            if _contains_any_pattern(text_lower, DEFINITION_PATTERNS):
                bonus += 0.16

        elif qtype == "remedy":
            if _contains_any_pattern(text_lower, REMEDY_PATTERNS):
                bonus += 0.18

        elif qtype == "norm_reference":
            if _contains_any_pattern(text_lower, LITERAL_NORM_TEXT_PATTERNS):
                bonus += 0.20
            if _contains_any_pattern(text_lower, NORMATIVE_PATTERNS):
                bonus += 0.18
            if "artículo" in text_lower or "articulo" in text_lower:
                bonus += 0.08

        if asks_normative:
            if _contains_any_pattern(text_lower, LITERAL_NORM_TEXT_PATTERNS):
                bonus += 0.15
            if _contains_any_pattern(text_lower, HISTORY_OR_CONTEXT_PATTERNS):
                bonus -= 0.18

        if article_refs and _contains_any_pattern(text_lower, ARTICLE_NEIGHBOR_PATTERNS):
            if not any(ref.lower() in text_lower for ref in article_refs):
                bonus -= 0.28

        if _contains_any_pattern(text_lower, GENERIC_HEADER_PATTERNS):
            bonus -= 0.05

        if qtype in {"deadline", "norm_reference", "remedy"} and _contains_any_pattern(text_lower, LOW_SIGNAL_PATTERNS):
            bonus -= 0.08

        final_score = c.score + bonus
        rescored.append((final_score, c))

    rescored.sort(key=lambda x: x[0], reverse=True)
    ordered = [c for _, c in rescored]

    if top_k is not None:
        ordered = ordered[:top_k]

    return ordered