from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence


# ============================================================
# Dataclasses
# ============================================================

@dataclass
class NormUnit:
    unit_type: str                  # article | paragraph | apartado | fraccion | inciso
    unit_id: str                    # p1 | pre | A | I | a
    label: str                      # Párrafo 1 | Apartado A | Fracción I | Inciso a)
    text: str
    path: str                       # art:69-B/ap:A/fr:I/inc:a
    parent_path: Optional[str]
    order_index: int
    source_order: int
    article_key: str
    apartado_id: Optional[str] = None
    fraccion_id: Optional[str] = None
    inciso_id: Optional[str] = None
    cited_as: Optional[str] = None


@dataclass
class ParsedArticle:
    ordenamiento: str
    abreviatura: str
    articulo: str
    titulo: str
    full_text: str
    body_text: str
    units: List[NormUnit] = field(default_factory=list)


@dataclass
class NormArticle:
    ordenamiento: str
    abreviatura: str
    articulo: str
    titulo: str
    contenido: str


# ============================================================
# Regex comunes
# ============================================================

ARTICLE_HEADER_RE = re.compile(
    r"""(?imx)
    ^
    (?P<header>
        art[ií]culo
        \s+
        (?P<article>
            \d+(?:o\.)?
            (?:
                \s*-\s*[A-Z]
                |
                -[A-Z]
                |
                \s+[A-Z]
            )?
            (?:
                \s*-\s*
                |
                \s+
                |
                -
            )?
            (?P<suffix>Bis|Ter|Qu[aá]ter|Quater)?
        )
    )
    (?P<ending>\.-|\.)\s*
    """
)

FRACCION_HEADER_RE = re.compile(
    r"""(?imx)
    ^
    (?P<roman>
        XLVI|XLV|XLIV|XLIII|XLII|XLI|XL|
        XXXIX|XXXVIII|XXXVII|XXXVI|XXXV|XXXIV|XXXIII|XXXII|XXXI|XXX|
        XXIX|XXVIII|XXVII|XXVI|XXV|XXIV|XXIII|XXII|XXI|XX|
        XIX|XVIII|XVII|XVI|XV|XIV|XIII|XII|XI|X|
        IX|VIII|VII|VI|V|IV|III|II|I
    )
    \.\s+
    """
)

INCISO_HEADER_RE = re.compile(
    r"""(?imx)
    ^
    (?P<letter>[a-z])
    \)\s+
    """
)

# Candidatos generales de apartado
APARTADO_CANDIDATE_RE = re.compile(
    r"""(?imx)
    ^
    \s*
    (?P<letter>[A-Z])
    \.
    \s*$
    """
)

# Apartados reales para CPEUM
APARTADO_REAL_RE = re.compile(
    r"""(?imx)
    ^
    \s*
    (?P<letter>[A-D])
    \.
    \s*$
    """
)

EDITORIAL_LINE_RE = re.compile(
    r"""(?imx)
    ^
    \s*
    (?:
        art[ií]culo(?:\s+\w+)?(?:\s*\([^)]+\))?\s+(?:reformado|adicionado|derogado)\s+DOF
        |
        p[aá]rrafo(?:\s+\w+)?(?:\s*\([^)]+\))?\s+(?:reformado|adicionado|derogado)\s+DOF
        |
        fracci[oó]n(?:\s+\w+)?(?:\s*\([^)]+\))?\s+(?:reformada|adicionada|derogada)\s+DOF
        |
        inciso(?:\s+\w+)?(?:\s*\([^)]+\))?\s+(?:reformado|adicionado|derogado)\s+DOF
        |
        apartado(?:\s+[A-Z])?(?:\s*\([^)]+\))?\s+(?:reformado|adicionado|derogado)\s+DOF
        |
        reforma\s+DOF
        |
        fe\s+de\s+erratas
        |
        nota\s+de\s+editor
    )
    .*$
    """
)

TRANSITORIOS_START_RE = re.compile(
    r"""(?imx)
    ^
    \s*
    transitorios
    \s*$
    """
)

REFORMA_TRANSITORIOS_RE = re.compile(
    r"""(?imx)
    ^
    \s*
    art[íi]culos?\s+transitorios\s+de\s+decretos?\s+de\s+reforma
    \s*$
    """
)


# ============================================================
# Normalización / Helpers
# ============================================================

def normalize_norm_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\ufeff", "")
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _clean_block(text: str) -> str:
    return normalize_norm_text(text).strip()


def _normalize_article_key(article_num: str) -> str:
    value = article_num.strip()
    value = value.replace("–", "-").replace("—", "-")
    value = re.sub(r"\s*-\s*", "-", value)
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"\.$", "", value)

    value = re.sub(r"\bBIS\b", "Bis", value, flags=re.IGNORECASE)
    value = re.sub(r"\bTER\b", "Ter", value, flags=re.IGNORECASE)
    value = re.sub(r"\bQUÁTER\b", "Quáter", value, flags=re.IGNORECASE)
    value = re.sub(r"\bQUATER\b", "Quater", value, flags=re.IGNORECASE)
    value = re.sub(r"-(Bis|Ter|Quáter|Quater)\b", r"-\1", value, flags=re.IGNORECASE)

    return value.strip()


def _make_cited_as(
    abreviatura: str,
    article_key: str,
    apartado_id: Optional[str] = None,
    fraccion_id: Optional[str] = None,
    inciso_id: Optional[str] = None,
    paragraph_label: Optional[str] = None,
) -> str:
    parts = [f"{abreviatura}, art. {article_key}"]
    if apartado_id:
        parts.append(f"ap. {apartado_id}")
    if fraccion_id:
        parts.append(f"frac. {fraccion_id}")
    if inciso_id:
        parts.append(f"inc. {inciso_id})")
    if paragraph_label:
        parts.append(paragraph_label)
    return ", ".join(parts)


def _is_editorial_line(line: str) -> bool:
    return bool(EDITORIAL_LINE_RE.match(line.strip()))


def _remove_editorial_lines(text: str) -> str:
    lines = text.splitlines()
    cleaned = [line for line in lines if not _is_editorial_line(line)]
    return "\n".join(cleaned)


def _candidate_apartado_letters(text: str, apartado_re: re.Pattern[str]) -> List[str]:
    return [m.group("letter").upper() for m in apartado_re.finditer(text)]


def _looks_like_apartado_context_cff(text: str) -> bool:
    letters = _candidate_apartado_letters(text, APARTADO_CANDIDATE_RE)
    if len(letters) < 2:
        return False

    allowed = {"A", "B", "C", "D"}
    seen_allowed = [x for x in letters if x in allowed]

    return len(set(seen_allowed)) >= 2


def _looks_like_apartado_context_cpeum(text: str) -> bool:
    matches = list(APARTADO_REAL_RE.finditer(text))
    if not matches:
        return False

    letters = [m.group("letter").upper() for m in matches]
    seen_allowed = [x for x in letters if x in {"A", "B", "C", "D"}]

    if len(set(seen_allowed)) >= 2:
        return True

    if len(seen_allowed) == 1:
        first = matches[0]
        tail = text[first.end():]
        if FRACCION_HEADER_RE.search(tail):
            return True

    return False


def _find_first_cut_position(text: str, patterns: Sequence[re.Pattern[str]]) -> Optional[int]:
    cut_positions: List[int] = []
    for pattern in patterns:
        m = pattern.search(text)
        if m:
            cut_positions.append(m.start())
    return min(cut_positions) if cut_positions else None


# ============================================================
# Preprocesadores por ordenamiento
# ============================================================

CPEUM_TAIL_CUT_PATTERNS = [
    re.compile(r"(?im)^\s*Art[ií]culos?\s+Transitorios\s*$"),
    re.compile(r"(?im)^\s*Art[íi]culos?\s+Transitorios\s+de\s+los?\s+Decretos?\s+de\s+Reforma\s*$"),
    re.compile(r"(?im)^\s*DECRETO\s+por\s+el\s+que\s+se\s+reforman"),
    re.compile(r"(?im)^\s*DECRETO\s+por\s+el\s+que\s+se\s+adicionan"),
    re.compile(r"(?im)^\s*DECRETO\s+por\s+el\s+que\s+se\s+derogan"),
]

def _preprocess_cpeum_text(raw_text: str) -> str:
    text = normalize_norm_text(raw_text)
    text = _remove_editorial_lines(text)

    art_136_match = re.search(
        r"(?im)^\s*art[ií]culo\s+136(?:\.-|\.|\s)",
        text
    )
    if not art_136_match:
        return text.strip()

    tail = text[art_136_match.start():]

    cut_positions: List[int] = []
    for pattern in CPEUM_TAIL_CUT_PATTERNS:
        m = pattern.search(tail)
        if m:
            cut_positions.append(m.start())

    if cut_positions:
        cut_at = art_136_match.start() + min(cut_positions)
        text = text[:cut_at]

    return text.strip()


def _preprocess_cpeum_text(raw_text: str) -> str:
    text = normalize_norm_text(raw_text)
    text = _remove_editorial_lines(text)
    return text.strip()


# ============================================================
# Splitters comunes
# ============================================================

def _split_incisos(
    fraccion_text: str,
    abreviatura: str,
    article_key: str,
    fraccion_id: str,
    source_order_start: int,
    parent_path: Optional[str] = None,
    apartado_id: Optional[str] = None,
) -> tuple[List[NormUnit], int]:
    matches = list(INCISO_HEADER_RE.finditer(fraccion_text))
    if not matches:
        return [], source_order_start

    if parent_path is None:
        parent_path = f"art:{article_key}/fr:{fraccion_id}"

    units: List[NormUnit] = []
    source_order = source_order_start

    first_match_start = matches[0].start()
    if first_match_start > 0:
        preamble = fraccion_text[:first_match_start].strip()
        if preamble:
            source_order += 1
            units.append(
                NormUnit(
                    unit_type="paragraph",
                    unit_id="pre",
                    label=f"Preámbulo Fracción {fraccion_id}",
                    text=_clean_block(preamble),
                    path=f"{parent_path}/p:pre",
                    parent_path=parent_path,
                    order_index=0,
                    source_order=source_order,
                    article_key=article_key,
                    apartado_id=apartado_id,
                    fraccion_id=fraccion_id,
                    cited_as=_make_cited_as(
                        abreviatura=abreviatura,
                        article_key=article_key,
                        apartado_id=apartado_id,
                        fraccion_id=fraccion_id,
                        paragraph_label="párr. inicial",
                    ),
                )
            )

    for i, match in enumerate(matches, start=1):
        start = match.start()
        end = matches[i].start() if i < len(matches) else len(fraccion_text)
        block = fraccion_text[start:end].strip()

        letter = match.group("letter").lower()
        source_order += 1

        units.append(
            NormUnit(
                unit_type="inciso",
                unit_id=letter,
                label=f"Inciso {letter})",
                text=_clean_block(block),
                path=f"{parent_path}/inc:{letter}",
                parent_path=parent_path,
                order_index=i,
                source_order=source_order,
                article_key=article_key,
                apartado_id=apartado_id,
                fraccion_id=fraccion_id,
                inciso_id=letter,
                cited_as=_make_cited_as(
                    abreviatura=abreviatura,
                    article_key=article_key,
                    apartado_id=apartado_id,
                    fraccion_id=fraccion_id,
                    inciso_id=letter,
                ),
            )
        )

    return units, source_order


def _split_fracciones(
    body_text: str,
    abreviatura: str,
    article_key: str,
    source_order_start: int,
    parent_path: Optional[str] = None,
    apartado_id: Optional[str] = None,
) -> tuple[List[NormUnit], int]:
    matches = list(FRACCION_HEADER_RE.finditer(body_text))
    if not matches:
        return [], source_order_start

    if parent_path is None:
        parent_path = f"art:{article_key}"

    units: List[NormUnit] = []
    source_order = source_order_start

    first_match_start = matches[0].start()
    if first_match_start > 0:
        preamble = body_text[:first_match_start].strip()
        if preamble:
            source_order += 1
            label = (
                f"Preámbulo Apartado {apartado_id}"
                if apartado_id
                else f"Preámbulo Artículo {article_key}"
            )
            units.append(
                NormUnit(
                    unit_type="paragraph",
                    unit_id="pre",
                    label=label,
                    text=_clean_block(preamble),
                    path=f"{parent_path}/p:pre",
                    parent_path=parent_path,
                    order_index=0,
                    source_order=source_order,
                    article_key=article_key,
                    apartado_id=apartado_id,
                    cited_as=_make_cited_as(
                        abreviatura=abreviatura,
                        article_key=article_key,
                        apartado_id=apartado_id,
                        paragraph_label="párr. inicial",
                    ),
                )
            )

    for i, match in enumerate(matches, start=1):
        start = match.start()
        end = matches[i].start() if i < len(matches) else len(body_text)
        block = body_text[start:end].strip()

        roman = match.group("roman").upper()
        fr_path = f"{parent_path}/fr:{roman}"

        source_order += 1
        units.append(
            NormUnit(
                unit_type="fraccion",
                unit_id=roman,
                label=f"Fracción {roman}",
                text=_clean_block(block),
                path=fr_path,
                parent_path=parent_path,
                order_index=i,
                source_order=source_order,
                article_key=article_key,
                apartado_id=apartado_id,
                fraccion_id=roman,
                cited_as=_make_cited_as(
                    abreviatura=abreviatura,
                    article_key=article_key,
                    apartado_id=apartado_id,
                    fraccion_id=roman,
                ),
            )
        )

        inciso_units, source_order = _split_incisos(
            fraccion_text=block,
            abreviatura=abreviatura,
            article_key=article_key,
            fraccion_id=roman,
            source_order_start=source_order,
            parent_path=fr_path,
            apartado_id=apartado_id,
        )
        units.extend(inciso_units)

    return units, source_order


def _split_paragraphs(
    body_text: str,
    abreviatura: str,
    article_key: str,
    source_order_start: int,
    parent_path: Optional[str] = None,
    apartado_id: Optional[str] = None,
) -> tuple[List[NormUnit], int]:
    paragraphs = [p.strip() for p in body_text.split("\n\n") if p.strip()]
    if not paragraphs:
        return [], source_order_start

    if parent_path is None:
        parent_path = f"art:{article_key}"

    units: List[NormUnit] = []
    source_order = source_order_start

    if len(paragraphs) == 1:
        source_order += 1
        label = f"Artículo {article_key}" if not apartado_id else f"Apartado {apartado_id}"
        units.append(
            NormUnit(
                unit_type="article" if not apartado_id else "paragraph",
                unit_id=article_key if not apartado_id else "p1",
                label=label,
                text=_clean_block(paragraphs[0]),
                path=parent_path if not apartado_id else f"{parent_path}/p:p1",
                parent_path=None if not apartado_id else parent_path,
                order_index=1,
                source_order=source_order,
                article_key=article_key,
                apartado_id=apartado_id,
                cited_as=_make_cited_as(
                    abreviatura=abreviatura,
                    article_key=article_key,
                    apartado_id=apartado_id,
                ),
            )
        )
        return units, source_order

    for idx, paragraph in enumerate(paragraphs, start=1):
        source_order += 1
        units.append(
            NormUnit(
                unit_type="paragraph",
                unit_id=f"p{idx}",
                label=f"Párrafo {idx}",
                text=_clean_block(paragraph),
                path=f"{parent_path}/p:p{idx}",
                parent_path=parent_path,
                order_index=idx,
                source_order=source_order,
                article_key=article_key,
                apartado_id=apartado_id,
                cited_as=_make_cited_as(
                    abreviatura=abreviatura,
                    article_key=article_key,
                    apartado_id=apartado_id,
                    paragraph_label=f"párr. {idx}",
                ),
            )
        )

    return units, source_order


def _split_apartados(
    body_text: str,
    abreviatura: str,
    article_key: str,
    source_order_start: int,
    apartado_header_re: re.Pattern[str],
) -> tuple[List[NormUnit], int]:
    matches = list(apartado_header_re.finditer(body_text))
    if not matches:
        return [], source_order_start

    units: List[NormUnit] = []
    source_order = source_order_start
    article_parent = f"art:{article_key}"

    first_match_start = matches[0].start()
    if first_match_start > 0:
        preamble = body_text[:first_match_start].strip()
        if preamble:
            source_order += 1
            units.append(
                NormUnit(
                    unit_type="paragraph",
                    unit_id="pre",
                    label=f"Preámbulo Artículo {article_key}",
                    text=_clean_block(preamble),
                    path=f"{article_parent}/p:pre",
                    parent_path=article_parent,
                    order_index=0,
                    source_order=source_order,
                    article_key=article_key,
                    cited_as=_make_cited_as(
                        abreviatura=abreviatura,
                        article_key=article_key,
                        paragraph_label="párr. inicial",
                    ),
                )
            )

    for i, match in enumerate(matches, start=1):
        start = match.start()
        end = matches[i].start() if i < len(matches) else len(body_text)
        block = body_text[start:end].strip()

        letter = match.group("letter").upper()
        apartado_path = f"{article_parent}/ap:{letter}"

        source_order += 1
        units.append(
            NormUnit(
                unit_type="apartado",
                unit_id=letter,
                label=f"Apartado {letter}",
                text=_clean_block(block),
                path=apartado_path,
                parent_path=article_parent,
                order_index=i,
                source_order=source_order,
                article_key=article_key,
                apartado_id=letter,
                cited_as=_make_cited_as(
                    abreviatura=abreviatura,
                    article_key=article_key,
                    apartado_id=letter,
                ),
            )
        )

        apartado_body = block[match.end() - match.start():].strip()

        fracciones, source_order = _split_fracciones(
            body_text=apartado_body,
            abreviatura=abreviatura,
            article_key=article_key,
            source_order_start=source_order,
            parent_path=apartado_path,
            apartado_id=letter,
        )
        if fracciones:
            units.extend(fracciones)
        else:
            paragraphs, source_order = _split_paragraphs(
                body_text=apartado_body,
                abreviatura=abreviatura,
                article_key=article_key,
                source_order_start=source_order,
                parent_path=apartado_path,
                apartado_id=letter,
            )
            units.extend(paragraphs)

    return units, source_order


# ============================================================
# Builders por ordenamiento
# ============================================================

def _build_cff_article_units(
    abreviatura: str,
    article_key: str,
    body_text: str,
) -> List[NormUnit]:
    body_text = _clean_block(body_text)
    if not body_text:
        return []

    source_order = 0

    if _looks_like_apartado_context_cff(body_text):
        apartados, source_order = _split_apartados(
            body_text=body_text,
            abreviatura=abreviatura,
            article_key=article_key,
            source_order_start=source_order,
            apartado_header_re=APARTADO_REAL_RE,
        )
        if apartados:
            return apartados

    fracciones, source_order = _split_fracciones(
        body_text=body_text,
        abreviatura=abreviatura,
        article_key=article_key,
        source_order_start=source_order,
    )
    if fracciones:
        return fracciones

    paragraphs, _ = _split_paragraphs(
        body_text=body_text,
        abreviatura=abreviatura,
        article_key=article_key,
        source_order_start=source_order,
    )
    return paragraphs


def _build_cpeum_article_units(
    abreviatura: str,
    article_key: str,
    body_text: str,
) -> List[NormUnit]:
    body_text = _clean_block(body_text)
    if not body_text:
        return []

    source_order = 0

    if _looks_like_apartado_context_cpeum(body_text):
        apartados, source_order = _split_apartados(
            body_text=body_text,
            abreviatura=abreviatura,
            article_key=article_key,
            source_order_start=source_order,
            apartado_header_re=APARTADO_REAL_RE,
        )
        if apartados:
            return apartados

    fracciones, source_order = _split_fracciones(
        body_text=body_text,
        abreviatura=abreviatura,
        article_key=article_key,
        source_order_start=source_order,
    )
    if fracciones:
        return fracciones

    paragraphs, _ = _split_paragraphs(
        body_text=body_text,
        abreviatura=abreviatura,
        article_key=article_key,
        source_order_start=source_order,
    )
    return paragraphs


def _build_generic_article_units(
    abreviatura: str,
    article_key: str,
    body_text: str,
) -> List[NormUnit]:
    body_text = _clean_block(body_text)
    if not body_text:
        return []

    source_order = 0

    fracciones, source_order = _split_fracciones(
        body_text=body_text,
        abreviatura=abreviatura,
        article_key=article_key,
        source_order_start=source_order,
    )
    if fracciones:
        return fracciones

    paragraphs, _ = _split_paragraphs(
        body_text=body_text,
        abreviatura=abreviatura,
        article_key=article_key,
        source_order_start=source_order,
    )
    return paragraphs


# ============================================================
# Parser base reusable
# ============================================================

def _parse_articles_generic(
    raw_text: str,
    *,
    ordenamiento: str,
    abreviatura: str,
    preprocess_fn: Optional[Callable[[str], str]] = None,
    build_units_fn: Optional[Callable[[str, str, str], List[NormUnit]]] = None,
) -> List[ParsedArticle]:
    text = raw_text
    if preprocess_fn is not None:
        text = preprocess_fn(text)

    text = normalize_norm_text(text)
    matches = list(ARTICLE_HEADER_RE.finditer(text))
    if not matches:
        return []

    if build_units_fn is None:
        build_units_fn = _build_generic_article_units

    articles: List[ParsedArticle] = []

    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

        block = text[start:end].strip()
        header = match.group("header").strip()
        article_num_raw = match.group("article").strip()
        article_key = _normalize_article_key(article_num_raw)

        header_len = match.end() - match.start()
        body_text = block[header_len:].strip()

        article = ParsedArticle(
            ordenamiento=ordenamiento,
            abreviatura=abreviatura,
            articulo=article_key,
            titulo=header,
            full_text=_clean_block(block),
            body_text=_clean_block(body_text),
            units=build_units_fn(
                abreviatura,
                article_key,
                body_text,
            ),
        )
        articles.append(article)

    return articles


# ============================================================
# APIs públicas por ordenamiento
# ============================================================

def parse_cff_articles(
    raw_text: str,
    ordenamiento: str = "Código Fiscal de la Federación",
    abreviatura: str = "CFF",
) -> List[ParsedArticle]:
    return _parse_articles_generic(
        raw_text,
        ordenamiento=ordenamiento,
        abreviatura=abreviatura,
        preprocess_fn=_preprocess_cff_text,
        build_units_fn=_build_cff_article_units,
    )


def parse_cpeum_articles(
    raw_text: str,
    ordenamiento: str = "Constitución Política de los Estados Unidos Mexicanos",
    abreviatura: str = "CPEUM",
) -> List[ParsedArticle]:
    return _parse_articles_generic(
        raw_text,
        ordenamiento=ordenamiento,
        abreviatura=abreviatura,
        preprocess_fn=_preprocess_cpeum_text,
        build_units_fn=_build_cpeum_article_units,
    )


def parse_lfpca_articles(
    raw_text: str,
    ordenamiento: str = "Ley Federal de Procedimiento Contencioso Administrativo",
    abreviatura: str = "LFPCA",
) -> List[ParsedArticle]:
    return _parse_articles_generic(
        raw_text,
        ordenamiento=ordenamiento,
        abreviatura=abreviatura,
        preprocess_fn=normalize_norm_text,
        build_units_fn=_build_generic_article_units,
    )


def parse_ley_amparo_articles(
    raw_text: str,
    ordenamiento: str = "Ley de Amparo",
    abreviatura: str = "LA",
) -> List[ParsedArticle]:
    return _parse_articles_generic(
        raw_text,
        ordenamiento=ordenamiento,
        abreviatura=abreviatura,
        preprocess_fn=normalize_norm_text,
        build_units_fn=_build_generic_article_units,
    )


# ============================================================
# Compatibilidad temporal
# ============================================================

def split_articles(
    raw_text: str,
    ordenamiento: str,
    abreviatura: str,
) -> List[NormArticle]:
    parser_map = {
        "CFF": parse_cff_articles,
        "CPEUM": parse_cpeum_articles,
        "LFPCA": parse_lfpca_articles,
        "LA": parse_ley_amparo_articles,
    }

    parser = parser_map.get(abreviatura.upper(), None)
    if parser is None:
        parsed = _parse_articles_generic(
            raw_text,
            ordenamiento=ordenamiento,
            abreviatura=abreviatura,
            preprocess_fn=normalize_norm_text,
            build_units_fn=_build_generic_article_units,
        )
    else:
        parsed = parser(
            raw_text=raw_text,
            ordenamiento=ordenamiento,
            abreviatura=abreviatura,
        )

    return [
        NormArticle(
            ordenamiento=a.ordenamiento,
            abreviatura=a.abreviatura,
            articulo=a.articulo,
            titulo=a.titulo,
            contenido=a.full_text,
        )
        for a in parsed
    ]