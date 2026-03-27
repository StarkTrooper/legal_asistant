from __future__ import annotations

import re


def normalize_norm_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_article_key(article_num: str) -> str:
    article_num = article_num.strip()
    article_num = re.sub(r"\s+", " ", article_num)
    return article_num.upper()


def clean_block(text: str) -> str:
    return normalize_norm_text(text).strip()


def extract_apartado_from_path(path: str | None) -> str | None:
    if not path:
        return None
    m = re.search(r"/ap:([A-Z])(?:/|$)", path)
    return m.group(1).upper() if m else None


def extract_fraccion_from_path(path: str | None) -> str | None:
    if not path:
        return None
    m = re.search(r"/fr:([IVXLCDM]+)(?:/|$)", path)
    return m.group(1).upper() if m else None


def extract_inciso_from_path(path: str | None) -> str | None:
    if not path:
        return None
    m = re.search(r"/inc:([a-z])(?:/|$)", path)
    return m.group(1).lower() if m else None