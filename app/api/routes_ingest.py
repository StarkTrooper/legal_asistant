from __future__ import annotations

import json
from datetime import date
from typing import Optional, Dict, Any, List

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel, Field
from pypdf import PdfReader

from app.services.ingest_service import (
    upsert_document_and_reindex,
    extract_identifiers_from_text,
    extract_identifiers_from_json,
    json_to_raw_text,
    normalize_input_document,
)

router = APIRouter()


class IngestTextRequest(BaseModel):
    source: str
    authority: str
    doc_type: str
    canonical_url: Optional[str] = None
    publication_date: Optional[date] = None
    identifiers: Optional[Dict[str, Any]] = None
    raw_text: str = Field(..., min_length=20)
    source_format: str = "text"
    mime_type: str = "text/plain"
    jurisdiction: str = "MX"
    collection: str = "general"


@router.post("/ingest/text")
def ingest_text(req: IngestTextRequest):
    try:
        meta = normalize_input_document(
            title=_build_title_from_identifiers(req.identifiers, req.doc_type),
            raw_text=req.raw_text,
            source=req.source,
            authority=req.authority,
            jurisdiction=req.jurisdiction,
            collection=req.collection,
            canonical_url=req.canonical_url,
            identifiers=req.identifiers,
            publication_date=req.publication_date.isoformat() if req.publication_date else None,
            effective_date=None,
            doc_type=req.doc_type,
            source_format=req.source_format,
            mime_type=req.mime_type,
        )

        document_id = upsert_document_and_reindex(
            title=meta.title,
            raw_text=meta.raw_text,
            source=meta.source,
            authority=meta.authority,
            jurisdiction=meta.jurisdiction,
            collection=meta.collection,
            canonical_url=meta.canonical_url,
            identifiers=meta.identifiers,
            publication_date=meta.publication_date,
            effective_date=meta.effective_date,
            prebuilt_chunks=None,
            chunk_max_chars=1800,
            chunk_overlap=100,
            doc_type=meta.doc_type,
            source_format=meta.source_format,
            mime_type=meta.mime_type,
        )

        return {
            "status": "success",
            "document_id": document_id,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/ingest/files")
async def ingest_files(
    files: List[UploadFile] = File(...),
    source: Optional[str] = Form(None),
    authority: Optional[str] = Form(None),
    doc_type: Optional[str] = Form(None),
):
    results = []
    errors = []

    allowed_extensions = {".pdf", ".txt", ".json"}

    for file in files:
        filename = file.filename or "unknown"
        ext = f".{filename.split('.')[-1].lower()}" if "." in filename else ""

        if ext not in allowed_extensions:
            errors.append({
                "filename": filename,
                "status": "error",
                "detail": f"Formato no soportado: {ext}. Solo se permite PDF, TXT o JSON.",
            })
            continue

        try:
            raw_text = ""
            identifiers: Dict[str, Any] = {}
            data: Dict[str, Any] = {}

            if ext == ".pdf":
                reader = PdfReader(file.file)
                parts = []
                for page in reader.pages:
                    t = page.extract_text()
                    if t and t.strip():
                        parts.append(t.strip())
                raw_text = "\n\n".join(parts)

            elif ext == ".json":
                content = await file.read()
                data = json.loads(content.decode("utf-8"))
                raw_text = json_to_raw_text(data)
                identifiers = extract_identifiers_from_json(data, filename=filename)

            elif ext == ".txt":
                content = await file.read()
                raw_text = content.decode("utf-8", errors="ignore")

            if not raw_text.strip():
                raise ValueError(
                    "El archivo está vacío o es un PDF escaneado sin capa de texto legible."
                )

            if not identifiers:
                identifiers = extract_identifiers_from_text(raw_text, filename=filename)

            final_source = data.get("source") if isinstance(data, dict) else None
            final_authority = data.get("authority") if isinstance(data, dict) else None
            final_doc_type = data.get("doc_type") if isinstance(data, dict) else None
            final_url = data.get("url") if isinstance(data, dict) else None
            final_pub_date = _safe_date(data.get("publication_date")) if isinstance(data, dict) else None

            meta = normalize_input_document(
                title=_build_title_from_identifiers(identifiers, final_doc_type or doc_type or "document"),
                raw_text=raw_text,
                source=final_source or source or "upload",
                authority=final_authority or authority or "unknown",
                jurisdiction="MX",
                collection="general",
                canonical_url=final_url,
                identifiers=identifiers,
                publication_date=final_pub_date.isoformat() if final_pub_date else None,
                effective_date=None,
                doc_type=final_doc_type or doc_type or "document",
                source_format=ext.replace(".", ""),
                mime_type=_guess_mime_type(ext),
            )

            document_id = upsert_document_and_reindex(
                title=meta.title,
                raw_text=meta.raw_text,
                source=meta.source,
                authority=meta.authority,
                jurisdiction=meta.jurisdiction,
                collection=meta.collection,
                canonical_url=meta.canonical_url,
                identifiers=meta.identifiers,
                publication_date=meta.publication_date,
                effective_date=meta.effective_date,
                prebuilt_chunks=None,
                chunk_max_chars=1800,
                chunk_overlap=100,
                doc_type=meta.doc_type,
                source_format=meta.source_format,
                mime_type=meta.mime_type,
            )

            results.append({
                "filename": filename,
                "status": "success",
                "document_id": document_id,
            })

        except Exception as e:
            errors.append({
                "filename": filename,
                "status": "error",
                "detail": str(e),
            })

    return {
        "documents_received": len(files),
        "documents_ingested": len(results),
        "documents_failed": len(errors),
        "results": results,
        "errors": errors,
    }


def _safe_date(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    try:
        if isinstance(value, date):
            return value
        return date.fromisoformat(value)
    except ValueError:
        return None


def _guess_mime_type(ext: str) -> str:
    mapping = {
        ".pdf": "application/pdf",
        ".txt": "text/plain",
        ".json": "application/json",
    }
    return mapping.get(ext.lower(), "application/octet-stream")


def _build_title_from_identifiers(
    identifiers: Optional[Dict[str, Any]],
    doc_type: str,
) -> str:
    identifiers = identifiers or {}

    abreviatura = identifiers.get("abreviatura")
    articulo = identifiers.get("articulo")
    expediente = identifiers.get("expediente")
    ordenamiento = identifiers.get("ordenamiento")

    if abreviatura and articulo:
        return f"{abreviatura} {articulo}"
    if ordenamiento and articulo:
        return f"{ordenamiento} {articulo}"
    if expediente:
        return f"{doc_type} {expediente}"
    if ordenamiento:
        return ordenamiento
    return doc_type