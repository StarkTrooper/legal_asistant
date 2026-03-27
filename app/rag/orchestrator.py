# app/rag/orchestrator.py
import uuid
from datetime import datetime
from app.db.session import SessionLocal
from app.rag.retriever import lexical_search, vector_search, merge_rerank
from app.rag.providers import get_embeddings_provider, get_llm_provider
from app.audit.audit_service import save_audit_record

def query_orchestrator(user_id: str | None, mode: str, user_input: str):
    query_id = uuid.uuid4()
    ts = datetime.utcnow().isoformat()

    db = SessionLocal()
    try:
        # 1) Embedding del query
        emb_provider = get_embeddings_provider()
        q_emb = emb_provider.embed(user_input)

        # 2) Búsqueda híbrida
        lex = lexical_search(db, user_input, limit=80)
        vec = vector_search(db, q_emb, limit=80)
        top = merge_rerank(lex, vec, top_k=20)

        # 3) Evidence packet simple
        evidence = [{
            "chunk_id": r["chunk_id"],
            "document_id": r["document_id"],
            "chunk_text_exact": r["chunk_text"],
            "score_final": r["score_final"],
        } for r in top]

        # 4) Prompt base (salida estructurada la metemos después)
        prompt = build_prompt(user_input, evidence)

        llm = get_llm_provider()
        llm_out = llm.generate(prompt)

        response = {
            "query_id": str(query_id),
            "timestamp": ts,
            "answer": llm_out,
            "evidence": [{"chunk_id": e["chunk_id"], "document_id": e["document_id"]} for e in evidence],
            "disclaimer": "Esto es orientación general y no sustituye asesoría legal profesional."
        }

        # 5) Audit
        save_audit_record(
            db=db,
            query_id=query_id,
            user_id=user_id,
            mode=mode,
            user_input=user_input,
            parsed_intent={},
            retrieved_chunks=evidence,
            evidence_packet={"evidence": evidence},
            model_info=llm.model_info(),
            response=response,
            verification={"status": "PENDING"},
            index_version="v0"
        )

        return response

    finally:
        db.close()

def build_prompt(user_input: str, evidence: list[dict]) -> str:
    ctx = "\n\n".join([f"[Chunk {e['chunk_id']}]\n{e['chunk_text_exact']}" for e in evidence])
    return f"""
Eres un asistente jurídico mexicano. Responde en español.
Reglas:
- No inventes tesis, artículos, registros ni autoridades.
- Si no hay evidencia suficiente, dilo y pide datos faltantes.
- Basa tu respuesta SOLO en los chunks.

PREGUNTA:
{user_input}

EVIDENCIA:
{ctx}

RESPUESTA:
""".strip()
