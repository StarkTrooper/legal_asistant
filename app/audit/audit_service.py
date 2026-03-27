
from sqlalchemy import text
import uuid

def save_audit_record(db, query_id: uuid.UUID, user_id, mode, user_input,
                      parsed_intent, retrieved_chunks, evidence_packet,
                      model_info, response, verification, index_version: str):
    sql = text("""
        INSERT INTO queries_audit(
          query_id, user_id, mode, user_input,
          parsed_intent_json, retrieved_chunks_json, evidence_packet_json,
          model_info_json, response_json, verification_json,
          index_version
        ) VALUES (
          :query_id, :user_id, :mode, :user_input,
          :parsed_intent::jsonb, :retrieved_chunks::jsonb, :evidence_packet::jsonb,
          :model_info::jsonb, :response::jsonb, :verification::jsonb,
          :index_version
        )
    """)
    db.execute(sql, {
        "query_id": str(query_id),
        "user_id": user_id,
        "mode": mode,
        "user_input": user_input,
        "parsed_intent": to_json(parsed_intent),
        "retrieved_chunks": to_json(retrieved_chunks),
        "evidence_packet": to_json(evidence_packet),
        "model_info": to_json(model_info),
        "response": to_json(response),
        "verification": to_json(verification),
        "index_version": index_version
    })
    db.commit()

def to_json(obj):
    import json
    return json.dumps(obj, ensure_ascii=False)
