from dotenv import load_dotenv
load_dotenv()

import os
from openai import OpenAI
from sqlalchemy import create_engine, text, event

from pgvector.psycopg import register_vector
from pgvector import Vector

DB_URL = os.environ["DATABASE_URL"]
EMB_MODEL = os.environ.get("EMBEDDINGS_MODEL", "text-embedding-3-small")

engine = create_engine(DB_URL)

# Registrar el tipo vector en cada conexión nueva
@event.listens_for(engine, "connect")
def connect(dbapi_connection, connection_record):
    register_vector(dbapi_connection)

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

q = "¿Cómo se defiende alguien ante el 69-B?"
q_emb = client.embeddings.create(model=EMB_MODEL, input=q).data[0].embedding
print("Dim embedding generado:", len(q_emb))

with engine.connect() as conn:
    dbname = conn.execute(text("SELECT current_database()")).scalar_one()
    user = conn.execute(text("SELECT current_user")).scalar_one()
    host = conn.execute(text("SELECT inet_server_addr()")).scalar_one_or_none()
    port = conn.execute(text("SELECT inet_server_port()")).scalar_one_or_none()
    print("DB:", dbname, "| user:", user, "| host:", host, "| port:", port)

    docs = conn.execute(text("SELECT COUNT(*) FROM documents")).scalar_one()
    chunks = conn.execute(text("SELECT COUNT(*) FROM chunks")).scalar_one()
    vecs = conn.execute(text("SELECT COUNT(*) FROM chunk_vectors")).scalar_one()
    print("counts -> documents:", docs, "chunks:", chunks, "chunk_vectors:", vecs)

    rows2 = conn.execute(text("""
    SELECT c.chunk_id, c.chunk_text
    FROM chunk_vectors v
    JOIN chunks c ON c.chunk_id = v.chunk_id
    """)).mappings().all()
    print("JOIN sin vector ops:", len(rows2), rows2)

    # 👇 Ahora pasamos un Vector real (no string)
    emb_vec = Vector(q_emb)

    rows = conn.execute(text("""
    SELECT c.chunk_id, c.chunk_text,
           (1 - (v.embedding <=> (:emb)::vector)) AS score
    FROM chunk_vectors v
    JOIN chunks c ON c.chunk_id = v.chunk_id
    ORDER BY v.embedding <=> (:emb)::vector
    LIMIT 5
    """), {"emb": emb_vec}).mappings().all()

    #  rows = conn.execute(text("""
    #     SELECT c.chunk_id, c.chunk_text,
    #            (1 - (v.embedding <=> :emb)) AS score
    #     FROM chunk_vectors v
    #     JOIN chunks c ON c.chunk_id = v.chunk_id
    #     ORDER BY v.embedding <=> :emb
    #     LIMIT 5
    # """), {"emb": emb_vec}).mappings().all() 

print("Filas encontradas:", len(rows))
for r in rows:
    print("\n---")
    print("chunk_id:", r["chunk_id"], "score:", float(r["score"]))
    print(r["chunk_text"])