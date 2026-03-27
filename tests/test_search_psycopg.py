from dotenv import load_dotenv
load_dotenv()

import os
import psycopg
from openai import OpenAI

db_url = os.environ["DATABASE_URL"].replace("postgresql+psycopg://", "postgresql://")

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
EMB_MODEL = os.environ.get("EMBEDDINGS_MODEL", "text-embedding-3-small")

q = "¿Cómo se defiende alguien ante el 69-B?"
q_emb = client.embeddings.create(model=EMB_MODEL, input=q).data[0].embedding
emb_str = "[" + ",".join(str(x) for x in q_emb) + "]"

print("Dim embedding generado:", len(q_emb))
print("emb_str len:", len(emb_str))
print("emb_str head:", emb_str[:60])
print("emb_str tail:", emb_str[-60:])

with psycopg.connect(db_url) as conn:
    with conn.cursor() as cur:
        # 1) listados básicos
        cur.execute("SELECT chunk_id FROM chunks;")
        chunks_rows = cur.fetchall()
        print("chunks rows:", chunks_rows)

        cur.execute("SELECT chunk_id, embedding IS NULL FROM chunk_vectors;")
        vec_rows = cur.fetchall()
        print("chunk_vectors rows:", vec_rows)

        # 2) operador vectorial SIN JOIN (debe regresar 1 fila)
        cur.execute("""
            SELECT chunk_id, (embedding <=> (%s)::vector) AS dist
            FROM chunk_vectors;
        """, (emb_str,))
        dist_rows = cur.fetchall()
        print("dist_rows:", dist_rows)

        # 3) JOIN sin ORDER BY (debe regresar 1 fila)
        cur.execute("""
            SELECT c.chunk_id, c.chunk_text
            FROM chunk_vectors v
            JOIN chunks c ON c.chunk_id = v.chunk_id
            LIMIT 5;
        """)
        join_rows = cur.fetchall()
        print("join_rows:", len(join_rows))

        # 4) JOIN + ORDER BY (debe regresar 1 fila)
        cur.execute("""
        SELECT c.chunk_id, c.chunk_text,
            (v.embedding <=> (%s)::vector) AS dist,
            (1 - (v.embedding <=> (%s)::vector)) AS score
        FROM chunk_vectors v
        JOIN chunks c ON c.chunk_id = v.chunk_id
        ORDER BY dist
            LIMIT 5;
        """, (emb_str, emb_str))
        rows = cur.fetchall()
        print("final rows:", len(rows))
        print(rows)

print("final rows:", len(rows))
for r in rows:
    print("\n---")
    print("chunk_id:", r[0], "score:", r[2])
    print(r[1])