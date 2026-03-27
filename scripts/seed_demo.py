from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os
from openai import OpenAI
import hashlib

load_dotenv()

DB_URL = os.environ["DATABASE_URL"]
EMB_MODEL = os.environ.get("EMBEDDINGS_MODEL", "text-embedding-3-small")

engine = create_engine(DB_URL)
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

DOC_TEXT = """
Artículo 69-B del Código Fiscal de la Federación (CFF).
Procedimiento de presunción de inexistencia de operaciones:
- Autoridad presume operaciones inexistentes.
- Publicación/listado.
- Contribuyente puede aportar pruebas para desvirtuar.
- Resolución y efectos fiscales.
"""

def sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def chunk_text(texto: str, max_len: int = 350):
    # chunker simple por párrafos
    parts = [p.strip() for p in texto.split("\n") if p.strip()]
    chunks = []
    cur = ""
    for p in parts:
        if len(cur) + len(p) + 1 <= max_len:
            cur = (cur + "\n" + p).strip()
        else:
            chunks.append(cur)
            cur = p
    if cur:
        chunks.append(cur)
    return chunks

def main():
    doc_hash = sha(DOC_TEXT)

    with engine.begin() as conn:
        # insertar documento
        doc_id = conn.execute(text("""
            INSERT INTO documents(source, doc_type, authority, publication_date, identifiers, canonical_url, raw_text, document_hash, version)
            VALUES (:source, :doc_type, :authority, NULL, '{}'::jsonb, :url, :raw_text, :doc_hash, 1)
            RETURNING id
        """), {
            "source": "DEMO",
            "doc_type": "articulo",
            "authority": "CFF",
            "url": "https://demo.local/cff/69-b",
            "raw_text": DOC_TEXT,
            "doc_hash": doc_hash
        }).scalar_one()

        # chunks + embeddings
        chunks = chunk_text(DOC_TEXT)
        for i, ch in enumerate(chunks):
            ch_hash = sha(ch + doc_hash + str(i))

            chunk_id = conn.execute(text("""
                INSERT INTO chunks(document_id, chunk_text, chunk_hash, start_offset, end_offset)
                VALUES (:doc_id, :chunk_text, :chunk_hash, NULL, NULL)
                RETURNING chunk_id
            """), {
                "doc_id": doc_id,
                "chunk_text": ch,
                "chunk_hash": ch_hash
            }).scalar_one()

            emb = client.embeddings.create(model=EMB_MODEL, input=ch).data[0].embedding

            conn.execute(text("""
                INSERT INTO chunk_vectors(chunk_id, embedding)
                VALUES (:chunk_id, :embedding)
            """), {
                "chunk_id": chunk_id,
                "embedding": emb
            })

    print("Seed listo. Documento:", doc_id, "Chunks:", len(chunks))

if __name__ == "__main__":
    main()
