"""Belgeyi okuma, parçalara ayırma ve SQLite veritabanına kaydetme işlemleri."""

import hashlib
import json
import re
import sqlite3
from pathlib import Path

from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent
DATABASE_PATH = BASE_DIR / "rag_storage.db"
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DOCUMENT_PATTERN = "*.txt"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 80


def split_into_chunks(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Metni cümle sınırlarını koruyarak, örtüşen parçalara ayırır."""
    normalized_text = re.sub(r"\s+", " ", text).strip()
    if not normalized_text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", normalized_text)
    chunks = []
    current_sentences = []

    for sentence in sentences:
        candidate = " ".join([*current_sentences, sentence])
        if current_sentences and len(candidate) > chunk_size:
            chunks.append(" ".join(current_sentences))

            overlap_sentences = []
            overlap_length = 0
            for previous_sentence in reversed(current_sentences):
                overlap_sentences.append(previous_sentence)
                overlap_length += len(previous_sentence) + 1
                if overlap_length >= overlap:
                    break

            current_sentences = list(reversed(overlap_sentences))

        current_sentences.append(sentence)

    if current_sentences:
        final_chunk = " ".join(current_sentences)
        if not chunks or final_chunk != chunks[-1]:
            chunks.append(final_chunk)

    return chunks


def load_documents():
    """docs/ klasöründeki veya proje kökündeki bütün metin belgelerini okur."""
    docs_folder = BASE_DIR / "docs"
    if docs_folder.exists() and list(docs_folder.glob(DOCUMENT_PATTERN)):
        search_dir = docs_folder
    else:
        search_dir = BASE_DIR

    document_paths = sorted(
        search_dir.glob(DOCUMENT_PATTERN),
        key=lambda path: path.name.lower(),
    )
    if not document_paths:
        raise FileNotFoundError(
            f"{search_dir} klasöründe indekslenecek bir .txt belgesi bulunamadı."
        )

    return [
        {
            "path": path,
            "content": path.read_text(encoding="utf-8"),
        }
        for path in document_paths
    ]


def calculate_index_hash(documents):
    """Bütün belgeler ve indeksleme ayarları için ortak parmak izi üretir."""
    document_content = "".join(
        f"\n--- {document['path'].name} ---\n{document['content']}"
        for document in documents
    )
    index_input = (
        f"{EMBEDDING_MODEL_NAME}|{CHUNK_SIZE}|{CHUNK_OVERLAP}|{document_content}"
    )
    return hashlib.sha256(index_input.encode("utf-8")).hexdigest()


def load_embedding_model():
    """Model cache'te varsa yerelden yükler; yoksa bir kez indirir."""
    try:
        print("Embedding modeli yerel cache'te aranıyor...")
        return SentenceTransformer(
            EMBEDDING_MODEL_NAME,
            local_files_only=True,
        )
    except Exception:
        print("Embedding modeli bulunamadı. Hugging Face'ten indiriliyor...")
        return SentenceTransformer(EMBEDDING_MODEL_NAME)


def init_database(db_path=DATABASE_PATH):
    """SQLite veritabanı tablosunu ve indeksini oluşturur."""
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                title TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                embedding TEXT NOT NULL,
                index_hash TEXT NOT NULL
            )
            """
        )
        conn.commit()


def prepare_database(embedding_model, db_path=DATABASE_PATH):
    """SQLite veritabanını açar ve gerekirse güncel belgelerle yeniden kurar."""
    init_database(db_path)
    documents = load_documents()
    current_index_hash = calculate_index_hash(documents)

    stored_index_hash = None
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT index_hash FROM documents LIMIT 1")
        row = cursor.fetchone()
        if row:
            stored_index_hash = row[0]
        cursor.execute("SELECT COUNT(*) FROM documents")
        collection_count = cursor.fetchone()[0]

    if collection_count == 0 or stored_index_hash != current_index_hash:
        if collection_count > 0:
            print("Belge veya indeksleme ayarları değişmiş. SQLite indeksi yenileniyor...")
            with sqlite3.connect(db_path) as conn:
                conn.execute("DELETE FROM documents")
                conn.commit()
        else:
            print("Belgeler işleniyor ve SQLite veritabanına kaydediliyor...")

        chunks = []
        rows = []

        for document_index, document in enumerate(documents):
            document_chunks = split_into_chunks(document["content"])

            for chunk_index, chunk in enumerate(document_chunks):
                chunks.append(chunk)
                chunk_id = f"doc_{document_index}_chunk_{chunk_index}"
                rows.append((
                    chunk_id,
                    document["path"].name,
                    document["path"].stem,
                    chunk_index,
                    chunk,
                    current_index_hash,
                ))

        print(f"{len(chunks)} parça için embedding vektörleri hesaplanıyor...")
        embeddings = embedding_model.encode(
            chunks,
            normalize_embeddings=True,
        ).tolist()

        insert_data = [
            (
                row[0],
                row[1],
                row[2],
                row[3],
                row[4],
                json.dumps(embedding),
                row[5],
            )
            for row, embedding in zip(rows, embeddings)
        ]

        with sqlite3.connect(db_path) as conn:
            conn.executemany(
                """
                INSERT INTO documents (
                    id, source, title, chunk_index, content, embedding, index_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                insert_data,
            )
            conn.commit()

        print(
            f"SQLite veritabanı {len(documents)} belge ve "
            f"{len(chunks)} parça ile hazırlandı.\n"
        )
    else:
        print(
            f"Belgeler değişmemiş. SQLite veritabanındaki {collection_count} parça kullanılıyor.\n"
        )

    return db_path


# Geriye dönük uyumluluk için alias
prepare_collection = prepare_database
