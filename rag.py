"""Kullanıcı sorusu için SQLite retrieval ve yerel LLM cevap üretme işlemleri."""

import json
import os
import shutil
import sqlite3
import subprocess
import time
from pathlib import Path

import numpy as np
import requests


BASE_DIR = Path(__file__).resolve().parent
DATABASE_PATH = BASE_DIR / "rag_storage.db"

FOUNDRY_PORT = int(os.getenv("FOUNDRY_PORT", "52268"))
FOUNDRY_BASE_URL = os.getenv(
    "FOUNDRY_BASE_URL",
    f"http://127.0.0.1:{FOUNDRY_PORT}",
).rstrip("/")
FOUNDRY_URL = f"{FOUNDRY_BASE_URL}/v1/chat/completions"
CHAT_MODEL_NAME = os.getenv("FOUNDRY_CHAT_MODEL", "phi-3.5-mini")
TOP_K = 2
MAX_RETRIEVAL_DISTANCE = float(os.getenv("RAG_MAX_DISTANCE", "0.65"))
FALLBACK_ANSWER = "Bu bilgi belgelerde bulunmuyor."

SYSTEM_PROMPT = """Sen yalnızca sağlanan <context> içerisindeki belgelere dayanarak doğrudan ve kesin cevap veren bir asistansın.

Kurallar:
1. Yalnızca <context> etiketleri arasındaki açık bilgileri kullan. Kendi genel bilgini veya tahminlerini kesinlikle ekleme.
2. Soruya ait bilgi context içinde açıkça yoksa sadece "Bu bilgi belgelerde bulunmuyor." yaz.
3. Soruda bir tanım isteniyorsa doğrudan tanım cümlesini, bir platform veya araç soruluyorsa adını doğrudan yaz. Türkçe, net ve en fazla 2 cümleyle cevap ver.
4. Cevabının sonuna [Kaynak: dosya_adı, Parça: no] biçiminde kaynak ekle.
"""

# Bellek içi parça önbelleği
_CHUNKS_METADATA = None
_EMBEDDINGS_MATRIX = None
_CACHE_TIMESTAMP = 0


def is_foundry_server_ready():
    """Foundry'nin yerel HTTP servisine erişilip erişilemediğini kontrol eder."""
    try:
        response = requests.get(f"{FOUNDRY_BASE_URL}/v1/models", timeout=2)
        return response.ok
    except requests.RequestException:
        return False


def is_chat_model_loaded():
    """İstenen sohbet modelinin Foundry belleğinde yüklü olup olmadığını kontrol eder."""
    try:
        response = requests.get(f"{FOUNDRY_BASE_URL}/v1/models", timeout=2)
        response.raise_for_status()
        models = response.json().get("data", [])
    except (requests.RequestException, ValueError):
        return False

    expected_name = CHAT_MODEL_NAME.lower()
    for model in models:
        model_id = str(model.get("id", "")).lower()
        parent_name = str(model.get("parent", "")).lower()
        if expected_name in {model_id, parent_name}:
            return True

    return False


def run_foundry_command(*arguments):
    """Foundry CLI komutunu çalıştırır ve hata oluşursa anlaşılır mesaj üretir."""
    foundry_executable = shutil.which("foundry")
    if foundry_executable is None:
        raise RuntimeError(
            "Foundry Local kurulu değil veya 'foundry' komutu PATH içinde bulunamadı."
        )

    result = subprocess.run(
        [foundry_executable, *arguments],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
    )
    if result.returncode != 0:
        error_message = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"Foundry komutu başarısız oldu: {error_message}")

    return result.stdout.strip()


def ensure_foundry_ready():
    """Gerekirse Foundry sunucusunu başlatır ve sohbet modelini belleğe yükler."""
    if is_foundry_server_ready():
        print("Foundry Local sunucusu zaten çalışıyor.")
    else:
        print("Foundry Local sunucusu başlatılıyor...")
        run_foundry_command(
            "server",
            "start",
            "--port",
            str(FOUNDRY_PORT),
            "--idle-timeout",
            "0",
        )

        for _ in range(10):
            if is_foundry_server_ready():
                break
            time.sleep(0.5)
        else:
            raise RuntimeError("Foundry Local başlatıldı ancak HTTP servisine erişilemiyor.")

    if is_chat_model_loaded():
        print(f"{CHAT_MODEL_NAME} modeli zaten bellekte yüklü.")
    else:
        print(f"{CHAT_MODEL_NAME} modeli belleğe yükleniyor...")
        run_foundry_command("model", "load", CHAT_MODEL_NAME)
    print("Foundry Local kullanıma hazır.\n")


def _get_cached_database_records(db_path):
    """SQLite'taki parçaları ve vektörleri hızlı arama için belleğe yükler."""
    global _CHUNKS_METADATA, _EMBEDDINGS_MATRIX, _CACHE_TIMESTAMP

    resolved_path = Path(db_path)
    if not resolved_path.exists():
        return [], np.zeros((0, 384), dtype=np.float32)

    current_mtime = resolved_path.stat().st_mtime
    if _CHUNKS_METADATA is None or current_mtime != _CACHE_TIMESTAMP:
        with sqlite3.connect(resolved_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, source, title, chunk_index, content, embedding FROM documents"
            )
            rows = cursor.fetchall()

        metadata_list = []
        vectors = []
        for row in rows:
            metadata_list.append(
                {
                    "id": row[0],
                    "source": row[1],
                    "title": row[2],
                    "chunk_index": row[3],
                    "content": row[4],
                }
            )
            vectors.append(json.loads(row[5]))

        _CHUNKS_METADATA = metadata_list
        _EMBEDDINGS_MATRIX = (
            np.array(vectors, dtype=np.float32)
            if vectors
            else np.zeros((0, 384), dtype=np.float32)
        )
        _CACHE_TIMESTAMP = current_mtime

    return _CHUNKS_METADATA, _EMBEDDINGS_MATRIX


def retrieve_chunks(db_target, embedding_model, question, top_k=TOP_K):
    """Soruya en yakın parçaları SQLite ve kosinüs benzerliği ile döndürür."""
    db_path = db_target if isinstance(db_target, (str, Path)) else DATABASE_PATH
    chunks_meta, matrix = _get_cached_database_records(db_path)

    if len(matrix) == 0:
        return []

    question_vector = embedding_model.encode(
        [question],
        normalize_embeddings=True,
    )[0]

    # Normalize vektörlerde kosinüs benzerliği = nokta çarpımı
    similarities = np.dot(matrix, question_vector)
    distances = 1.0 - similarities

    top_indices = np.argsort(distances)[:top_k]

    return [
        {
            "document": chunks_meta[idx]["content"],
            "metadata": {
                "source": chunks_meta[idx]["source"],
                "chunk_index": chunks_meta[idx]["chunk_index"],
                "title": chunks_meta[idx]["title"],
            },
            "distance": float(distances[idx]),
        }
        for idx in top_indices
    ]


def format_context(chunks):
    """Bulunan parçaları kaynak etiketli LLM context'ine dönüştürür."""
    return "\n\n".join(
        (
            f"[Kaynak: {chunk['metadata']['source']}, "
            f"Parça: {chunk['metadata']['chunk_index']}]\n"
            f"{chunk['document']}"
        )
        for chunk in chunks
    )


def is_context_relevant(chunks, max_distance=MAX_RETRIEVAL_DISTANCE):
    """En iyi eşleşmenin cevap üretmek için yeterince yakın olup olmadığını belirler."""
    return bool(chunks) and chunks[0]["distance"] <= max_distance


def retrieve_context(db_target, embedding_model, question, top_k=TOP_K):
    """Soruya en yakın parçaları, kaynak bilgileriyle birlikte döndürür."""
    chunks = retrieve_chunks(db_target, embedding_model, question, top_k)

    sources = []
    for chunk in chunks:
        source = chunk["metadata"]["source"]
        if source not in sources:
            sources.append(source)

    return format_context(chunks), sources, is_context_relevant(chunks)


def stream_answer(question, context):
    """Soruyu ve bulunan context'i Foundry Local'a gönderip cevabı parça parça üretir."""
    augmented_prompt = (
        "Aşağıdaki context'e dayanarak soruyu cevapla:\n\n"
        f"<context>\n{context}\n</context>\n\n"
        f"Kullanıcı sorusu: {question}"
    )
    parameters = {
        "model": CHAT_MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": augmented_prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 160,
        "stream": True,
    }

    response = requests.post(
        FOUNDRY_URL,
        json=parameters,
        stream=True,
        timeout=(5, 120),
    )
    response.raise_for_status()

    for line in response.iter_lines():
        if not line:
            continue

        line_text = line.decode("utf-8")
        if not line_text.startswith("data: "):
            continue

        data_content = line_text[6:]
        if data_content.strip() == "[DONE]":
            break

        try:
            chunk_json = json.loads(data_content)
        except json.JSONDecodeError:
            continue

        choices = chunk_json.get("choices") or []
        if not choices:
            continue

        delta = choices[0].get("delta") or {}
        if "content" in delta:
            yield delta["content"]
