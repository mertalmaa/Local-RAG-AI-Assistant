"""ONNX Runtime ile hafif ve hızlı cümle embedding üretimi."""

from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models" / "paraphrase-multilingual-MiniLM-L12-v2"
ONNX_MODEL_PATH = MODEL_DIR / "embedding.onnx"
TOKENIZER_PATH = MODEL_DIR / "tokenizer.json"
MAX_SEQUENCE_LENGTH = 128
EMBEDDING_DIMENSION = 384
DEFAULT_BATCH_SIZE = 32


class OnnxEmbeddingModel:
    """Sentence Transformers ile aynı mean-pooling embedding'ini üretir."""

    def __init__(self):
        missing_files = [
            path for path in (ONNX_MODEL_PATH, TOKENIZER_PATH) if not path.exists()
        ]
        if missing_files:
            missing_names = ", ".join(str(path) for path in missing_files)
            raise FileNotFoundError(f"ONNX embedding dosyaları bulunamadı: {missing_names}")

        self.tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
        self.tokenizer.enable_truncation(max_length=MAX_SEQUENCE_LENGTH)
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        self.session = ort.InferenceSession(
            str(ONNX_MODEL_PATH),
            providers=["CPUExecutionProvider"],
        )

    def encode(self, texts, normalize_embeddings=True, batch_size=DEFAULT_BATCH_SIZE):
        """Metinleri 384 boyutlu NumPy vektörlerine dönüştürür."""
        text_list = list(texts)
        if not text_list:
            return np.empty((0, EMBEDDING_DIMENSION), dtype=np.float32)

        batches = []
        for start in range(0, len(text_list), batch_size):
            encoded = self.tokenizer.encode_batch(text_list[start : start + batch_size])
            input_ids = np.asarray([item.ids for item in encoded], dtype=np.int64)
            attention_mask = np.asarray(
                [item.attention_mask for item in encoded], dtype=np.int64
            )
            token_type_ids = np.asarray(
                [item.type_ids for item in encoded], dtype=np.int64
            )

            token_embeddings = self.session.run(
                None,
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "token_type_ids": token_type_ids,
                },
            )[0]
            mask = attention_mask[..., None].astype(np.float32)
            embeddings = (token_embeddings * mask).sum(axis=1) / mask.sum(
                axis=1
            ).clip(min=1e-9)

            if normalize_embeddings:
                embeddings /= np.linalg.norm(
                    embeddings, axis=1, keepdims=True
                ).clip(min=1e-12)
            batches.append(embeddings)

        return np.concatenate(batches, axis=0)
