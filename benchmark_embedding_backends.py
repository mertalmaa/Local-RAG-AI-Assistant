"""PyTorch ve ONNX embedding başlangıç sürelerini ayrı süreçlerde ölçer."""

import json
import os
import sys
import time
from pathlib import Path


MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
QUESTION = "PCR nedir?"
ONNX_PATH = Path(os.environ.get("TEMP", ".")) / "local-rag-benchmark" / "embedding.onnx"


def find_snapshot():
    cache_root = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / "models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2"
        / "snapshots"
    )
    snapshots = [path for path in cache_root.iterdir() if path.is_dir()]
    if not snapshots:
        raise FileNotFoundError("Embedding modeli Hugging Face cache'inde bulunamadı.")
    return snapshots[0]


def export_onnx():
    started = time.perf_counter()
    import torch
    from sentence_transformers import SentenceTransformer

    ONNX_PATH.parent.mkdir(parents=True, exist_ok=True)
    sentence_model = SentenceTransformer(MODEL_NAME, local_files_only=True)
    transformer = sentence_model[0].auto_model.eval()

    class Encoder(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_ids, attention_mask, token_type_ids):
            return self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                return_dict=False,
            )[0]

    dummy_shape = (1, 16)
    torch.onnx.export(
        Encoder(transformer),
        (
            torch.ones(dummy_shape, dtype=torch.long),
            torch.ones(dummy_shape, dtype=torch.long),
            torch.zeros(dummy_shape, dtype=torch.long),
        ),
        ONNX_PATH,
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["token_embeddings"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "token_type_ids": {0: "batch", 1: "sequence"},
            "token_embeddings": {0: "batch", 1: "sequence"},
        },
        opset_version=17,
        dynamo=False,
    )
    print(json.dumps({"export_seconds": time.perf_counter() - started, "path": str(ONNX_PATH)}))


class OnnxEmbeddingModel:
    def __init__(self):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        snapshot = find_snapshot()
        self.tokenizer = Tokenizer.from_file(str(snapshot / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=128)
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        self.session = ort.InferenceSession(
            str(ONNX_PATH),
            providers=["CPUExecutionProvider"],
        )

    def encode(self, texts, normalize_embeddings=True):
        import numpy as np

        encoded = self.tokenizer.encode_batch(list(texts))
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
        embeddings = (token_embeddings * mask).sum(axis=1) / mask.sum(axis=1).clip(
            min=1e-9
        )
        if normalize_embeddings:
            embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True).clip(
                min=1e-12
            )
        return embeddings


def benchmark(backend):
    process_started = time.perf_counter()
    import numpy as np

    imports_finished = time.perf_counter()
    if backend == "pytorch":
        from sentence_transformers import SentenceTransformer

        backend_imported = time.perf_counter()
        model = SentenceTransformer(MODEL_NAME, local_files_only=True)
    elif backend == "onnx":
        import onnxruntime  # noqa: F401
        import tokenizers  # noqa: F401

        backend_imported = time.perf_counter()
        model = OnnxEmbeddingModel()
    else:
        raise ValueError(f"Bilinmeyen backend: {backend}")

    model_loaded = time.perf_counter()
    vector = model.encode([QUESTION], normalize_embeddings=True)[0]
    query_finished = time.perf_counter()
    print(
        "BENCHMARK_JSON="
        + json.dumps(
            {
                "backend": backend,
                "base_import_seconds": imports_finished - process_started,
                "backend_import_seconds": backend_imported - imports_finished,
                "model_load_seconds": model_loaded - backend_imported,
                "first_query_seconds": query_finished - model_loaded,
                "internal_total_seconds": query_finished - process_started,
                "vector_norm": float(np.linalg.norm(vector)),
                "vector_head": vector[:8].tolist(),
            }
        )
    )


def load_model(backend):
    if backend == "pytorch":
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(MODEL_NAME, local_files_only=True)
    if backend == "onnx":
        return OnnxEmbeddingModel()
    raise ValueError(f"Bilinmeyen backend: {backend}")


def benchmark_app(backend):
    process_started = time.perf_counter()
    from ingest import prepare_database
    from rag import ensure_foundry_ready, retrieve_chunks

    imports_finished = time.perf_counter()
    ensure_foundry_ready()
    foundry_finished = time.perf_counter()
    model = load_model(backend)
    model_loaded = time.perf_counter()
    db_path = prepare_database(model)
    database_ready = time.perf_counter()
    chunks = retrieve_chunks(db_path, model, QUESTION)
    query_finished = time.perf_counter()
    print(
        "APP_BENCHMARK_JSON="
        + json.dumps(
            {
                "backend": backend,
                "app_import_seconds": imports_finished - process_started,
                "foundry_ready_seconds": foundry_finished - imports_finished,
                "embedding_load_seconds": model_loaded - foundry_finished,
                "database_ready_seconds": database_ready - model_loaded,
                "first_retrieval_seconds": query_finished - database_ready,
                "prompt_ready_seconds": database_ready - process_started,
                "first_result_seconds": query_finished - process_started,
                "top_source": chunks[0]["metadata"]["source"],
                "top_distance": chunks[0]["distance"],
            }
        )
    )


def compare_vectors():
    import numpy as np

    pytorch_vector = load_model("pytorch").encode(
        [QUESTION], normalize_embeddings=True
    )[0]
    onnx_vector = load_model("onnx").encode([QUESTION], normalize_embeddings=True)[0]
    print(
        json.dumps(
            {
                "cosine_similarity": float(np.dot(pytorch_vector, onnx_vector)),
                "max_absolute_difference": float(
                    np.max(np.abs(pytorch_vector - onnx_vector))
                ),
            }
        )
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(
            "Kullanım: benchmark_embedding_backends.py "
            "export|pytorch|onnx|app-pytorch|app-onnx|compare"
        )
    if sys.argv[1] == "export":
        export_onnx()
    elif sys.argv[1].startswith("app-"):
        benchmark_app(sys.argv[1].removeprefix("app-"))
    elif sys.argv[1] == "compare":
        compare_vectors()
    else:
        benchmark(sys.argv[1])
