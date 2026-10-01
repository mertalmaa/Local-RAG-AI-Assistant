import sys
import requests

from ingest import load_embedding_model, prepare_database
from rag import FALLBACK_ANSWER, ensure_foundry_ready, retrieve_context, stream_answer

if sys.platform == "win32":
    try:
        sys.stdin.reconfigure(encoding="utf-8")
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass


EXIT_COMMANDS = {"çıkış", "cikis", "exit", "quit", "q"}


def run_chat_loop(db_path, embedding_model):
    """Kullanıcı çıkış komutu verene kadar yeni sorular alır."""
    print("Sorunuzu yazın. Programdan çıkmak için 'çıkış' yazabilirsiniz.\n")

    while True:
        try:
            question = input("Soru: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nProgram kapatılıyor.")
            break

        if question.lower() in EXIT_COMMANDS:
            print("Program kapatılıyor.")
            break

        if not question:
            print("Lütfen boş olmayan bir soru girin.\n")
            continue

        try:
            context, sources, is_relevant = retrieve_context(
                db_path,
                embedding_model,
                question,
            )
            print(f"İncelenen kaynaklar: {', '.join(sources)}")

            if not is_relevant:
                print(f"Cevap: {FALLBACK_ANSWER}\n")
                continue

            print("Cevap: ", end="", flush=True)
            for answer_piece in stream_answer(question, context):
                print(answer_piece, end="", flush=True)
            print("\n")
        except requests.RequestException as error:
            print(f"\nFoundry Local bağlantı hatası: {error}\n")
        except (KeyError, IndexError, ValueError) as error:
            print(f"\nModel cevabı işlenemedi: {error}\n")


def main():
    print("Uygulama başlatılıyor...", flush=True)
    try:
        ensure_foundry_ready()
        embedding_model = load_embedding_model()
        db_path = prepare_database(embedding_model)
    except (RuntimeError, OSError, requests.RequestException) as error:
        print(f"Uygulama başlatılamadı: {error}")
        return

    run_chat_loop(db_path, embedding_model)


if __name__ == "__main__":
    main()
