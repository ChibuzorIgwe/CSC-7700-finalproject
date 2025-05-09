import json
import argparse
import pathlib
import sentencepiece as spm


def build_corpus(json_path: pathlib.Path, corpus_path: pathlib.Path):
    """Extract prompt + completion text lines and write them to a text corpus."""
    with json_path.open("r", encoding="utf-8") as f_in, corpus_path.open("w", encoding="utf-8") as f_out:
        data = json.load(f_in)
        for item in data:
            # Expect each item to have 'prompt' and 'response' or 'completion'
            prompt = item.get("prompt", "")
            response = item.get("response", item.get("completion", ""))
            line = f"{prompt} {response}".strip()
            if line:
                f_out.write(line.replace("\n", " ") + "\n")


def train_tokenizer(corpus_file: pathlib.Path, model_prefix: str, vocab_size: int):
    spm.SentencePieceTrainer.Train(
        input=str(corpus_file),
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        user_defined_symbols="",
        hard_vocab_limit=False,  # allow trainer to pick smaller vocab if corpus is small
    )


def main():
    parser = argparse.ArgumentParser(description="Train SentencePiece BPE tokenizer from JSON dataset.")
    parser.add_argument("--data", type=pathlib.Path, default=pathlib.Path("train.json"), help="Path to train.json file")
    parser.add_argument("--vocab_size", type=int, default=8000, help="Vocabulary size (will back off if corpus too small)")
    parser.add_argument("--model_prefix", type=str, default="bpe_tokenizer", help="Output model prefix (creates .model and .vocab)")
    parser.add_argument("--tmp_corpus", type=pathlib.Path, default=pathlib.Path("corpus.txt"), help="Temporary corpus file")

    args = parser.parse_args()

    print(f"Building corpus from {args.data} → {args.tmp_corpus}")
    build_corpus(args.data, args.tmp_corpus)

    print("Training SentencePiece tokenizer…")
    train_tokenizer(args.tmp_corpus, args.model_prefix, args.vocab_size)
    print(f"Tokenizer saved as {args.model_prefix}.model and {args.model_prefix}.vocab")

    # Cleanup corpus file if you don't need it
    try:
        args.tmp_corpus.unlink()
    except Exception:
        pass


if __name__ == "__main__":
    main()
