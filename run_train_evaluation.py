# Language Model Training & Evaluation with Validation Loss (RNN, LSTM, Transformer)
"""Train/evaluate RNN, LSTM, Transformer and compare metrics with/without retrieval-augmented generation (RAG)."""

import torch
from tokenizer import load_jsonl, BPEWrapper, TextDataset
from train_and_evaluation import (
    train_model,
    evaluate_bleu,
    compute_perplexity,
    plot_loss_curve_with_validation,
    evaluate_prompt_response,
)
from rag_utils import FaissRetriever, rag_generate
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from lstm_module import LSTMLanguageModel
from rnn_module import RNNLanguageModel
from Transformer_module import TransformerLanguageModel

# === CONFIGURATION ===
# Use raw strings so backslashes aren't treated as escape sequences
train_file = r"C:\Users\USER\Desktop\Spring 2025\CSC-7700-Project2-main\train.json"
test_file = r"C:\Users\USER\Desktop\Spring 2025\CSC-7700-Project2-main\test.json"
tokenizer_model = r"C:\Users\USER\Desktop\Spring 2025\CSC-7700-Project2-main\bpe_tokenizer.model"

# Hyperparameters
embed_dim = 256
hidden_dim = 512
num_layers = 2
num_heads = 8
max_len = 512
epochs = 30
batch_size = 128
lr = 5e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# === LOAD DATA ===
tokenizer = BPEWrapper(tokenizer_model)
# Fetch the real vocabulary size from the SentencePiece model
vocab_size = tokenizer.sp.get_piece_size()
train_data = load_jsonl(train_file)
test_data = load_jsonl(test_file)
train_dataset = TextDataset(train_data, tokenizer, max_len=max_len)
test_dataset = TextDataset(test_data, tokenizer, max_len=max_len)

# === LOAD RETRIEVER FOR RAG ===
retriever = FaissRetriever()
try:
    retriever.load()  # uses default rag_index.faiss / rag_docs.json
except Exception as e:
    print("[WARN] Could not load retrieval index – RAG metrics will be skipped.", e)
    retriever = None

# List of models to train sequentially
model_types = ['rnn', 'lstm', 'transformer']

# ---------------- Helper for RAG BLEU ----------------

def evaluate_bleu_rag(model, tokenizer, data_list, retriever, device, model_type, k=5, max_gen_len=50):
    """Compute BLEU on generated answers that include retrieved context."""
    if retriever is None:
        return None

    model.eval()
    refs, hyps = [], []
    with torch.no_grad():
        for item in data_list:  # each item is a dict with at least "prompt"/"response"
            prompt_text = item.get("prompt") or item.get("question") or item.get("input")
            target_text = item.get("response") or item.get("answer") or item.get("output")
            if not prompt_text or not target_text:
                continue

            gen_text = rag_generate(
                model,
                tokenizer,
                prompt_text,
                retriever,
                k=k,
                max_ctx_tokens=max_len - 50,
                max_seq_length=max_gen_len,
            )

            refs.append([target_text.split()])
            hyps.append(gen_text.split())

    smooth = SmoothingFunction().method4
    return corpus_bleu(refs, hyps, smoothing_function=smooth)

# === SEQUENTIAL TRAINING ===
for model_type in model_types:
    print(f"\nTraining {model_type.upper()} model...")

    # === MODEL SETUP ===
    if model_type == 'rnn':
        model = RNNLanguageModel(vocab_size, embed_dim, hidden_dim, num_layers)
    elif model_type == 'lstm':
        model = LSTMLanguageModel(vocab_size, embed_dim, hidden_dim, num_layers)
    elif model_type == 'transformer':
        model = TransformerLanguageModel(vocab_size, embed_dim, num_heads, num_layers)

    # === TRAINING WITH VALIDATION ===
    print(f"Training {model_type.upper()} model...")
    loss_history, val_loss_history = train_model(model, train_dataset, val_dataset=test_dataset,
                                                 epochs=epochs, batch_size=batch_size, lr=lr, device=device, model_type=model_type)

    plot_loss_curve_with_validation(loss_history, val_loss_history, model_type)

    # === EVALUATION ===
    ppl = compute_perplexity(model, test_dataset, batch_size, device=device, model_type=model_type)
    bleu = evaluate_bleu(model, test_dataset, batch_size, device=device, model_type=model_type)
    bleu_rag = evaluate_bleu_rag(model, tokenizer, test_data, retriever, device, model_type)

    print("\nModel Evaluation Metrics (plain vs RAG):")
    header = f"{'Model':<15} {'PPL':<10} {'BLEU':<10} {'BLEU+RAG':<10}"
    print(header)
    rag_val = f"{bleu_rag:.4f}" if bleu_rag is not None else "n/a"
    print(f"{model_type:<15} {ppl:<10.2f} {bleu:<10.4f} {rag_val:<10}")

    # === PROMPT RESPONSES ===
    prompt1 = "What is IMPES?"
    custom_prompt = "What is GDIMS?"
    response1 = evaluate_prompt_response(model, tokenizer, prompt1, device=device)
    response2 = evaluate_prompt_response(model, tokenizer, custom_prompt, device=device)

    print(f"\nPrompt 1: {prompt1}\nResponse: {response1}")
    print(f"\nPrompt 2: {custom_prompt}\nResponse: {response2}")
