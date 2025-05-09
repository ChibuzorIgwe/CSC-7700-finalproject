import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
import matplotlib.pyplot as plt
import numpy as np
import sentencepiece as spm
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from tqdm import tqdm

# Load the BPE tokenizer
sp = spm.SentencePieceProcessor()
sp.load("bpe_tokenizer.model")

def collate_fn(batch):
    xs, ys = zip(*batch)
    xs = pad_sequence(xs, batch_first=True, padding_value=0)
    ys = pad_sequence(ys, batch_first=True, padding_value=0)
    return xs, ys

def decode_sequence(seq):
    return sp.decode([int(x) for x in seq if int(x) != 0])

def train_model(model, dataset, val_dataset=None, epochs=30, batch_size=128, lr=5e-4, device='cuda', model_type="rnn"):
    model.to(device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    criterion = CrossEntropyLoss(ignore_index=0)
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=1)

    loss_history, val_loss_history = [], []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in tqdm(dataloader, desc=f"[{model_type}] Epoch {epoch+1}/{epochs} - Training"):
            x, y = x.to(device), y.to(device)
            if model_type == "rnn" :
                logits,_ = model(x)
            elif model_type == "lstm":
                logits,_ = model(x)
            else:
                logits = model(x)    
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        loss_history.append(total_loss / len(dataloader))

        if val_dataset:
            val_loss = compute_validation_loss(model, val_dataset, batch_size, criterion, device,model_type)
            val_loss_history.append(val_loss)

        # Implementing Early stopping
            
            if len(val_loss_history) >= 2 and val_loss_history[-1] > val_loss_history[-2]:
                print("Validation loss increased. Early stopping Activated.")
                break
            print(f"Epoch {epoch+1}, Train Loss: {loss_history[-1]:.4f}, Val Loss: {val_loss:.4f}")

            # Update learning rate based on validation loss
            scheduler.step(val_loss)
        else:
            print(f"Epoch {epoch+1}, Train Loss: {loss_history[-1]:.4f}, Val Loss: {val_loss:.4f}")

    return loss_history, val_loss_history

def compute_validation_loss(model, dataset, batch_size, criterion, device, model_type):
    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    total_loss = 0
    with torch.no_grad():
        for x, y in tqdm(dataloader, desc=f"[{model_type}] Evaluating"):
            x, y = x.to(device), y.to(device)
            if model_type == "rnn" :
                logits,_ = model(x)
            elif model_type == "lstm":
                logits,_ = model(x)
            else:
                logits = model(x) 
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate_bleu(model, dataset, batch_size, device, model_type="rnn"):
    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    criterion = CrossEntropyLoss(ignore_index=0)
    references = []
    hypotheses = []
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            if model_type == "rnn" :
                logits,_ = model(x)
            elif model_type == "lstm":
                logits,_ = model(x)
            else:
                logits = model(x) 
            preds = torch.argmax(logits, dim=-1)

            for ref_seq, pred_seq in zip(y.cpu().numpy(), preds.cpu().numpy()):
                ref_text = decode_sequence(ref_seq).split()
                hyp_text = decode_sequence(pred_seq).split()
                if len(ref_text) > 0 and len(hyp_text) > 0:
                    references.append([ref_text])
                    hypotheses.append(hyp_text)

    # Apply smoothing to avoid zero BLEU when no 4-gram overlap exists
    smoothing = SmoothingFunction().method4
    bleu = corpus_bleu(references, hypotheses, smoothing_function=smoothing)
    print(f"BLEU Score: {bleu:.4f}")
    return bleu

def compute_perplexity(model, dataset, batch_size, device, model_type="rnn"):
    model.to(device)
    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    criterion = CrossEntropyLoss(ignore_index=0, reduction='sum')
    total_loss, total_tokens = 0, 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            if model_type == "rnn" :
                logits,_ = model(x)
            elif model_type == "lstm":
                logits,_ = model(x)
            else:
                logits = model(x) 
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss += loss.item()
            total_tokens += (y != 0).sum().item()

    perplexity = np.exp(total_loss / total_tokens)
    print(f"Perplexity: {perplexity:.4f}")
    return perplexity

def plot_loss_curve_with_validation(train_loss, val_loss, model_name):
    plt.figure()
    plt.plot(train_loss, label="Training Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training and Validation Loss Curve for {model_name}")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"Losscurve_{model_name}.png")
    plt.close()

def evaluate_prompt_response(model, tokenizer, prompt, device='cuda'):
    model.to(device)
    model.eval()
    return model.generate(tokenizer, prompt, max_seq_length=50, temperature=1.0)