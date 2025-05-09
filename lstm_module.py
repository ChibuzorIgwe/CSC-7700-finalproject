import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout=0.2, device='cuda'):
        super(LSTMLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        # Output projection
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids, hidden=None, temperature=1.0):
        # Compute actual lengths (non-padding tokens) for efficient packing
        lengths = (input_ids != 0).sum(dim=1).cpu()

        embeds = self.embedding(input_ids)
        packed = nn.utils.rnn.pack_padded_sequence(embeds, lengths, batch_first=True, enforce_sorted=False)
        packed_out, hidden = self.lstm(packed, hidden)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

        logits = self.fc_out(output) / temperature
        return logits, hidden

    def generate(self, tokenizer, prompt, max_seq_length=50, temperature=1.0, eos_token='<eos>'):
        self.eval()
        input_ids = tokenizer.encode(prompt)
        input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(next(self.parameters()).device)
        generated = input_ids[:]

        hidden = None
        for _ in range(max_seq_length - len(input_ids)):
            logits, hidden = self.forward(input_tensor, hidden, temperature)
            next_token_logits = logits[:, -1, :]
            probs = F.softmax(next_token_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1).item()

            if tokenizer.decode([next_token_id]) == eos_token:
                break

            generated.append(next_token_id)
            input_tensor = torch.tensor([[next_token_id]], dtype=torch.long).to(next(self.parameters()).device)

        return tokenizer.decode(generated)
