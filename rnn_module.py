
import torch
import torch.nn as nn
import torch.nn.functional as F

class RNNLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout=0.2, device='cuda'):
        super(RNNLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids, hidden=None, temperature=1.0):
        embeds = self.embedding(input_ids)  # (batch, seq_len, embed_dim)
        output, hidden = self.rnn(embeds, hidden)  # output: (batch, seq_len, hidden_dim)
        logits = self.fc_out(output)  # (batch, seq_len, vocab_size)
        logits = logits / temperature
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
