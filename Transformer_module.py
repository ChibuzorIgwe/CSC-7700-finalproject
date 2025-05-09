import torch
import torch.nn as nn
import torch.nn.functional as F


# Add PositionalEncoding class
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerLanguageModel(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=2, dim_feedforward=2048, dropout=0.1, max_len=512, pad_idx=0):
        super(TransformerLanguageModel, self).__init__()
        self.pad_idx = pad_idx
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_embed = nn.Embedding(max_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Decoder shares weights with input embedding to keep vocabulary consistent
        self.fc = nn.Linear(d_model, vocab_size, bias=False)
        self.fc.weight = self.embed.weight  # weight tying

    def forward(self, x, temperature=None):
        seq_len = x.size(1)
        pos = torch.arange(0, seq_len, device=x.device).unsqueeze(0)
        x_embed = self.embed(x) + self.pos_embed(pos)

        # Create attention mask to prevent attending to padding tokens
        attention_mask = (x != self.pad_idx)

        # Causal Masking (Look-Ahead Masking)
        causal_mask = self._generate_causal_mask(seq_len).to(x.device)

        # Expand the causal mask to match the batch size
        mask = causal_mask.unsqueeze(0).expand(x.size(0), -1, -1)

        # Apply padding mask
        out = self.transformer(x_embed, src_key_padding_mask=~attention_mask)

        logits = self.fc(out)

        if temperature is not None:
            logits = logits / temperature  # Apply temperature if it's passed

        return logits

    def _generate_causal_mask(self, size):
        """Generate a mask for causal (look-ahead) masking."""
        mask = torch.triu(torch.ones(size, size), diagonal=1)
        return mask == 0  # Set the upper triangle to 0 (don't allow attention to future tokens)

    def generate(self, tokenizer, prompt, max_seq_length=50, temperature=1.0, eos_id=2):
        self.eval()
        generated = tokenizer.encode(prompt)  # Convert prompt to token IDs

        real_vocab = tokenizer.sp.get_piece_size()

        for _ in range(max_seq_length):
            input_tensor = torch.tensor([generated], dtype=torch.long).to(self.fc.weight.device)  # Add device compatibility
            logits = self.forward(input_tensor, temperature=temperature)  # Pass temperature to the forward pass
            next_token_logits = logits[0, -1, :]  # Get logits for the last token
            
            # Apply temperature to the logits here if needed
            next_token_logits = next_token_logits / temperature

            # Apply softmax to get probabilities and truncate to valid vocab size
            probs = F.softmax(next_token_logits[:real_vocab], dim=-1)
            
            # Sample from the probabilities
            next_token = torch.multinomial(probs, num_samples=1).item()
            generated.append(next_token)

            if next_token == eos_id:
                break

        return tokenizer.decode(generated)  # Decode the generated tokens back to text
