import torch
import torch.nn as nn


class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_dim, nhead=4, dim_feedforward=hidden_dim, dropout=0.1),
            num_layers=2,
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.2)
        self.projection = nn.Linear(embed_dim, 128)

    def forward(self, token_ids, attention_mask=None):
        x = self.embedding(token_ids)
        x = self.transformer(x, src_key_padding_mask=attention_mask)
        x = x.mean(dim=1)
        x = self.norm(x)
        x = self.dropout(x)
        return self.projection(x)


class EmbeddingService:
    def __init__(self, checkpoint_path, device="cuda"):
        self.device = torch.device(device)
        self.model = TextEncoder(vocab_size=30000, embed_dim=256, hidden_dim=512)
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)

    def encode_single(self, token_ids):
        tensor = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        with torch.inference_mode():
            embedding = self.model(tensor)
        return embedding.squeeze(0).cpu().numpy()

    def encode_batch(self, batch_token_ids, batch_masks):
        ids = torch.tensor(batch_token_ids, dtype=torch.long, device=self.device)
        masks = torch.tensor(batch_masks, dtype=torch.bool, device=self.device)
        with torch.inference_mode():
            embeddings = self.model(ids, attention_mask=masks)
        return embeddings.cpu().numpy()

    def compute_similarity(self, tokens_a, tokens_b):
        a = torch.tensor([tokens_a], dtype=torch.long, device=self.device)
        b = torch.tensor([tokens_b], dtype=torch.long, device=self.device)
        with torch.inference_mode():
            emb_a = self.model(a)
            emb_b = self.model(b)
        cosine_sim = nn.functional.cosine_similarity(emb_a, emb_b)
        return cosine_sim.item()
