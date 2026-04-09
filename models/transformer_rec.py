"""
Transformer-based sequential recommendation model.

Architecture overview:
  - Movie embedding layer: maps movie index → dense vector (embed_dim=128)
  - Learned positional embedding: encodes position within the sequence
  - Transformer encoder (2 layers, 2 heads): attends over the sequence with a
    causal (upper-triangular) mask so position t can only see positions ≤ t
  - Linear output head: projects each position's hidden state → logits over
    the full movie vocabulary

Training objective:
  Given [m_0, m_1, …, m_{T-1}], predict [m_1, m_2, …, m_T].
  This is next-item prediction via shifted cross-entropy loss — the same
  formulation used by language models, applied to watch histories.

Why a causal mask?
  Without it the model could "cheat" by attending to the answer (m_{t+1})
  when predicting m_{t+1}.  The mask forces it to predict each next item
  using only the movies that came before it.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerRecommender(nn.Module):
    """
    Causal Transformer encoder for next-item sequential recommendation.

    Args:
        vocab_size:    number of distinct movies (after preprocessing)
        embed_dim:     dimensionality of movie + positional embeddings
        num_heads:     number of self-attention heads
        num_layers:    number of Transformer encoder layers
        ffn_dim:       inner dimensionality of the feed-forward sublayer
        max_seq_len:   maximum sequence length (window size during training)
        dropout:       dropout probability applied after embeddings and inside layers
        pad_idx:       index used for padding (set to vocab_size by convention)
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        num_heads: int = 2,
        num_layers: int = 2,
        ffn_dim: int = 256,
        max_seq_len: int = 50,
        dropout: float = 0.1,
        pad_idx: int | None = None,
    ) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        # Padding index is one past the last real movie index
        self.pad_idx = pad_idx if pad_idx is not None else vocab_size

        # Movie embedding — padding index gets a zero vector that never updates
        self.movie_embedding = nn.Embedding(
            vocab_size + 1,  # +1 for the pad token
            embed_dim,
            padding_idx=self.pad_idx,
        )
        # Positional embedding — learned, one vector per position
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)

        self.embed_dropout = nn.Dropout(dropout)

        # Stack of Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,  # (batch, seq, embed) — more intuitive than (seq, batch, embed)
            norm_first=True,  # Pre-LN is more stable than post-LN for small models
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Project from embed_dim → vocabulary logits at every sequence position
        self.output_head = nn.Linear(embed_dim, vocab_size)

        self._init_weights()

    def _init_weights(self) -> None:
        """
        Small normal initialisation for embeddings; Xavier uniform for linear layers.
        Keeps gradient magnitudes reasonable at the start of training.
        """
        nn.init.normal_(self.movie_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)
        nn.init.xavier_uniform_(self.output_head.weight)
        nn.init.zeros_(self.output_head.bias)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Upper-triangular boolean mask of shape (seq_len, seq_len).

        PyTorch's TransformerEncoder treats True positions as "ignore this",
        so we set the upper triangle (future positions) to True.

            Position:  0  1  2  3
            Attend to: ✓  ✗  ✗  ✗   ← predicting position 0
                       ✓  ✓  ✗  ✗   ← predicting position 1
                       ✓  ✓  ✓  ✗   ← predicting position 2
                       ✓  ✓  ✓  ✓   ← predicting position 3
        """
        return torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
            diagonal=1,
        )

    def forward(
        self,
        movie_ids: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            movie_ids:             (batch, seq_len) long tensor of movie indices
            src_key_padding_mask:  (batch, seq_len) bool tensor; True = padding position

        Returns:
            logits: (batch, seq_len, vocab_size) — unnormalised scores for each
                    next-item position
        """
        batch_size, seq_len = movie_ids.shape

        # Positions 0, 1, …, seq_len-1 broadcast across batch
        positions = torch.arange(seq_len, device=movie_ids.device).unsqueeze(
            0
        )  # (1, seq_len)

        # Combine movie content and positional information
        x = self.movie_embedding(movie_ids) + self.position_embedding(positions)
        x = self.embed_dropout(x)  # (batch, seq_len, embed_dim)

        causal_mask = self._causal_mask(seq_len, movie_ids.device)

        # Transformer encoder with causal mask
        x = self.transformer(
            x,
            mask=causal_mask,
            src_key_padding_mask=src_key_padding_mask,
            is_causal=True,  # hint to PyTorch to use flash-attention if available
        )

        # Project to vocabulary size at every position
        logits = self.output_head(x)  # (batch, seq_len, vocab_size)
        return logits

    @torch.no_grad()
    def recommend(
        self,
        history: list[int],
        top_k: int = 10,
        seen_movie_ids: list[int] | None = None,
        device: torch.device | None = None,
    ) -> list[tuple[int, float]]:
        """
        Given a user's watch history (ordered list of movie indices), return
        top-k predicted next movies.

        Args:
            history:        ordered list of movie indices (most recent last)
            top_k:          number of recommendations to return
            seen_movie_ids: movies to exclude from results (typically == history)
            device:         torch device (defaults to model's device)

        Returns:
            list of (movie_idx, probability) tuples, highest prob first
        """
        if device is None:
            device = next(self.parameters()).device

        # Truncate to max_seq_len (keep most recent context)
        history = history[-self.max_seq_len :]
        seq_tensor = torch.tensor([history], dtype=torch.long, device=device)  # (1, T)

        logits = self.forward(seq_tensor)  # (1, T, vocab_size)
        # Take logits at the last position — this is the "next item" prediction
        last_logits = logits[0, -1, :]  # (vocab_size,)

        # Mask out seen movies
        if seen_movie_ids:
            for mid in seen_movie_ids:
                if 0 <= mid < self.vocab_size:
                    last_logits[mid] = float("-inf")

        probs = F.softmax(last_logits, dim=-1)
        top_indices = probs.argsort(descending=True)[:top_k]
        return [(int(idx), float(probs[idx])) for idx in top_indices]
