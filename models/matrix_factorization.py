"""
Matrix Factorization — baseline recommendation model.

Each user and movie gets a dense embedding vector of dimension `embed_dim`.
The predicted affinity between a user and a movie is the dot product of their
embeddings.  This is the classic approach from SVD-style collaborative filtering
(see Koren et al., 2009).

What it captures well:  global user taste profiles and item popularity.
What it misses:         the ORDER in which a user watched movies — a user who
                        just finished a thriller trilogy is more likely to watch
                        another thriller next, which MF cannot model.
"""

import torch
import torch.nn as nn


class MatrixFactorization(nn.Module):
    """
    Dot-product matrix factorization model.

    Args:
        n_users:   number of distinct users in the dataset
        n_movies:  vocabulary size (number of distinct movies after encoding)
        embed_dim: dimensionality of user and movie embedding vectors
    """

    def __init__(self, n_users: int, n_movies: int, embed_dim: int = 64) -> None:
        super().__init__()

        # One row per user / movie — learned during training
        self.user_embeddings = nn.Embedding(n_users, embed_dim)
        self.movie_embeddings = nn.Embedding(n_movies, embed_dim)

        # Small initial weights → stable early training
        nn.init.normal_(self.user_embeddings.weight, std=0.01)
        nn.init.normal_(self.movie_embeddings.weight, std=0.01)

    def forward(self, user_ids: torch.Tensor, movie_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute predicted ratings for (user, movie) pairs.

        Args:
            user_ids:  (batch,) long tensor of user indices
            movie_ids: (batch,) long tensor of movie indices

        Returns:
            (batch,) float tensor of predicted affinity scores
        """
        u = self.user_embeddings(user_ids)  # (batch, embed_dim)
        m = self.movie_embeddings(movie_ids)  # (batch, embed_dim)

        # Element-wise product summed across embedding dimension → scalar per pair
        return (u * m).sum(dim=-1)

    @torch.no_grad()
    def recommend(
        self,
        user_id: int,
        seen_movie_ids: list[int],
        top_k: int = 10,
        device: torch.device | None = None,
    ) -> list[tuple[int, float]]:
        """
        Return top-k unseen movies for a given user, ranked by predicted score.

        Args:
            user_id:        integer user index
            seen_movie_ids: list of movie indices already watched (excluded from results)
            top_k:          number of recommendations to return
            device:         torch device to run on (defaults to cpu)

        Returns:
            list of (movie_idx, score) tuples, highest score first
        """
        if device is None:
            device = next(self.parameters()).device

        n_movies = self.movie_embeddings.weight.shape[0]

        uid_tensor = torch.tensor([user_id], device=device)
        u = self.user_embeddings(uid_tensor)  # (1, embed_dim)

        # Score against all movies at once — cheaper than a loop
        all_movies = self.movie_embeddings.weight  # (n_movies, embed_dim)
        scores = (u @ all_movies.T).squeeze(0)  # (n_movies,)

        # Zero out already-seen movies so they don't appear in results
        seen_set = set(seen_movie_ids)
        mask = torch.tensor([i in seen_set for i in range(n_movies)], device=device)
        scores[mask] = float("-inf")

        top_indices = scores.argsort(descending=True)[:top_k]
        return [(int(idx), float(scores[idx])) for idx in top_indices]
