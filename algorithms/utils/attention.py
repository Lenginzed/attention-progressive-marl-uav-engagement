import torch
import torch.nn as nn

from .flatten import build_flattener
from .mlp import MLPLayer


def _get_activation(activation_id: int):
    return [nn.Tanh, nn.ReLU, nn.LeakyReLU, nn.ELU][activation_id]


def infer_token_dims(flat_dim: int, group_dim: int = 0):
    if group_dim and flat_dim % group_dim == 0 and flat_dim > group_dim:
        return [group_dim] * (flat_dim // group_dim)
    if flat_dim > 9 and (flat_dim - 9) % 6 == 0:
        return [9] + [6] * ((flat_dim - 9) // 6)
    return [flat_dim]


class MultiDimensionalAttentionBase(nn.Module):
    def __init__(
        self,
        obs_space,
        hidden_size,
        activation_id,
        use_feature_normalization,
        embed_dim=128,
        num_heads=4,
        dropout=0.0,
        sparse_topk=2,
        group_dim=0,
        query_token_count=1,
    ):
        super(MultiDimensionalAttentionBase, self).__init__()
        self._use_feature_normalization = use_feature_normalization
        self.obs_flattener = build_flattener(obs_space)
        input_dim = self.obs_flattener.size
        self.token_dims = infer_token_dims(input_dim, group_dim)
        self.query_token_count = min(max(query_token_count, 1), len(self.token_dims))
        self.sparse_topk = sparse_topk

        if self._use_feature_normalization:
            self.feature_norm = nn.LayerNorm(input_dim)

        activation_cls = _get_activation(activation_id)
        self.token_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(token_dim, embed_dim),
                activation_cls(),
                nn.LayerNorm(embed_dim),
            )
            for token_dim in self.token_dims
        ])
        self.token_embeddings = nn.Parameter(torch.zeros(1, len(self.token_dims), embed_dim))

        self.self_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.sparse_gate = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            activation_cls(),
            nn.Linear(embed_dim, 1),
        )

        self.fusion_logits = nn.Parameter(torch.zeros(3))
        self.residual_norm = nn.LayerNorm(embed_dim)
        self.mlp = MLPLayer(embed_dim, hidden_size, activation_id)

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        if self._use_feature_normalization:
            x = self.feature_norm(x)

        chunks = torch.split(x, self.token_dims, dim=-1)
        tokens = [token_mlp(chunk) for token_mlp, chunk in zip(self.token_mlps, chunks)]
        tokens = torch.stack(tokens, dim=1) + self.token_embeddings

        self_context, self_weights = self.self_attention(
            tokens,
            tokens,
            tokens,
            need_weights=return_aux,
            average_attn_weights=False,
        )
        self_context = self_context.mean(dim=1)

        if len(self.token_dims) > self.query_token_count:
            query = tokens[:, :self.query_token_count].mean(dim=1, keepdim=True)
            context = tokens[:, self.query_token_count:]
            cross_context, cross_weights = self.cross_attention(
                query,
                context,
                context,
                need_weights=return_aux,
                average_attn_weights=False,
            )
            cross_context = cross_context.squeeze(1)
        else:
            cross_context = tokens.mean(dim=1)
            cross_weights = None

        sparse_logits_raw = self.sparse_gate(tokens).squeeze(-1)
        sparse_logits = sparse_logits_raw
        if 0 < self.sparse_topk < len(self.token_dims):
            topk = min(self.sparse_topk, len(self.token_dims))
            keep_indices = torch.topk(sparse_logits, k=topk, dim=1).indices
            keep_mask = torch.zeros_like(sparse_logits, dtype=torch.bool)
            keep_mask.scatter_(1, keep_indices, True)
            sparse_logits = sparse_logits.masked_fill(~keep_mask, float("-inf"))
        sparse_weights = torch.softmax(sparse_logits, dim=1).unsqueeze(-1)
        sparse_context = (sparse_weights * tokens).sum(dim=1)

        fusion_weights = torch.softmax(self.fusion_logits, dim=0)
        fused_context = (
            fusion_weights[0] * self_context
            + fusion_weights[1] * cross_context
            + fusion_weights[2] * sparse_context
        )
        fused_context = self.residual_norm(fused_context + tokens.mean(dim=1))
        output = self.mlp(fused_context)

        if not return_aux:
            return output

        aux = {
            "token_dims": list(self.token_dims),
            "query_token_count": int(self.query_token_count),
            "tokens": tokens.detach(),
            "self_attention_weights": None if self_weights is None else self_weights.detach(),
            "cross_attention_weights": None if cross_weights is None else cross_weights.detach(),
            "sparse_logits_raw": sparse_logits_raw.detach(),
            "sparse_weights": sparse_weights.squeeze(-1).detach(),
            "fusion_weights": fusion_weights.detach(),
        }
        return output, aux

    @property
    def output_size(self) -> int:
        return self.mlp.output_size
