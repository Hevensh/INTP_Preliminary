import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed(nn.Module):
    def __init__(self, img_size: int = 32, patch_size: int = 4, in_chans: int = 3, embed_dim: int = 128):
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError(f"img_size ({img_size}) must be divisible by patch_size ({patch_size})")

        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # (B, D, H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        return x


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, attn_dropout: float = 0.0, proj_dropout: float = 0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.attn_dropout = float(attn_dropout)

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.proj_drop = nn.Dropout(proj_dropout)

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor | None = None) -> torch.Tensor:
        bsz, seq_len, dim = x.shape
        qkv = self.qkv(x)  # (B, N, 3D)
        qkv = qkv.reshape(bsz, seq_len, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, N, Hd)

        if attn_bias is not None:
            static_shape = (self.num_heads, seq_len, seq_len)
            dynamic_shape = (bsz, self.num_heads, seq_len, seq_len)
            if tuple(attn_bias.shape) == static_shape:
                attn_bias = attn_bias.unsqueeze(0)
            elif tuple(attn_bias.shape) != dynamic_shape:
                raise ValueError(
                    f"attn_bias must have shape {static_shape} or {dynamic_shape}, "
                    f"got {tuple(attn_bias.shape)}"
                )
            # CUDA SDPA requires a well-aligned head stride for additive masks;
            # dynamically constructed per-layer look bias may be non-contiguous.
            attn_bias = attn_bias.to(device=x.device, dtype=q.dtype).contiguous()

        if hasattr(F, "scaled_dot_product_attention") and attn_bias is None:
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.attn_dropout if self.training else 0.0,
                is_causal=False,
            )
        else:
            # PyTorch's CUDA fused SDPA backward currently rejects some
            # learnable additive-mask layouts ("LSE ... strideH").  The
            # explicit path is reliable and modest at our ~196-token length.
            # Keep learnable-bias logits in fp32: fp16 backward through the
            # additive mask can overflow even when GradScaler is enabled.
            with torch.autocast(device_type=x.device.type, enabled=False):
                q_float, k_float, v_float = q.float(), k.float(), v.float()
                attn = (q_float @ k_float.transpose(-2, -1)) * self.scale
                if attn_bias is not None:
                    attn = attn + attn_bias.float()
                attn = attn.softmax(dim=-1)
                if self.attn_dropout:
                    attn = F.dropout(attn, p=self.attn_dropout, training=self.training)
                out = attn @ v_float  # (B, H, N, Hd)

        out = out.transpose(1, 2).reshape(bsz, seq_len, dim)  # (B, N, D)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=norm_eps)
        self.attn = MultiHeadSelfAttention(dim, num_heads, attn_dropout=attn_dropout, proj_dropout=dropout)
        self.norm2 = nn.LayerNorm(dim, eps=norm_eps)
        self.mlp = MLP(dim, hidden_dim=int(dim * mlp_ratio), dropout=dropout)

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), attn_bias=attn_bias)
        x = x + self.mlp(self.norm2(x))
        return x


def init_vit_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
