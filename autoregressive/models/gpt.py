# Modified from:
#   VQGAN:    https://github.com/CompVis/taming-transformers/blob/master/taming/modules/transformer/mingpt.py
#   DiT:      https://github.com/facebookresearch/DiT/blob/main/models.py  
#   nanoGPT:  https://github.com/karpathy/nanoGPT/blob/master/model.py
#   llama:    https://github.com/facebookresearch/llama/blob/main/llama/model.py
#   gpt-fast: https://github.com/pytorch-labs/gpt-fast/blob/main/model.py
#   PixArt:   https://github.com/PixArt-alpha/PixArt-alpha/blob/master/diffusion/model/nets/PixArt_blocks.py
from dataclasses import dataclass
from typing import Optional, List


import torch
import torch.nn as nn
from torch.nn import functional as F
from utils.drop_path import DropPath


def find_multiple(n: int, k: int):
    if n % k == 0:
        return n
    return n + k - (n % k)

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layer: int = 32
    n_head: int = 32
    n_kv_head: Optional[int] = None
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    rope_base: float = 10000
    norm_eps: float = 1e-5
    initializer_range: float = 0.02
    
    token_dropout_p: float = 0.1
    attn_dropout_p: float = 0.0
    resid_dropout_p: float = 0.1
    ffn_dropout_p: float = 0.1
    drop_path_rate: float = 0.0

    num_classes: int = 1000
    caption_dim: int = 2048
    class_dropout_prob: float = 0.1
    model_type: str = 'c2i'

    vocab_size: int = 16384
    cls_token_num: int = 1
    block_size: int = 256
    max_batch_size: int = 32
    max_seq_len: int = 2048


#################################################################################
#                      Embedding Layers for Class Labels                        #
#################################################################################
class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels).unsqueeze(1)
        return embeddings


#################################################################################
#                      Embedding Layers for Text Feature                        #
#################################################################################
class CaptionEmbedder(nn.Module):
    """
    Embeds text caption into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, in_channels, hidden_size, uncond_prob, token_num=120):
        super().__init__()
        self.cap_proj = MLP(in_features=in_channels, hidden_features=hidden_size, out_features=hidden_size)
        self.register_buffer("uncond_embedding", nn.Parameter(torch.randn(token_num, in_channels) / in_channels ** 0.5))
        self.uncond_prob = uncond_prob

    def token_drop(self, caption, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(caption.shape[0], device=caption.device) < self.uncond_prob
        else:
            drop_ids = force_drop_ids == 1
        caption = torch.where(drop_ids[:, None, None], self.uncond_embedding, caption)
        return caption

    def forward(self, caption, train, force_drop_ids=None):
        use_dropout = self.uncond_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            caption = self.token_drop(caption, force_drop_ids)
        embeddings = self.cap_proj(caption)
        return embeddings


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=False)
        self.act = nn.GELU(approximate='tanh')
        self.fc2 = nn.Linear(hidden_features, out_features, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


#################################################################################
#                                  GPT Model                                    #
#################################################################################
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class FeedForward(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        hidden_dim = 4 * config.dim
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if config.ffn_dim_multiplier is not None:
            hidden_dim = int(config.ffn_dim_multiplier * hidden_dim)
        hidden_dim = find_multiple(hidden_dim, config.multiple_of)

        self.w1 = nn.Linear(config.dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(config.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config.dim, bias=False)
        self.ffn_dropout = nn.Dropout(config.ffn_dropout_p)

    def forward(self, x):
        return self.ffn_dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class KVCache(nn.Module):
    def __init__(self, max_batch_size, max_seq_length, n_head, head_dim, dtype):
        super().__init__()
        cache_shape = (max_batch_size, n_head, max_seq_length, head_dim)
        self.register_buffer('k_cache', torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer('v_cache', torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos, k_val, v_val):
        # input_pos: [S], k_val: [B, H, S, D]
        assert input_pos.shape[0] == k_val.shape[2]
        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val

        return k_out, v_out


class Attention(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        assert config.dim % config.n_head == 0
        self.dim = config.dim
        self.head_dim = config.dim // config.n_head
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head if config.n_kv_head is not None else config.n_head
        total_kv_dim = (self.n_head + 2 * self.n_kv_head) * self.head_dim

        # key, query, value projections for all heads, but in a batch
        self.wqkv = nn.Linear(config.dim, total_kv_dim, bias=False)
        self.wo = nn.Linear(config.dim, config.dim, bias=False)
        self.kv_cache = None

        # regularization
        self.attn_dropout_p = config.attn_dropout_p
        self.resid_dropout = nn.Dropout(config.resid_dropout_p)

    def forward(
        self, x: torch.Tensor, freqs_cis: torch.Tensor = None, 
        input_pos: Optional[torch.Tensor] = None, 
        mask: Optional[torch.Tensor] = None
    ):
        bsz, seqlen, _ = x.shape
        kv_size = self.n_kv_head * self.head_dim
        xq, xk, xv = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)

        xq = xq.view(bsz, seqlen, self.n_head, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_head, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_head, self.head_dim)
        
        xq = apply_rotary_emb(xq, freqs_cis)
        xk = apply_rotary_emb(xk, freqs_cis)

        xq, xk, xv = map(lambda x: x.transpose(1, 2), (xq, xk, xv))

        if self.kv_cache is not None:
            keys, values = self.kv_cache.update(input_pos, xk, xv)
        else:
            keys, values = xk, xv
        keys = keys.repeat_interleave(self.n_head // self.n_kv_head, dim=1)
        values = values.repeat_interleave(self.n_head // self.n_kv_head, dim=1)

        output = F.scaled_dot_product_attention(
            xq, keys, values, 
            attn_mask=mask, 
            is_causal=True if mask is None else False, # is_causal=False is for KV cache
            dropout_p=self.attn_dropout_p if self.training else 0)            
        
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)

        output = self.resid_dropout(self.wo(output))
        return output


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelArgs, drop_path: float):
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(
        self, x: torch.Tensor, freqs_cis: torch.Tensor, start_pos: int, mask: Optional[torch.Tensor] = None):
        h = x + self.drop_path(self.attention(self.attention_norm(x), freqs_cis, start_pos, mask))
        out = h + self.drop_path(self.feed_forward(self.ffn_norm(h)))
        return out


class Transformer(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.n_layer = config.n_layer
        self.block_size = config.block_size
        self.num_classes = config.num_classes
        self.model_type = config.model_type
        self.cls_token_num = config.cls_token_num
        if self.model_type == 'c2i':
            self.cls_embedding = LabelEmbedder(config.num_classes, config.dim, config.class_dropout_prob)
        elif self.model_type == 't2i':
            self.cls_embedding = CaptionEmbedder(config.caption_dim, config.dim, config.class_dropout_prob)
        else:
            raise Exception("please check model type")
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.tok_dropout = nn.Dropout(config.token_dropout_p)

        # transformer blocks
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.n_layer)]
        self.layers = torch.nn.ModuleList()
        for layer_id in range(config.n_layer):
            self.layers.append(TransformerBlock(config, dpr[layer_id]))

        # output layer
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        # 2d rotary pos embedding
        grid_size = int(self.block_size ** 0.5)
        assert grid_size * grid_size == self.block_size
        self.freqs_cis = precompute_freqs_cis_2d(grid_size, self.config.dim // self.config.n_head, self.config.rope_base, self.cls_token_num)
        
        # KVCache
        self.max_batch_size = -1
        self.max_seq_length = -1

        self.initialize_weights()

    def create_2d_window_mask(self, max_seq_length, grid_size, window_radius):
        """Create a 2D spatial window mask for local attention."""
        mask = torch.zeros(max_seq_length, max_seq_length, dtype=torch.bool)
        
        for i in range(max_seq_length):
            # Always include class token (position 0)
            mask[i, 0] = True
            
            if i == 0:  # Class token position
                mask[i, i] = True
                continue
                
            # Current position's 2D coordinates
            curr_row = (i - 1) // grid_size
            curr_col = (i - 1) % grid_size
            
            # Check all previous positions
            for j in range(1, i + 1):  # Include current position
                prev_row = (j - 1) // grid_size
                prev_col = (j - 1) % grid_size
                
                # Check if within 2D window (square window)
                if (abs(curr_row - prev_row) <= window_radius and 
                    abs(curr_col - prev_col) <= window_radius):
                    mask[i, j] = True
                    
        return mask

    def create_1d_strided_mask(self, max_seq_length, stride):
        """Create a 1D strided mask where only positions at multiples of stride are visible."""
        mask = torch.zeros(max_seq_length, max_seq_length, dtype=torch.bool)
        
        for i in range(max_seq_length):
            # Always include class token (position 0)
            mask[i, 0] = True
            
            if i == 0:  # Class token position
                mask[i, i] = True
                continue
                
            # For image tokens, only see positions that are multiples of stride
            for j in range(1, i + 1):
                if (j - 1) % stride == 0:  # See positions 1, 1+stride, 1+2*stride, ...
                    mask[i, j] = True
                    
        return mask
    
    def create_2d_strided_mask(self, max_seq_length, grid_size, stride):
        """Create a 2D strided mask where only positions at grid multiples of stride are visible."""
        mask = torch.zeros(max_seq_length, max_seq_length, dtype=torch.bool)
        
        for i in range(max_seq_length):
            # Always include class token (position 0)
            mask[i, 0] = True
            
            if i == 0:  # Class token position
                mask[i, i] = True
                continue
                
            # Current position's 2D coordinates
            curr_row = (i - 1) // grid_size
            curr_col = (i - 1) % grid_size
            
            # Check all previous positions
            for j in range(1, i + 1):
                prev_row = (j - 1) // grid_size
                prev_col = (j - 1) % grid_size
                
                # Check if both row and col are at multiples of stride
                if prev_row % stride == 0 and prev_col % stride == 0:
                    mask[i, j] = True
                    
        return mask
    
    def create_checkerboard_mask(self, max_seq_length, grid_size):
        """Create a checkerboard mask where positions follow a checkerboard pattern."""
        mask = torch.zeros(max_seq_length, max_seq_length, dtype=torch.bool)
        
        for i in range(max_seq_length):
            # Always include class token (position 0)
            mask[i, 0] = True
            
            if i == 0:  # Class token position
                mask[i, i] = True
                continue
                
            # Current position's 2D coordinates
            curr_row = (i - 1) // grid_size
            curr_col = (i - 1) % grid_size
            
            # Check all previous positions
            for j in range(1, i + 1):
                prev_row = (j - 1) // grid_size
                prev_col = (j - 1) % grid_size
                
                # Checkerboard pattern: (row + col) % 2 == 0
                if (prev_row + prev_col) % 2 == 0:
                    mask[i, j] = True
                    
        return mask

    def initialize_weights(self):        
        # Initialize nn.Linear and nn.Embedding
        self.apply(self._init_weights)

        # Zero-out output layers:
        nn.init.constant_(self.output.weight, 0)

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)

    def setup_caches(self, max_batch_size, max_seq_length, dtype, window_parameter=64, window_type="1d"):
        # if self.max_seq_length >= max_seq_length and self.max_batch_size >= max_batch_size:
        #     return
        head_dim = self.config.dim // self.config.n_head
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        for b in self.layers:
            b.attention.kv_cache = KVCache(max_batch_size, max_seq_length, self.config.n_head, head_dim, dtype)

        causal_mask = torch.tril(torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool))
        self.causal_mask = causal_mask.unsqueeze(0).repeat(self.max_batch_size, 1, 1)
        
        # Create recent window mask for local guidance
        grid_size = int(self.config.block_size ** 0.5)
        assert grid_size * grid_size == self.block_size
        
        if window_type == "2d":
            # For 2D window, window_parameter is the radius
            self.recent_window_mask = self.create_2d_window_mask(
                self.max_seq_length, grid_size, window_radius=window_parameter
            )
        elif window_type == "1d_strided":
            # For 1D strided, window_parameter is the stride
            self.recent_window_mask = self.create_1d_strided_mask(
                self.max_seq_length, stride=window_parameter
            )
        elif window_type == "2d_strided":
            # For 2D strided, window_parameter is the stride
            self.recent_window_mask = self.create_2d_strided_mask(
                self.max_seq_length, grid_size, stride=window_parameter
            )
        elif window_type == "checkerboard":
            # For checkerboard, ignore recent_window_size
            self.recent_window_mask = self.create_checkerboard_mask(
                self.max_seq_length, grid_size
            )
        else:  # 1d
            self.recent_window_mask = torch.zeros(self.max_seq_length, self.max_seq_length, dtype=torch.bool)
            for i in range(self.max_seq_length):
                # Always include class token (position 0)
                self.recent_window_mask[i, 0] = True
                
                # Apply window only to image tokens
                if i > 0:  # For image token positions
                    start = max(1, i - window_parameter + 1)  # Start from 1 (first image token)
                    self.recent_window_mask[i, start:i+1] = True
                else:  # For class token position
                    self.recent_window_mask[i, i] = True  # Only see itself
                    
        self.recent_window_mask = self.recent_window_mask.unsqueeze(0).repeat(self.max_batch_size, 1, 1)
        self.window_parameter = window_parameter
        self.window_type = window_type
        
        self.freqs_cis = precompute_freqs_cis_2d(grid_size, self.config.dim // self.config.n_head, self.config.rope_base, self.cls_token_num)

    def forward(
        self, 
        idx: torch.Tensor, 
        cond_idx: torch.Tensor,  # cond_idx_or_embed
        input_pos:  Optional[torch.Tensor] = None, 
        targets: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        valid: Optional[torch.Tensor] = None,
    ):
        if idx is not None and cond_idx is not None: # training or naive inference
            cond_embeddings = self.cls_embedding(cond_idx, train=self.training)[:,:self.cls_token_num]
            token_embeddings = self.tok_embeddings(idx)
            token_embeddings = torch.cat((cond_embeddings, token_embeddings), dim=1)
            h = self.tok_dropout(token_embeddings)
            self.freqs_cis = self.freqs_cis.to(h.device)
        else:
            if cond_idx is not None: # prefill in inference
                token_embeddings = self.cls_embedding(cond_idx, train=self.training)[:,:self.cls_token_num]
            else: # decode_n_tokens(kv cache) in inference
                token_embeddings = self.tok_embeddings(idx)
            
            bs = token_embeddings.shape[0]
            
            # Check if using local guidance (batch size is multiple of 3)
            if hasattr(self, 'recent_window_mask') and bs % 3 == 0 and bs > 1:
                # Create different masks for different parts of the batch
                batch_per_type = bs // 3
                mask_causal = self.causal_mask[:batch_per_type * 2, None, input_pos]
                mask_local = self.recent_window_mask[:batch_per_type, None, input_pos]
                mask = torch.cat([mask_causal, mask_local], dim=0)
            else:
                mask = self.causal_mask[:bs, None, input_pos]
                
            h = self.tok_dropout(token_embeddings)
            self.freqs_cis = self.freqs_cis
        
        if self.training:
            freqs_cis = self.freqs_cis[:token_embeddings.shape[1]]
        else:
            freqs_cis = self.freqs_cis[input_pos]
        # transformer blocks
        for layer in self.layers:
            h = layer(h, freqs_cis, input_pos, mask)
        
        # output layers
        h = self.norm(h)
        logits = self.output(h).float()
        
        if self.training:
            logits = logits[:, self.cls_token_num - 1:].contiguous()

        # if we are given some desired targets also calculate the loss
        loss = None
        if valid is not None:
            loss_all = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction='none')
            valid_all = valid[:,None].repeat(1, targets.shape[1]).view(-1)
            loss = (loss_all * valid_all).sum() / max(valid_all.sum(), 1)
        elif targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss


    def get_fsdp_wrap_module_list(self) -> List[nn.Module]:
        return list(self.layers)



#################################################################################
#                      Rotary Positional Embedding Functions                    #
#################################################################################
# https://github.com/pytorch-labs/gpt-fast/blob/main/model.py 
def precompute_freqs_cis(seq_len: int, n_elem: int, base: int = 10000, cls_token_num=120):
    freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem))
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs) # (seq_len, head_dim // 2)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1) # (cls_token_num+seq_len, head_dim // 2, 2)
    cond_cache = torch.cat([torch.zeros(cls_token_num, n_elem // 2, 2), cache]) # (cls_token_num+seq_len, head_dim // 2, 2)
    return cond_cache 


def precompute_freqs_cis_2d(grid_size: int, n_elem: int, base: int = 10000, cls_token_num=120):
    # split the dimension into half, one for x and one for y
    half_dim = n_elem // 2
    freqs = 1.0 / (base ** (torch.arange(0, half_dim, 2)[: (half_dim // 2)].float() / half_dim))
    t = torch.arange(grid_size, device=freqs.device)
    freqs = torch.outer(t, freqs) # (grid_size, head_dim // 2)
    freqs_grid = torch.concat([
        freqs[:, None, :].expand(-1, grid_size, -1),
        freqs[None, :, :].expand(grid_size, -1, -1),
    ], dim=-1)  # (grid_size, grid_size, head_dim // 2)
    cache_grid = torch.stack([torch.cos(freqs_grid), torch.sin(freqs_grid)], dim=-1) # (grid_size, grid_size, head_dim // 2, 2)
    cache = cache_grid.flatten(0, 1)
    cond_cache = torch.cat([torch.zeros(cls_token_num, n_elem // 2, 2), cache]) # (cls_token_num+grid_size**2, head_dim // 2, 2)
    return cond_cache 


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor):
    # x: (bs, seq_len, n_head, head_dim)
    # freqs_cis (seq_len, head_dim // 2, 2)
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2) # (bs, seq_len, n_head, head_dim//2, 2)
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2) # (1, seq_len, 1, head_dim//2, 2)
    x_out2 = torch.stack([
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
    ], dim=-1)
    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)



#################################################################################
#                                GPT Configs                                    #
#################################################################################
### text-conditional
def GPT_7B(**kwargs):
    return Transformer(ModelArgs(n_layer=32, n_head=32, dim=4096, **kwargs)) # 6.6B

def GPT_3B(**kwargs):
    return Transformer(ModelArgs(n_layer=24, n_head=32, dim=3200, **kwargs)) # 3.1B

def GPT_1B(**kwargs):
    return Transformer(ModelArgs(n_layer=22, n_head=32, dim=2048, **kwargs)) # 1.2B

### class-conditional
def GPT_XXXL(**kwargs):
    return Transformer(ModelArgs(n_layer=48, n_head=40, dim=2560, **kwargs)) # 3.9B

def GPT_XXL(**kwargs):
    return Transformer(ModelArgs(n_layer=48, n_head=24, dim=1536, **kwargs)) # 1.4B

def GPT_XL(**kwargs):
    return Transformer(ModelArgs(n_layer=36, n_head=20, dim=1280, **kwargs)) # 775M

def GPT_L(**kwargs):
    return Transformer(ModelArgs(n_layer=24, n_head=16, dim=1024, **kwargs)) # 343M

def GPT_B(**kwargs):
    return Transformer(ModelArgs(n_layer=12, n_head=12, dim=768, **kwargs)) # 111M
        

GPT_models = {
    'GPT-B': GPT_B, 'GPT-L': GPT_L, 'GPT-XL': GPT_XL, 'GPT-XXL': GPT_XXL, 'GPT-XXXL': GPT_XXXL,
    'GPT-1B': GPT_1B, 'GPT-3B': GPT_3B, 'GPT-7B': GPT_7B, 
}