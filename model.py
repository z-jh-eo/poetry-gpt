import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.embd_d % config.n_head == 0,\
            "embd_d must be divisible by n_head"

        self.n_head  = config.n_head
        self.embd_d  = config.embd_d
        self.head_d  = config.embd_d // config.n_head
        self.dropout = config.dropout
        self.causal_mask = config.causal_mask #"flash" | "manual"

        self.c_attn  = nn.Linear(config.embd_d, 3 * config.embd_d, bias=config.bias)
        self.c_proj  = nn.Linear(config.embd_d,     config.embd_d, bias=config.bias)
        
        self.attn_drop  = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.shape  #B:batch size, T:nb of tokens, C:embd dim

        # project to QKV
        q, k, v = self.c_attn(x).split(self.embd_d, dim=2) 
                                #split the QKV that we initilised together
        def split_heads(t):
            return t.view(B, T, self.n_head, self.head_d).transpose(1,2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        # SDPA
        dropout_p = self.dropout if self.training else 0.0

        if self.causal_mask == "flash":
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=dropout_p,
                is_causal=True,
            )
        elif self.causal_mask == "manual":
            mask = torch.ones(T, T, dtype=torch.bool, deive=x.device)
            mask = mask.view(1, 1, T, T)

            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask,
                dropout_p=dropout_p,
                is_causal=False,
            )
        else:
            raise ValueError(
                "Unknown causal mask. Choose 'flash' or 'manual'."
            )
        
        # reassemble heads
        y = y.transpose(1,2).contiguous().view(B, T, C)
        y = self.resid_drop(self.c_proj(y))
        return y


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.ff   = nn.Sequential(
            nn.Linear(config.embd_d, 4*config.embd_d, bias=config.bias),
            nn.GELU(),
            nn.Linear(4*config.embd_d, config.embd_d, bias=config.bias)
        )
        self.ln1  = nn.LayerNorm(config.embd_d)
        self.ln2  = nn.LayerNorm(config.embd_d)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return(x)

@dataclass
class GPTConfig:
    block_size: int  = 256
    vocab_size: int  = 100
    n_layer: int     = 6
    n_head: int      = 8
    embd_d: int      = 256
    dropout: float   = 0.1
    bias: bool       = True
    causal_mask: str = "flash"   #("flash" | "manual")


class GPT(nn.Module):
    def __init__(self,config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config =  config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.embd_d), # token embedding table
            wpe = nn.Embedding(config.block_size, config.embd_d), # position embedding table
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([TransformerBlock(config) for _  in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.embd_d)
        ))

        self.lm_head = nn.Linear(config.embd_d, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self,  non_embedding=True):
        n_params = sum(p.numel()  for p in self.parameters())
        if  non_embedding:
            n_params -=  self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None): 
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size

        pos = torch.arange(0, t, dtype=torch.long, device=device)  # (t,)
 
        tok_emb = self.transformer.wte(idx)   # (b, t, embd_d)
        pos_emb = self.transformer.wpe(pos)   # (t, embd_d)
        x = self.transformer.drop(tok_emb + pos_emb)
 
        for block in self.transformer.h:
            x = block(x)
 
        x = self.transformer.ln_f(x)
 
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
        else:
            # At inference time only compute logits for the last position
            logits = self.lm_head(x[:, [-1], :])
            loss = None
 
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None): 
        """
        Autoregressive generation.
        idx: (b, t) LongTensor of conditioning token indices.
        """
        for _ in range(max_new_tokens):
            # Crop context to block_size if needed
            idx_cond = idx if idx.size(1) <= self.config.block_size\
                           else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature  # (b, vocab_size)
 
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
 
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
 
        return idx

