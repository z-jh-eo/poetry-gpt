import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            config.embd_d,
            config.n_head,
            batch_first=True
        )
        self.ff   = nn.Sequential(
            nn.Linear(config.embd_d, 4*config.embd_d, bias=config.bias),
            nn.GELU(),
            nn.Linear(4*config.embd_d, config.embd_d, bias=config.bias)
        )
        self.ln1  = nn.LayerNorm(config.embd_d)
        self.ln2  = nn.LayerNorm(config.embd_d)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x):
        normed_x = self.ln1(x)
        x = x + self.drop(
            self.attn(
                query=normed_x,
                key  =normed_x,
                value=normed_x,
                is_causal=True,
                need_weights=False,
            )[0]
        )

        x = x + self.drop(self.ff(self.ln2(x)))

        return(x)

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int    = 12
    n_head: int     = 12
    embd_d: int     = 768
    dropout: float  = 0.1
    bias: bool      = True


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
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature  # (b, vocab_size)
 
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
 
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
 
        return idx

