import torch
import torch.nn as nn
import math

class ScaledDotProductAttention(nn.Module):
    def forward(self, q, k, v, mask=None):
        d_k = q.size(-1)
        # compute QK^T / sqrt(d_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        
        # apply causal mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attn = torch.softmax(scores, dim=-1)
        return torch.matmul(attn, v)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention()
        
    def forward(self, q, k, v, mask=None):
        B = q.size(0)
        
        # project and reshape to (B, heads, S, d_k)
        q = self.W_q(q).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(k).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_v(v).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        if mask is not None:
            # mask is (S, S), we need it to broadcast to (B, num_heads, S, S)
            mask = mask.unsqueeze(0).unsqueeze(1)
            
        out = self.attention(q, k, v, mask)
        
        # concatenate heads
        out = out.transpose(1, 2).contiguous().view(B, -1, self.d_model)
        return self.W_o(out)

class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, memory, tgt_mask):
        # self attention (masked)
        attn_out = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # cross attention with visual memory (unmasked)
        attn_out = self.cross_attn(x, memory, memory, mask=None)
        x = self.norm2(x + self.dropout(attn_out))
        
        # feed forward
        ff_out = self.ff(x)
        x = self.norm3(x + self.dropout(ff_out))
        
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class LatexDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_heads=8, num_layers=4, d_ff=2048, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        self.layers = nn.ModuleList([
            DecoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.fc_out = nn.Linear(d_model, vocab_size)
        
    def generate_causal_mask(self, sz):
        # lower triangular matrix of ones (prevents looking ahead)
        return torch.tril(torch.ones(sz, sz))
        
    def forward(self, tgt, memory):
        # generate causal mask
        seq_len = tgt.size(1)
        tgt_mask = self.generate_causal_mask(seq_len).to(tgt.device)
        
        # embed and encode position
        x = self.embedding(tgt)
        x = self.pos_encoder(x)
        
        # decode layers
        for layer in self.layers:
            x = layer(x, memory, tgt_mask)
            
        # final vocabulary projection
        return self.fc_out(x)

if __name__ == "__main__":
    # dummy encoder memory (B=4, Seq_len=44, Features=512)
    memory = torch.randn(4, 44, 512)
    
    # dummy target tokens (B=4, Seq_len=12)
    tgt = torch.randint(0, 4000, (4, 12))
    
    # initialize and forward
    decoder = LatexDecoder(vocab_size=4000, d_model=512, num_heads=8, num_layers=4)
    logits = decoder(tgt, memory)
    
    print("=== Text Decoder Verification ===")
    print(f"Memory shape:      {list(memory.shape)}")
    print(f"Target IDs shape:  {list(tgt.shape)}")
    print(f"Logits shape:      {list(logits.shape)}")
