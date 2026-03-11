import torch
import torch.nn as nn


class ReportDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=4, num_layers=3, dim_feedforward=512, dropout=0.1, max_len=300):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.max_len = max_len

    def forward(self, input_ids, memory, memory_key_padding_mask=None):
        # input_ids: BxT
        b, t = input_ids.shape
        pos = torch.arange(t, device=input_ids.device).unsqueeze(0).expand(b, t)
        x = self.token_embed(input_ids) + self.pos_embed(pos)

        # causal mask
        tgt_mask = torch.triu(torch.ones(t, t, device=input_ids.device), diagonal=1).bool()

        out = self.decoder(
            x,
            memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        logits = self.lm_head(out)
        return logits

