import torch
import torch.nn as nn

from src.models.encoder import ImageEncoder
from src.models.decoder import ReportDecoder
from src.utils.labels import CHEXPERT_LABELS


class XAIReportModel(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=4, num_layers=3, dim_feedforward=512, dropout=0.1, max_len=300, encoder_name="resnet50"):
        super().__init__()
        self.encoder = ImageEncoder(name=encoder_name, d_model=d_model)
        self.concept_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, len(CHEXPERT_LABELS)),
        )
        self.concept_embed = nn.Embedding(len(CHEXPERT_LABELS), d_model)
        self.decoder = ReportDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_len=max_len,
        )

    def forward(self, images, input_ids):
        # Encoder features
        memory = self.encoder(images)  # BxNxD

        # Concept logits from pooled image features
        pooled = memory.mean(dim=1)
        concept_logits = self.concept_head(pooled)

        # Append concept embeddings as extra memory tokens
        concept_ids = torch.arange(concept_logits.shape[-1], device=images.device)
        concept_tokens = self.concept_embed(concept_ids).unsqueeze(0).expand(images.size(0), -1, -1)
        memory = torch.cat([memory, concept_tokens], dim=1)

        logits = self.decoder(input_ids, memory)
        return logits, concept_logits

    @torch.no_grad()
    def generate(self, images, vocab, max_len=300, device=None):
        if device is None:
            device = images.device
        memory = self.encoder(images)
        pooled = memory.mean(dim=1)
        concept_logits = self.concept_head(pooled)
        concept_ids = torch.arange(concept_logits.shape[-1], device=device)
        concept_tokens = self.concept_embed(concept_ids).unsqueeze(0).expand(images.size(0), -1, -1)
        memory = torch.cat([memory, concept_tokens], dim=1)

        bos = vocab.stoi["<bos>"]
        eos = vocab.stoi["<eos>"]
        input_ids = torch.full((images.size(0), 1), bos, dtype=torch.long, device=device)
        finished = torch.zeros(images.size(0), dtype=torch.bool, device=device)

        for _ in range(max_len - 1):
            logits = self.decoder(input_ids, memory)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            finished |= (next_token.squeeze(1) == eos)
            if finished.all():
                break

        return input_ids, concept_logits

