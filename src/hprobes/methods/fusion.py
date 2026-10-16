"""Fusion: hidden-state layer tokens + pretrained text-encoder tokens of the (question, answer) pair.

Image side: (pooled) layers become tokens, standardized per coordinate with source-train
statistics, projected by a shared Linear(hidden -> d) plus a learned layer-position embedding.
Text side: a pretrained encoder (default BGE-base) reads question + generated answer as one
sentence pair; frozen (states computed once) or fine-tuned end to end (--finetune-text-encoder).
A pre-LN transformer over [CLS] + layer tokens + text tokens (with modality embeddings) and a
linear head on [CLS] predict BLEURT correctness. --fusion-mode image / text keeps one modality.
"""

from __future__ import annotations

import os

import numpy as np

from hprobes.features import pool_layers, pooled_layer_count
from hprobes.losses import make_loss
from hprobes.training import fit, predict, seed_everything

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def encode_frozen_text(domains, args, device):
    """Run the frozen encoder once; returns per-domain lists of float16 (tokens, width) arrays."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.text_encoder, revision=args.text_encoder_revision)
    encoder = AutoModel.from_pretrained(args.text_encoder, revision=args.text_encoder_revision,
                                        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32)
    encoder.to(device).eval()
    out = {}
    with torch.inference_mode():
        for domain in domains:
            questions = [row["input_text"] for row in domain.rows]
            answers = [row["output_text"] for row in domain.rows]
            states = []
            for start in range(0, len(questions), 256):
                encoded = tokenizer(questions[start:start + 256], answers[start:start + 256], truncation=True,
                                    max_length=args.max_text_tokens, padding=True, return_tensors="pt").to(device)
                hidden = encoder(**encoded).last_hidden_state.float().cpu().numpy().astype(np.float16)
                for h, mask in zip(hidden, encoded["attention_mask"].cpu().numpy()):
                    states.append(h[:int(mask.sum())])
            out[domain.name] = states
    del encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def run(args, data, device):
    import torch
    from torch import nn
    from torch.utils.data import DataLoader

    if args.pca_dim:
        raise SystemExit("--pca-dim is supported by --method mlp and sala")
    use_image, use_text = args.fusion_mode in ("fused", "image"), args.fusion_mode in ("fused", "text")
    finetune = use_text and args.finetune_text_encoder
    tokens = pooled_layer_count(data.n_layers, args.pooling, args.pool_kernel, args.pool_stride, args.pool_layers)
    domains = [*data.sources, *data.targets]
    seed_everything(args.seed)
    text, width = {}, 0
    if finetune:
        from transformers import AutoConfig, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.text_encoder, revision=args.text_encoder_revision)
        for d in domains:
            enc = tokenizer([r["input_text"] for r in d.rows], [r["output_text"] for r in d.rows],
                            truncation=True, max_length=args.max_text_tokens)
            text[d.name] = [np.asarray([i, t], dtype=np.int64)
                            for i, t in zip(enc["input_ids"], enc["token_type_ids"])]
        width = AutoConfig.from_pretrained(args.text_encoder, revision=args.text_encoder_revision).hidden_size
    elif use_text:
        text = encode_frozen_text(domains, args, device)
        width = next(iter(text.values()))[0].shape[1]

    class FusionDataset:
        def __init__(self, split_domains, split):
            self.domains = split_domains
            self.entries = [(k, int(i), int(d.labels[i])) for k, d in enumerate(split_domains) for i in d.split[split]]

        def __len__(self):
            return len(self.entries)

        def __getitem__(self, position):
            k, i, label = self.entries[position]
            d = self.domains[k]
            states = (np.asarray(d.features[i], dtype=np.float16).copy() if use_image
                      else np.zeros((0,), dtype=np.float16))
            t = text[d.name][i] if use_text else np.zeros((0, 0), dtype=np.float16)
            return states, t, np.float32(label)

    def collate(batch):
        states = torch.from_numpy(np.stack([b[0] for b in batch]))
        if finetune:
            lengths = torch.tensor([b[1].shape[1] for b in batch])
            t = torch.zeros(len(batch), 2, int(lengths.max()), dtype=torch.long)
            for k, b in enumerate(batch):
                t[k, :, :b[1].shape[1]] = torch.from_numpy(b[1])
        else:
            lengths = torch.tensor([len(b[1]) for b in batch])
            t = torch.zeros(len(batch), int(lengths.max()) if use_text else 0, width, dtype=torch.float16)
            for k, b in enumerate(batch):
                if use_text:
                    t[k, :len(b[1])] = torch.from_numpy(b[1])
        return states, t, lengths, torch.tensor([b[2] for b in batch])

    splits = {"train": FusionDataset(data.sources, "train"), "val": FusionDataset(data.sources, "val"),
              "test": FusionDataset(data.sources, "test")}
    splits.update({f"target:{d.name}": FusionDataset([d], "all") for d in data.targets})
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {name: DataLoader(ds, batch_size=args.batch_size if name == "train" else 256,
                                shuffle=name == "train", generator=generator if name == "train" else None,
                                collate_fn=collate, num_workers=args.num_workers, pin_memory=device.type == "cuda")
               for name, ds in splits.items()}

    feature_mean = feature_std = None
    if use_image:
        total = torch.zeros(tokens, data.hidden_size, dtype=torch.float64)
        total_sq = torch.zeros_like(total)
        with torch.inference_mode():
            for states, _, _, _ in DataLoader(splits["train"], batch_size=256, collate_fn=collate):
                image = pool_layers(states.to(device).float(), args.pooling, args.pool_kernel,
                                    args.pool_stride, args.pool_layers).cpu().double()
                total += image.sum(0)
                total_sq += image.square().sum(0)
        n = len(splits["train"])
        feature_mean = (total / n).float()
        feature_std = (total_sq / n - (total / n).square()).clamp(min=0).sqrt().float().clamp(min=1e-4)

    class FusionModel(nn.Module):
        def __init__(self):
            super().__init__()
            d = args.model_dim
            self.cls = nn.Parameter(torch.zeros(1, 1, d))
            self.type_embed = nn.Embedding(2, d)  # 0 = layer token, 1 = text token
            if use_image:
                self.register_buffer("feature_mean", feature_mean.clone())
                self.register_buffer("feature_std", feature_std.clone())
                self.layer_proj = nn.Linear(data.hidden_size, d)
                self.layer_pos = nn.Parameter(torch.zeros(1, tokens, d))
                nn.init.trunc_normal_(self.layer_pos, std=0.02)
            if finetune:
                from transformers import AutoModel

                self.text_encoder = AutoModel.from_pretrained(args.text_encoder, revision=args.text_encoder_revision)
            if use_text:
                self.text_proj = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, d))
            self.input_dropout = nn.Dropout(args.dropout)
            layer = nn.TransformerEncoderLayer(d, args.heads, 4 * d, args.dropout, activation="gelu",
                                               batch_first=True, norm_first=True)
            self.fusion = nn.TransformerEncoder(layer, args.fusion_layers, norm=nn.LayerNorm(d),
                                                enable_nested_tensor=False)
            self.head = nn.Linear(d, 1)
            nn.init.trunc_normal_(self.cls, std=0.02)

        def forward(self, states, t, lengths, drop_text=None, drop_image=False):
            b = lengths.shape[0]
            parts = [self.cls.expand(b, -1, -1)]
            masks = [torch.zeros(b, 1, dtype=torch.bool, device=lengths.device)]
            if use_image:
                image = pool_layers(states.float(), args.pooling, args.pool_kernel, args.pool_stride, args.pool_layers)
                image = ((image - self.feature_mean) / self.feature_std).clamp(-args.feature_clip, args.feature_clip)
                parts.append(self.layer_proj(image) + self.layer_pos + self.type_embed.weight[0])
                masks.append(torch.full((b, tokens), drop_image, dtype=torch.bool, device=lengths.device))
            if use_text:
                positions = torch.arange(t.shape[-1] if finetune else t.shape[1], device=lengths.device)
                mask = positions.unsqueeze(0) >= lengths.unsqueeze(1)
                if finetune:
                    t = self.text_encoder(input_ids=t[:, 0], token_type_ids=t[:, 1],
                                          attention_mask=(~mask).long()).last_hidden_state
                if drop_text is not None:
                    mask = mask | drop_text.unsqueeze(1)
                parts.append(self.text_proj(t.float()) + self.type_embed.weight[1])
                masks.append(mask)
            sequence = self.input_dropout(torch.cat([p.float() for p in parts], dim=1))
            hidden = self.fusion(sequence, src_key_padding_mask=torch.cat(masks, dim=1))
            return self.head(hidden[:, 0]).squeeze(-1).float()

    model = FusionModel().to(device)
    text_dropout = args.text_dropout if args.fusion_mode == "fused" else 0.0
    mask_state = {"text": False, "image": False}

    def step(batch):
        states, t, lengths, labels = (x.to(device, non_blocking=True) for x in batch)
        drop = None
        if use_text:
            drop = (torch.rand(len(labels), device=device) < text_dropout) if model.training else \
                torch.full_like(lengths, mask_state["text"], dtype=torch.bool)
        return model(states, t, lengths, drop_text=drop, drop_image=mask_state["image"]), labels

    groups = {}
    for name, p in model.named_parameters():
        no_decay = p.ndim < 2 or name in ("cls", "layer_pos")
        lr = args.text_encoder_lr if name.startswith("text_encoder.") else args.lr
        groups.setdefault((lr, no_decay), []).append(p)
    param_groups = [{"params": ps, "lr": lr, "weight_decay": 0.0 if nd else args.weight_decay}
                    for (lr, nd), ps in groups.items()]
    print(f"fusion: mode={args.fusion_mode}; {tokens} layer tokens; text encoder={args.text_encoder if use_text else None}"
          f" ({'fine-tuned' if finetune else 'frozen'}); parameters={sum(p.numel() for p in model.parameters()):,}",
          flush=True)
    best_auc, best_epoch, history = fit(model, loaders["train"], loaders["val"], data.labels("val"), step,
                                        make_loss(args, data.labels("train")), args, device, param_groups)
    logits = {name: predict(model, loader, step, device, args) for name, loader in loaders.items() if name != "train"}
    reliance = {}
    if args.fusion_mode == "fused":
        from sklearn.metrics import roc_auc_score

        for key in ("text", "image"):
            mask_state[key] = True
            reliance[f"{key}_masked_source_val_auroc"] = float(roc_auc_score(
                data.labels("val"), predict(model, loaders["val"], step, device, args)))
            for d in data.targets:
                reliance[f"{key}_masked_{d.name}_auroc"] = float(roc_auc_score(
                    d.labels, predict(model, loaders[f"target:{d.name}"], step, device, args)))
            mask_state[key] = False
    summary = {"fusion_mode": args.fusion_mode, "layer_tokens": tokens, "text_encoder": args.text_encoder,
               "text_encoder_finetuned": finetune, "text_dropout": text_dropout,
               "parameters": sum(p.numel() for p in model.parameters()), "reliance_checks": reliance,
               "best_source_val_auroc": best_auc, "best_epoch": best_epoch, "history": history}
    return summary, logits["val"], logits["test"], {d.name: logits[f"target:{d.name}"] for d in data.targets}
