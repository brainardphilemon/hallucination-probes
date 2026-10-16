"""From-scratch vision-language model: the (pooled) layers x hidden matrix is an image.

The standardized hidden-state matrix is cut into non-overlapping (patch-height x patch-width)
patches, encoded by a small ViT (patch embedding + patch positions + transformer), projected by
an MLP, and fused by a single-stream transformer with the question and generated-answer tokens
(Qwen BPE pieces remapped to a source-train vocabulary, embeddings learned from scratch).
A linear head on [CLS] predicts BLEURT correctness. Every weight is randomly initialized.
--vlm-text none trains on the image only. Use --patch-width = hidden size for one token per layer.
"""

from __future__ import annotations

from collections import Counter
import os

import numpy as np

from hprobes.extract import DEFAULT_MODEL, DEFAULT_MODEL_REVISION
from hprobes.features import pool_layers, pooled_layer_count
from hprobes.losses import make_loss
from hprobes.training import fit, predict, seed_everything

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
PAD, UNK, CLS, SEP = 0, 1, 2, 3
PATCH_TYPE, QUESTION_TYPE, ANSWER_TYPE = 0, 1, 2


def run(args, data, device):
    import torch
    from torch import nn
    from torch.nn import functional as F
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    if args.pca_dim:
        raise SystemExit("--pca-dim is supported by --method mlp and sala")
    rows_in_image = pooled_layer_count(data.n_layers, args.pooling, args.pool_kernel, args.pool_stride, args.pool_layers)
    if data.hidden_size % args.patch_width:
        raise SystemExit(f"--patch-width must divide the hidden size {data.hidden_size}")
    grid_rows, grid_cols = -(-rows_in_image // args.patch_height), data.hidden_size // args.patch_width
    num_patches = grid_rows * grid_cols
    domains = [*data.sources, *data.targets]
    seed_everything(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL, revision=DEFAULT_MODEL_REVISION)
    bpe = {d.name: (tokenizer([r["input_text"] for r in d.rows], add_special_tokens=False)["input_ids"],
                    tokenizer([r["output_text"] for r in d.rows], add_special_tokens=False)["input_ids"])
           for d in domains}
    counts = Counter()
    for d in data.sources:
        for i in d.split["train"]:
            counts.update(bpe[d.name][0][i])
            counts.update(bpe[d.name][1][i])
    vocab = {token: k + 4 for k, token in enumerate(sorted(counts))}
    text = {}
    for d in domains:
        text[d.name] = []
        for q, a in zip(*bpe[d.name]):
            q = [vocab.get(i, UNK) for i in q[:args.max_question_tokens]]
            a = [vocab.get(i, UNK) for i in a[:args.max_answer_tokens]]
            if args.vlm_text == "none":
                q, a = [], []
            text[d.name].append((q + [SEP] + a, [QUESTION_TYPE] * (len(q) + 1) + [ANSWER_TYPE] * len(a)))
    max_text = args.max_question_tokens + 1 + args.max_answer_tokens

    class VLMDataset:
        def __init__(self, split_domains, split):
            self.domains = split_domains
            self.entries = [(k, int(i), int(d.labels[i])) for k, d in enumerate(split_domains) for i in d.split[split]]

        def __len__(self):
            return len(self.entries)

        def __getitem__(self, position):
            k, i, label = self.entries[position]
            d = self.domains[k]
            ids, types = text[d.name][i]
            return np.asarray(d.features[i], dtype=np.float16).copy(), ids, types, np.float32(label)

    def collate(batch):
        width = max(len(b[1]) for b in batch)
        ids = torch.full((len(batch), width), PAD, dtype=torch.long)
        types = torch.full((len(batch), width), ANSWER_TYPE, dtype=torch.long)
        for k, b in enumerate(batch):
            ids[k, :len(b[1])] = torch.tensor(b[1])
            types[k, :len(b[2])] = torch.tensor(b[2])
        return torch.from_numpy(np.stack([b[0] for b in batch])), ids, types, torch.tensor([b[3] for b in batch])

    splits = {"train": VLMDataset(data.sources, "train"), "val": VLMDataset(data.sources, "val"),
              "test": VLMDataset(data.sources, "test")}
    splits.update({f"target:{d.name}": VLMDataset([d], "all") for d in data.targets})
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {name: DataLoader(ds, batch_size=args.batch_size if name == "train" else 256,
                                shuffle=name == "train", generator=generator if name == "train" else None,
                                collate_fn=collate, num_workers=args.num_workers, pin_memory=device.type == "cuda")
               for name, ds in splits.items()}

    def image_of(states):
        return pool_layers(states.float(), args.pooling, args.pool_kernel, args.pool_stride, args.pool_layers)

    total = torch.zeros(rows_in_image, data.hidden_size, dtype=torch.float64)
    total_sq = torch.zeros_like(total)
    with torch.inference_mode():
        for states, _, _, _ in DataLoader(splits["train"], batch_size=256, collate_fn=collate):
            image = image_of(states.to(device)).cpu().double()
            total += image.sum(0)
            total_sq += image.square().sum(0)
    n = len(splits["train"])
    feature_mean = (total / n).float()
    feature_std = (total_sq / n - (total / n).square()).clamp(min=0).sqrt().float().clamp(min=1e-4)

    def encoder(layers):
        layer = nn.TransformerEncoderLayer(args.model_dim, args.heads, 4 * args.model_dim, args.dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        return nn.TransformerEncoder(layer, layers, norm=nn.LayerNorm(args.model_dim), enable_nested_tensor=False)

    class ScratchVLM(nn.Module):
        def __init__(self):
            super().__init__()
            d = args.model_dim
            self.register_buffer("feature_mean", feature_mean.clone())
            self.register_buffer("feature_std", feature_std.clone())
            self.patch_embed = nn.Linear(args.patch_height * args.patch_width, d)
            self.patch_pos = nn.Parameter(torch.zeros(1, num_patches, d))
            self.vision = encoder(args.vision_layers)
            self.projector = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
            self.token_embed = nn.Embedding(len(vocab) + 4, d, padding_idx=PAD)
            self.text_pos = nn.Parameter(torch.zeros(1, max_text, d))
            self.type_embed = nn.Embedding(3, d)
            self.cls = nn.Parameter(torch.zeros(1, 1, d))
            self.fusion = encoder(args.fusion_layers)
            self.head = nn.Linear(d, 1)
            for p in (self.patch_pos, self.text_pos, self.cls):
                nn.init.trunc_normal_(p, std=0.02)

        def forward(self, states, ids, types):
            image = ((image_of(states) - self.feature_mean) / self.feature_std).clamp(-args.feature_clip,
                                                                                      args.feature_clip)
            image = F.pad(image, (0, 0, 0, grid_rows * args.patch_height - rows_in_image))
            patches = image.reshape(-1, grid_rows, args.patch_height, grid_cols, args.patch_width)
            patches = patches.permute(0, 1, 3, 2, 4).reshape(-1, num_patches, args.patch_height * args.patch_width)
            patches = self.projector(self.vision(self.patch_embed(patches) + self.patch_pos))
            patches = patches + self.type_embed.weight[PATCH_TYPE]
            t = self.token_embed(ids) + self.text_pos[:, :ids.shape[1]] + self.type_embed(types)
            b = states.shape[0]
            sequence = torch.cat((self.cls.expand(b, -1, -1), patches.to(t.dtype), t), dim=1)
            padding = torch.cat((torch.zeros(b, 1 + num_patches, dtype=torch.bool, device=states.device),
                                 ids == PAD), dim=1)
            return self.head(self.fusion(sequence, src_key_padding_mask=padding)[:, 0]).squeeze(-1).float()

    model = ScratchVLM().to(device)

    def step(batch):
        states, ids, types, labels = (x.to(device, non_blocking=True) for x in batch)
        return model(states, ids, types), labels

    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (no_decay if p.ndim < 2 or name.endswith(("_pos", "cls")) else decay).append(p)
    param_groups = [{"params": decay}, {"params": no_decay, "weight_decay": 0.0}]
    print(f"vlm: image {rows_in_image}x{data.hidden_size} -> {num_patches} patches of "
          f"{args.patch_height}x{args.patch_width}; text={args.vlm_text}; vocab={len(vocab) + 4}; "
          f"parameters={sum(p.numel() for p in model.parameters()):,}", flush=True)
    best_auc, best_epoch, history = fit(model, loaders["train"], loaders["val"], data.labels("val"), step,
                                        make_loss(args, data.labels("train")), args, device, param_groups)
    logits = {name: predict(model, loader, step, device, args) for name, loader in loaders.items() if name != "train"}
    summary = {"image_rows": rows_in_image, "patch_size": [args.patch_height, args.patch_width],
               "patches": num_patches, "vlm_text": args.vlm_text, "text_vocab_size": len(vocab) + 4,
               "parameters": sum(p.numel() for p in model.parameters()),
               "best_source_val_auroc": best_auc, "best_epoch": best_epoch, "history": history}
    return summary, logits["val"], logits["test"], {d.name: logits[f"target:{d.name}"] for d in data.targets}
