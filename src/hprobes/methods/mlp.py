"""SALA without projections: hidden states -> [layer pooling] -> [PCA] -> flatten -> residual MLP."""

from __future__ import annotations

from torch.utils.data import DataLoader

from hprobes.data import StateDataset
from hprobes.features import LayerPCA, pool_layers, pooled_layer_count, pooled_matrix
from hprobes.losses import make_loss
from hprobes.training import fit, predict, residual_mlp, seed_everything, tensor_loader


LAST_MODEL: dict = {}  # filled by run(); cli.train() writes it to <output-dir>/model.pt


def run(args, data, device):
    import torch

    counts = {name: pooled_layer_count(layers, args.pooling, args.pool_kernel, args.pool_stride, args.pool_layers)
              for name, (layers, _) in data.models.items()}
    if len(set(counts.values())) != 1:
        raise SystemExit(f"LLMs end with different pooled layer counts {counts}; pass --pool-layers 15 (or similar)")
    pooled = next(iter(counts.values()))
    if len(data.models) > 1 and not args.pca_dim:
        raise SystemExit("Several LLMs have different hidden sizes: use --pca-dim (one PCA per LLM and layer)")
    seed_everything(args.seed)
    splits = {"train": StateDataset(data.sources, "train"), "val": StateDataset(data.sources, "val"),
              "test": StateDataset(data.sources, "test")}
    splits.update({f"target:{d.name}": StateDataset([d], "all") for d in data.targets})
    labels = {name: [entry[2] for entry in ds.entries] for name, ds in splits.items()}
    pcas = {}
    if args.pca_dim:
        # One PCA per LLM, fitted on that LLM's pooled source-train features only; the MLP is shared.
        for llm in data.models:
            domains = [d for d in data.sources if d.model == llm]
            if not domains:
                raise SystemExit(f"LLM {llm!r} has no source dataset to fit its PCA on")
            pcas[llm] = LayerPCA(args.pca_dim, args.pca_scope, args.seed)
            pcas[llm].fit_transform(pooled_matrix(StateDataset(domains, "train"), args, device), device)

        def project(domains, split):
            return torch.cat([pcas[d.model].transform(pooled_matrix(StateDataset([d], split), args, device),
                                                      device).flatten(1) for d in domains])

        x = {"train": project(data.sources, "train"), "val": project(data.sources, "val"),
             "test": project(data.sources, "test")}
        x.update({f"target:{d.name}": project([d], "all") for d in data.targets})
        input_dim = x["train"].shape[1]
        print(f"PCA per LLM {[p.summary() for p in pcas.values()]}; MLP input={input_dim}", flush=True)
        loaders = {name: tensor_loader(x[name], labels[name], args.batch_size if name == "train" else 256,
                                       shuffle=name == "train", seed=args.seed) for name in splits}

        def step(batch):
            return model(batch[0].to(device, non_blocking=True)), batch[1].to(device, non_blocking=True)
    else:
        # Stream float16 states from the memmaps and pool on the device inside each step.
        input_dim = pooled * data.hidden_size
        generator = torch.Generator().manual_seed(args.seed)
        loaders = {name: DataLoader(ds, batch_size=args.batch_size if name == "train" else 256,
                                    shuffle=name == "train", generator=generator if name == "train" else None,
                                    num_workers=args.num_workers, pin_memory=device.type == "cuda")
                   for name, ds in splits.items()}

        def step(batch):
            states = pool_layers(batch[0].to(device, non_blocking=True).float(), args.pooling,
                                 args.pool_kernel, args.pool_stride, args.pool_layers)
            return model(states.flatten(1)), batch[1].to(device, non_blocking=True)
    print(f"mlp: pooling={args.pooling} -> {pooled} layers; LLMs={dict(data.models)}; input={input_dim}; "
          f"head={args.hidden_layers}x{args.hidden_dim}; train/val/test="
          f"{len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])}", flush=True)
    model = residual_mlp(input_dim, args.hidden_dim, args.hidden_layers, args.dropout).to(device)
    best_auc, best_epoch, history = fit(model, loaders["train"], loaders["val"], data.labels("val"), step,
                                        make_loss(args, data.labels("train")), args, device)
    logits = {name: predict(model, loader, step, device, args) for name, loader in loaders.items()}
    LAST_MODEL.clear()
    LAST_MODEL.update({
        "pooling": args.pooling, "pool_kernel": args.pool_kernel, "pool_stride": args.pool_stride,
        "pool_layers": args.pool_layers,
        "models": {name: {"n_layers": layers, "hidden_size": hidden,
                          "pca": pcas[name].state() if pcas else None}
                   for name, (layers, hidden) in data.models.items()},
        "head": {"input_dim": input_dim, "hidden_dim": args.hidden_dim, "hidden_layers": args.hidden_layers,
                 "dropout": args.dropout},
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
    })
    summary = {"pooled_layers": pooled, "mlp_input_dimension": input_dim,
               "pca": {name: p.summary() for name, p in pcas.items()} or None,
               "parameters": sum(p.numel() for p in model.parameters()),
               "best_source_val_auroc": best_auc, "best_epoch": best_epoch, "history": history}
    return summary, logits["val"], logits["test"], {d.name: logits[f"target:{d.name}"] for d in data.targets}
