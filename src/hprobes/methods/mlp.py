"""SALA without projections: hidden states -> [layer pooling] -> [PCA] -> flatten -> residual MLP."""

from __future__ import annotations

from torch.utils.data import DataLoader

from hprobes.data import StateDataset
from hprobes.features import LayerPCA, pool_layers, pooled_layer_count, pooled_matrix
from hprobes.losses import make_loss
from hprobes.training import fit, predict, residual_mlp, seed_everything, tensor_loader


def run(args, data, device):
    pooled = pooled_layer_count(data.n_layers, args.pooling, args.pool_kernel, args.pool_stride)
    seed_everything(args.seed)
    splits = {"train": StateDataset(data.sources, "train"), "val": StateDataset(data.sources, "val"),
              "test": StateDataset(data.sources, "test")}
    splits.update({f"target:{d.name}": StateDataset([d], "all") for d in data.targets})
    labels = {name: [entry[2] for entry in ds.entries] for name, ds in splits.items()}
    pca = None
    if args.pca_dim:
        # Fit PCA on pooled source-train features, then train the MLP on the components.
        pca = LayerPCA(args.pca_dim, args.pca_scope, args.seed)
        x = {"train": pca.fit_transform(pooled_matrix(splits["train"], args, device), device).flatten(1)}
        for name in splits:
            if name != "train":
                x[name] = pca.transform(pooled_matrix(splits[name], args, device), device).flatten(1)
        input_dim = x["train"].shape[1]
        print(f"PCA {pca.summary()}; MLP input={input_dim}", flush=True)
        loaders = {name: tensor_loader(x[name], labels[name], args.batch_size if name == "train" else 256,
                                       shuffle=name == "train", seed=args.seed) for name in splits}

        def step(batch):
            return model(batch[0].to(device, non_blocking=True)), batch[1].to(device, non_blocking=True)
    else:
        # Stream float16 states from the memmaps and pool on the device inside each step.
        input_dim = pooled * data.hidden_size
        import torch

        generator = torch.Generator().manual_seed(args.seed)
        loaders = {name: DataLoader(ds, batch_size=args.batch_size if name == "train" else 256,
                                    shuffle=name == "train", generator=generator if name == "train" else None,
                                    num_workers=args.num_workers, pin_memory=device.type == "cuda")
                   for name, ds in splits.items()}

        def step(batch):
            states = pool_layers(batch[0].to(device, non_blocking=True).float(), args.pooling,
                                 args.pool_kernel, args.pool_stride)
            return model(states.flatten(1)), batch[1].to(device, non_blocking=True)
    print(f"mlp: pooling={args.pooling} -> {pooled} layers x {data.hidden_size}; input={input_dim}; "
          f"head={args.hidden_layers}x{args.hidden_dim}; train/val/test="
          f"{len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])}", flush=True)
    model = residual_mlp(input_dim, args.hidden_dim, args.hidden_layers, args.dropout).to(device)
    best_auc, best_epoch, history = fit(model, loaders["train"], loaders["val"], data.labels("val"), step,
                                        make_loss(args, data.labels("train")), args, device)
    logits = {name: predict(model, loader, step, device, args) for name, loader in loaders.items()}
    summary = {"pooled_layers": pooled, "mlp_input_dimension": input_dim,
               "pca": pca.summary() if pca else None,
               "parameters": sum(p.numel() for p in model.parameters()),
               "best_source_val_auroc": best_auc, "best_epoch": best_epoch, "history": history}
    return summary, logits["val"], logits["test"], {d.name: logits[f"target:{d.name}"] for d in data.targets}
