"""Command-line interface: train, sample, evaluate, and explore VAE latent space.

Examples
--------
    molgen train --data molecules.smi --model molgpt --epochs 20 --out model.pt
    molgen sample --checkpoint model.pt --num 1000 --out generated.smi
    molgen eval --generated generated.smi --reference molecules.smi
    molgen vae-train --data molecules.smi --epochs 20 --out vae.pt
    molgen vae-sample --checkpoint vae.pt --seed-smiles "c1ccccc1" --num 100
"""

from __future__ import annotations

import argparse
import json


def _read_smiles(path: str) -> list[str]:
    if path.endswith(".csv"):
        from molgen.data import read_smiles_csv

        return read_smiles_csv(path)
    with open(path, encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def _build_tokenizer(kind: str, smiles):
    if kind == "selfies":
        from molgen.selfies_tokenizer import SelfiesTokenizer

        return SelfiesTokenizer.from_smiles(smiles)
    from molgen.tokenizers import SmilesTokenizer

    return SmilesTokenizer.from_smiles(smiles)


def cmd_train(args: argparse.Namespace) -> None:
    from molgen.char_rnn import CharRNN
    from molgen.checkpoint import save_checkpoint
    from molgen.data import build_dataloaders
    from molgen.molgpt import MolGPT
    from molgen.trainer import TrainConfig, train_language_model
    from molgen.utils import get_device, set_seed

    set_seed(args.seed)
    smiles = _read_smiles(args.data)
    tokenizer = _build_tokenizer(args.tokenizer, smiles)
    train_loader, val_loader = build_dataloaders(
        smiles, tokenizer, batch_size=args.batch_size, augment=args.augment, seed=args.seed
    )
    if args.model == "charrnn":
        kwargs = dict(
            vocab_size=tokenizer.vocab_size,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            pad_idx=tokenizer.pad_id,
        )
        model = CharRNN(**kwargs)
    else:
        kwargs = dict(
            vocab_size=tokenizer.vocab_size,
            embedding_dim=args.embedding_dim,
            nhead=args.nhead,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            pad_idx=tokenizer.pad_id,
        )
        model = MolGPT(**kwargs)
    config = TrainConfig(epochs=args.epochs, lr=args.lr)
    train_language_model(model, train_loader, val_loader, config, get_device(), tokenizer.pad_id)
    save_checkpoint(args.out, model, args.model, kwargs, tokenizer)
    print(f"Saved checkpoint to {args.out}")


def cmd_sample(args: argparse.Namespace) -> None:
    from molgen.checkpoint import load_checkpoint
    from molgen.sampling import sample
    from molgen.utils import get_device

    device = get_device()
    model, tokenizer = load_checkpoint(args.checkpoint, device=device)
    smiles = sample(
        model,
        tokenizer,
        num_samples=args.num,
        max_len=args.max_len,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        device=device,
    )
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write("\n".join(smiles) + "\n")
        print(f"Wrote {len(smiles)} SMILES to {args.out}")
    else:
        print("\n".join(smiles))


def cmd_eval(args: argparse.Namespace) -> None:
    from molgen.metrics import evaluate_generation

    generated = _read_smiles(args.generated)
    reference = _read_smiles(args.reference) if args.reference else None
    print(json.dumps(evaluate_generation(generated, reference), indent=2))


def cmd_vae_train(args: argparse.Namespace) -> None:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from molgen.checkpoint import save_checkpoint
    from molgen.utils import get_device, set_seed
    from molgen.vae import BetaTCVAE, train

    set_seed(args.seed)
    device = get_device()
    smiles = _read_smiles(args.data)
    tokenizer = _build_tokenizer(args.tokenizer, smiles)
    data = tokenizer.encode_batch(smiles, max_len=None)

    kwargs = dict(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        nhead=args.nhead,
        num_layers=args.num_layers,
        pad_idx=tokenizer.pad_id,
        device=device,
    )
    model = BetaTCVAE(**kwargs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loader = DataLoader(TensorDataset(data), batch_size=args.batch_size, shuffle=True)
    for epoch in range(args.epochs):
        loss = train(
            model, loader, optimizer, device, beta=args.beta, gamma=args.gamma, alpha=args.alpha
        )
        print(f"Epoch {epoch + 1}/{args.epochs}: loss={loss:.4f}")
    save_checkpoint(args.out, model, "vae", kwargs, tokenizer)
    print(f"Saved checkpoint to {args.out}")


def cmd_vae_sample(args: argparse.Namespace) -> None:
    from molgen.checkpoint import load_checkpoint
    from molgen.generate import generate_nearby_smiles
    from molgen.utils import get_device

    device = get_device()
    model, tokenizer = load_checkpoint(args.checkpoint, device=device)
    model.to(device)
    generated = generate_nearby_smiles(
        model,
        tokenizer,
        args.seed_smiles,
        args.max_len,
        args.num,
        device,
        temperature=args.temperature,
        distance_multiplier=args.distance_multiplier,
    )
    _write_or_print(generated, args.out)


def cmd_vae_interpolate(args: argparse.Namespace) -> None:
    from molgen.checkpoint import load_checkpoint
    from molgen.interpolate import interpolate_smiles
    from molgen.utils import get_device

    device = get_device()
    model, tokenizer = load_checkpoint(args.checkpoint, device=device)
    model.to(device)
    path = interpolate_smiles(
        model,
        tokenizer,
        args.start,
        args.end,
        args.max_len,
        device,
        num_steps=args.num_steps,
        temperature=args.temperature,
    )
    _write_or_print(path, args.out)


def _write_or_print(smiles, out) -> None:
    if out:
        with open(out, "w", encoding="utf-8") as handle:
            handle.write("\n".join(smiles) + "\n")
        print(f"Wrote {len(smiles)} molecules to {out}")
    else:
        for s in smiles:
            print(s)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="molgen", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="train a generator on a SMILES file (.smi/.csv)")
    train.add_argument("--data", required=True)
    train.add_argument("--model", choices=["charrnn", "molgpt"], default="molgpt")
    train.add_argument("--tokenizer", choices=["smiles", "selfies"], default="smiles")
    train.add_argument("--epochs", type=int, default=10)
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--lr", type=float, default=1e-3)
    train.add_argument("--embedding-dim", type=int, default=256)
    train.add_argument("--hidden-dim", type=int, default=512)
    train.add_argument("--num-layers", type=int, default=3)
    train.add_argument("--nhead", type=int, default=8)
    train.add_argument("--augment", action="store_true")
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--out", default="model.pt")
    train.set_defaults(func=cmd_train)

    sample = sub.add_parser("sample", help="sample SMILES from a trained checkpoint")
    sample.add_argument("--checkpoint", required=True)
    sample.add_argument("--num", type=int, default=100)
    sample.add_argument("--max-len", type=int, default=120)
    sample.add_argument("--temperature", type=float, default=1.0)
    sample.add_argument("--top-k", type=int, default=None)
    sample.add_argument("--top-p", type=float, default=None)
    sample.add_argument("--out", default=None)
    sample.set_defaults(func=cmd_sample)

    evaluate = sub.add_parser("eval", help="compute metrics for a file of generated SMILES")
    evaluate.add_argument("--generated", required=True)
    evaluate.add_argument("--reference", default=None)
    evaluate.set_defaults(func=cmd_eval)

    vae_train = sub.add_parser("vae-train", help="train a VAE for latent-space exploration")
    vae_train.add_argument("--data", required=True)
    vae_train.add_argument("--tokenizer", choices=["smiles", "selfies"], default="selfies")
    vae_train.add_argument("--epochs", type=int, default=10)
    vae_train.add_argument("--batch-size", type=int, default=64)
    vae_train.add_argument("--lr", type=float, default=1e-3)
    vae_train.add_argument("--embedding-dim", type=int, default=64)
    vae_train.add_argument("--latent-dim", type=int, default=32)
    vae_train.add_argument("--hidden-dim", type=int, default=256)
    vae_train.add_argument("--num-layers", type=int, default=2)
    vae_train.add_argument("--nhead", type=int, default=4)
    vae_train.add_argument(
        "--beta", type=float, default=4.0, help="total-correlation weight (disentanglement)"
    )
    vae_train.add_argument("--gamma", type=float, default=1.0, help="dimension-wise KL weight")
    vae_train.add_argument(
        "--alpha", type=float, default=1.0, help="index-code mutual-information weight"
    )
    vae_train.add_argument("--seed", type=int, default=42)
    vae_train.add_argument("--out", default="vae_model.pt")
    vae_train.set_defaults(func=cmd_vae_train)

    vae_sample = sub.add_parser("vae-sample", help="sample molecules near a seed in latent space")
    vae_sample.add_argument("--checkpoint", required=True)
    vae_sample.add_argument("--seed-smiles", required=True, help="seed molecule to explore around")
    vae_sample.add_argument("--num", type=int, default=100)
    vae_sample.add_argument("--max-len", type=int, default=120)
    vae_sample.add_argument("--temperature", type=float, default=1.0)
    vae_sample.add_argument("--distance-multiplier", type=float, default=0.5)
    vae_sample.add_argument("--out", default=None)
    vae_sample.set_defaults(func=cmd_vae_sample)

    vae_interp = sub.add_parser(
        "vae-interpolate", help="interpolate between two molecules in latent space"
    )
    vae_interp.add_argument("--checkpoint", required=True)
    vae_interp.add_argument("--start", required=True, help="start molecule (SMILES)")
    vae_interp.add_argument("--end", required=True, help="end molecule (SMILES)")
    vae_interp.add_argument("--num-steps", type=int, default=10)
    vae_interp.add_argument("--max-len", type=int, default=120)
    vae_interp.add_argument("--temperature", type=float, default=1.0)
    vae_interp.add_argument("--out", default=None)
    vae_interp.set_defaults(func=cmd_vae_interpolate)

    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
