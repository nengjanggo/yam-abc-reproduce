"""Merge an ABC-DiT LoRA checkpoint into a plain (deployable) checkpoint.

LoRALinear (abc_minimal/dit.py): y = base(x) + scale * lora_b(lora_a(x)),
scale = (alpha or rank)/rank = 1.0 for our runs (apply_lora passes no alpha).
So merged weight W = base.weight + lora_b.weight @ lora_a.weight ; bias = base.bias.
Folds every ``*.base.weight`` (+ its lora_a/lora_b siblings) back to ``*.weight``
so the checkpoint loads into a non-LoRA DiT.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from yam_abc_reproduce.storage import StoragePaths, add_save_root_argument  # noqa: E402


def merge_checkpoint(src: Path, dst: Path) -> None:
    import torch

    dst.parent.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    sd = {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v for k, v in sd.items()}

    prefixes = sorted({k[: -len(".base.weight")] for k in sd if k.endswith(".base.weight")})
    print(f"{len(prefixes)} LoRA-wrapped layers to merge")

    merged, consumed = {}, set()
    for p in prefixes:
        bw = sd[f"{p}.base.weight"]
        la, lb = sd.get(f"{p}.lora_a.weight"), sd.get(f"{p}.lora_b.weight")
        w = bw.float()
        if la is not None and lb is not None:
            w = w + (lb.float() @ la.float())  # scale = 1.0
        merged[f"{p}.weight"] = w.to(bw.dtype)
        consumed |= {f"{p}.base.weight", f"{p}.lora_a.weight", f"{p}.lora_b.weight"}
        bb = sd.get(f"{p}.base.bias")
        if bb is not None:
            merged[f"{p}.bias"] = bb
            consumed.add(f"{p}.base.bias")

    for k, v in sd.items():
        if k not in consumed:
            merged[k] = v

    # sanity: no lora/base residue left
    leftover = [k for k in merged if ".lora_" in k or ".base." in k]
    assert not leftover, f"residual lora keys: {leftover[:5]}"

    # preserve non-model checkpoint payload (norm_stats etc.), drop the huge optimizer.
    out = {"model": merged}
    if isinstance(ckpt, dict):
        for k, v in ckpt.items():
            if k not in ("model", "optimizer"):
                out[k] = v
    torch.save(out, dst)
    print(f"merged {len(sd)} -> {len(merged)} model keys; carried over: "
          f"{[k for k in out if k != 'model']}; saved to {dst}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_save_root_argument(parser)
    parser.add_argument("src", type=Path, help="source ABC LoRA checkpoint")
    parser.add_argument("dst", type=Path, nargs="?",
                        help="output override (default: <save-root>/model/abc/merged/<name>_merged.pt)")
    args = parser.parse_args(argv)
    paths = StoragePaths.from_arg(args.save_root)
    paths.configure_environment()
    src = args.src.expanduser()
    dst = paths.resolve(args.dst or paths.model / "abc" / "merged" / f"{src.stem}_merged.pt")
    merge_checkpoint(src, dst)


if __name__ == "__main__":
    main()
