#!/usr/bin/env python3
import argparse
import gc
import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch

try:
    from safetensors.torch import safe_open, save_file
except Exception as e:  # pragma: no cover
    raise SystemExit(f"Missing dependency safetensors: {e}")

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None


def _rewrite_key(old: str) -> Optional[str]:
    if ".lora_" in old or ".lora_A." in old or ".lora_B." in old:
        return None
    name = old
    if name.startswith("base_model.model."):
        name = name[len("base_model.model.") :]
    elif name.startswith("base_model."):
        name = name[len("base_model.") :]
    # PEFT wraps Linear with a base_layer module.
    name = name.replace(".base_layer.", ".")
    # Some configs double-nest as model.model.*
    name = name.replace("model.model.", "model.")
    return name


def _is_peft_index(index_obj: dict) -> bool:
    wm = (index_obj or {}).get("weight_map", {}) or {}
    for k in wm.keys():
        if k.startswith("base_model.") or ".lora_" in k or ".base_layer." in k:
            return True
    return False


def _load_index(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _copy_non_weight_files(src_dir: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for p in src_dir.iterdir():
        if p.is_dir():
            continue
        name = p.name
        if name.endswith(".safetensors"):
            continue
        if name.endswith(".safetensors.index.json"):
            continue
        shutil.copy2(p, dst_dir / name)


def _iter_weight_files(src_dir: Path, index_obj: dict) -> List[Path]:
    files = sorted({v for v in (index_obj.get("weight_map", {}) or {}).values()})
    return [src_dir / f for f in files]


def _tensor_nbytes(t: torch.Tensor) -> int:
    try:
        return int(t.numel() * t.element_size())
    except Exception:
        return 0


@dataclass
class ConvertStats:
    dropped_lora: int = 0
    dropped_unmatched: int = 0
    rewritten: int = 0
    total_out_tensors: int = 0
    total_out_bytes: int = 0
    out_shards: int = 0


def convert_one_dir(
    src_dir: Path,
    *,
    in_place: bool,
    out_dir_name: str,
    out_dir_override: Optional[Path],
    max_shard_size_gb: float,
    overwrite: bool,
    keep_original_weights: bool,
    progress: bool,
) -> Tuple[Path, ConvertStats]:
    index_path = src_dir / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing index: {index_path}")
    index_obj = _load_index(index_path)
    if not _is_peft_index(index_obj):
        # Still copy to out dir if requested, otherwise no-op.
        if in_place:
            return src_dir, ConvertStats()
        out_dir = out_dir_override or (src_dir.parent / out_dir_name)
        if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
            return out_dir, ConvertStats()
        out_dir.mkdir(parents=True, exist_ok=True)
        _copy_non_weight_files(src_dir, out_dir)
        shutil.copy2(index_path, out_dir / index_path.name)
        for wf in _iter_weight_files(src_dir, index_obj):
            shutil.copy2(wf, out_dir / wf.name)
        return out_dir, ConvertStats()

    max_bytes = int(max_shard_size_gb * (1024**3))
    if max_bytes <= 0:
        max_bytes = 1 * (1024**3)

    # Stage in a temp dir first, then optionally swap into src_dir.
    out_dir = out_dir_override or (src_dir.parent / out_dir_name)
    stage_dir = out_dir.parent / f".{out_dir.name}.tmp_{int(time.time())}"
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)
    _copy_non_weight_files(src_dir, stage_dir)

    weight_map: Dict[str, str] = index_obj.get("weight_map", {}) or {}
    by_file: Dict[str, List[str]] = {}
    for k, fname in weight_map.items():
        by_file.setdefault(fname, []).append(k)

    tmp_files: List[Path] = []
    new_weight_map: Dict[str, str] = {}
    stats = ConvertStats()

    current: Dict[str, torch.Tensor] = {}
    current_bytes = 0

    def flush(shard_idx: int) -> None:
        nonlocal current, current_bytes, tmp_files
        if not current:
            return
        tmp_name = f"model-tmp-{shard_idx:05d}.safetensors"
        tmp_path = stage_dir / tmp_name
        save_file(current, str(tmp_path))
        tmp_files.append(tmp_path)
        stats.out_shards += 1
        current = {}
        current_bytes = 0
        gc.collect()

    shard_idx = 1

    file_items: Iterable[Tuple[str, List[str]]] = sorted(by_file.items(), key=lambda x: x[0])
    if progress and tqdm is not None:
        file_items = tqdm(list(file_items), desc=f"convert shards: {src_dir.name}")

    for fname, keys in file_items:
        src_file = src_dir / fname
        if not src_file.exists():
            raise FileNotFoundError(f"Missing shard: {src_file}")
        with safe_open(str(src_file), framework="pt", device="cpu") as f:
            for old_key in keys:
                if ".lora_" in old_key or ".lora_A." in old_key or ".lora_B." in old_key:
                    stats.dropped_lora += 1
                    continue
                new_key = _rewrite_key(old_key)
                if new_key is None:
                    stats.dropped_unmatched += 1
                    continue
                if new_key in new_weight_map or new_key in current:
                    raise RuntimeError(f"Key collision after rewrite: {old_key} -> {new_key}")

                t = f.get_tensor(old_key)
                current[new_key] = t
                stats.rewritten += 1
                stats.total_out_tensors += 1
                nb = _tensor_nbytes(t)
                stats.total_out_bytes += nb
                current_bytes += nb

                # We'll patch the filename later after we know total shard count.
                new_weight_map[new_key] = f"model-tmp-{shard_idx:05d}.safetensors"

                if current_bytes >= max_bytes:
                    flush(shard_idx)
                    shard_idx += 1

    flush(shard_idx)

    # Rename tmp shard names to HuggingFace-style names with total shards.
    total = len(tmp_files)
    final_names: Dict[str, str] = {}
    for i, tmp_path in enumerate(tmp_files, start=1):
        final = f"model-{i:05d}-of-{total:05d}.safetensors"
        final_names[tmp_path.name] = final
        tmp_path.rename(stage_dir / final)

    new_weight_map = {k: final_names.get(v, v) for k, v in new_weight_map.items()}
    index_out = {
        "metadata": {"total_size": int(stats.total_out_bytes)},
        "weight_map": new_weight_map,
    }
    (stage_dir / "model.safetensors.index.json").write_text(json.dumps(index_out, ensure_ascii=False), encoding="utf-8")

    # Quick sanity check.
    bad_prefix = 0
    for k in new_weight_map.keys():
        if k.startswith("base_model.") or ".lora_" in k or ".base_layer." in k:
            bad_prefix += 1
    if bad_prefix:
        raise RuntimeError(f"Converted index still contains {bad_prefix} PEFT keys; refusing to swap in-place.")

    if not in_place:
        if out_dir.exists():
            if not overwrite:
                raise FileExistsError(f"Output exists: {out_dir} (use --overwrite)")
            shutil.rmtree(out_dir)
        stage_dir.rename(out_dir)
        return out_dir, stats

    # In-place: move converted weights/index into src_dir, optionally keep originals.
    original_files = _iter_weight_files(src_dir, index_obj)
    original_index = src_dir / "model.safetensors.index.json"

    # Remove originals first only if we keep a staging dir intact.
    if not keep_original_weights:
        for p in original_files:
            try:
                p.unlink()
            except FileNotFoundError:
                pass
    try:
        original_index.unlink()
    except FileNotFoundError:
        pass

    # Move new files into src_dir.
    for p in stage_dir.iterdir():
        target = src_dir / p.name
        if target.exists():
            if target.is_file():
                target.unlink()
            else:
                shutil.rmtree(target)
        p.rename(target)
    stage_dir.rmdir()

    return src_dir, stats


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert PEFT-wrapped HF checkpoints to plain HF safetensors (lmdeploy-friendly).")
    ap.add_argument("--run-dir", required=True, help="Run directory containing global_step_*/actor/huggingface.")
    ap.add_argument("--glob", default="global_step_*/actor/huggingface", help='Glob pattern under run-dir (default: "global_step_*/actor/huggingface").')
    ap.add_argument("--in-place", action="store_true", help="Rewrite each huggingface dir in-place (default: false).")
    ap.add_argument("--out-dir-name", default="huggingface_plain", help="Output dir name when not --in-place (default: huggingface_plain).")
    ap.add_argument(
        "--out-root",
        default=None,
        help="Write converted checkpoints under this directory (writable). When set, ignores --in-place and mirrors paths under --run-dir.",
    )
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip conversion when output directory already exists (useful for resuming).",
    )
    ap.add_argument("--max-shard-size-gb", type=float, default=1.0, help="Max output shard size in GB (default: 1.0).")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite output dir when not --in-place.")
    ap.add_argument("--keep-original-weights", action="store_true", help="When --in-place, keep original weight shards (default: delete).")
    ap.add_argument("--progress", action="store_true", help="Show progress bars (requires tqdm).")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    out_root = Path(args.out_root).resolve() if args.out_root else None
    if out_root is not None:
        out_root.mkdir(parents=True, exist_ok=True)
    targets = sorted(run_dir.glob(args.glob))
    if not targets:
        raise SystemExit(f"No targets found under {run_dir} with glob {args.glob!r}")

    t0 = time.time()
    ok = 0
    for src in targets:
        if not src.is_dir():
            continue
        try:
            out_dir_override = None
            in_place = bool(args.in_place)
            if out_root is not None:
                in_place = False
                rel = src.relative_to(run_dir)
                out_dir_override = out_root / rel
                if args.skip_existing and out_dir_override.exists() and (out_dir_override / "model.safetensors.index.json").exists():
                    print(f"[SKIP] {src} -> {out_dir_override} (exists)")
                    continue
            out, stats = convert_one_dir(
                src,
                in_place=in_place,
                out_dir_name=str(args.out_dir_name),
                out_dir_override=out_dir_override,
                max_shard_size_gb=float(args.max_shard_size_gb),
                overwrite=bool(args.overwrite),
                keep_original_weights=bool(args.keep_original_weights),
                progress=bool(args.progress),
            )
            ok += 1
            if stats.total_out_tensors:
                print(
                    f"[OK] {src} -> {out} | shards={stats.out_shards} tensors={stats.total_out_tensors} "
                    f"out_gb={stats.total_out_bytes/(1024**3):.2f} dropped_lora={stats.dropped_lora} dropped_other={stats.dropped_unmatched}"
                )
            else:
                print(f"[SKIP] {src} (already plain or copied)")
        except Exception as e:
            print(f"[FAIL] {src}: {e}")
            raise

    dt = time.time() - t0
    print(f"Done. converted={ok}/{len(targets)} elapsed_s={dt:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
