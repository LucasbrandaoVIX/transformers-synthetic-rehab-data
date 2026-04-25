"""
Merge two MotionGPT3 sample folders into a unified data/synthetic/ dataset.

Use case: we generated 150 samples in one run (under the 500-prompt schedule)
and 100 samples in a second run (extra_prompts.txt covering missing exercises).
This script combines both into a clean dataset with consistent metadata.

Usage:
    python scripts/merge_synthetic.py \\
        --old_dir MotionGPT3/results/motgpt/.../samples_2026-04-23-16-19-01 \\
        --old_count 150 \\
        --new_dir MotionGPT3/results/motgpt/.../samples_2026-04-23-XX-XX-XX \\
        --output_dir data/synthetic
"""

import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np


# Old run used num_per_exercise=100 schedule (500 prompts total).
# Indices 0-99 -> lateral_arm_elevation, 100-199 -> arm_flexion_elbows, etc.
OLD_EXERCISES_PER_BLOCK = 100
OLD_EXERCISE_ORDER = [
    "lateral_arm_elevation",
    "arm_flexion_elbows",
    "trunk_rotation",
    "pelvis_rotation",
    "squatting",
]

# New run had this fixed mapping: 33 trunk + 33 pelvis + 34 squat
NEW_PLAN = [
    ("trunk_rotation", 33),
    ("pelvis_rotation", 33),
    ("squatting", 34),
]


def old_exercise_for_idx(idx: int) -> str:
    block = idx // OLD_EXERCISES_PER_BLOCK
    if block >= len(OLD_EXERCISE_ORDER):
        return "unknown"
    return OLD_EXERCISE_ORDER[block]


def new_exercise_for_idx(idx: int) -> str:
    cumulative = 0
    for name, count in NEW_PLAN:
        if idx < cumulative + count:
            return name
        cumulative += count
    return "unknown"


def read_prompts(path: Path) -> list:
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def collect_indexed(dir_path: Path) -> list:
    """Return [(prompt_idx, joints_path), ...] sorted by prompt_idx."""
    pattern = re.compile(r"^(\d+)_out\.npy$")
    out = []
    for p in dir_path.iterdir():
        m = pattern.match(p.name)
        if m:
            out.append((int(m.group(1)), p))
    out.sort(key=lambda x: x[0])
    return out


def copy_one(joints_path: Path, output_stem: str,
             motions_dir: Path, features_dir: Path, viz_dir: Path):
    """Copy joints + features + gif for one sample."""
    motion = np.load(str(joints_path))
    np.save(str(motions_dir / f"{output_stem}.npy"), motion)

    src_dir = joints_path.parent
    prompt_idx = int(joints_path.stem.split("_")[0])

    feats_path = src_dir / f"{prompt_idx}_out_feats.npy"
    feats_shape = None
    if feats_path.exists():
        feats = np.load(str(feats_path))
        np.save(str(features_dir / f"{output_stem}.npy"), feats)
        feats_shape = list(feats.shape)

    gif_path = src_dir / f"{prompt_idx}_out.gif"
    if gif_path.exists():
        shutil.copy2(str(gif_path), str(viz_dir / f"{output_stem}.gif"))

    if motion.ndim == 4:
        n_frames = int(motion.shape[1])
    else:
        n_frames = int(motion.shape[0])

    return {
        "n_frames": n_frames,
        "joints_shape": list(motion.shape),
        "feats_shape": feats_shape,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old_dir", required=True,
                    help="Samples folder from first generation run")
    ap.add_argument("--old_count", type=int, default=150,
                    help="How many samples to take from old run (default: 150)")
    ap.add_argument("--old_prompts", default="data/synthetic/rehab_prompts.txt",
                    help="500-prompt list used for the old run")
    ap.add_argument("--new_dir", required=True,
                    help="Samples folder from extra/second generation run")
    ap.add_argument("--new_prompts", default="data/synthetic/extra_prompts.txt",
                    help="100-prompt list used for the new run")
    ap.add_argument("--output_dir", default="data/synthetic")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    motions_dir = output_dir / "motions"
    features_dir = output_dir / "features"
    viz_dir = output_dir / "visualizations"
    motions_dir.mkdir(parents=True, exist_ok=True)
    features_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    old_prompts = read_prompts(Path(args.old_prompts)) if Path(args.old_prompts).exists() else []
    new_prompts = read_prompts(Path(args.new_prompts)) if Path(args.new_prompts).exists() else []

    metadata = []

    # Process old samples (take first `old_count`)
    old_indexed = collect_indexed(Path(args.old_dir))
    old_indexed = [x for x in old_indexed if x[0] < args.old_count]
    print(f"Old run: {len(old_indexed)} samples to copy from {args.old_dir}")
    for prompt_idx, joints_path in old_indexed:
        ex = old_exercise_for_idx(prompt_idx)
        stem = f"synthetic_{ex}_{prompt_idx:05d}"
        info = copy_one(joints_path, stem, motions_dir, features_dir, viz_dir)
        metadata.append({
            "filename": f"{stem}.npy",
            "exercise": ex,
            "prompt": old_prompts[prompt_idx] if prompt_idx < len(old_prompts) else "unknown",
            "source": "run1",
            **info,
        })

    # Process new samples (renumber to start at old_count)
    new_indexed = collect_indexed(Path(args.new_dir))
    print(f"New run: {len(new_indexed)} samples to copy from {args.new_dir}")
    for local_idx, joints_path in new_indexed:
        ex = new_exercise_for_idx(local_idx)
        global_idx = args.old_count + local_idx
        stem = f"synthetic_{ex}_{global_idx:05d}"
        info = copy_one(joints_path, stem, motions_dir, features_dir, viz_dir)
        metadata.append({
            "filename": f"{stem}.npy",
            "exercise": ex,
            "prompt": new_prompts[local_idx] if local_idx < len(new_prompts) else "unknown",
            "source": "run2",
            **info,
        })

    # Sort metadata by exercise then idx for nicer browsing
    metadata.sort(key=lambda m: (m["exercise"], m["filename"]))

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Write prompts.txt as union
    all_prompts = [m["prompt"] for m in metadata]
    with open(output_dir / "prompts.txt", "w") as f:
        for p in all_prompts:
            f.write(p + "\n")

    # Distribution summary
    counts = {}
    for m in metadata:
        counts[m["exercise"]] = counts.get(m["exercise"], 0) + 1
    print(f"\nMerged dataset: {len(metadata)} samples")
    for ex, n in sorted(counts.items()):
        print(f"  {ex}: {n}")
    print(f"Output: {output_dir}/")


if __name__ == "__main__":
    main()
