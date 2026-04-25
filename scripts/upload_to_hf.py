"""
Upload synthetic rehabilitation dataset to HuggingFace Hub.

Prepares the dataset directory with README (dataset card), LICENSE,
metadata, and uploads to a HuggingFace dataset repository.

Usage:
    # First time: login or export HF_TOKEN
    huggingface-cli login

    python scripts/upload_to_hf.py \
        --repo_id LucasbrandaoVIX/PhysioMotion-Synthetic-Baseline \
        --source_dir data/synthetic \
        --create_repo
"""

import os
import json
import shutil
import argparse
from pathlib import Path
from textwrap import dedent

from huggingface_hub import HfApi, create_repo


LICENSE_CC_BY_NC_4 = """Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)

You are free to:
  - Share: copy and redistribute the material in any medium or format
  - Adapt: remix, transform, and build upon the material

Under the following terms:
  - Attribution: You must give appropriate credit, provide a link to the license,
    and indicate if changes were made.
  - NonCommercial: You may not use the material for commercial purposes.

Full text: https://creativecommons.org/licenses/by-nc/4.0/legalcode
"""


def build_dataset_card(
    repo_id: str,
    metadata: list,
    evaluation: dict | None,
    num_samples: int,
    exercises: dict,
) -> str:
    """Build the HuggingFace dataset card (README.md) content."""

    # Evaluation summary (may be None if eval not run yet)
    eval_section = ""
    if evaluation:
        summary = evaluation.get("summary", {})
        jerk = summary.get("jerk_stats", {})
        dur = summary.get("duration_stats", {})
        bio = summary.get("biomechanical_validity", {})
        comp = evaluation.get("comparison_vs_real", {})

        # Minimum validity across joints
        bio_pcts = [v.get("mean_valid_pct", 100.0) for v in bio.values()] if bio else [100.0]
        min_bio = min(bio_pcts) if bio_pcts else 100.0

        comp_rows = ""
        if comp:
            comp_rows = (
                f"| Synthetic mean joint velocity (m/s) | {comp.get('synthetic_velocity_mean', 0):.4f} |\n"
                f"| Real KIMORE mean joint velocity (m/s) | {comp.get('real_velocity_mean', 0):.4f} |\n"
                f"| KL divergence (real ‖ synthetic velocity) | {comp.get('kl_divergence_velocity', 0):.4f} |\n"
            )

        comp_block = (comp_rows + "\n") if comp_rows else "\n"
        eval_section = (
            "## Quality Metrics\n\n"
            f"Biomechanical evaluation over all {summary.get('total_files', 'N/A')} synthetic samples "
            "(and distribution comparison against the real KIMORE dataset when available):\n\n"
            "| Metric | Value |\n"
            "|---|---|\n"
            f"| Jerk (mean, lower = smoother) | {jerk.get('mean', 0):.2f} ± {jerk.get('std', 0):.2f} |\n"
            f"| Duration (s) | {dur.get('mean', 0):.2f} ± {dur.get('std', 0):.2f} |\n"
            f"| Biomechanical validity (joint angles in ROM, min across joints) | {min_bio:.1f}% |\n"
            f"{comp_block}"
            "Synthetic motions are smoother than Kinect-captured ones because the generator "
            "produces clean trajectories (no sensor noise). Per-joint biomechanical validity "
            "is available in `evaluation_results.json`."
        )

    # Exercise distribution
    exercise_lines = "\n".join(
        f"| {name} | {count} |"
        for name, count in exercises.items()
    )

    yaml_frontmatter = (
        "---\n"
        "license: cc-by-nc-4.0\n"
        "language:\n"
        "- en\n"
        "tags:\n"
        "- motion-generation\n"
        "- rehabilitation\n"
        "- physiotherapy\n"
        "- kimore\n"
        "- motiongpt3\n"
        "- smpl\n"
        "- humanml3d\n"
        "- synthetic\n"
        "task_categories:\n"
        "- text-to-3d\n"
        "- other\n"
        "size_categories:\n"
        "- n<1K\n"
        f"pretty_name: PhysioMotion Synthetic Baseline\n"
        "---"
    )

    body = f"""# PhysioMotion-Synthetic-Baseline

Synthetic rehabilitation movement dataset generated with **MotionGPT3** (pre-trained,
no fine-tuning). Produced as part of an undergraduate research project (Iniciação
Científica) at UFES focused on augmenting rehabilitation datasets with transformer-based
motion generation.

This is the **baseline** release — a fine-tuned version trained on KIMORE will follow
once GPU resources become available.

## Dataset Summary

- **Samples**: {num_samples} motion sequences
- **Exercises**: 5 rehabilitation exercises (KIMORE protocol)
- **Format**: SMPL 22-joint skeleton, 20 fps, positions in meters
- **Generator**: MotionGPT3 (pre-trained checkpoint, no fine-tuning)
- **Conditioning**: text prompts describing each rehab exercise

## Exercise Distribution

| Exercise | Samples |
|---|---|
{exercise_lines}

## Data Structure

```
PhysioMotion-Synthetic-Baseline/
├── README.md                # This dataset card
├── LICENSE                  # CC-BY-NC-4.0
├── metadata.json            # Per-sample: filename, exercise, prompt, n_frames, shape
├── evaluation_results.json  # Biomechanical metrics vs real KIMORE
├── prompts.txt              # All text prompts used for generation
├── motions/                 # .npy files, shape (1, n_frames, 22, 3) — SMPL joint positions
└── features/                # .npy files, shape (n_frames, 263) — HumanML3D features
```

### Loading a sample

```python
import numpy as np
from huggingface_hub import hf_hub_download

# Download one motion
path = hf_hub_download(
    repo_id="{repo_id}",
    filename="motions/synthetic_squatting_00400.npy",
    repo_type="dataset",
)
motion = np.load(path)  # shape: (1, n_frames, 22, 3)
print(motion.shape)
```

### Joint order (SMPL-22)

`pelvis, l_hip, r_hip, spine1, l_knee, r_knee, spine2, l_ankle, r_ankle,
spine3, l_foot, r_foot, neck, l_collar, r_collar, head, l_shoulder, r_shoulder,
l_elbow, r_elbow, l_wrist, r_wrist`

## Exercises (KIMORE Protocol)

| ID | Name | Target |
|---|---|---|
| Es1 | Lateral arm elevation | Shoulder |
| Es2 | Arm flexion with elbows at hips | Elbow |
| Es3 | Trunk rotation (seated) | Thoracic spine |
| Es4 | Pelvis rotation (standing) | Lumbar spine |
| Es5 | Squatting | Lower limbs |

{eval_section}

## Generation Details

- Model: **MotionGPT3** pre-trained checkpoint (MoT architecture, GPT2 backbone + VAE + diffusion head)
- Task: Text-to-Motion (t2m)
- Inference: CPU (Mac M1), ~20s per sample
- Prompts: 10 text-template variations per exercise (cycled through repetitions to
  produce diverse generations within each exercise class)

See [source repo](https://github.com/LucasbrandaoVIX/PhysioMotion-Synthetic)
for generation scripts and full pipeline.

## Intended Use

- **Research only** (CC-BY-NC-4.0): academic study of synthetic motion augmentation
  for rehabilitation analysis, movement classification, and generative model benchmarking.
- **Not for clinical use**: these motions are generated by a generic motion model and
  have not been validated by physiotherapists. Do not use them for diagnosis, therapy
  planning, or patient-facing applications.

## Limitations

- **No pathology conditioning**: the generator is pre-trained on HumanML3D (healthy-like
  general motion). It does not capture patient-specific compensations or restricted ROM
  even when the prompt asks for them.
- **Smoother than reality**: synthetic jerk is lower than sensor-captured motion because
  there is no Kinect noise.
- **Limited prompt diversity**: 50 base prompts (10 per exercise). More variation will
  come with the fine-tuned release.

## Citation

If you use this dataset, please cite:

```bibtex
@misc{{physiomotion-synthetic-baseline,
  title  = {{PhysioMotion-Synthetic-Baseline: Rehabilitation Motion Dataset via MotionGPT3}},
  author = {{Brandão, Lucas Dantas}},
  year   = {{2026}},
  howpublished = {{\\url{{https://huggingface.co/datasets/{repo_id}}}}},
  note   = {{Undergraduate research project, UFES}}
}}
```

And the upstream works:

```bibtex
@article{{motiongpt3,
  title   = {{MotionGPT3: Human Motion as a Second Modality}},
  journal = {{arXiv:2506.24086}},
  year    = {{2025}}
}}

@article{{kimore,
  title   = {{KIMORE: Kinematic Assessment of Movement and Clinical Scores
             for Remote Monitoring of Physical Rehabilitation}},
  author  = {{Capecci, M. et al.}},
  journal = {{IEEE Transactions on Neural Systems and Rehabilitation Engineering}},
  year    = {{2019}}
}}
```

## License

Released under **CC-BY-NC-4.0**. You may share and adapt the material with attribution,
for non-commercial purposes only.
"""

    return yaml_frontmatter + "\n\n" + body


def prepare_upload_dir(source_dir: Path, staging_dir: Path, repo_id: str):
    """Copy/prepare files into staging_dir for upload."""
    staging_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata
    metadata_path = source_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {source_dir}")
    with open(metadata_path) as f:
        metadata = json.load(f)

    # Load evaluation (optional)
    evaluation = None
    eval_path = source_dir / "evaluation_results.json"
    if eval_path.exists():
        with open(eval_path) as f:
            evaluation = json.load(f)

    # Exercise distribution
    exercises = {}
    for m in metadata:
        ex = m.get("exercise", "unknown")
        exercises[ex] = exercises.get(ex, 0) + 1

    # Copy motions and features directories
    for sub in ("motions", "features"):
        src = source_dir / sub
        dst = staging_dir / sub
        if src.exists():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            print(f"  Copied {sub}/ ({len(list(dst.iterdir()))} files)")

    # Copy metadata and evaluation
    shutil.copy2(metadata_path, staging_dir / "metadata.json")
    if evaluation:
        shutil.copy2(eval_path, staging_dir / "evaluation_results.json")

    # Copy prompts file if exists
    prompts_src = source_dir / "rehab_prompts.txt"
    if prompts_src.exists():
        shutil.copy2(prompts_src, staging_dir / "prompts.txt")

    # Write LICENSE
    (staging_dir / "LICENSE").write_text(LICENSE_CC_BY_NC_4)

    # Write dataset card
    card = build_dataset_card(
        repo_id=repo_id,
        metadata=metadata,
        evaluation=evaluation,
        num_samples=len(metadata),
        exercises=exercises,
    )
    (staging_dir / "README.md").write_text(card)

    print(f"  Dataset card written ({len(card)} chars)")
    print(f"  Total samples: {len(metadata)}")
    print(f"  Exercises: {exercises}")


def main():
    parser = argparse.ArgumentParser(description="Upload synthetic dataset to HuggingFace Hub")
    parser.add_argument("--repo_id", required=True,
                        help="HF repo id, e.g. 'username/dataset-name'")
    parser.add_argument("--source_dir", default="data/synthetic",
                        help="Local synthetic data directory")
    parser.add_argument("--staging_dir", default="data/hf_staging",
                        help="Temporary directory for upload staging")
    parser.add_argument("--create_repo", action="store_true",
                        help="Create the HF repo if it doesn't exist")
    parser.add_argument("--private", action="store_true",
                        help="Create repo as private (default: public)")
    parser.add_argument("--token", default=None,
                        help="HF token (default: reads from env or ~/.cache/huggingface)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Prepare staging dir but skip upload")
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    staging_dir = Path(args.staging_dir)

    print(f"Preparing upload staging at: {staging_dir}")
    prepare_upload_dir(source_dir, staging_dir, args.repo_id)

    if args.dry_run:
        print(f"\n[dry run] Staging ready. Would upload to: {args.repo_id}")
        return

    api = HfApi(token=args.token)

    if args.create_repo:
        print(f"\nCreating (or reusing) repo: {args.repo_id}")
        create_repo(
            repo_id=args.repo_id,
            repo_type="dataset",
            private=args.private,
            exist_ok=True,
            token=args.token,
        )

    print(f"\nUploading folder to huggingface.co/datasets/{args.repo_id} ...")
    api.upload_folder(
        folder_path=str(staging_dir),
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message="Upload PhysioMotion-Synthetic-Baseline v1",
    )

    print(f"\n✅ Done!")
    print(f"   https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
