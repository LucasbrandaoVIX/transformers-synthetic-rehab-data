"""
Generate synthetic rehabilitation movement data using fine-tuned MotionGPT3.

This script generates synthetic motion data from text descriptions of rehabilitation
exercises, producing .npy files with joint positions that can be used for:
- Data augmentation of rehabilitation datasets
- Training downstream models (exercise classification, quality assessment)
- Visualization and analysis

Usage:
    python scripts/generate_synthetic.py --num_samples 100 --output_dir data/synthetic
    python scripts/generate_synthetic.py --prompts_file prompts.txt --output_dir data/synthetic
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parent.parent
MGPT_ROOT = PROJ_ROOT / "MotionGPT3"
sys.path.insert(0, str(MGPT_ROOT))


# Rehabilitation exercise prompt templates for diverse generation
REHAB_PROMPTS = {
    "lateral_arm_elevation": [
        "a person raises both arms laterally to shoulder height",
        "a person performs lateral arm raises from sides to shoulder level",
        "a person slowly lifts arms sideways as a shoulder rehabilitation exercise",
        "a person elevates both arms laterally in the frontal plane",
        "a person performs shoulder abduction raising arms to the sides",
        "a person raises arms out to the sides and lowers them back down",
        "a person performs bilateral lateral arm raises for shoulder recovery",
        "a person lifts arms sideways up to shoulder height with controlled movement",
        "a person performs lateral arm elevation with limited range of motion",
        "a person slowly performs lateral arm raises with compensatory movement",
    ],
    "arm_flexion_elbows": [
        "a person flexes both forearms upward with elbows at the hips",
        "a person performs bicep curls with elbows fixed at the sides",
        "a person bends arms at the elbow keeping upper arms stationary",
        "a person raises forearms by flexing at the elbow with arms held close",
        "a person performs elbow flexion exercise with controlled movement",
        "a person curls both forearms upward while elbows remain at the hips",
        "a person performs bilateral elbow flexion for arm rehabilitation",
        "a person bends and straightens arms at the elbows while standing",
        "a person performs slow elbow flexion with reduced range of motion",
        "a person flexes forearms with slight compensatory body movement",
    ],
    "trunk_rotation": [
        "a seated person rotates their trunk to the left and right",
        "a person sitting performs trunk rotations side to side",
        "a seated person twists upper body alternately left and right",
        "a person performs seated torso rotations for spinal mobility",
        "a seated person performs controlled trunk rotation exercise",
        "a person sitting rotates their torso to each side slowly",
        "a seated person performs alternating trunk rotations",
        "a person performs slow seated trunk rotations for rehabilitation",
        "a seated person rotates trunk with limited range of motion",
        "a seated person performs trunk rotation with compensatory movement",
    ],
    "pelvis_rotation": [
        "a standing person rotates their pelvis from side to side",
        "a person performs hip rotations while standing upright",
        "a standing person rotates hips in the transversal plane",
        "a person performs pelvic rotations while maintaining upright posture",
        "a standing person performs controlled pelvic rotation exercise",
        "a person rotates the pelvis left and right while standing",
        "a person performs transversal plane pelvis rotations",
        "a standing person performs pelvic rotations for lumbar rehabilitation",
        "a standing person performs pelvis rotation with limited amplitude",
        "a person performs slow pelvic rotations with slight compensation",
    ],
    "squatting": [
        "a person performs a squat by bending the knees and lowering the body",
        "a standing person squats down and returns to standing position",
        "a person bends at the knees and hips to perform a squat",
        "a person performs a controlled squat lowering the hips",
        "a person squats by flexing the knees with back straight",
        "a person performs a rehabilitation squat with controlled movement",
        "a person lowers body by bending knees and then stands back up",
        "a person performs a partial squat for lower extremity rehabilitation",
        "a person performs slow controlled squats for leg strengthening",
        "a person performs a squat with reduced depth due to limited mobility",
    ],
}


def create_prompt_file(output_path: str, num_per_exercise: int = 20):
    """Create a text file with rehabilitation prompts for batch generation."""
    prompts = []
    for exercise_name, exercise_prompts in REHAB_PROMPTS.items():
        for i in range(num_per_exercise):
            prompt = exercise_prompts[i % len(exercise_prompts)]
            prompts.append(prompt)

    with open(output_path, "w") as f:
        for prompt in prompts:
            f.write(prompt + "\n")

    print(f"Created prompt file with {len(prompts)} prompts: {output_path}")
    return prompts


def generate_with_motiongpt3(
    prompts_file: str,
    config_path: str,
    output_dir: str,
    task: str = "t2m",
):
    """
    Generate synthetic motions using MotionGPT3 demo.py.

    This wraps the existing MotionGPT3 demo pipeline.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve prompts_file to absolute path (we cd into MGPT_ROOT)
    prompts_file_abs = str(Path(prompts_file).resolve())

    # Use MotionGPT3's demo.py for generation
    cmd = (
        f"cd {MGPT_ROOT} && "
        f"python demo.py "
        f"--cfg {config_path} "
        f"--example {prompts_file_abs} "
        f"--task {task}"
    )

    print(f"Running MotionGPT3 generation...")
    print(f"  Config: {config_path}")
    print(f"  Prompts: {prompts_file}")
    print(f"  Task: {task}")
    print(f"  Command: {cmd}")

    ret = os.system(cmd)

    if ret != 0:
        print(f"Error: Generation failed with return code {ret}")
        return False

    print(f"Generation complete!")
    return True


def organize_outputs(
    mgpt_output_dir: str,
    final_output_dir: str,
    prompts: list,
):
    """
    Organize generated motion files into a structured dataset.

    MotionGPT3 demo.py writes for each prompt:
      <idx>_out.npy        — joint positions, shape (1, nframes, 22, 3)
      <idx>_out_feats.npy  — HumanML3D features, shape (nframes, 263)
      <idx>_out.gif        — visualization

    Creates:
        final_output_dir/
            motions/          - (1, nframes, 22, 3) joint positions
            features/         - (nframes, 263) HumanML3D features
            visualizations/   - .gif files
            metadata.json     - Mapping of filenames to prompts and exercise types
    """
    import re
    import shutil

    mgpt_output_dir = Path(mgpt_output_dir)
    final_output_dir = Path(final_output_dir)
    motions_dir = final_output_dir / "motions"
    features_dir = final_output_dir / "features"
    viz_dir = final_output_dir / "visualizations"
    motions_dir.mkdir(parents=True, exist_ok=True)
    features_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    # Find all *_out.npy joint files and extract numeric prompt index
    joint_files = list(mgpt_output_dir.glob("*_out.npy"))
    idx_re = re.compile(r"^(\d+)_out\.npy$")

    indexed = []
    for f in joint_files:
        m = idx_re.match(f.name)
        if m:
            indexed.append((int(m.group(1)), f))

    # Sort numerically by prompt index
    indexed.sort(key=lambda x: x[0])

    if not indexed:
        print(f"No <idx>_out.npy files found in {mgpt_output_dir}")
        return

    metadata = []
    exercise_names = list(REHAB_PROMPTS.keys())
    prompts_per_exercise = (
        len(prompts) // len(exercise_names) if exercise_names else 1
    )

    for prompt_idx, joints_path in indexed:
        motion = np.load(str(joints_path))

        # Determine exercise type from prompt index position
        exercise_idx = min(prompt_idx // prompts_per_exercise, len(exercise_names) - 1)
        exercise_name = exercise_names[exercise_idx]

        # Output filename uses the original prompt index for reproducibility
        output_stem = f"synthetic_{exercise_name}_{prompt_idx:05d}"
        motion_out = motions_dir / f"{output_stem}.npy"
        np.save(str(motion_out), motion)

        # Companion features file
        feats_path = joints_path.with_name(f"{prompt_idx}_out_feats.npy")
        feats_shape = None
        if feats_path.exists():
            feats = np.load(str(feats_path))
            np.save(str(features_dir / f"{output_stem}.npy"), feats)
            feats_shape = list(feats.shape)

        # Companion gif
        gif_path = joints_path.with_name(f"{prompt_idx}_out.gif")
        if gif_path.exists():
            shutil.copy2(str(gif_path), str(viz_dir / f"{output_stem}.gif"))

        prompt_text = prompts[prompt_idx] if prompt_idx < len(prompts) else "unknown"

        # Number of frames: if shape is (1, T, 22, 3) take T, else take 0-dim
        if motion.ndim == 4:
            n_frames = int(motion.shape[1])
        else:
            n_frames = int(motion.shape[0])

        entry = {
            "filename": f"{output_stem}.npy",
            "exercise": exercise_name,
            "prompt": prompt_text,
            "n_frames": n_frames,
            "joints_shape": list(motion.shape),
        }
        if feats_shape is not None:
            entry["feats_shape"] = feats_shape
        metadata.append(entry)

    # Save metadata
    with open(str(final_output_dir / "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nOrganized {len(metadata)} synthetic motion files")
    print(f"  Output: {final_output_dir}")
    print(f"  Motions: {motions_dir}")
    print(f"  Metadata: {final_output_dir / 'metadata.json'}")

    # Print summary
    exercise_counts = {}
    for m in metadata:
        ex = m["exercise"]
        exercise_counts[ex] = exercise_counts.get(ex, 0) + 1

    print("\nSamples per exercise:")
    for ex, count in exercise_counts.items():
        print(f"  {ex}: {count}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic rehabilitation motion data with MotionGPT3"
    )
    parser.add_argument(
        "--num_per_exercise", type=int, default=20,
        help="Number of samples to generate per exercise type (default: 20)"
    )
    parser.add_argument(
        "--prompts_file", type=str, default=None,
        help="Path to custom prompts file (one prompt per line). If not provided, uses built-in rehab prompts."
    )
    parser.add_argument(
        "--config", type=str, default="configs/test.yaml",
        help="MotionGPT3 config file (default: configs/test.yaml for pre-trained, use configs/rehab_finetune.yaml for fine-tuned)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="data/synthetic",
        help="Output directory for synthetic data"
    )
    parser.add_argument(
        "--task", type=str, default="t2m",
        choices=["t2m", "pred", "inbetween"],
        help="Generation task (default: t2m)"
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Create prompts file
    if args.prompts_file is None:
        prompts_file = str(output_dir / "rehab_prompts.txt")
        prompts = create_prompt_file(prompts_file, args.num_per_exercise)
    else:
        prompts_file = args.prompts_file
        with open(prompts_file) as f:
            prompts = [line.strip() for line in f if line.strip()]

    # Step 2: Generate motions
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = str(MGPT_ROOT / config_path)

    success = generate_with_motiongpt3(
        prompts_file=prompts_file,
        config_path=config_path,
        output_dir=str(output_dir),
        task=args.task,
    )

    if not success:
        print("\nGeneration failed. Check the error messages above.")
        print("Make sure you have:")
        print("  1. Set up the environment (bash setup.sh)")
        print("  2. Downloaded the pre-trained model (checkpoints/motiongpt3.ckpt)")
        print("  3. Or fine-tuned a model (use --config configs/rehab_finetune.yaml)")
        return

    # Step 3: Organize outputs
    # MotionGPT3 demo.py writes to:
    #   MotionGPT3/results/motgpt/<run_name>/samples_<timestamp>/
    # We pick the most recently modified samples_* dir under results/.
    mgpt_results = MGPT_ROOT / "results"
    samples_dirs = sorted(
        mgpt_results.glob("**/samples_*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    ) if mgpt_results.exists() else []

    if samples_dirs:
        latest = samples_dirs[0]
        print(f"\nUsing latest samples dir: {latest}")
        organize_outputs(str(latest), str(output_dir), prompts)
    else:
        print(f"\nNote: Could not find any samples_* folder under {mgpt_results}")
        print("Check the config TEST.FOLDER setting for the actual output location.")


if __name__ == "__main__":
    main()
