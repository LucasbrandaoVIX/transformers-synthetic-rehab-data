"""
Evaluate quality of synthetic rehabilitation motion data.

Computes biomechanical plausibility metrics:
1. Joint angle ranges (are they anatomically valid?)
2. Motion smoothness (jerk metric)
3. Joint velocity statistics
4. Comparison with real data distributions (if available)

Usage:
    python scripts/evaluate_synthetic.py --synthetic_dir data/synthetic/motions
    python scripts/evaluate_synthetic.py --synthetic_dir data/synthetic/motions --real_dir data/rehab/joints
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict


# SMPL joint names for reference
SMPL_JOINTS = [
    "pelvis", "l_hip", "r_hip", "spine1", "l_knee", "r_knee",
    "spine2", "l_ankle", "r_ankle", "spine3", "l_foot", "r_foot",
    "neck", "l_collar", "r_collar", "head", "l_shoulder", "r_shoulder",
    "l_elbow", "r_elbow", "l_wrist", "r_wrist"
]

# Anatomically plausible joint angle ranges (degrees)
# Based on typical ROM (Range of Motion) values
JOINT_ANGLE_LIMITS = {
    "shoulder_abduction": (0, 180),     # l/r shoulder
    "elbow_flexion": (0, 150),          # l/r elbow
    "knee_flexion": (0, 140),           # l/r knee
    "hip_flexion": (-30, 120),          # l/r hip
    "trunk_rotation": (-90, 90),        # spine
}


def compute_joint_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
    """
    Compute angle at joint p2 formed by segments p1-p2 and p2-p3.

    Args:
        p1, p2, p3: Joint positions, shape (nframes, 3) or (3,)

    Returns:
        angles: Angle in degrees, shape (nframes,) or scalar
    """
    v1 = p1 - p2
    v2 = p3 - p2

    # Normalize
    v1_norm = np.linalg.norm(v1, axis=-1, keepdims=True)
    v2_norm = np.linalg.norm(v2, axis=-1, keepdims=True)

    v1_norm = np.where(v1_norm < 1e-8, 1.0, v1_norm)
    v2_norm = np.where(v2_norm < 1e-8, 1.0, v2_norm)

    v1 = v1 / v1_norm
    v2 = v2 / v2_norm

    cos_angle = np.clip(np.sum(v1 * v2, axis=-1), -1.0, 1.0)
    angles = np.degrees(np.arccos(cos_angle))
    return angles


def compute_jerk(positions: np.ndarray, fps: float = 20.0) -> float:
    """
    Compute normalized jerk (smoothness metric) for a motion sequence.
    Lower jerk = smoother motion.

    Args:
        positions: (nframes, n_joints, 3)
        fps: Frames per second

    Returns:
        mean_jerk: Average normalized jerk across all joints
    """
    dt = 1.0 / fps

    # Velocity (nframes-1, joints, 3)
    vel = np.diff(positions, axis=0) / dt

    # Acceleration (nframes-2, joints, 3)
    acc = np.diff(vel, axis=0) / dt

    # Jerk (nframes-3, joints, 3)
    jerk = np.diff(acc, axis=0) / dt

    # Compute normalized jerk
    jerk_magnitude = np.linalg.norm(jerk, axis=-1)  # (nframes-3, joints)
    mean_jerk = jerk_magnitude.mean()

    return float(mean_jerk)


def compute_velocity_stats(positions: np.ndarray, fps: float = 20.0) -> dict:
    """Compute velocity statistics for each joint."""
    dt = 1.0 / fps
    vel = np.diff(positions, axis=0) / dt
    vel_magnitude = np.linalg.norm(vel, axis=-1)  # (nframes-1, joints)

    stats = {}
    for j, name in enumerate(SMPL_JOINTS):
        if j >= vel_magnitude.shape[1]:
            break
        stats[name] = {
            "mean_vel": float(vel_magnitude[:, j].mean()),
            "max_vel": float(vel_magnitude[:, j].max()),
            "std_vel": float(vel_magnitude[:, j].std()),
        }

    return stats


def check_biomechanical_plausibility(positions: np.ndarray) -> dict:
    """
    Check if joint angles are within anatomically plausible ranges.

    Returns dict with percentage of frames within valid range per joint angle.
    """
    results = {}

    nframes = positions.shape[0]

    # Left elbow angle (shoulder -> elbow -> wrist)
    angles = compute_joint_angle(
        positions[:, 16],  # l_shoulder
        positions[:, 18],  # l_elbow
        positions[:, 20],  # l_wrist
    )
    valid = np.logical_and(angles >= 0, angles <= 180)
    results["l_elbow_flexion"] = {
        "valid_pct": float(valid.mean() * 100),
        "mean_angle": float(angles.mean()),
        "min_angle": float(angles.min()),
        "max_angle": float(angles.max()),
    }

    # Right elbow angle
    angles = compute_joint_angle(
        positions[:, 17],  # r_shoulder
        positions[:, 19],  # r_elbow
        positions[:, 21],  # r_wrist
    )
    valid = np.logical_and(angles >= 0, angles <= 180)
    results["r_elbow_flexion"] = {
        "valid_pct": float(valid.mean() * 100),
        "mean_angle": float(angles.mean()),
        "min_angle": float(angles.min()),
        "max_angle": float(angles.max()),
    }

    # Left knee angle (hip -> knee -> ankle)
    angles = compute_joint_angle(
        positions[:, 1],   # l_hip
        positions[:, 4],   # l_knee
        positions[:, 7],   # l_ankle
    )
    valid = np.logical_and(angles >= 0, angles <= 180)
    results["l_knee_flexion"] = {
        "valid_pct": float(valid.mean() * 100),
        "mean_angle": float(angles.mean()),
        "min_angle": float(angles.min()),
        "max_angle": float(angles.max()),
    }

    # Right knee angle
    angles = compute_joint_angle(
        positions[:, 2],   # r_hip
        positions[:, 5],   # r_knee
        positions[:, 8],   # r_ankle
    )
    valid = np.logical_and(angles >= 0, angles <= 180)
    results["r_knee_flexion"] = {
        "valid_pct": float(valid.mean() * 100),
        "mean_angle": float(angles.mean()),
        "min_angle": float(angles.min()),
        "max_angle": float(angles.max()),
    }

    return results


def evaluate_motion_file(npy_path: str, fps: float = 20.0) -> dict:
    """Evaluate a single motion .npy file."""
    motion = np.load(npy_path)

    # Handle different shapes
    if motion.ndim == 2 and motion.shape[1] == 263:
        # This is a feature vector, not raw joints - skip biomechanical eval
        return {
            "filename": os.path.basename(npy_path),
            "format": "features_263d",
            "n_frames": motion.shape[0],
            "feature_mean": float(motion.mean()),
            "feature_std": float(motion.std()),
        }

    if motion.ndim == 2:
        # Try to reshape as (nframes, 22, 3)
        if motion.shape[1] == 66:
            motion = motion.reshape(-1, 22, 3)
        else:
            return {
                "filename": os.path.basename(npy_path),
                "format": f"unknown_{motion.shape}",
                "n_frames": motion.shape[0],
            }

    if motion.shape[1] != 22 or motion.shape[2] != 3:
        return {
            "filename": os.path.basename(npy_path),
            "format": f"unexpected_shape_{motion.shape}",
            "n_frames": motion.shape[0],
        }

    result = {
        "filename": os.path.basename(npy_path),
        "format": "joints_22x3",
        "n_frames": motion.shape[0],
        "duration_sec": motion.shape[0] / fps,
        "jerk": compute_jerk(motion, fps),
        "biomechanical": check_biomechanical_plausibility(motion),
        "velocity_stats": compute_velocity_stats(motion, fps),
    }

    return result


def evaluate_dataset(motion_dir: str, fps: float = 20.0) -> dict:
    """Evaluate all motion files in a directory."""
    motion_dir = Path(motion_dir)
    npy_files = sorted(motion_dir.glob("*.npy"))

    if not npy_files:
        print(f"No .npy files found in {motion_dir}")
        return {}

    print(f"Evaluating {len(npy_files)} motion files...")

    results = []
    for npy_file in npy_files:
        result = evaluate_motion_file(str(npy_file), fps)
        results.append(result)

    # Aggregate statistics
    jerks = [r["jerk"] for r in results if "jerk" in r]
    durations = [r["duration_sec"] for r in results if "duration_sec" in r]

    summary = {
        "total_files": len(results),
        "files_with_joints": sum(1 for r in results if r.get("format") == "joints_22x3"),
        "files_with_features": sum(1 for r in results if r.get("format") == "features_263d"),
    }

    if jerks:
        summary["jerk_stats"] = {
            "mean": float(np.mean(jerks)),
            "std": float(np.std(jerks)),
            "min": float(np.min(jerks)),
            "max": float(np.max(jerks)),
        }

    if durations:
        summary["duration_stats"] = {
            "mean": float(np.mean(durations)),
            "std": float(np.std(durations)),
            "min": float(np.min(durations)),
            "max": float(np.max(durations)),
        }

    # Aggregate biomechanical plausibility
    bio_results = defaultdict(list)
    for r in results:
        if "biomechanical" in r:
            for joint, data in r["biomechanical"].items():
                bio_results[joint].append(data["valid_pct"])

    if bio_results:
        summary["biomechanical_validity"] = {}
        for joint, pcts in bio_results.items():
            summary["biomechanical_validity"][joint] = {
                "mean_valid_pct": float(np.mean(pcts)),
                "min_valid_pct": float(np.min(pcts)),
            }

    return {
        "summary": summary,
        "per_file": results,
    }


def compare_distributions(synthetic_dir: str, real_dir: str, fps: float = 20.0):
    """Compare velocity and angle distributions between synthetic and real data."""
    print("\n--- Distribution Comparison ---")

    syn_files = sorted(Path(synthetic_dir).glob("*.npy"))
    real_files = sorted(Path(real_dir).glob("*.npy"))

    def collect_velocities(files):
        all_vels = []
        for f in files:
            motion = np.load(str(f))
            if motion.ndim == 3 and motion.shape[1] == 22:
                vel = np.diff(motion, axis=0) * fps
                vel_mag = np.linalg.norm(vel, axis=-1).mean(axis=1)
                all_vels.extend(vel_mag.tolist())
        return np.array(all_vels)

    syn_vels = collect_velocities(syn_files)
    real_vels = collect_velocities(real_files)

    if len(syn_vels) > 0 and len(real_vels) > 0:
        print(f"  Synthetic velocities: mean={syn_vels.mean():.4f}, std={syn_vels.std():.4f}")
        print(f"  Real velocities:      mean={real_vels.mean():.4f}, std={real_vels.std():.4f}")

        # KL divergence approximation via histogram
        bins = np.linspace(0, max(syn_vels.max(), real_vels.max()), 50)
        syn_hist, _ = np.histogram(syn_vels, bins=bins, density=True)
        real_hist, _ = np.histogram(real_vels, bins=bins, density=True)

        # Add small epsilon to avoid log(0)
        eps = 1e-10
        syn_hist = syn_hist + eps
        real_hist = real_hist + eps

        kl_div = np.sum(real_hist * np.log(real_hist / syn_hist)) * (bins[1] - bins[0])
        print(f"  KL divergence (real||synthetic): {kl_div:.4f}")
    else:
        print("  Insufficient data for comparison")


def main():
    parser = argparse.ArgumentParser(description="Evaluate synthetic rehabilitation motion data")
    parser.add_argument("--synthetic_dir", type=str, required=True,
                        help="Directory with synthetic motion .npy files")
    parser.add_argument("--real_dir", type=str, default=None,
                        help="Directory with real motion .npy files for comparison")
    parser.add_argument("--fps", type=float, default=20.0,
                        help="Frames per second (default: 20)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file for results")

    args = parser.parse_args()

    # Evaluate synthetic data
    results = evaluate_dataset(args.synthetic_dir, args.fps)

    if not results:
        return

    # Print summary
    summary = results["summary"]
    print(f"\n=== Evaluation Summary ===")
    print(f"Total files: {summary['total_files']}")
    print(f"Joint format: {summary.get('files_with_joints', 0)}")
    print(f"Feature format: {summary.get('files_with_features', 0)}")

    if "jerk_stats" in summary:
        js = summary["jerk_stats"]
        print(f"\nMotion Smoothness (jerk, lower=smoother):")
        print(f"  Mean: {js['mean']:.2f}, Std: {js['std']:.2f}")
        print(f"  Min: {js['min']:.2f}, Max: {js['max']:.2f}")

    if "duration_stats" in summary:
        ds = summary["duration_stats"]
        print(f"\nDuration (seconds):")
        print(f"  Mean: {ds['mean']:.1f}, Std: {ds['std']:.1f}")

    if "biomechanical_validity" in summary:
        print(f"\nBiomechanical Validity (% frames in valid range):")
        for joint, data in summary["biomechanical_validity"].items():
            print(f"  {joint}: {data['mean_valid_pct']:.1f}% (min: {data['min_valid_pct']:.1f}%)")

    # Compare with real data if provided
    if args.real_dir:
        compare_distributions(args.synthetic_dir, args.real_dir, args.fps)

    # Save results
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
