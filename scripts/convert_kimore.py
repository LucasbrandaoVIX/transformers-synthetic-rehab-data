"""
Convert KIMORE dataset to HumanML3D format for MotionGPT3 fine-tuning.

KIMORE dataset contains skeleton data from Kinect v2 (25 joints) that needs to be
mapped to SMPL 22-joint format and converted to 263-dimensional features.

Supports two KIMORE formats:
  1. EGCN pre-processed (.skeleton files):
     skeleton/G{group}S{subject}E{exercise}R{repetition}.skeleton
  2. Original KIMORE (JointPosition.csv in hierarchical dirs):
     CG/Subject/Es1/JointPosition.csv

Output format (HumanML3D compatible):
    data/rehab/
        joints/         - Raw 3D joint positions (nframes, 22, 3)
        new_joint_vecs/ - 263D feature vectors (nframes, 263)
        texts/          - Text descriptions
        Mean.npy, Std.npy
        train.txt, val.txt, test.txt
"""

import os
import sys
import glob
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import torch

# Add MotionGPT3 to path for using its processing utilities
PROJ_ROOT = Path(__file__).resolve().parent.parent
MGPT_ROOT = PROJ_ROOT / "MotionGPT3"
sys.path.insert(0, str(MGPT_ROOT))

try:
    from motGPT.data.humanml.utils.paramUtil import (
        t2m_raw_offsets, t2m_kinematic_chain
    )
    from motGPT.data.humanml.common.skeleton import Skeleton
    from motGPT.data.humanml.common.quaternion import *
    HAS_MGPT = True
except ImportError:
    print("Warning: Could not import MotionGPT3 modules.")
    print("Feature extraction (263D) will not be available.")
    print("Only raw joint positions will be saved.")
    HAS_MGPT = False


# KIMORE Kinect v2 joint indices (25 joints) -> SMPL 22 joint mapping
# Kinect joints: https://learn.microsoft.com/en-us/azure/kinect-dk/body-joints
# SMPL joints: pelvis, l_hip, r_hip, spine1, l_knee, r_knee, spine2, l_ankle, r_ankle,
#              spine3, l_foot, r_foot, neck, l_collar, r_collar, head, l_shoulder,
#              r_shoulder, l_elbow, r_elbow, l_wrist, r_wrist

KINECT_TO_SMPL = {
    0: 0,    # SpineBase -> pelvis
    12: 1,   # HipLeft -> l_hip
    16: 2,   # HipRight -> r_hip
    1: 3,    # SpineMid -> spine1
    13: 4,   # KneeLeft -> l_knee
    17: 5,   # KneeRight -> r_knee
    20: 6,   # SpineShoulder -> spine2
    14: 7,   # AnkleLeft -> l_ankle
    18: 8,   # AnkleRight -> r_ankle
    1: 9,    # SpineMid -> spine3 (approximate)
    15: 10,  # FootLeft -> l_foot
    19: 11,  # FootRight -> r_foot
    2: 12,   # Neck -> neck (using SpineShoulder area)
    4: 13,   # ShoulderLeft -> l_collar
    8: 14,   # ShoulderRight -> r_collar
    3: 15,   # Head -> head
    4: 16,   # ShoulderLeft -> l_shoulder
    8: 17,   # ShoulderRight -> r_shoulder
    5: 18,   # ElbowLeft -> l_elbow
    9: 19,   # ElbowRight -> r_elbow
    6: 20,   # WristLeft -> l_wrist
    10: 21,  # WristRight -> r_wrist
}

# Direct index-based mapping (Kinect index for each SMPL joint 0-21)
KINECT_IDX_FOR_SMPL = [
    0,   # 0: pelvis <- SpineBase
    12,  # 1: l_hip <- HipLeft
    16,  # 2: r_hip <- HipRight
    1,   # 3: spine1 <- SpineMid
    13,  # 4: l_knee <- KneeLeft
    17,  # 5: r_knee <- KneeRight
    20,  # 6: spine2 <- SpineShoulder
    14,  # 7: l_ankle <- AnkleLeft
    18,  # 8: r_ankle <- AnkleRight
    1,   # 9: spine3 <- SpineMid (interpolated later)
    15,  # 10: l_foot <- FootLeft
    19,  # 11: r_foot <- FootRight
    2,   # 12: neck <- Neck (Kinect joint 2 or 20)
    4,   # 13: l_collar <- ShoulderLeft
    8,   # 14: r_collar <- ShoulderRight
    3,   # 15: head <- Head
    4,   # 16: l_shoulder <- ShoulderLeft
    8,   # 17: r_shoulder <- ShoulderRight
    5,   # 18: l_elbow <- ElbowLeft
    9,   # 19: r_elbow <- ElbowRight
    6,   # 20: l_wrist <- WristLeft
    10,  # 21: r_wrist <- WristRight
]


def load_skeleton_file(filepath: str) -> np.ndarray:
    """
    Load NTU-style .skeleton file and return joint positions.

    Format per frame:
        body_count (1)
        body_info (tracking id line)
        n_joints (25)
        25 lines: x y z depth_x qw qx qy qz

    Returns:
        positions: np.ndarray of shape (nframes, 25, 3)
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    idx = 0
    n_frames = int(lines[idx].strip())
    idx += 1

    all_positions = []

    for _ in range(n_frames):
        body_count = int(lines[idx].strip())
        idx += 1

        for b in range(body_count):
            # Body info line (tracking id, etc.)
            idx += 1
            n_joints = int(lines[idx].strip())
            idx += 1

            joints = np.zeros((n_joints, 3))
            for j in range(n_joints):
                vals = lines[idx].strip().split()
                joints[j] = [float(vals[0]), float(vals[1]), float(vals[2])]
                idx += 1

            # Only use first body
            if b == 0:
                all_positions.append(joints)

    return np.array(all_positions)


def load_kimore_csv(csv_path: str) -> np.ndarray:
    """
    Load KIMORE JointPosition.csv and return joint positions.

    Returns:
        positions: np.ndarray of shape (nframes, 25, 3)
    """
    try:
        data = np.loadtxt(csv_path, delimiter=',')
    except ValueError:
        data = np.loadtxt(csv_path, delimiter=';')

    if data.ndim == 1:
        data = data.reshape(1, -1)

    n_cols = data.shape[1]
    if n_cols == 75:
        positions = data.reshape(-1, 25, 3)
    elif n_cols == 100:
        data_reshaped = data.reshape(-1, 25, 4)
        positions = data_reshaped[:, :, :3]
    else:
        raise ValueError(f"Unexpected number of columns: {n_cols} in {csv_path}")

    return positions


def kinect_to_smpl_joints(kinect_positions: np.ndarray) -> np.ndarray:
    """
    Map Kinect v2 25-joint skeleton to SMPL 22-joint skeleton.

    Args:
        kinect_positions: (nframes, 25, 3) Kinect joint positions

    Returns:
        smpl_positions: (nframes, 22, 3) SMPL joint positions
    """
    nframes = kinect_positions.shape[0]
    smpl_positions = np.zeros((nframes, 22, 3))

    for smpl_idx, kinect_idx in enumerate(KINECT_IDX_FOR_SMPL):
        smpl_positions[:, smpl_idx] = kinect_positions[:, kinect_idx]

    # Interpolate spine3 as midpoint between spine2 and neck
    smpl_positions[:, 9] = (smpl_positions[:, 6] + smpl_positions[:, 12]) / 2.0

    # Collar bones: interpolate between neck and shoulder
    smpl_positions[:, 13] = 0.6 * smpl_positions[:, 12] + 0.4 * smpl_positions[:, 16]  # l_collar
    smpl_positions[:, 14] = 0.6 * smpl_positions[:, 12] + 0.4 * smpl_positions[:, 17]  # r_collar

    return smpl_positions


def resample_motion(positions: np.ndarray, src_fps: float, tgt_fps: float = 20.0) -> np.ndarray:
    """Resample motion from source FPS to target FPS (HumanML3D uses 20 FPS)."""
    if abs(src_fps - tgt_fps) < 0.1:
        return positions

    nframes_src = len(positions)
    duration = nframes_src / src_fps
    nframes_tgt = int(duration * tgt_fps)

    if nframes_tgt < 2:
        return positions[:2]

    src_times = np.linspace(0, duration, nframes_src)
    tgt_times = np.linspace(0, duration, nframes_tgt)

    # Interpolate each joint coordinate
    result = np.zeros((nframes_tgt, positions.shape[1], positions.shape[2]))
    for j in range(positions.shape[1]):
        for c in range(positions.shape[2]):
            result[:, j, c] = np.interp(tgt_times, src_times, positions[:, j, c])

    return result


def compute_features_from_positions(positions: np.ndarray) -> np.ndarray:
    """
    Compute 263D HumanML3D features from 3D joint positions.

    Uses the same feature extraction pipeline as HumanML3D:
    - Root data: 4D (angular vel, linear vel x/z, root height)
    - Joint positions (RIC): 21*3 = 63D
    - Joint rotations (6D continuous): 21*6 = 126D
    - Joint velocities: 22*3 = 66D
    - Foot contacts: 4D
    Total: 263D

    Args:
        positions: (nframes, 22, 3) joint positions

    Returns:
        features: (nframes-1, 263) feature vector, or None if MotionGPT3 not available
    """
    if not HAS_MGPT:
        return None

    # Use only body kinematic chain (22 joints, no hands)
    n_raw_offsets = torch.from_numpy(t2m_raw_offsets)
    kinematic_chain = t2m_kinematic_chain

    # Face direction indices: r_hip, l_hip, sdr_r, sdr_l
    face_joint_indx = [2, 1, 17, 16]
    fid_r = [8, 11]  # r_ankle, r_foot
    fid_l = [7, 10]  # l_ankle, l_foot
    feet_thre = 0.002

    # Build skeleton for this sample
    src_skel = Skeleton(n_raw_offsets, kinematic_chain, 'cpu')

    # Get target offsets from the first frame of this sample
    tgt_offsets = src_skel.get_offsets_joints(
        torch.from_numpy(positions[0]).float()
    )

    # --- Uniform skeleton ---
    positions = uniform_skeleton_22j(src_skel, positions, tgt_offsets, kinematic_chain, face_joint_indx)

    # --- Put on floor ---
    floor_height = positions.min(axis=0).min(axis=0)[1]
    positions[:, :, 1] -= floor_height

    # --- XZ at origin ---
    root_pos_init = positions[0]
    root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1])
    positions = positions - root_pose_init_xz

    # --- Face Z+ direction ---
    r_hip, l_hip, sdr_r, sdr_l = face_joint_indx
    across1 = root_pos_init[r_hip] - root_pos_init[l_hip]
    across2 = root_pos_init[sdr_r] - root_pos_init[sdr_l]
    across = across1 + across2
    across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]

    forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
    forward_init = forward_init / np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis]

    target = np.array([[0, 0, 1]])
    root_quat_init = qbetween_np(forward_init, target)
    root_quat_init = np.ones(positions.shape[:-1] + (4,)) * root_quat_init
    positions = qrot_np(root_quat_init, positions)

    global_positions = positions.copy()

    # --- Foot contacts ---
    def foot_detect(positions, thres):
        velfactor = np.array([thres, thres])
        feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
        feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
        feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
        feet_l = ((feet_l_x + feet_l_y + feet_l_z) < velfactor).astype(np.float64)

        feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
        feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
        feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
        feet_r = ((feet_r_x + feet_r_y + feet_r_z) < velfactor).astype(np.float64)
        return feet_l, feet_r

    feet_l, feet_r = foot_detect(positions, feet_thre)

    # --- Continuous 6D rotation parameters ---
    skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
    quat_params = skel.inverse_kinematics_np(positions, face_joint_indx, smooth_forward=True)
    cont_6d_params = quaternion_to_cont6d_np(quat_params)

    r_rot = quat_params[:, 0].copy()
    velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
    velocity = qrot_np(r_rot[1:], velocity)
    r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))

    # --- RIC (rotation invariant coordinates) ---
    positions[..., 0] -= positions[:, 0:1, 0]
    positions[..., 2] -= positions[:, 0:1, 2]
    positions = qrot_np(np.repeat(r_rot[:, None], positions.shape[1], axis=1), positions)

    # --- Assemble 263D feature vector ---
    root_y = positions[:, 0, 1:2]
    r_velocity_y = np.arcsin(r_velocity[:, 2:3])
    l_velocity = velocity[:, [0, 2]]
    root_data = np.concatenate([r_velocity_y, l_velocity, root_y[:-1]], axis=-1)  # 4D

    rot_data = cont_6d_params[:, 1:].reshape(len(cont_6d_params), -1)  # 21*6 = 126D
    ric_data = positions[:, 1:].reshape(len(positions), -1)  # 21*3 = 63D

    local_vel = qrot_np(
        np.repeat(r_rot[:-1, None], global_positions.shape[1], axis=1),
        global_positions[1:] - global_positions[:-1]
    )
    local_vel = local_vel.reshape(len(local_vel), -1)  # 22*3 = 66D

    data = np.concatenate([
        root_data,       # 4
        ric_data[:-1],   # 63
        rot_data[:-1],   # 126
        local_vel,       # 66
        feet_l,          # 2
        feet_r,          # 2
    ], axis=-1)  # Total: 263

    return data


def uniform_skeleton_22j(src_skel, positions, target_offset, kinematic_chain, face_joint_indx):
    """
    Normalize skeleton to target bone lengths (22-joint body only).
    Simplified version of uniform_skeleton from motion_process.py.
    """
    l_idx1, l_idx2 = 5, 8  # r_knee, r_ankle

    src_skel_obj = Skeleton(torch.from_numpy(t2m_raw_offsets), kinematic_chain, 'cpu')
    src_offset = src_skel_obj.get_offsets_joints(torch.from_numpy(positions[0]).float())
    src_offset = src_offset.numpy()
    tgt_offset = target_offset.numpy()

    src_leg_len = np.abs(src_offset[l_idx1]).max() + np.abs(src_offset[l_idx2]).max()
    tgt_leg_len = np.abs(tgt_offset[l_idx1]).max() + np.abs(tgt_offset[l_idx2]).max()

    if src_leg_len < 1e-6:
        return positions

    scale_rt = tgt_leg_len / src_leg_len
    src_root_pos = positions[:, 0]
    tgt_root_pos = src_root_pos * scale_rt

    quat_params = src_skel_obj.inverse_kinematics_np(positions, face_joint_indx)
    src_skel_obj.set_offset(target_offset)
    new_joints = src_skel_obj.forward_kinematics_np(quat_params, tgt_root_pos)

    return new_joints


def find_kimore_files(kimore_root: str) -> list:
    """
    Find all KIMORE skeleton files (supports both .skeleton and .csv formats).

    Returns list of tuples: (file_path, sample_id, exercise_id, group, format)

    EGCN .skeleton naming: G{group}S{subject}E{exercise}R{repetition}.skeleton
    KIMORE exercise mapping: E002-E006 -> Es1-Es5 (EGCN uses different IDs)
    """
    results = []
    kimore_root = Path(kimore_root)

    # Try EGCN .skeleton format first
    skeleton_dir = kimore_root / "skeleton"
    if not skeleton_dir.exists():
        skeleton_dir = kimore_root  # Files might be directly in root

    skeleton_files = sorted(skeleton_dir.glob("*.skeleton"))

    if skeleton_files:
        # EGCN format: G{group}S{subject}E{exercise}R{repetition}.skeleton
        # KIMORE exercises in EGCN: E002=Es1, E003=Es2, E004=Es3, E005=Es4, E006=Es5
        # (E007-E009 may be additional exercises or other movements)
        EGCN_TO_EXERCISE = {
            "E002": "Es1",  # Lateral arm elevation
            "E003": "Es2",  # Arm flexion with elbows at hips
            "E004": "Es3",  # Trunk rotation (seated)
            "E005": "Es4",  # Pelvis rotation (standing)
            "E006": "Es5",  # Squatting
            # E007-E009: other movements, map generically
            "E007": "Es1",
            "E008": "Es2",
            "E009": "Es3",
        }

        for skel_path in skeleton_files:
            fname = skel_path.stem  # e.g., G001S001E002R001
            try:
                # Parse filename
                parts = fname.split('S')
                group_str = parts[0]  # G001
                rest = parts[1]  # 001E002R001
                subj_and_rest = rest.split('E')
                subject_str = subj_and_rest[0]  # 001
                ex_and_rep = subj_and_rest[1].split('R')
                exercise_egcn = f"E{ex_and_rep[0]}"  # E002
                repetition = ex_and_rep[1]  # 001

                group_num = int(group_str[1:])
                # Group mapping: G001=healthy, G002+=pathological
                group = "CG" if group_num == 1 else f"GP{group_num - 1}"

                exercise = EGCN_TO_EXERCISE.get(exercise_egcn, exercise_egcn)
                sample_id = f"{group}_S{subject_str}_{exercise}_R{repetition}"

                results.append((str(skel_path), sample_id, exercise, group, "skeleton"))
            except (IndexError, ValueError) as e:
                print(f"  Warning: Could not parse filename '{fname}': {e}")
                continue

        print(f"Found {len(results)} .skeleton files (EGCN format)")
        return results

    # Fallback: Original KIMORE CSV format
    for csv_path in sorted(kimore_root.rglob("JointPosition.csv")):
        parts = csv_path.relative_to(kimore_root).parts
        if len(parts) >= 3:
            group = parts[0]
            subject = parts[1]
            exercise = parts[2]
            sample_id = f"{group}_{subject}_{exercise}"
            results.append((str(csv_path), sample_id, exercise, group, "csv"))

    if results:
        print(f"Found {len(results)} JointPosition.csv files (original KIMORE format)")

    return results


def convert_kimore_dataset(
    kimore_root: str,
    output_root: str,
    src_fps: float = 30.0,
    tgt_fps: float = 20.0,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
):
    """
    Convert entire KIMORE dataset to HumanML3D format.

    Args:
        kimore_root: Path to KIMORE raw data
        output_root: Path to output directory (data/rehab/)
        src_fps: KIMORE recording FPS (Kinect v2 = 30 FPS)
        tgt_fps: Target FPS for HumanML3D (20 FPS)
        train_ratio: Fraction of data for training
        val_ratio: Fraction of data for validation
    """
    output_root = Path(output_root)
    joints_dir = output_root / "joints"
    vecs_dir = output_root / "new_joint_vecs"
    texts_dir = output_root / "texts"

    for d in [joints_dir, vecs_dir, texts_dir, output_root / "tmp"]:
        d.mkdir(parents=True, exist_ok=True)

    # Find all KIMORE files
    files = find_kimore_files(kimore_root)
    if not files:
        print(f"No JointPosition.csv files found in {kimore_root}")
        print("Expected structure: <kimore_root>/<Group>/<Subject>/<Exercise>/JointPosition.csv")
        return

    print(f"Found {len(files)} motion files in KIMORE dataset")

    successful_ids = []
    all_features = []

    for file_path, sample_id, exercise, group, fmt in tqdm(files, desc="Converting KIMORE"):
        try:
            # 1. Load Kinect skeleton
            if fmt == "skeleton":
                kinect_positions = load_skeleton_file(file_path)
            else:
                kinect_positions = load_kimore_csv(file_path)

            if len(kinect_positions) < 10:
                print(f"  Skipping {sample_id}: too short ({len(kinect_positions)} frames)")
                continue

            # 2. Map to SMPL joints
            smpl_positions = kinect_to_smpl_joints(kinect_positions)

            # 3. Resample to target FPS
            smpl_positions = resample_motion(smpl_positions, src_fps, tgt_fps)

            if len(smpl_positions) < 20:  # min_motion_length
                print(f"  Skipping {sample_id}: too short after resampling ({len(smpl_positions)} frames)")
                continue

            if len(smpl_positions) > 200:  # max_motion_length
                smpl_positions = smpl_positions[:200]

            # 4. Save raw joint positions
            np.save(str(joints_dir / f"{sample_id}.npy"), smpl_positions)

            # 5. Compute 263D features
            features = compute_features_from_positions(smpl_positions)

            if features is None or len(features) < 20:
                print(f"  Skipping {sample_id}: feature extraction failed")
                continue

            np.save(str(vecs_dir / f"{sample_id}.npy"), features)
            all_features.append(features)
            successful_ids.append(sample_id)

        except Exception as e:
            print(f"  Error processing {sample_id}: {e}")
            continue

    if not successful_ids:
        print("No samples were successfully converted!")
        return

    print(f"\nSuccessfully converted {len(successful_ids)} / {len(files)} samples")

    # 6. Compute Mean and Std
    all_features_concat = np.concatenate(all_features, axis=0)
    mean = all_features_concat.mean(axis=0)
    std = all_features_concat.std(axis=0)
    std[std < 1e-6] = 1.0  # Avoid division by zero

    np.save(str(output_root / "Mean.npy"), mean)
    np.save(str(output_root / "Std.npy"), std)
    print(f"Saved Mean.npy and Std.npy (feature dim: {mean.shape[0]})")

    # 7. Create train/val/test splits
    np.random.seed(42)
    indices = np.random.permutation(len(successful_ids))
    n_train = int(len(indices) * train_ratio)
    n_val = int(len(indices) * val_ratio)

    train_ids = [successful_ids[i] for i in indices[:n_train]]
    val_ids = [successful_ids[i] for i in indices[n_train:n_train + n_val]]
    test_ids = [successful_ids[i] for i in indices[n_train + n_val:]]

    for split_name, split_ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        with open(str(output_root / f"{split_name}.txt"), "w") as f:
            for sid in split_ids:
                f.write(sid + "\n")
        print(f"  {split_name}: {len(split_ids)} samples")

    print(f"\nConversion complete! Output saved to {output_root}")
    print("Next step: Run generate_rehab_texts.py to create text annotations")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert KIMORE dataset to HumanML3D format")
    parser.add_argument("--kimore_root", type=str, default="data/kimore/raw",
                        help="Path to raw KIMORE data")
    parser.add_argument("--output_root", type=str, default="data/rehab",
                        help="Output directory for converted data")
    parser.add_argument("--src_fps", type=float, default=30.0,
                        help="KIMORE recording FPS (default: 30)")
    parser.add_argument("--tgt_fps", type=float, default=20.0,
                        help="Target FPS for HumanML3D (default: 20)")

    args = parser.parse_args()
    convert_kimore_dataset(args.kimore_root, args.output_root, args.src_fps, args.tgt_fps)
