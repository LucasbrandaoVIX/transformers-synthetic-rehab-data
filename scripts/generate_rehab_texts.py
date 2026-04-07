"""
Generate text annotations for KIMORE rehabilitation exercises in HumanML3D format.

KIMORE exercises:
    Es1: Lateral elevation of arms on the frontal plane
    Es2: Flexion of the arms with elbows at the hips
    Es3: Trunk rotation while seated
    Es4: Pelvis rotations on the transversal plane (standing)
    Es5: Squatting

Text format (HumanML3D):
    Each .txt file contains one or more lines with:
    caption#tokenized_caption#start_time#end_time

    For full-sequence descriptions, start_time and end_time are 0.0.
"""

import os
import json
import shutil
import random
import argparse
from pathlib import Path

# Rehabilitation exercise descriptions (English, multiple variations per exercise)
# Each exercise has: base descriptions, quality modifiers, and clinical context

EXERCISE_DESCRIPTIONS = {
    "Es1": {
        "name": "lateral arm elevation",
        "descriptions": [
            "a person raises both arms laterally to shoulder height on the frontal plane",
            "a person performs lateral arm raises from the sides of the body to shoulder level",
            "a person lifts arms sideways up to shoulder height and then lowers them back down",
            "a person elevates both arms laterally in the frontal plane as a rehabilitation exercise",
            "a standing person raises their arms out to the sides until they are parallel to the ground",
            "a person performs shoulder abduction by raising arms laterally to the sides",
            "a person slowly raises both arms sideways to shoulder height and returns to starting position",
            "a person stands and performs bilateral lateral arm raises for shoulder rehabilitation",
            "a person lifts both arms away from the body in a lateral raising motion",
            "lateral arm elevation exercise where a person raises arms to the sides repeatedly",
        ],
        "clinical": [
            "shoulder abduction exercise for range of motion recovery",
            "rehabilitation exercise targeting deltoid and supraspinatus muscles",
            "lateral arm raise exercise for shoulder mobility rehabilitation",
        ]
    },
    "Es2": {
        "name": "arm flexion with elbows at hips",
        "descriptions": [
            "a person flexes both forearms upward while keeping elbows pinned at the hips",
            "a person performs bicep curls with elbows fixed at the sides of the body",
            "a person bends both arms at the elbow while maintaining elbows at hip level",
            "a person raises forearms by flexing at the elbow with upper arms held against the body",
            "a person performs elbow flexion exercise keeping upper arms stationary at the sides",
            "a standing person curls both forearms upward while elbows remain at the hips",
            "a person performs bilateral elbow flexion with arms held close to the torso",
            "a person flexes arms at the elbows repeatedly with upper arms fixed at hip height",
            "elbow flexion rehabilitation exercise with elbows anchored at the hips",
            "a person bends and straightens both arms at the elbows while standing upright",
        ],
        "clinical": [
            "elbow flexion exercise for upper limb rehabilitation",
            "bicep strengthening exercise with controlled elbow movement",
            "forearm flexion exercise for arm mobility recovery",
        ]
    },
    "Es3": {
        "name": "seated trunk rotation",
        "descriptions": [
            "a seated person rotates their trunk to the left and right",
            "a person sitting on a chair performs trunk rotations side to side",
            "a seated person twists their upper body alternately to the left and right",
            "a person performs seated torso rotations for spinal mobility",
            "a person sitting down rotates their trunk from side to side",
            "a seated person performs controlled rotational movements of the trunk",
            "seated trunk rotation exercise where the person twists the upper body left and right",
            "a person sits and rotates their torso to each side as a rehabilitation exercise",
            "a seated person performs alternating trunk rotations for core rehabilitation",
            "a person sitting performs slow controlled rotations of the upper body",
        ],
        "clinical": [
            "trunk rotation exercise for spinal mobility and core rehabilitation",
            "seated rotational exercise for low back pain rehabilitation",
            "thoracic rotation exercise for trunk mobility recovery",
        ]
    },
    "Es4": {
        "name": "standing pelvis rotation",
        "descriptions": [
            "a standing person rotates their pelvis on the transversal plane",
            "a person performs hip rotations while standing upright",
            "a standing person rotates their hips from side to side",
            "a person performs pelvic rotations in the horizontal plane while standing",
            "a standing person moves their pelvis in rotational movements",
            "a person rotates the pelvis left and right while maintaining upright posture",
            "pelvis rotation exercise performed while standing on both feet",
            "a standing person performs controlled pelvic rotations for rehabilitation",
            "a person performs transversal plane pelvis rotations while standing still",
            "a person standing performs alternating pelvic rotations to each side",
        ],
        "clinical": [
            "pelvic rotation exercise for lumbar spine rehabilitation",
            "standing hip rotation exercise for lower back mobility",
            "transversal pelvic movement exercise for core stability",
        ]
    },
    "Es5": {
        "name": "squatting",
        "descriptions": [
            "a person performs a squat by bending the knees and lowering the body",
            "a standing person squats down and then returns to standing position",
            "a person bends at the knees and hips to perform a squat exercise",
            "a person performs a controlled squat lowering the hips toward the ground",
            "a person squats by flexing the knees while keeping the back straight",
            "a standing person performs a rehabilitation squat with controlled movement",
            "a person lowers their body by bending the knees and then stands back up",
            "a person performs a partial squat for lower extremity rehabilitation",
            "squatting exercise where the person bends knees to lower and raise the body",
            "a person performs slow controlled squats for leg strengthening rehabilitation",
        ],
        "clinical": [
            "squat exercise for lower extremity strengthening and rehabilitation",
            "knee flexion-extension exercise through squatting motion",
            "functional squatting exercise for lower limb rehabilitation",
        ]
    }
}

# Quality/performance modifiers to increase data diversity
QUALITY_MODIFIERS = {
    "normal": [
        "",  # no modifier (correct execution)
    ],
    "slow": [
        "slowly and carefully",
        "at a slow controlled pace",
        "with slow deliberate movements",
    ],
    "limited_rom": [
        "with limited range of motion",
        "with reduced amplitude",
        "with restricted movement range",
    ],
    "compensatory": [
        "with slight compensatory movements",
        "with some body compensation",
        "with minor postural compensation",
    ]
}


def tokenize_simple(text: str) -> str:
    """Simple whitespace tokenization with POS-like tags (simplified)."""
    tokens = text.lower().strip().split()
    # Add simple verb/noun tags (approximate)
    tagged = []
    for t in tokens:
        clean = t.strip(".,;!?")
        if clean:
            tagged.append(f"{clean}/OTHER")
    return " ".join(tagged)


def generate_text_file(sample_id: str, exercise_id: str, group: str, output_dir: Path):
    """
    Generate a text annotation file for a given sample.

    Args:
        sample_id: Unique sample identifier
        exercise_id: Exercise type (Es1-Es5)
        group: Subject group (CG=control, GPx=pathological)
        output_dir: Output texts/ directory
    """
    if exercise_id not in EXERCISE_DESCRIPTIONS:
        print(f"  Warning: Unknown exercise {exercise_id} for {sample_id}")
        return

    ex_info = EXERCISE_DESCRIPTIONS[exercise_id]

    # Select descriptions based on group
    is_pathological = group.startswith("GP")

    lines = []

    # Pick 2-4 random descriptions
    n_descriptions = random.randint(2, 4)
    selected_descs = random.sample(ex_info["descriptions"], min(n_descriptions, len(ex_info["descriptions"])))

    for desc in selected_descs:
        # Add quality modifier for pathological groups
        if is_pathological and random.random() < 0.5:
            modifier_type = random.choice(["slow", "limited_rom", "compensatory"])
            modifier = random.choice(QUALITY_MODIFIERS[modifier_type])
            if modifier:
                desc = f"{desc} {modifier}"

        tokens = tokenize_simple(desc)
        # Format: caption#tokens#start_time#end_time
        line = f"{desc}#{tokens}#0.0#0.0"
        lines.append(line)

    # Add one clinical description
    clinical_desc = random.choice(ex_info["clinical"])
    tokens = tokenize_simple(clinical_desc)
    lines.append(f"{clinical_desc}#{tokens}#0.0#0.0")

    # Write text file
    output_path = output_dir / f"{sample_id}.txt"
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def generate_all_texts(data_root: str):
    """
    Generate text annotations for all converted KIMORE samples.

    Reads sample IDs from train.txt, val.txt, test.txt splits.
    """
    data_root = Path(data_root)
    texts_dir = data_root / "texts"
    texts_dir.mkdir(parents=True, exist_ok=True)

    # Read all sample IDs from splits
    all_ids = []
    for split in ["train", "val", "test"]:
        split_file = data_root / f"{split}.txt"
        if split_file.exists():
            with open(split_file) as f:
                ids = [line.strip() for line in f if line.strip()]
                all_ids.extend(ids)

    if not all_ids:
        print("No sample IDs found. Run convert_kimore.py first.")
        return

    print(f"Generating text annotations for {len(all_ids)} samples...")

    random.seed(42)  # Reproducibility

    for sample_id in all_ids:
        # Parse exercise and group from sample_id (format: GROUP_SUBJECT_EXERCISE)
        parts = sample_id.split("_")
        if len(parts) >= 3:
            group = parts[0]
            exercise = parts[-1]  # Last part is exercise (Es1-Es5)
        else:
            print(f"  Warning: Cannot parse sample_id '{sample_id}', skipping")
            continue

        generate_text_file(sample_id, exercise, group, texts_dir)

    print(f"Generated text files in {texts_dir}")
    print(f"Total: {len(all_ids)} text annotation files")

    # Copy instruction templates to data_root (required by MotionGPT3 training)
    copy_instruction_templates(data_root)


def copy_instruction_templates(data_root: str):
    """
    Copy instruction template JSONs to the data directory.

    MotionGPT3 expects instruction templates in the data_root directory:
    - template_t2m_instructions.json (for lm_finetune stage)
    - template_t2m_pretrain.json (for lm_pretrain stage)
    """
    data_root = Path(data_root)
    proj_root = Path(__file__).resolve().parent.parent
    mgpt_instructions = proj_root / "MotionGPT3" / "prepare" / "instructions"

    # Copy rehabilitation-specific templates
    rehab_template = mgpt_instructions / "template_rehab_instructions.json"
    if rehab_template.exists():
        # Use rehab template as the t2m instructions
        shutil.copy(str(rehab_template), str(data_root / "template_t2m_instructions.json"))
        print(f"Copied rehab instruction template -> template_t2m_instructions.json")

        # Also create pretrain version
        shutil.copy(str(rehab_template), str(data_root / "template_t2m_pretrain.json"))
        print(f"Copied rehab instruction template -> template_t2m_pretrain.json")
    else:
        # Fallback: copy original templates
        for template_name in ["template_t2m_pretrain.json", "template_instructions.json"]:
            src = mgpt_instructions / template_name
            if src.exists():
                dst_name = template_name
                if template_name == "template_instructions.json":
                    dst_name = "template_t2m_instructions.json"
                shutil.copy(str(src), str(data_root / dst_name))
                print(f"Copied {template_name} -> {dst_name}")

    # Also copy the 'all' variants for flexibility
    for template_name in ["template_pretrain.json", "template_witht2t_instructions.json"]:
        src = mgpt_instructions / template_name
        if src.exists():
            shutil.copy(str(src), str(data_root / template_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text annotations for KIMORE rehab data")
    parser.add_argument("--data_root", type=str, default="data/rehab",
                        help="Path to converted rehab data directory")

    args = parser.parse_args()
    generate_all_texts(args.data_root)
