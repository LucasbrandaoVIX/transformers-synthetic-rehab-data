# Synthetic Rehabilitation Data with Transformers

Geração de dados sintéticos de movimento de reabilitação usando MotionGPT3.

Projeto de Iniciação Científica (IC) que utiliza o framework MotionGPT3 para gerar movimentos de reabilitação realistas a partir de descrições textuais, com fine-tuning em dados do KIMORE dataset.

## Estrutura do Projeto

```
.
├── MotionGPT3/                  # Framework MotionGPT3 (submodule)
│   ├── configs/
│   │   ├── rehab_finetune.yaml  # Config de fine-tuning para reabilitação
│   │   └── ...
│   └── prepare/
│       └── instructions/
│           └── template_rehab_instructions.json  # Templates de instrução
├── scripts/
│   ├── convert_kimore.py        # Conversão KIMORE -> formato HumanML3D
│   ├── generate_rehab_texts.py  # Geração de anotações textuais
│   ├── generate_synthetic.py    # Geração de dados sintéticos
│   └── evaluate_synthetic.py    # Avaliação de qualidade biomecânica
├── data/
│   ├── kimore/raw/              # Dados brutos KIMORE (download manual)
│   ├── rehab/                   # Dados convertidos (formato HumanML3D)
│   └── synthetic/               # Dados sintéticos gerados
├── setup.sh                     # Script de setup automatizado
└── README.md
```

## Quick Start

### 1. Setup do Ambiente

```bash
bash setup.sh
conda activate mgpt
```

### 2. Download do KIMORE Dataset

Baixe o KIMORE dataset e coloque os dados em `data/kimore/raw/`:
- [KIMORE Dataset](https://vrai.dii.univpm.it/content/kimore-dataset)

Estrutura esperada:
```
data/kimore/raw/
├── CG/          # Grupo controle
│   ├── Subject1/
│   │   ├── Es1/JointPosition.csv
│   │   ├── Es2/JointPosition.csv
│   │   └── ...
│   └── ...
└── GP1/         # Grupo patológico
    └── ...
```

### 3. Converter Dados para Formato HumanML3D

```bash
# Converter skeleton data do KIMORE
python scripts/convert_kimore.py --kimore_root data/kimore/raw --output_root data/rehab

# Gerar anotações textuais
python scripts/generate_rehab_texts.py --data_root data/rehab
```

### 4. Fine-tuning do MotionGPT3

```bash
cd MotionGPT3

# Fine-tuning com dados de reabilitação
python -m train --cfg configs/rehab_finetune.yaml --nodebug
```

### 5. Gerar Dados Sintéticos

```bash
# Com modelo pré-treinado (sem fine-tuning)
python scripts/generate_synthetic.py --num_per_exercise 20 --output_dir data/synthetic

# Com modelo fine-tuned
python scripts/generate_synthetic.py \
    --num_per_exercise 50 \
    --config configs/rehab_finetune.yaml \
    --output_dir data/synthetic_finetuned
```

### 6. Avaliar Qualidade

```bash
# Avaliar dados sintéticos
python scripts/evaluate_synthetic.py --synthetic_dir data/synthetic/motions

# Comparar com dados reais
python scripts/evaluate_synthetic.py \
    --synthetic_dir data/synthetic/motions \
    --real_dir data/rehab/joints \
    --output evaluation_results.json
```

## Exercícios KIMORE

| ID  | Exercício                                  | Foco                    |
|-----|-------------------------------------------|-------------------------|
| Es1 | Elevação lateral dos braços (plano frontal)| Ombro                  |
| Es2 | Flexão dos braços com cotovelos no quadril | Cotovelo                |
| Es3 | Rotação do tronco (sentado)                | Coluna torácica         |
| Es4 | Rotação pélvica (em pé)                    | Coluna lombar           |
| Es5 | Agachamento                                | Membros inferiores      |

## Pipeline

```
KIMORE (Kinect 25 joints, 30fps)
    → Mapeamento para SMPL (22 joints)
    → Resample para 20fps
    → Extração de features 263D (HumanML3D)
    → Anotações textuais
    → Fine-tuning MotionGPT3
    → Geração text-to-motion
    → Avaliação biomecânica
```

## Referências

- [MotionGPT3](https://arxiv.org/abs/2506.24086) - Human Motion as a Second Modality
- [KIMORE Dataset](https://www.researchgate.net/publication/333791841) - Kinematic Assessment of Movement
- [HumanML3D](https://github.com/EricGuo5513/HumanML3D) - Text-Motion Dataset
