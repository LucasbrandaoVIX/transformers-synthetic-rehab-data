# Makefile for Synthetic Rehabilitation Data Pipeline
# Usage: make <target>

PYTHON = python
MGPT_DIR = MotionGPT3
DATA_DIR = data
KIMORE_RAW = $(DATA_DIR)/kimore/raw
REHAB_DIR = $(DATA_DIR)/rehab
SYNTHETIC_DIR = $(DATA_DIR)/synthetic

.PHONY: help setup convert texts finetune generate evaluate upload-hf clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup: ## Setup conda environment and download dependencies
	bash setup.sh

convert: ## Convert KIMORE dataset to HumanML3D format
	$(PYTHON) scripts/convert_kimore.py \
		--kimore_root $(KIMORE_RAW) \
		--output_root $(REHAB_DIR)

texts: ## Generate text annotations for rehabilitation exercises
	$(PYTHON) scripts/generate_rehab_texts.py \
		--data_root $(REHAB_DIR)

prepare: convert texts ## Run full data preparation (convert + texts)

finetune: ## Fine-tune MotionGPT3 on rehabilitation data
	cd $(MGPT_DIR) && $(PYTHON) -m train \
		--cfg configs/rehab_finetune.yaml --nodebug

generate: ## Generate synthetic rehabilitation data (pre-trained model)
	$(PYTHON) scripts/generate_synthetic.py \
		--num_per_exercise 20 \
		--output_dir $(SYNTHETIC_DIR)

generate-finetuned: ## Generate synthetic data with fine-tuned model
	$(PYTHON) scripts/generate_synthetic.py \
		--num_per_exercise 50 \
		--config configs/rehab_finetune.yaml \
		--output_dir $(SYNTHETIC_DIR)_finetuned

evaluate: ## Evaluate synthetic data quality
	$(PYTHON) scripts/evaluate_synthetic.py \
		--synthetic_dir $(SYNTHETIC_DIR)/motions \
		--output $(SYNTHETIC_DIR)/evaluation.json

evaluate-compare: ## Evaluate and compare with real data
	$(PYTHON) scripts/evaluate_synthetic.py \
		--synthetic_dir $(SYNTHETIC_DIR)/motions \
		--real_dir $(REHAB_DIR)/joints \
		--output $(SYNTHETIC_DIR)/evaluation_results.json

upload-hf: ## Upload synthetic dataset to HuggingFace Hub (requires HF login)
	$(PYTHON) scripts/upload_to_hf.py \
		--repo_id lucasbrandao/PhysioMotion-Synthetic-Baseline \
		--source_dir $(SYNTHETIC_DIR) \
		--create_repo

demo: ## Run MotionGPT3 web demo
	cd $(MGPT_DIR) && $(PYTHON) app.py

test-demo: ## Run a quick text-to-motion demo
	cd $(MGPT_DIR) && $(PYTHON) demo.py \
		--cfg ./configs/test.yaml \
		--example ./assets/texts/t2m.txt

clean: ## Remove generated data and temporary files
	rm -rf $(REHAB_DIR)/tmp
	rm -rf $(SYNTHETIC_DIR)
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Full pipeline
all: prepare finetune generate-finetuned evaluate-compare ## Run full pipeline
