PYTHON ?= python
GPU_ENV = CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$(or $(CUDA_VISIBLE_DEVICES),0)

.PHONY: setup test lint preprocess smoke train-128 train-128-mask train-256-mask experiments

setup:
	pip install -r requirements.txt && pip install -e .

test:
	CUDA_VISIBLE_DEVICES="" $(PYTHON) -m pytest -q

lint:
	ruff check synthmri scripts tests

preprocess:
	$(PYTHON) scripts/preprocess.py --config configs/ldm128_maskcond.yaml
	$(PYTHON) scripts/preprocess.py --config configs/ldm256_maskcond.yaml

smoke:
	$(GPU_ENV) $(PYTHON) scripts/train.py --config configs/smoke.yaml
	$(GPU_ENV) $(PYTHON) scripts/sample.py --run runs/smoke --num_images 16 --steps 10
	$(GPU_ENV) $(PYTHON) scripts/evaluate.py --run runs/smoke --samples runs/smoke/samples_ddim10_cfg2_seed0 --max_real 64 --skip_fid

train-128:
	$(GPU_ENV) $(PYTHON) scripts/train.py --config configs/ldm128_uncond.yaml

train-128-mask:
	$(GPU_ENV) $(PYTHON) scripts/train.py --config configs/ldm128_maskcond.yaml

train-256-mask:
	$(GPU_ENV) $(PYTHON) scripts/train.py --config configs/ldm256_maskcond.yaml

experiments:
	bash scripts/run_experiments.sh
