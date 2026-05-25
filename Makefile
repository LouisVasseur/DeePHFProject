SEED ?= 43
SUBSETS = water alkanes qm7b_T gdb13_T

.PHONY: all features grid ablations figures clean

all: grid ablations figures

features:
	@for s in $(SUBSETS); do \
	  python prepare/prepare_mobml.py --subset $$s ; \
	  python prepare/compute_descriptors.py --subset $$s ; \
	done

grid:
	python scripts/train_grid.py --seed $(SEED) --resume

ablations:
	python scripts/ablate_edge_bias.py
	python scripts/ablate_edge_features.py --dataset alkanes
	python scripts/test_invariance.py
	python scripts/k_sweep.py

figures:
	python scripts/make_artifacts.py

clean:
	rm -rf figures tables
