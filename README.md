<div align="center">

# [Categorical Flow Maps](https://arxiv.org/abs/2602.12233)

_[Daan Roos*](https://mrroose.github.io/), [Oscar Davis*](https://olsdavis.github.io), [Floor Eijkelboom*](https://flooreijkelboom.github.io/),<br> [Michael Bronstein](https://www.cs.ox.ac.uk/people/michael.bronstein/), [Max Welling](https://amlab.science.uva.nl/people/MaxWelling/), [İsmail İlkan Ceylan](https://www.cs.ox.ac.uk/people/ismaililkan.ceylan/), [Luca Ambrogioni](https://www.artcogsys.com/team/luca), [Jan-Willem van de Meent](https://jwvdm.github.io/)_

Official implementation of the graph experiments. :rocket:

[![arXiv](https://img.shields.io/badge/arXiv-2602.12233-red.svg)](https://arxiv.org/abs/2602.12233)

<img src="res/overview.png" width="80%">
</div>

## :question: About
This repository contains the code for the graph/molecule experiments from the Categorical Flow Maps paper.
For text experiments, refer to [olsdavis/semicat](https://github.com/olsdavis/semicat/).

- Training and experiment entrypoint: [`main.py`](main.py)
- Batch/epoch training logic (velocity + distillation objectives): [`train_map.py`](train_map.py)
- Graph transformer model: [`models/transformer.py`](models/transformer.py)
- Loss implementations (instantaneous velocity + Naive/CSD/ECLD): [`loss_functions.py`](loss_functions.py)
- Graph sampling code (Euler + flow-map rollout): [`generation.py`](generation.py)

## :gear: Running the code
CLI entrypoint: `python main.py`

1. Install/sync dependencies:
```sh
uv sync
```

2. Run experiments:
- QM9 run:
```sh
uv run python main.py --task qm9_wo_H --velocity_loss kld --distill_objective csd
```
- Quick smoke test:
```sh
uv run python main.py --task qm9_wo_H --dataset_size tiny --epochs 10 --gen_interval 10 --ckpt_interval 10
```
- ZINC run:
```sh
uv run python main.py --task zinc --velocity_loss kld --distill_objective ecld
```
- Naive flow map baseline:
```sh
uv run python main.py --task qm9_wo_H --velocity_loss mse --distill_objective mse
```
For the naive baseline, set both `--velocity_loss` and `--distill_objective` to `mse`.

3. Optional logging:
- Add `--log` to any run command to enable Weights & Biases logging.

4. Resume from checkpoint:
```sh
uv run python main.py --resume_ckpt <checkpoint_name>.pt ...
```

## :bar_chart: Data
- On first run, datasets are downloaded automatically via PyG (`QM9`/`ZINC`) into `data/`.
- Processed cache files are stored as:
  - `data/{task}_{split}_graphs_full.pt`
  - `data/{task}_{split}_smiles_full.pt`
- Dataset/task selectors:
  - `--task`: `qm9_wo_H` or `zinc`

Output/data interfaces:
- `data/` for dataset and processed cache artifacts
- `checkpoints/` for model checkpoints
- `wandb/` for run metadata when logging is enabled

## :mag: Evaluation and logging
- Periodic generation/evaluation is controlled by `--gen_interval`.
- Euler sampler is evaluated at steps `[1, 2, 5, 10, 50]`.
- Flow-map sampler is evaluated at steps `[1, 2, 5, 10]` when distillation is enabled (`--distill_objective != none`).
- Distillation objective choices: `none|mse|csd|ecld`.
- Reported metrics include validity, uniqueness, novelty, and FCD.

## :blue_book: Citation
To cite the paper or the code, please use the following:
```bibtex
@misc{roos2026categoricalflowmaps,
    title={Categorical Flow Maps}, 
    author={Daan Roos and Oscar Davis and Floor Eijkelboom and Michael Bronstein and Max Welling and İsmail İlkan Ceylan and Luca Ambrogioni and Jan-Willem van de Meent},
    year={2026},
    eprint={2602.12233},
    archivePrefix={arXiv},
    primaryClass={cs.LG},
    url={https://arxiv.org/abs/2602.12233}, 
}
```
