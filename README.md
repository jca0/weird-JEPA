# my-JEPA

A JEPA-based model for robotic control tasks.

## Setup

```bash
uv venv --python=3.10
source .venv/bin/activate
uv pip install stable-worldmodel[train,env]
```

## Data

Datasets are downloaded automatically on first use. To pre-download:

```python
import stable_worldmodel as swm
swm.data.load_dataset("pusht_expert_train.h5")
```

Available datasets: `pusht_expert_train.h5`, `tworoom`, `ogb_cube`.

## Train

```bash
python train.py
```

This uses the default config (`config/train/lewm.yaml` with `data: pusht`). Override settings via Hydra:

```bash
python train.py trainer.max_epochs=50 loader.batch_size=128 data=tworoom
```

Checkpoints are saved to `.stable_worldmodel/checkpoints/`.

## Evaluate

```bash
python eval.py policy=<checkpoint_name>.pt
```

The `.pt` file and its `config.json` must be in `.stable_worldmodel/checkpoints/`. To run a random baseline:

```bash
python eval.py
```

To evaluate with a different environment config:

```bash
python eval.py --config-name=cube policy=<checkpoint_name>.pt
```

Results are saved to `.stable_worldmodel/<env>_results.txt`.
