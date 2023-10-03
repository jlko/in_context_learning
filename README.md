# In-Context Learning Learns Label Relationships but Is Not Conventional Learning

This repository contains the code necessary to reproduce the results of ['In-Context Learning Learns Label Relationships but Is Not Conventional Learning'](https://arxiv.org/abs/2307.12375).


## Setup

We recommend you set up a conda environment like so:

```
conda-env update -f environment.yaml
conda activate llm_unc
```

## Reproducing the Experiments

We give a full list of commands necessary to execute all main experiments across all models and tasks in [runs.md](runs.md).

As an example, the commands below compute ICL training curves for LLaMa-2-70B on SST-2 with (1) default labels, (2) randomized labels, (3) replacement labels, (4) dynamically changing labels.

```
# 1. Default Labels
export CLASS_NAMES=DEFAULT
python change_labels.py --debug=False --model.name=Llama-2-70b --note=random --dataset.name=sst2  --dataset.config.num_prompt_examples=140 --n_repeats=500

# 2. Random Labels
export CLASS_NAMES=DEFAULT
python change_labels.py --debug=False --model.name=Llama-2-70b --note=default --dataset.name=sst2 --dataset.config.num_prompt_examples=140 --n_repeats=500 --dataset.config.random_labels=True

# 3. Replacement Labels
export CLASS_NAMES=ALL
python change_labels.py --debug=False --model.name=Llama-2-70b --note=change_labels --dataset.name=sst2 --dataset.config.num_prompt_examples=140 --n_repeats=100

# 4. Dynamic Label Flip
## Flip Half-Way
export CLASS_NAMES=DEFAULT_AND_FLIP
python change_labels.py --debug=False --model.name=Llama-2-70b --note=centered_flip_labels_both_directions --dataset.name=sst2 --dataset.config.num_prompt_examples=140 --dataset.config.flip_labels_after=60 --n_repeats=500
## Alternating
export CLASS_NAMES=DEFAULT_ONLY
python change_labels.py --debug=False --model.name=Llama-2-70b --note=flip_after_1 --dataset.name=sst2  --dataset.config.num_prompt_examples=140 --n_repeats=500 --dataset.config.flip_labels_after=1 --dataset.config.flip_labels_multi=True
```

In [example_evaluation.ipynb](example_evaluation.ipynb) we further showcase how to evaluate the above runs.


### Details

* Our experiments rely on [wandb](https://wandb.ai/) for logging results and currently do not support execution without it. Your wandb entity should be set in the `WANDB_API_ENTITY` environment variable.

* To access LLaMa-2 models, you need to set the `HUGGING_FACE_HUB_TOKEN` environment variable with your API token.

## Citation

If you find this codebase useful to a project, we would appreciate a citation:

```
@article{kossen2023context,
  title={In-Context Learning Learns Label Relationships but Is Not Conventional Learning},
  author={Kossen, Jannik and Gal, Yarin and Rainforth, Tom},
  journal={arXiv:2307.12375},
  year={2023}
}
```
