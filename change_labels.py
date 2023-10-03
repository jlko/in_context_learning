"""Run Fewshot In-Context-Learning Experiment."""
import os
import sys
import logging

import pickle
import numpy as np
import wandb

# Local imports.
from llm import models
from llm import utils
from llm import plot_utils
from llm.datasets import datasets
from llm.config import get_config
from llm import config_utils


def main(args):
    """Run experiment."""

    def default_assemble(data):
        return data.get_fewshot_prompt

    def get_with_new_assemble(
            *,
            assemble_raw=default_assemble,
            assemble_with_label=False,
            data_kwargs=None,
            random_seed=None,
    ):
        """Get ICL predictions from LLM for a single input."""

        # Initialize dataset given config and random seed.
        rng = np.random.default_rng(random_seed)
        data_kwargs = {**args.dataset.config, **data_kwargs}
        data = datasets[args.dataset.name](rng=rng, **data_kwargs)
        assemble_with_repeat = assemble_raw(data)
        logging.info('class names %s', data.class_names)

        # Get indices of label tokens.
        try:
            logit_idxs, _ = utils.get_logit_idxs(model, data, skip=True)
        except:
            return {}, False
        logging.info(logit_idxs)

        # Always compute calibration temperature, even if not used.
        calibrate, prior_log_probs = utils.get_calibrate(
            model, logit_idxs, data=data)
        if not args.model.calibrate:
            calibrate = None

        # Execute forward pass.
        out = utils.get_log_probs(
            model, data, logit_idxs, assemble=assemble_with_repeat,
            calibrate=calibrate, assemble_with_label=assemble_with_label,
            print_first_prompt=True, all_logit_idxs=all_logit_idxs,
            extra_pos=f'\n{data.answer_string}:'
            )
        all_log_probs, uncalibrated_log_probs, labels, all_extra_probs, all_extra_pos = out

        # Return model predictions.
        return dict(
            prior_log_probs=prior_log_probs,
            all_log_probs=all_log_probs,
            uncalibrated_log_probs=uncalibrated_log_probs,
            labels=labels,
            train_idxs=data.fewshot_indices,
            train_labels=[
                data.train_data[i][data.label_name] for i in data.fewshot_indices],
            logit_idxs=logit_idxs,
            class_names=data.class_names,
            all_extra_probs=all_extra_probs,
            all_extra_pos=all_extra_pos), True

    # <Initialization>
    user = os.getenv('USER')
    slurm_jobid = os.getenv('SLURM_JOB_ID')
    scratch_dir = os.getenv('SCRATCH_DIR', '~')
    filename = 'results.pkl'
    path = f'{scratch_dir}/{user}/llm'

    os.system(f'mkdir -p {path}')
    wandb.init(
        project='llm' if not args.debug else 'llm_debug',
        dir=path,
        config=plot_utils.flatten_dict(args.to_dict()),
        notes=f'args.note: {args.note}, slurm_jobid: {slurm_jobid}',
    )

    # Initialize model.
    model = models.HuggingfaceModel(args.model.name, **args.model.config)

    # Grab indices for label tokens.
    class_names = datasets[args.dataset.name].class_name_variants
    all_classes = datasets[args.dataset.name].all_classes
    all_logit_idxs, success_classes = utils.get_logit_idxs(
        model, all_classes, skip=True)
    out_dict = {
        'results': {},
        'setup': {'all_logit_idxs': all_logit_idxs, 'all_classes': success_classes},
    }
    # </Initialization>.

    # Iterate over different random dataset subsets.
    for repetition in range(args.n_repeats):
        out_dict['results'][repetition] = {}

        # Iterate over different label names.
        for cname, cconfig in class_names.items():

            logging.info(80 * '#')
            logging.info('Repetition: %d, name: %s', repetition, cname)

            # Get predictions from forward pass.
            res, success = get_with_new_assemble(
                data_kwargs={'class_names': cconfig},
                random_seed=args.random_seed + repetition,
            )

            if success:
                out_dict['results'][repetition][cname] = res

        with open(f'{wandb.run.dir}/{filename}', 'wb') as file:
            pickle.dump(out_dict, file)
        wandb.save(filename)


if __name__ == '__main__':
    utils.setup_logger()
    margs = get_config()
    os.system('mkdir -p log')

    # Update default config with command line args.
    for arg in sys.argv[1:]:
        # remove '--', split into name and value
        update_dict = config_utils.get_nested_dict(*arg[2:].split('='))
        margs.update(update_dict)

    margs['experiment_name'] = 'change-labels'

    main(margs)
