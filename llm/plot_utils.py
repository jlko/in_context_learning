"""Utils and imports for eval in notebook."""
import os
import pickle
from collections.abc import MutableMapping

import numpy as np
import torch

import wandb


api = wandb.Api()
api.entity = os.environ['WANDB_API_ENTITY']


metrics = ['accuracy', 'log_lik', 'entropy']
metrics_dict = dict(zip(
    metrics,
    [r'Accuracy $\uparrow$', r'Log Likelihood $\uparrow$', r'Entropy', ]
))


def flatten_dict(dictionary, parent_key='', separator='_'):
    """Flatten dictionay."""
    items = []
    for key, value in dictionary.items():
        new_key = parent_key + separator + str(key) if parent_key else str(key)
        if isinstance(value, MutableMapping):
            items.extend(flatten_dict(
                value, new_key, separator=separator).items())
        else:
            items.append((new_key, value))
    return dict(items)


def wandb_restore(runid, filenames='results.pkl'):
    files_dir = 'wandb_restored'
    os.system(f'mkdir -p {files_dir}')

    if not isinstance(filenames, list):
        filenames = [filenames]
        single_out = True
    else:
        single_out = False

    outs = {}

    for filename in filenames:
        run = api.run(f'llm/{runid}')
        os.system(f'rm -rf {files_dir}/{filename}')
        run.file(filename).download(
            root=files_dir, replace=True, exist_ok=False)
        try:
            with open(f'{files_dir}/{filename}', 'rb') as f:
                outs[filename] = pickle.load(f)
        except Exception as exception:   # pylint: disable=broad-exception-caught
            print(f'Failed for {runid} with exception {exception}.')
            outs[filename] = {}

    if single_out:
        return outs[filenames[0]]
    else:
        return outs


def get_values(
        results, all_classes_sorted, class_names, model, label_setup, value,
        average=True, smooth_n=5, choose_logits=None, invert_logits=False):

    if choose_logits is None:
        choose_logits = label_setup

    yis = []
    repetitions = list(results[model]['results'].keys())

    for rep in repetitions:
        exp = results[model]['results'][rep][label_setup]
        log_liks = exp['all_extra_probs'][0]
        poss = exp['all_extra_pos'][0] - 1
        idxs = [all_classes_sorted.index(j) for j in class_names[choose_logits]]

        if invert_logits:
            idxs = [idxs[(i + 1) % len(idxs)] for i in range(len(idxs))]

        tmp = torch.nn.functional.log_softmax(log_liks[poss][:-1, idxs].float(), -1)
        labels = exp['train_labels']

        if value == 'log_lik':
            # get class names that we care about in this instance
            # first need to sort
            yi = tmp[range(len(tmp)), labels]
        elif value == 'accuracy':
            yi = torch.argmax(tmp, -1).numpy() == labels
        elif value == 'entropy':
            yi = - (torch.exp(tmp) * tmp).sum(-1)
        else:
            raise
        yis.append(np.array(yi))

    yis = np.stack(yis, 0)

    if not average:
        return yis
    else:
        yis = np.mean(yis, 0)
        if smooth_n == 1:
            return yis
        else:
            return np.convolve(yis, np.ones(smooth_n)/smooth_n, mode='valid')
