import logging
import functools
import numpy as np
import torch

from llm.datasets import DEFAULT_STRINGS

# noqa: E501
# pylint: disable=C0116
# pylint: disable=E1101
# pylint: disable=C0301


def construct_zeroshot_input(sentence, prompt_top, prompt_bottom):
    return f"{prompt_top}'{sentence}'{prompt_bottom}"


def get_log_probs(
        model, data, logit_idxs, calibrate=None, indices=None,
        assemble=construct_zeroshot_input, split='test_data',
        assemble_with_label=False, print_first_prompt=False,
        all_logit_idxs=None, extra_pos=None):

    if indices is None:
        indices = data.test_indices

    if all_logit_idxs is not None:
        all_extra_probs = []
    if extra_pos is not None:
        all_extra_pos = []

    data_split = getattr(data, split)

    # Gather log probs of predictions over dataset.
    all_log_probs, labels = [], []
    for i, idx in enumerate(indices):
        datapoint = data_split[idx]
        sentence, label = data.get_feature(datapoint), datapoint[data.label_name]
        labels.append(label)

        if assemble_with_label:
            input_text = assemble(sentence, label)
        else:
            input_text = assemble(sentence)

        if print_first_prompt and i == 0:
            logging.info('First prompt: %s', input_text)

        out = model.get_token_prob(
            input_text, logit_idxs, extra_indices=all_logit_idxs,
            extra_pos=extra_pos)

        if all_logit_idxs is not None or extra_pos is not None:
            log_probs = out['log_probs']
        else:
            log_probs = out

        if all_logit_idxs is not None:
            all_extra_probs.append(out['extra_probs'])
        if extra_pos is not None:
            all_extra_pos.append(out['extra_pos'])

        all_log_probs.append(log_probs)
    all_log_probs = torch.stack(all_log_probs, 0)

    labels = np.array(labels)

    # First return value is always the one to be used, the second is for debugging.
    all_uncalibrated_log_probs = all_log_probs
    if calibrate:
        all_log_probs = calibrate(all_log_probs)

    if all_logit_idxs is not None:
        return (
            all_log_probs, all_uncalibrated_log_probs, labels,
            all_extra_probs, all_extra_pos)
    else:
        return all_log_probs, all_uncalibrated_log_probs, labels



def get_logit_idxs(model, data, skip=False):
    """Extract logit indices corresponding to label tokens."""

    if isinstance(data, list):
        class_names = data
    else:
        class_names = data.class_names
    # get tokenizer indices of class.
    if 'falcon' in model.model_name:
        # Prepend space to class name.
        class_to_tokens = {k: model.tokenizer(' ' + k)['input_ids'] for k in class_names}
    elif 'llama' in model.model_name.lower():
        # Skip start token.
        class_to_tokens = {k: model.tokenizer(k)['input_ids'][1:] for k in class_names}
        # Llama has 'autospace'. don't need space in query, and also don't need it in token.
    else:
        class_to_tokens = {k: model.tokenizer(k)['input_ids'] for k in class_names}

    if not skip:
        assert all([len(v) == 1 for _, v in class_to_tokens.items()])
        logit_idxs = np.concatenate(list(class_to_tokens.values()))
        return logit_idxs
    else:
        class_to_tokens_out = {}
        success_classes = []
        for k, v in class_to_tokens.items():
            if len(v) != 1:
                logging.warning('Only taking first token for classes: %s %s', k, v)
            class_to_tokens_out[k] = v[:1]
            success_classes.append(k)

        logit_idxs = np.concatenate(list(class_to_tokens_out.values()))

        return logit_idxs, success_classes


def get_calibrate(model, logit_idxs, data=None):
    """Compute calibration temperature."""
    if data is not None:
        sentence = data.text_string
        answer = data.answer_string
    else:
        sentence = DEFAULT_STRINGS['sentence']
        answer = DEFAULT_STRINGS['answer']

    # look at probability of pos/neg for following 'Answer:' as well as just as the next token
    input_texts = [
        f'{answer}: ',
        f"{sentence}: 'A neutral sentence.'\n{answer}: ",
        f"{sentence}: .'\n{answer}: "]

    prior_log_probs = []
    for input_text in input_texts:
        prior_log_probs.append(model.get_token_prob(input_text, list(logit_idxs)))
    prior_log_probs = torch.stack(prior_log_probs, 0).mean(0)
    # Now re-weight: divide by prior prob.
    def calibrate(log_probs, prior_log_probs):
        if prior_log_probs.ndim == 1:
            prior_log_probs = prior_log_probs.unsqueeze(0)
        reweight = log_probs - prior_log_probs
        normalize = torch.nn.functional.log_softmax(reweight, -1)
        return normalize

    calibrate = functools.partial(calibrate, prior_log_probs=prior_log_probs)
    logging.info(
        'prior probs: %s, calibrated probs: %s',
        torch.exp(prior_log_probs).numpy(),
        torch.exp(calibrate(prior_log_probs)).numpy())

    return calibrate, prior_log_probs.numpy()


def setup_logger():
    """Setup logger to always print time and level."""
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')
    logging.getLogger().setLevel(logging.INFO)  # logging.DEBUG
