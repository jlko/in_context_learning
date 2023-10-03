"""Implement HuggingfaceModel models."""
from collections import Counter
import numpy as np
import torch
import torch.utils._pytree as pytree

import accelerate

from transformers import AutoTokenizer
from transformers import AutoConfig
from transformers import AutoModelForCausalLM
from transformers import BitsAndBytesConfig

from huggingface_hub import snapshot_download


def remove_split_layer(device_map):
    """Modify device maps s.t. individual layers are not spread across devices."""

    destinations = list(device_map.keys())

    counts = Counter(['.'.join(i.split('.')[:2]) for i in destinations])

    found_split = False
    for layer, count in counts.items():
        if count == 1:
            continue

        if found_split:
            raise ValueError('More than one split layer')

        print(f'Split layer is {layer}')

        # remove split for that layer
        for name in list(device_map.keys()):
            if name.startswith(layer):
                print(f'pop {name}')
                device = device_map.pop(name)

        device_map[layer] = device
        found_split = True

    return device_map


class HuggingfaceModel():
    """HuggingfaceModel."""

    def __init__(self, model_name, stop_sequences=None):


        if 'llama' in model_name.lower():

            if model_name.endswith('-8bit'):
                kwargs = {'quantization_config': BitsAndBytesConfig(
                    load_in_8bit=True,)}
                model_name = model_name[:-len('-8bit')]
            else:
                kwargs = {}

            if 'Llama-2' in model_name:
                base = 'meta-llama'
                model_name = model_name + '-hf'
            else:
                base = 'huggyllama'

            self.tokenizer = AutoTokenizer.from_pretrained(
                f"{base}/{model_name}", device_map="auto",
                token_type_ids=None)

            llama65b = '65b' in model_name and base == 'huggyllama'
            llama2_70b = '70b' in model_name and base == 'meta-llama'

            if '7b' in model_name or '13b' in model_name:
                self.model = AutoModelForCausalLM.from_pretrained(
                    f"{base}/{model_name}", device_map="auto", **kwargs)

            elif llama2_70b or llama65b:
                path = snapshot_download(
                    repo_id=f'{base}/{model_name}',
                    allow_patterns=['*.json', '*.model', '*.safetensors'],
                    ignore_patterns=['pytorch_model.bin.index.json']
                )
                config = AutoConfig.from_pretrained(f"{base}/{model_name}")
                # config.load_in_8bit = True
                with accelerate.init_empty_weights():
                    self.model = AutoModelForCausalLM.from_config(config)
                self.model.tie_weights()

                max_mem = 15 * 4686198491 # 4G*15
                device_map = accelerate.infer_auto_device_map(
                    self.model.model,
                    max_memory={0: max_mem, 1: max_mem},
                    dtype='float16'
                )
                device_map = remove_split_layer(device_map)
                full_model_device_map = {f"model.{k}": v for k, v in device_map.items()}
                full_model_device_map["lm_head"] = 0

                # get snapshot folder
                self.model = accelerate.load_checkpoint_and_dispatch(
                    self.model, path, device_map=full_model_device_map,
                    dtype='float16')
            else:
                raise ValueError

        elif 'falcon' in model_name:
            model_id = f'tiiuae/{model_name}'
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id, device_map='auto', token_type_ids=None,
                clean_up_tokenization_spaces=False)

            kwargs = {'quantization_config': BitsAndBytesConfig(
                load_in_8bit=True,)}

            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                device_map='auto',
                **kwargs,
            )
        else:
            raise ValueError

        self.model_name = model_name
        self.stop_sequences = stop_sequences

    def get_token_prob(
            self, input_text, select_indices=slice(None),
            extra_indices=None, extra_pos=None):
        """LLM forward pass that returns log token probabilities.

        Args:
            input_text: String for which to compute forward pass.
            select_indices: Input positions at which to extract logits.
            extra_indices: Additional indices at which to extract logits.
            extra_pos: Record positions that match string `extra_pos'. We will
                often have `extra_pos='Answer:'` to identify positions of
                label tokens.
        """
        inputs = self.tokenizer(input_text, return_tensors='pt').to('cuda')

        if 'llama' in self.model_name.lower() or 'falcon' in self.model_name:
            if 'token_type_ids' in inputs:  # seems to have been updated
                del inputs['token_type_ids']

        with torch.no_grad():
            out = self.model(**inputs)


        out = pytree.tree_map(lambda x: x.cpu(), dict(out))
        logits = out['logits'][0, -1, select_indices]
        log_probs = torch.nn.functional.log_softmax(logits.float(), -1)

        res, return_res = {}, False
        if extra_indices is not None:
            return_res = True
            # Select logits at additional indices.
            res['extra_probs'] = out['logits'][0, :, extra_indices]

        if extra_pos is not None:
            return_res = True
            match = np.array(self.tokenizer.encode(extra_pos))
            if 'llama' in self.model_name.lower():
                match = match[1:]
            encoded = inputs['input_ids'][0].cpu().numpy()
            arange = range(len(encoded)-len(match) + 1)
            pos = [
                np.array_equal(match, encoded[i:i+len(match)]) for i in arange]
            answer_pos = np.where(pos)[0]
            # Offset to get the loglik of label prediction.
            answer_pos = answer_pos + len(match)
            res['extra_pos'] = answer_pos

        if return_res:
            res['log_probs'] = log_probs
            return res
        else:
            return log_probs
