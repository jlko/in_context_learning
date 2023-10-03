import os
import logging
from typing import List, Text, Mapping, Any

import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import wandb

from datasets import load_dataset

def split(ds, all_indices=None):
    all_labels = ds['train']['label']

    if all_indices is None:
        all_indices = range(ds.num_rows['train'])
    else:
        all_indices = [int(i) for i in all_indices]
        all_labels = [all_labels[idx] for idx in all_indices]

    train_idxs, test_idxs = train_test_split(
        all_indices, test_size=0.33, random_state=41,
        stratify=all_labels
    )
    return {
        'train': [ds['train'][i] for i in train_idxs],
        'test': [ds['train'][i] for i in test_idxs]
    }


DEFAULT_STRINGS = {'sentence': 'Sentence', 'answer': 'Answer'}


BINARY_REPLACEMENT_LABELS = {
    'rating_1': ['bad', 'good'],
    'rating_2': ['good', 'bad'],
    'options_1': ['A', 'B'],
    'options_2': ['B', 'A'],
    'colors_1': ['green', 'blue'],
    'colors_2': ['blue', 'green'],
    'sea_chair_1': ['sea', 'chair'],
    'sea_chair_2': ['chair', 'sea'],
}

class BaseDataset:
    class_name_variants: Mapping[Text, List[Text]]
    all_classes: List[Text]
    label_name: Text = 'label'
    feature_name: Text
    dataset: Any
    is_binary_classification: bool = True
    text_string: Text = DEFAULT_STRINGS['sentence']
    default_prompt_string: Text = None

    def __init__(
        self, *,
        rng=None,
        num_test_samples=None,
        num_prompt_examples=None,
        reverse_labels=None,
        class_names=None,
        use_model_preds_as_labels=None,
        fix_space_bug=True,
        answer_string=None,
        test_set_name=None,
        random_labels=False,
        flip_labels_after=-1,
        flip_labels_multi=False,
        replace_labels_by='',
        replace_labels_until_step=-1,
        prompt_string=None,
        use_default_prompt=False,
        flip_labels_after_halfway=False,
        inputs_off_by_one=False,
        flip_labels_multi_lim=int(1e19),
    ):
        if rng is None or num_test_samples is None or num_prompt_examples is None:
            raise ValueError

        if class_names is None:
            class_names = self.class_name_variants['default']

        if reverse_labels:
            class_names = [class_names[(i + 1) % len(class_names)] for i in range(len(class_names))]

        if answer_string is None:
            answer_string = DEFAULT_STRINGS['answer']

        self.rng = rng
        self.class_names = class_names
        self.num_classes = len(self.class_names)
        self.numlabel2text = dict(zip(range(self.num_classes), self.class_names))
        self.answer_string = answer_string

        self.use_model_preds_as_labels = use_model_preds_as_labels
        self.fix_space_bug = fix_space_bug

        self.test_data = self.dataset[test_set_name]
        self.test_indices = self.rng.choice(range(len(self.test_data)), num_test_samples, replace=False)
        self.test_indices = list(map(int, self.test_indices))

        self.train_data = self.dataset['train']
        self.num_prompt_examples = num_prompt_examples
        self.fewshot_indices = self.rng.choice(range(len(self.train_data)), self.num_prompt_examples)
        self.fewshot_indices = list(map(int, self.fewshot_indices))

        # Use with active learning.
        self.pred_labels = {}

        self.random_labels = random_labels
        if flip_labels_after_halfway:
            logging.warning("Enabling flip after halfway.")
            self.flip_labels_after_halfway = flip_labels_after_halfway
            assert flip_labels_after == -1
            flip_labels_after = self.num_prompt_examples // 2

        self.flip_labels_after = flip_labels_after
        self.flip_labels_multi = flip_labels_multi

        self.replace_labels_by = replace_labels_by
        self.replace_labels_until_step = replace_labels_until_step

        self.use_default_prompt = use_default_prompt

        if prompt_string == 'TRIGGER_FLIP':

            prompt_string = f'In the following, '
            for i in range(len(class_names)):
                prompt_string += f'`{class_names[i]}` means `{class_names[(i + 1) % self.num_classes]}`'

                if i != len(class_names) - 1:
                    prompt_string += ' and '
                else:
                    prompt_string += '.'

        prompt = ''
        if self.use_default_prompt and self.default_prompt_string:
            prompt = self.default_prompt_string
        if prompt_string:
            if prompt:
                prompt += '\n'
            prompt += prompt_string
        self.prompt_string = prompt
        logging.info('Prompt: %s', self.prompt_string)

        self.guessing_performance()
        self.inputs_off_by_one = inputs_off_by_one
        self.flip_labels_multi_lim = flip_labels_multi_lim
    @property
    def fewshot_examples(self):
        return self.get_fewshot_examples(indices=self.fewshot_indices)

    def get_feature(self, datapoint):
        return datapoint[self.feature_name]

    def add_example(self, sentence, label):
        if label is None:
            label = ''
            if self.fix_space_bug:
                return f"{self.text_string}: '{sentence}' \n{self.answer_string}:"

        return f"{self.text_string}: '{sentence}' \n{self.answer_string}: {label}"

    def get_fewshot_examples(self, indices, random_labels=False):
        prompt = ''

        if self.prompt_string:
            prompt = self.prompt_string + '\n'
        else:
            prompt = ''

        for i, idx in enumerate(indices):

            if self.inputs_off_by_one:
                # Need to cut input length by one for this.
                if i == len(indices) - 1:
                    break
                # Associate (x_{i+1}, y_i) --> x_2, y_1, x_3, y_2, x_4, y_4, ..
                sentence = self.get_feature(self.train_data[indices[i + 1]])
            else:
                sentence = self.get_feature(self.train_data[idx])

            if self.use_model_preds_as_labels:
                label = self.pred_labels[idx]
            else:
                label = self.train_data[idx][self.label_name]

            flip = False
            if (self.flip_labels_after != -1):
                if self.flip_labels_multi:
                    # Will switch between False and True every `self.flip_labels_after`.
                    # Will start with False.
                    if (i // self.flip_labels_after) % 2 == 1:
                        flip = True
                elif i >= self.flip_labels_after:
                    flip = True
                if i > self.flip_labels_multi_lim:
                    flip = False

            if flip:
                # Flips for binary, rotates for multiclass.
                label = (label + 1) % self.num_classes

            replace = True
            if self.replace_labels_until_step != -1:
                if i >= self.replace_labels_until_step:
                    replace = False

            if (random_labels or self.random_labels) and replace:
                ridx = int(self.rng.choice(range(len(self.train_data)), 1)[0])
                label = self.train_data[ridx][self.label_name]

            if self.replace_labels_by and replace:
                label = self.replace_labels_by
            else:
                label = self.numlabel2text[label]
            add = self.add_example(sentence, label) + "\n\n"
            prompt = prompt + add

        return prompt

    def get_fewshot_prompt(self, sentence, examples=None):
        if examples is None:
            examples = self.fewshot_examples

        return examples + self.add_example(sentence, None)

    def guessing_performance(self):
        label_counts = np.bincount([i[self.label_name] for i in self.dataset['train']])
        guessing = label_counts / label_counts.sum()
        wandb.log(
            {'label_frequencies': guessing, 'guessing_acc': guessing.max()})
        logging.info('Label freqs are %s', guessing)


class SST2Dataset(BaseDataset):

    if os.getenv("CLASS_NAMES") == 'DEFAULT_ONLY':
        class_name_variants = {
            'default': ['negative', 'positive'],
        }
    elif os.getenv("CLASS_NAMES") == 'DEFAULT_AND_FLIP':
        class_name_variants = {
            'default': ['negative', 'positive'],
            'flip': ['positive', 'negative'],
        }
    else:
        class_name_variants = {
            'default': ['negative', 'positive'],
            'flip': ['positive', 'negative'],
            **BINARY_REPLACEMENT_LABELS
            # 'colors_1': ['green', 'blue'],
            # 'colors_2': ['blue', 'green'],
            # 'feelings_1': ['sad', 'happy'],
            # 'feelings_2': ['happy', 'sad'],
            # 'rating_1': ['bad', 'good'],
            # 'rating_2': ['good', 'bad'],
            # 'sea_chair_1': ['sea', 'chair'],
            # 'sea_chair_2': ['chair', 'sea'],
            # 'options_1': ['A', 'B'],
            # 'options_2': ['B', 'A'],
            # 'movie_book_1': ['movie', 'book'],
            # 'movie_book_2': ['book', 'movie'],
            # 'movie_chair_1': ['movie', 'chair'],
            # 'movie_chair_2': ['chair', 'movie'],
            # 'good_movie_1': ['good', 'movie'],
            # 'good_movie_2': ['movie', 'good'],
        }
    all_classes = [
        num for sublist in class_name_variants.values() for num in sublist]
    all_classes = sorted(list(set(all_classes)))


    feature_name = 'sentence'
    default_prompt_string = "Is the sentiment of the following sentence `positive` or `negative`. Respond with `positive` or `negative."

    def __init__(self, **kwargs):

        self.dataset = load_dataset('sst2')
        kwargs['test_set_name'] = 'validation' if kwargs.pop('use_validation_if_possible') else 'test'

        super().__init__(**kwargs)

        # Zeroshot
        self.prompt_top = "Is the sentiment of the following sentence `positive` or 'negative`. Respond with `positive` or `negative`. \nSentence: "
        self.prompt_bottom = f"\n{self.answer_string}: "


class SubjDataset(BaseDataset):

    if os.getenv("CLASS_NAMES") == 'DEFAULT_ONLY':
        class_name_variants = {
            'default': ['objective', 'subjective'],
        }
    elif os.getenv("CLASS_NAMES") == 'DEFAULT_AND_FLIP':
        class_name_variants = {
            'default': ['objective', 'subjective'],
            'flip': ['subjective', 'objective'],
        }
    else:
        class_name_variants = {
            'default': ['objective', 'subjective'],
            'flip': ['subjective', 'objective'],
            **BINARY_REPLACEMENT_LABELS
        }
    all_classes = [
        num for sublist in class_name_variants.values() for num in sublist]
    all_classes = sorted(list(set(all_classes)))


    feature_name = 'text'
    default_prompt_string = "Is the following sentence `objective` or `subjective`. Respond with `objective` or `subjective`."

    def __init__(self, **kwargs):

        self.dataset = load_dataset('SetFit/subj')

        if kwargs.pop('use_validation_if_possible'):
            logging.warn(
                'No validation set for subj but `use_validation_if_possible == True`.')
        kwargs['test_set_name'] = 'test'

        super().__init__(**kwargs)


class FinancialPhrasebankDataset(BaseDataset):

    if os.getenv("CLASS_NAMES") == 'DEFAULT_ONLY':
        class_name_variants = {
            'default': ['negative', 'neutral', 'positive'],
        }
    elif os.getenv("CLASS_NAMES") == 'DEFAULT_AND_FLIP':
        class_name_variants = {
            'default': ['negative', 'neutral', 'positive'],
            # Choose perm_2 here, because rotated once more (by flip_label)
            # this becomes default again.
            'perm_2': ['positive', 'negative', 'neutral'],
        }
    else:
        class_name_variants = {
            'default': ['negative', 'neutral', 'positive'],
            'perm_1': ['neutral', 'positive', 'negative'],
            'perm_2': ['positive', 'negative', 'neutral'],
            'flip_outer': ['positive', 'neutral', 'negative'],
            'yesno_1': ['no', 'ok', 'yes'],
            'yesno_2': ['yes', 'ok', 'no'],
            'rating_1': ['bad', 'ok', 'good'],
            'rating_2': ['good', 'ok', 'bad'],
            # 'rating_perm_1': ['ok', 'good', 'bad'],
            # 'rating_perm_2': ['good', 'bad', 'ok'],
            # 'rating_flip_outer': ['good', 'ok', 'bad'],
            # 'feeling_default': ['sad', 'ok', 'happy'],
            'options_1': ['A', 'B', 'C'],
            'options_2': ['C', 'A', 'B'],
            'options_3': ['B', 'C', 'A'],
            'colors_1': ['blue', 'green', 'yellow'],
            'colors_2': ['yellow', 'blue', 'green'],
            'colors_3': ['green', 'yellow', 'blue'],
        }

    all_classes = [
        num for sublist in class_name_variants.values() for num in sublist]
    all_classes = sorted(list(set(all_classes)))


    feature_name = 'sentence'
    is_binary_classification = False
    default_prompt_string = "Is the sentiment of the following sentence `negative`, `neutral`, or `positive`. Respond with `negative`, `neutral`, or `positive`."

    def __init__(self, **kwargs):

        ds = load_dataset('financial_phrasebank', 'sentences_allagree')

        self.dataset = split(ds)
        if kwargs.pop('use_validation_if_possible'):
            logging.warn(
                'No validation set for subj but `use_validation_if_possible == True`.')
        kwargs['test_set_name'] = 'test'

        super().__init__(**kwargs)


class YesNoDataset(BaseDataset):
    if os.getenv("CLASS_NAMES") == 'DEFAULT_ONLY':
        class_name_variants = {
            'default': ['no', 'yes'],
        }
    elif os.getenv("CLASS_NAMES") == 'DEFAULT_AND_FLIP':
        class_name_variants = {
            'default': ['no', 'yes'],
            'flip': ['yes', 'no'],
        }
    else:
        class_name_variants = {
            'default': ['no', 'yes'],
            'flip': ['yes', 'no'],
            **BINARY_REPLACEMENT_LABELS
        }
    all_classes = [
        num for sublist in class_name_variants.values() for num in sublist]
    all_classes = sorted(list(set(all_classes)))
    is_binary_classification = True


class ParaphraseDataset(YesNoDataset):
    if not os.getenv('OLD_CLASS_NAMES') == 'TRUE':
        if os.getenv("CLASS_NAMES") == 'DEFAULT_ONLY':
            class_name_variants = {
                'default': ['not equivalent', 'equivalent'],
            }
        elif os.getenv("CLASS_NAMES") == 'DEFAULT_AND_FLIP':
            class_name_variants = {
                'default': ['not equivalent', 'equivalent'],
                'flip': ['equivalent', 'not equivalent'],
            }
        else:
            class_name_variants = {
                'default': ['not equivalent', 'equivalent'],
                'flip': ['equivalent', 'not equivalent'],
                **BINARY_REPLACEMENT_LABELS
            }
        all_classes = [
            num for sublist in class_name_variants.values() for num in sublist]
        all_classes = sorted(list(set(all_classes)))
        default_prompt_string = 'Are Sentence 1 and Sentence 2 equivalent? Respond with `equivalent` or `not equivalent`.'
    else:
        default_prompt_string = 'Are Sentence 1 and Sentence 2 equivalent? Respond with `yes` or `no`.'

    is_binary_classification = True

    feature_name = ['sentence1', 'sentence2']
    text_string = ['Sentence 1', 'Sentence 2']

    def get_feature(self, datapoint):
        return [datapoint[name] for name in self.feature_name]

    def add_example(self, sentence, label):
        feature = f"{self.text_string[0]}: '{sentence[0]}'\n{self.text_string[1]}: '{sentence[1]}'"
        if label is None:
            label = ''
            if self.fix_space_bug:
                return f"{feature} \n{self.answer_string}:"

        return f"{feature} \n{self.answer_string}: {label}"


class MedicalQuestionsPairs(ParaphraseDataset):
    text_string = ['Question 1', 'Question 2']
    feature_name = ['question_1', 'question_2']
    default_prompt_string = 'Are Question 1 and Question 2 equivalent? Respond with `equivalent` or `not equivalent`.'

    if not os.getenv('OLD_CLASS_NAMES') == 'TRUE':
        default_prompt_string = 'Are Question 1 and Question 2 equivalent? Respond with `equivalent` or `not equivalent`.'
    else:
        default_prompt_string = 'Are Question 1 and Question 2 equivalent? Respond with `yes` or `no`.'


    def __init__(self, **kwargs):

        ds = load_dataset('medical_questions_pairs')
        self.dataset = split(ds)

        if kwargs.pop('use_validation_if_possible'):
            logging.warn(
                'No validation set for subj but `use_validation_if_possible == True`.')
        kwargs['test_set_name'] = 'test'

        super().__init__(**kwargs)


class MRPCDataset(ParaphraseDataset):

    def __init__(self, **kwargs):
        self.dataset = load_dataset('glue', 'mrpc')
        kwargs['test_set_name'] = 'validation' if kwargs.pop('use_validation_if_possible') else 'test'
        super().__init__(**kwargs)


class EntailmentDataset(ParaphraseDataset):
    # change-1
    # change-1
    if not os.getenv('OLD_CLASS_NAMES') == 'TRUE':
        if os.getenv("CLASS_NAMES") == 'DEFAULT_ONLY':
            class_name_variants = {
                'default': ['not entailment', 'entailment'],
            }
        elif os.getenv("CLASS_NAMES") == 'DEFAULT_AND_FLIP':
            class_name_variants = {
                'default': ['not entailment', 'entailment'],
                'flip':  ['entailment', 'not entailment'],
            }
        else:
            class_name_variants = {
                'default': ['not entailment', 'entailment'],
                'flip':  ['entailment', 'not entailment'],
                **BINARY_REPLACEMENT_LABELS
            }
        all_classes = [
            num for sublist in class_name_variants.values() for num in sublist]
        all_classes = sorted(list(set(all_classes)))
        default_prompt_string = 'Does Sentence 1 entail Sentence 1? Respond with `entailment` or `not entailment`.'
    else:
        default_prompt_string = 'Does Sentence 1 entail Sentence 1? Respond with `yes` or `no`.'

    is_binary_classification = True


class WNLIDataset(EntailmentDataset):
    def __init__(self, **kwargs):
        self.dataset = load_dataset('glue', 'wnli')
        kwargs['test_set_name'] = 'validation' if kwargs.pop('use_validation_if_possible') else 'test'
        super().__init__(**kwargs)


class RTEDataset(EntailmentDataset):
    if not os.getenv('OLD_CLASS_NAMES') == 'TRUE':
        if os.getenv("CLASS_NAMES") == 'DEFAULT_ONLY':
            class_name_variants = {
                'default': ['not entailment', 'entailment'][::-1],
            }
        elif os.getenv("CLASS_NAMES") == 'DEFAULT_AND_FLIP':
            class_name_variants = {
                'default': ['not entailment', 'entailment'][::-1],
                'flip':  ['entailment', 'not entailment'][::-1],
            }
        else:
            class_name_variants = {
                'default': ['not entailment', 'entailment'][::-1],
                'flip':  ['entailment', 'not entailment'][::-1],
                **BINARY_REPLACEMENT_LABELS
            }
        all_classes = [
            num for sublist in class_name_variants.values() for num in sublist]
        all_classes = sorted(list(set(all_classes)))
        default_prompt_string = 'Does Sentence 1 entail Sentence 1? Respond with `entailment` or `not entailment`.'
    else:
        if os.getenv("CLASS_NAMES") == 'DEFAULT_ONLY':
            class_name_variants = {
                'default': ['no', 'yes'][::-1],
            }
        elif os.getenv("CLASS_NAMES") == 'DEFAULT_AND_FLIP':
            class_name_variants = {
                'default': ['no', 'yes'][::-1],
                'flip': ['yes', 'no'][::-1],
            }
        else:
            class_name_variants = {
                'default': ['no', 'yes'][::-1],
                'flip': ['yes', 'no'][::-1],
                **BINARY_REPLACEMENT_LABELS
            }
        all_classes = [
            num for sublist in class_name_variants.values() for num in sublist]
        all_classes = sorted(list(set(all_classes)))

        default_prompt_string = 'Does Sentence 1 entail Sentence 1? Respond with `yes` or `no`.'

    def __init__(self, **kwargs):
        self.dataset = load_dataset('glue', 'rte')
        kwargs['test_set_name'] = 'validation' if kwargs.pop('use_validation_if_possible') else 'test'
        super().__init__(**kwargs)


class HateSpeechDataset(YesNoDataset):
    if not os.getenv('OLD_CLASS_NAMES') == 'TRUE':
        if os.getenv("CLASS_NAMES") == 'DEFAULT_ONLY':
            class_name_variants = {
                'default': ['not hate speech', 'hate speech'],
            }
        elif os.getenv("CLASS_NAMES") == 'DEFAULT_AND_FLIP':
            class_name_variants = {
                'default': ['not hate speech', 'hate speech'],
                'flip':  ['hate speech', 'not hate speech'],
            }
        else:
            class_name_variants = {
                'default': ['not hate speech', 'hate speech'],
                'flip':  ['hate speech', 'not hate speech'],
                **BINARY_REPLACEMENT_LABELS
            }
        all_classes = [
            num for sublist in class_name_variants.values() for num in sublist]
        all_classes = sorted(list(set(all_classes)))
        default_prompt_string = 'Does the sentence contain hate speech? Respond with `hate speech` or `not hate speech`.'
    else:
        default_prompt_string = 'Does the sentence contain hate speech? Respond with `yes` or `no`.'

    is_binary_classification = True

    # change-1
    feature_name = 'text'

    def __init__(self, **kwargs):
        ds = load_dataset('hate_speech18')

        # select 1000 datapoints with indices 0 and 1 (skip bad labels 1, 3)
        # also get rid of clas
        idxs = np.sort(np.concatenate([
            np.where(np.array(ds['train']['label']) == 0)[0][:1000],
            np.where(np.array(ds['train']['label']) == 1)[0][:1000],
        ]))
        self.dataset = split(ds, all_indices=idxs)

        if kwargs.pop('use_validation_if_possible'):
            logging.warn(
                'No validation set for subj but `use_validation_if_possible == True`.')
        kwargs['test_set_name'] = 'test'
        super().__init__(**kwargs)


class AuthorIdDataset(BaseDataset):
    if os.getenv("CLASS_NAMES") == 'DEFAULT_ONLY':
        class_name_variants = {
            'default': ['tom', 'jan'],
        }
    elif os.getenv("CLASS_NAMES") == 'DEFAULT_AND_FLIP':
        class_name_variants = {
            'default': ['tom', 'jan'],
            'flipped': ['jan', 'tom'],
        }
    else:
        class_name_variants = {
            'default': ['tom', 'jan'],
            'flipped': ['jan', 'tom'],
            'sentiment': ['negative', 'positive'],
            'sentiment_flipped': ['positive', 'negative'],
            'feelings_1': ['happy', 'sad'],
            'feelings_2': ['sad', 'happy'],
            'colors_1': ['green', 'blue'],
            'rating_1': ['bad', 'good'],
            'sea_chair_1': ['sea', 'chair'],
            'options_1': ['A', 'B'],
            'movie_book_1': ['movie', 'book'],
        }
    all_classes = [
        num for sublist in class_name_variants.values() for num in sublist]
    all_classes = sorted(list(set(all_classes)))

    label_name = 'author'
    feature_name = 'sentence'

    max_sentence_length = 200

    def __init__(self, **kwargs):

        self.dataset = self.load_dataset()
        logging.warn('No real test data for this dataset.')
        kwargs.pop('use_validation_if_possible')
        kwargs['test_set_name'] = 'test'
        super().__init__(**kwargs)

    def load_dataset(self):
        df = pd.read_csv('data/tomvjan/chat_cleaned.csv', sep='\t ')

        # remove low-quality sentences
        data = df[df['low_quality'] == False]

        # enforce max sequence_length
        msl = self.max_sentence_length
        data['sentence_cut'] = data['sentence'].map(
            lambda x: x if len(x) < msl - 3 else x[:msl - 3] + '...')

        txt2num = {'tom': 0, 'jan': 1}
        train_data = [
            {'sentence': i.sentence_cut, 'author': txt2num[i.author]}
            for _, i in data.iterrows()]
        train_data = list(np.random.permutation(train_data))
        dataset = {
            'test': [{'sentence': 'There is no test data', 'author': 0}],
            'train': train_data,
        }
        return dataset

class AGNewsDataset(BaseDataset):
# World (0), Sports (1), Business (2), Sci/Tech (3).

    if os.getenv("CLASS_NAMES") == 'DEFAULT_ONLY':
        class_name_variants = {
            'default': ['world', 'sports', 'business', 'science and technology'],
        }
    elif os.getenv("CLASS_NAMES") == 'DEFAULT_AND_FLIP':
        class_name_variants = {
            'default': ['world', 'sports', 'business', 'science and technology'],
            'perm_1': ['science and technology', 'world', 'sports', 'business'],
        }
    else:
        # sports business scienectech world
        class_name_variants = {
            'default': ['world', 'sports', 'business', 'science and technology'],
            'perm_1': ['science and technology', 'world', 'sports', 'business'],
            'perm_2': ['business', 'science and technology', 'world', 'sports'],
            'perm_3': ['sports', 'business', 'science and technology', 'world'],
            'flip': ['world', 'sports', 'business', 'science and technology'][::-1],
            'options_1': ['A', 'B', 'C', 'D'],
            'options_2': ['D', 'C', 'B', 'A'],
        }
    all_classes = [
        num for sublist in class_name_variants.values() for num in sublist]
    all_classes = sorted(list(set(all_classes)))

    feature_name = 'text'
    default_prompt_string = "To which category does the following news article belong: `world`, `sport`, `busines`, or `science and technology`?"
    is_binary_classification = False

    def __init__(self, **kwargs):

        self.dataset = load_dataset('ag_news')

        if kwargs.pop('use_validation_if_possible'):
            logging.warn(
                'No validation set for subj but `use_validation_if_possible == True`.')
        kwargs['test_set_name'] = 'test'

        super().__init__(**kwargs)



datasets = {
    'sst2': SST2Dataset,
    'subj': SubjDataset,
    'financial_phrasebank': FinancialPhrasebankDataset,
    'author_id': AuthorIdDataset,
    'medical_questions_pairs': MedicalQuestionsPairs,
    'mrpc': MRPCDataset,
    'wnli': WNLIDataset,
    'rte': RTEDataset,
    'hate_speech': HateSpeechDataset,
    'ag_news': AGNewsDataset,
}
