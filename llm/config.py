from ml_collections.config_dict import config_dict


def get_config():
    config = config_dict.ConfigDict()
    config.debug = False
    config.random_seed = 200 # Random seed for reproducibility. For now only affects data selection.
    config.experiment_name = 'average-fewshot'
    config.note = 'None'
    config.n_repeats = 15

    config.experiment = config_dict.ConfigDict()
    config.experiment.context_lengths = [1, 2, 3, 5, 7, 10, 15, 20]

    config.model = config_dict.ConfigDict()
    config.model.name = 'llama-7b' # LLM Model name.
    config.model.config = config_dict.ConfigDict()
    config.model.config.stop_sequences = ['\n']
    config.model.calibrate = True

    config.dataset = config_dict.ConfigDict()
    config.dataset.name = 'sst2' # Dataset.
    config.dataset.config = config_dict.ConfigDict()
    config.dataset.config.num_test_samples = 1

    config.dataset.config.num_prompt_examples = 20 # Number of in-context examples for fewshot prompt evaluation.
    config.dataset.config.use_validation_if_possible = True # Do not use test set until final experiments.
    config.dataset.config_to_model = config_dict.ConfigDict()
    config.dataset.config_to_model.max_new_tokens = 1
    config.dataset.config.reverse_labels = False
    config.dataset.config.use_model_preds_as_labels = False
    config.dataset.config.class_names = None
    config.dataset.config.answer_string = None
    config.dataset.config.random_labels = False
    config.dataset.config.flip_labels_after = -1
    config.dataset.config.flip_labels_multi_lim = int(1e19)
    config.dataset.config.flip_labels_after_halfway = False
    config.dataset.config.flip_labels_multi = False
    config.dataset.config.replace_labels_by = ''
    config.dataset.config.replace_labels_until_step = -1
    config.dataset.config.use_default_prompt = False
    config.dataset.config.inputs_off_by_one=False
    config.dataset.fix_space_bug = True

    return config
