# Experiment Scripts

Below we give a list of the commands necessary to reproduce the main experiments of the paper.


## Few-Shot ICL on Default Labels

```
export CLASS_NAMES=DEFAULT_ONLY
python change_labels.py --debug=False --model.name=falcon-7b --note=default --dataset.name=sst2  --dataset.config.num_prompt_examples=70 --n_repeats=500
python change_labels.py --debug=False --model.name=falcon-7b-instruct --note=default --dataset.name=sst2  --dataset.config.num_prompt_examples=70 --n_repeats=500
python change_labels.py --debug=False --model.name=falcon-40b --note=default --dataset.name=sst2  --dataset.config.num_prompt_examples=70 --n_repeats=500
python change_labels.py --debug=False --model.name=falcon-40b-instruct --note=default --dataset.name=sst2  --dataset.config.num_prompt_examples=70 --n_repeats=500
python change_labels.py --debug=False --model.name=llama-7b-8bit --note=default --dataset.name=sst2  --dataset.config.num_prompt_examples=70 --n_repeats=500
python change_labels.py --debug=False --model.name=llama-13b-8bit --note=default --dataset.name=sst2  --dataset.config.num_prompt_examples=70 --n_repeats=500
python change_labels.py --debug=False --model.name=llama-65b --note=default --dataset.name=sst2  --dataset.config.num_prompt_examples=70 --n_repeats=500
python change_labels.py --debug=False --model.name=falcon-7b --note=default --dataset.name=subj --dataset.config.num_prompt_examples=40 --n_repeats=500
python change_labels.py --debug=False --model.name=falcon-7b-instruct --note=default --dataset.name=subj --dataset.config.num_prompt_examples=40 --n_repeats=500
python change_labels.py --debug=False --model.name=falcon-40b --note=default --dataset.name=subj --dataset.config.num_prompt_examples=40 --n_repeats=500
python change_labels.py --debug=False --model.name=falcon-40b-instruct --note=default --dataset.name=subj --dataset.config.num_prompt_examples=40 --n_repeats=500
python change_labels.py --debug=False --model.name=llama-7b-8bit --note=default --dataset.name=subj --dataset.config.num_prompt_examples=40 --n_repeats=500
python change_labels.py --debug=False --model.name=llama-13b-8bit --note=default --dataset.name=subj --dataset.config.num_prompt_examples=40 --n_repeats=500
python change_labels.py --debug=False --model.name=llama-65b --note=default --dataset.name=subj --dataset.config.num_prompt_examples=40 --n_repeats=500
python change_labels.py --debug=False --model.name=falcon-7b --note=default --dataset.name=financial_phrasebank --dataset.config.num_prompt_examples=45 --n_repeats=500
python change_labels.py --debug=False --model.name=falcon-7b-instruct --note=default --dataset.name=financial_phrasebank --dataset.config.num_prompt_examples=45 --n_repeats=500
python change_labels.py --debug=False --model.name=falcon-40b --note=default --dataset.name=financial_phrasebank --dataset.config.num_prompt_examples=45 --n_repeats=500
python change_labels.py --debug=False --model.name=falcon-40b-instruct --note=default --dataset.name=financial_phrasebank --dataset.config.num_prompt_examples=45 --n_repeats=500
python change_labels.py --debug=False --model.name=llama-7b-8bit --note=default --dataset.name=financial_phrasebank --dataset.config.num_prompt_examples=45 --n_repeats=500
python change_labels.py --debug=False --model.name=llama-13b-8bit --note=default --dataset.name=financial_phrasebank --dataset.config.num_prompt_examples=45 --n_repeats=500
python change_labels.py --debug=False --model.name=llama-65b --note=default --dataset.name=financial_phrasebank --dataset.config.num_prompt_examples=45 --n_repeats=500
python change_labels.py --debug=False --model.name=falcon-7b --note=default --dataset.name=medical_questions_pairs --dataset.config.num_prompt_examples=25 --n_repeats=500
python change_labels.py --debug=False --model.name=falcon-7b-instruct --note=default --dataset.name=medical_questions_pairs --dataset.config.num_prompt_examples=25 --n_repeats=500
python change_labels.py --debug=False --model.name=falcon-40b --note=default --dataset.name=medical_questions_pairs --dataset.config.num_prompt_examples=25 --n_repeats=500
python change_labels.py --debug=False --model.name=falcon-40b-instruct --note=default --dataset.name=medical_questions_pairs --dataset.config.num_prompt_examples=25 --n_repeats=500
python change_labels.py --debug=False --model.name=llama-7b-8bit --note=default --dataset.name=medical_questions_pairs --dataset.config.num_prompt_examples=25 --n_repeats=500
python change_labels.py --debug=False --model.name=llama-13b-8bit --note=default --dataset.name=medical_questions_pairs --dataset.config.num_prompt_examples=25 --n_repeats=500
python change_labels.py --debug=False --model.name=llama-65b --note=default --dataset.name=medical_questions_pairs --dataset.config.num_prompt_examples=25 --n_repeats=500
python change_labels.py --debug=False --model.name=falcon-7b --note=default --dataset.name=mrpc --dataset.config.num_prompt_examples=20 --n_repeats=500
python change_labels.py --debug=False --model.name=falcon-7b-instruct --note=default --dataset.name=mrpc --dataset.config.num_prompt_examples=20 --n_repeats=500
python change_labels.py --debug=False --model.name=falcon-40b --note=default --dataset.name=mrpc --dataset.config.num_prompt_examples=20 --n_repeats=500
python change_labels.py --debug=False --model.name=falcon-40b-instruct --note=default --dataset.name=mrpc --dataset.config.num_prompt_examples=20 --n_repeats=500
python change_labels.py --debug=False --model.name=llama-7b-8bit --note=default --dataset.name=mrpc --dataset.config.num_prompt_examples=20 --n_repeats=500
python change_labels.py --debug=False --model.name=llama-13b-8bit --note=default --dataset.name=mrpc --dataset.config.num_prompt_examples=20 --n_repeats=500
python change_labels.py --debug=False --model.name=llama-65b --note=default --dataset.name=mrpc --dataset.config.num_prompt_examples=20 --n_repeats=500
python change_labels.py --debug=False --model.name=falcon-7b --note=default --dataset.name=wnli --dataset.config.num_prompt_examples=40 --n_repeats=500
python change_labels.py --debug=False --model.name=falcon-7b-instruct --note=default --dataset.name=wnli --dataset.config.num_prompt_examples=40 --n_repeats=500
python change_labels.py --debug=False --model.name=falcon-40b --note=default --dataset.name=wnli --dataset.config.num_prompt_examples=40 --n_repeats=500
python change_labels.py --debug=False --model.name=falcon-40b-instruct --note=default --dataset.name=wnli --dataset.config.num_prompt_examples=40 --n_repeats=500
python change_labels.py --debug=False --model.name=llama-7b-8bit --note=default --dataset.name=wnli --dataset.config.num_prompt_examples=40 --n_repeats=500
python change_labels.py --debug=False --model.name=llama-13b-8bit --note=default --dataset.name=wnli --dataset.config.num_prompt_examples=40 --n_repeats=500
python change_labels.py --debug=False --model.name=llama-65b --note=default --dataset.name=wnli --dataset.config.num_prompt_examples=40 --n_repeats=500
python change_labels.py --debug=False --model.name=falcon-7b --note=default --dataset.name=hate_speech --dataset.config.num_prompt_examples=40 --n_repeats=500
python change_labels.py --debug=False --model.name=falcon-7b-instruct --note=default --dataset.name=hate_speech --dataset.config.num_prompt_examples=40 --n_repeats=500
python change_labels.py --debug=False --model.name=falcon-40b --note=default --dataset.name=hate_speech --dataset.config.num_prompt_examples=40 --n_repeats=500
python change_labels.py --debug=False --model.name=falcon-40b-instruct --note=default --dataset.name=hate_speech --dataset.config.num_prompt_examples=40 --n_repeats=500
python change_labels.py --debug=False --model.name=llama-7b-8bit --note=default --dataset.name=hate_speech --dataset.config.num_prompt_examples=40 --n_repeats=500
python change_labels.py --debug=False --model.name=llama-13b-8bit --note=default --dataset.name=hate_speech --dataset.config.num_prompt_examples=40 --n_repeats=500
python change_labels.py --debug=False --model.name=llama-65b --note=default --dataset.name=hate_speech --dataset.config.num_prompt_examples=40 --n_repeats=500
python change_labels.py --debug=False --model.name=falcon-7b --note=default --dataset.name=rte --dataset.config.num_prompt_examples=15 --n_repeats=500
python change_labels.py --debug=False --model.name=falcon-7b-instruct --note=default --dataset.name=rte --dataset.config.num_prompt_examples=15 --n_repeats=500
python change_labels.py --debug=False --model.name=falcon-40b --note=default --dataset.name=rte --dataset.config.num_prompt_examples=15 --n_repeats=500
python change_labels.py --debug=False --model.name=falcon-40b-instruct --note=default --dataset.name=rte --dataset.config.num_prompt_examples=15 --n_repeats=500
python change_labels.py --debug=False --model.name=llama-7b-8bit --note=default --dataset.name=rte --dataset.config.num_prompt_examples=15 --n_repeats=500
python change_labels.py --debug=False --model.name=llama-13b-8bit --note=default --dataset.name=rte --dataset.config.num_prompt_examples=15 --n_repeats=500
python change_labels.py --debug=False --model.name=llama-65b --note=default --dataset.name=rte --dataset.config.num_prompt_examples=15 --n_repeats=500
python change_labels.py --debug=False --model.name=falcon-7b --note=default --dataset.name=ag_news --dataset.config.num_prompt_examples=30 --n_repeats=500
python change_labels.py --debug=False --model.name=falcon-7b-instruct --note=default --dataset.name=ag_news --dataset.config.num_prompt_examples=30 --n_repeats=500
python change_labels.py --debug=False --model.name=falcon-40b --note=default --dataset.name=ag_news --dataset.config.num_prompt_examples=30 --n_repeats=500
python change_labels.py --debug=False --model.name=falcon-40b-instruct --note=default --dataset.name=ag_news --dataset.config.num_prompt_examples=30 --n_repeats=500
python change_labels.py --debug=False --model.name=llama-7b-8bit --note=default --dataset.name=ag_news --dataset.config.num_prompt_examples=30 --n_repeats=500
python change_labels.py --debug=False --model.name=llama-13b-8bit --note=default --dataset.name=ag_news --dataset.config.num_prompt_examples=30 --n_repeats=500
python change_labels.py --debug=False --model.name=llama-65b --note=default --dataset.name=ag_news --dataset.config.num_prompt_examples=30 --n_repeats=500
python change_labels.py --debug=False --model.name=Llama-2-7b-8bit --note=default --dataset.name=sst2  --dataset.config.num_prompt_examples=140 --n_repeats=500
python change_labels.py --debug=False --model.name=Llama-2-13b-8bit --note=default --dataset.name=sst2  --dataset.config.num_prompt_examples=140 --n_repeats=500
python change_labels.py --debug=False --model.name=Llama-2-70b --note=default --dataset.name=sst2  --dataset.config.num_prompt_examples=140 --n_repeats=500
python change_labels.py --debug=False --model.name=Llama-2-7b-8bit --note=default --dataset.name=subj --dataset.config.num_prompt_examples=80 --n_repeats=500
python change_labels.py --debug=False --model.name=Llama-2-13b-8bit --note=default --dataset.name=subj --dataset.config.num_prompt_examples=80 --n_repeats=500
python change_labels.py --debug=False --model.name=Llama-2-70b --note=default --dataset.name=subj --dataset.config.num_prompt_examples=80 --n_repeats=500
python change_labels.py --debug=False --model.name=Llama-2-7b-8bit --note=default --dataset.name=financial_phrasebank --dataset.config.num_prompt_examples=80 --n_repeats=500
python change_labels.py --debug=False --model.name=Llama-2-13b-8bit --note=default --dataset.name=financial_phrasebank --dataset.config.num_prompt_examples=80 --n_repeats=500
python change_labels.py --debug=False --model.name=Llama-2-70b --note=default --dataset.name=financial_phrasebank --dataset.config.num_prompt_examples=80 --n_repeats=500
python change_labels.py --debug=False --model.name=Llama-2-7b-8bit --note=default --dataset.name=medical_questions_pairs --dataset.config.num_prompt_examples=50 --n_repeats=500
python change_labels.py --debug=False --model.name=Llama-2-13b-8bit --note=default --dataset.name=medical_questions_pairs --dataset.config.num_prompt_examples=50 --n_repeats=500
python change_labels.py --debug=False --model.name=Llama-2-70b --note=default --dataset.name=medical_questions_pairs --dataset.config.num_prompt_examples=50 --n_repeats=500
python change_labels.py --debug=False --model.name=Llama-2-7b-8bit --note=default --dataset.name=mrpc --dataset.config.num_prompt_examples=40 --n_repeats=500
python change_labels.py --debug=False --model.name=Llama-2-13b-8bit --note=default --dataset.name=mrpc --dataset.config.num_prompt_examples=40 --n_repeats=500
python change_labels.py --debug=False --model.name=Llama-2-70b --note=default --dataset.name=mrpc --dataset.config.num_prompt_examples=40 --n_repeats=500
python change_labels.py --debug=False --model.name=Llama-2-7b-8bit --note=default --dataset.name=wnli --dataset.config.num_prompt_examples=60 --n_repeats=500
python change_labels.py --debug=False --model.name=Llama-2-13b-8bit --note=default --dataset.name=wnli --dataset.config.num_prompt_examples=60 --n_repeats=500
python change_labels.py --debug=False --model.name=Llama-2-70b --note=default --dataset.name=wnli --dataset.config.num_prompt_examples=60 --n_repeats=500
python change_labels.py --debug=False --model.name=Llama-2-7b-8bit --note=default --dataset.name=hate_speech --dataset.config.num_prompt_examples=80 --n_repeats=500
python change_labels.py --debug=False --model.name=Llama-2-13b-8bit --note=default --dataset.name=hate_speech --dataset.config.num_prompt_examples=80 --n_repeats=500
python change_labels.py --debug=False --model.name=Llama-2-70b --note=default --dataset.name=hate_speech --dataset.config.num_prompt_examples=80 --n_repeats=500
python change_labels.py --debug=False --model.name=Llama-2-7b-8bit --note=default --dataset.name=rte --dataset.config.num_prompt_examples=30 --n_repeats=500
python change_labels.py --debug=False --model.name=Llama-2-13b-8bit --note=default --dataset.name=rte --dataset.config.num_prompt_examples=30 --n_repeats=500
python change_labels.py --debug=False --model.name=Llama-2-70b --note=default --dataset.name=rte --dataset.config.num_prompt_examples=30 --n_repeats=500
python change_labels.py --debug=False --model.name=Llama-2-7b-8bit --note=default --dataset.name=ag_news --dataset.config.num_prompt_examples=50 --n_repeats=500
python change_labels.py --debug=False --model.name=Llama-2-13b-8bit --note=default --dataset.name=ag_news --dataset.config.num_prompt_examples=50 --n_repeats=500
python change_labels.py --debug=False --model.name=Llama-2-70b --note=default --dataset.name=ag_news --dataset.config.num_prompt_examples=50 --n_repeats=500
```

## Few-Shot ICL on Randomized Labels
```
export CLASS_NAMES=DEFAULT_ONLY
python change_labels.py --debug=False --model.name=falcon-7b --note=random --dataset.name=sst2  --dataset.config.num_prompt_examples=70 --n_repeats=500 --dataset.config.random_labels=True
python change_labels.py --debug=False --model.name=falcon-7b-instruct --note=random --dataset.name=sst2  --dataset.config.num_prompt_examples=70 --n_repeats=500 --dataset.config.random_labels=True
python change_labels.py --debug=False --model.name=falcon-40b --note=random --dataset.name=sst2  --dataset.config.num_prompt_examples=70 --n_repeats=500 --dataset.config.random_labels=True
python change_labels.py --debug=False --model.name=falcon-40b-instruct --note=random --dataset.name=sst2  --dataset.config.num_prompt_examples=70 --n_repeats=500 --dataset.config.random_labels=True
python change_labels.py --debug=False --model.name=llama-7b-8bit --note=random --dataset.name=sst2  --dataset.config.num_prompt_examples=70 --n_repeats=500 --dataset.config.random_labels=True
python change_labels.py --debug=False --model.name=llama-13b-8bit --note=random --dataset.name=sst2  --dataset.config.num_prompt_examples=70 --n_repeats=500 --dataset.config.random_labels=True
python change_labels.py --debug=False --model.name=llama-65b --note=random --dataset.name=sst2  --dataset.config.num_prompt_examples=70 --n_repeats=500 --dataset.config.random_labels=True
python change_labels.py --debug=False --model.name=falcon-7b --note=random --dataset.name=subj --dataset.config.num_prompt_examples=40 --n_repeats=500 --dataset.config.random_labels=True
python change_labels.py --debug=False --model.name=falcon-7b-instruct --note=random --dataset.name=subj --dataset.config.num_prompt_examples=40 --n_repeats=500 --dataset.config.random_labels=True
python change_labels.py --debug=False --model.name=falcon-40b --note=random --dataset.name=subj --dataset.config.num_prompt_examples=40 --n_repeats=500 --dataset.config.random_labels=True
python change_labels.py --debug=False --model.name=falcon-40b-instruct --note=random --dataset.name=subj --dataset.config.num_prompt_examples=40 --n_repeats=500 --dataset.config.random_labels=True
python change_labels.py --debug=False --model.name=llama-7b-8bit --note=random --dataset.name=subj --dataset.config.num_prompt_examples=40 --n_repeats=500 --dataset.config.random_labels=True
python change_labels.py --debug=False --model.name=llama-13b-8bit --note=random --dataset.name=subj --dataset.config.num_prompt_examples=40 --n_repeats=500 --dataset.config.random_labels=True
python change_labels.py --debug=False --model.name=llama-65b --note=random --dataset.name=subj --dataset.config.num_prompt_examples=40 --n_repeats=500 --dataset.config.random_labels=True
python change_labels.py --debug=False --model.name=falcon-7b --note=random --dataset.name=financial_phrasebank --dataset.config.num_prompt_examples=45 --n_repeats=500 --dataset.config.random_labels=True
python change_labels.py --debug=False --model.name=falcon-7b-instruct --note=random --dataset.name=financial_phrasebank --dataset.config.num_prompt_examples=45 --n_repeats=500 --dataset.config.random_labels=True
python change_labels.py --debug=False --model.name=falcon-40b --note=random --dataset.name=financial_phrasebank --dataset.config.num_prompt_examples=45 --n_repeats=500 --dataset.config.random_labels=True
python change_labels.py --debug=False --model.name=falcon-40b-instruct --note=random --dataset.name=financial_phrasebank --dataset.config.num_prompt_examples=45 --n_repeats=500 --dataset.config.random_labels=True
python change_labels.py --debug=False --model.name=llama-7b-8bit --note=random --dataset.name=financial_phrasebank --dataset.config.num_prompt_examples=45 --n_repeats=500 --dataset.config.random_labels=True
python change_labels.py --debug=False --model.name=llama-13b-8bit --note=random --dataset.name=financial_phrasebank --dataset.config.num_prompt_examples=45 --n_repeats=500 --dataset.config.random_labels=True
python change_labels.py --debug=False --model.name=llama-65b --note=random --dataset.name=financial_phrasebank --dataset.config.num_prompt_examples=45 --n_repeats=500 --dataset.config.random_labels=True
python change_labels.py --debug=False --model.name=falcon-7b --note=random --dataset.name=medical_questions_pairs --dataset.config.num_prompt_examples=25 --n_repeats=500 --dataset.config.random_labels=True
python change_labels.py --debug=False --model.name=falcon-7b-instruct --note=random --dataset.name=medical_questions_pairs --dataset.config.num_prompt_examples=25 --n_repeats=500 --dataset.config.random_labels=True
python change_labels.py --debug=False --model.name=falcon-40b --note=random --dataset.name=medical_questions_pairs --dataset.config.num_prompt_examples=25 --n_repeats=500 --dataset.config.random_labels=True
python change_labels.py --debug=False --model.name=falcon-40b-instruct --note=random --dataset.name=medical_questions_pairs --dataset.config.num_prompt_examples=25 --n_repeats=500 --dataset.config.random_labels=True
python change_labels.py --debug=False --model.name=llama-7b-8bit --note=random --dataset.name=medical_questions_pairs --dataset.config.num_prompt_examples=25 --n_repeats=500 --dataset.config.random_labels=True
python change_labels.py --debug=False --model.name=llama-13b-8bit --note=random --dataset.name=medical_questions_pairs --dataset.config.num_prompt_examples=25 --n_repeats=500 --dataset.config.random_labels=True
python change_labels.py --debug=False --model.name=llama-65b --note=random --dataset.name=medical_questions_pairs --dataset.config.num_prompt_examples=25 --n_repeats=500 --dataset.config.random_labels=True
python change_labels.py --debug=False --model.name=falcon-7b --note=random --dataset.name=mrpc --dataset.config.num_prompt_examples=20 --n_repeats=500 --dataset.config.random_labels=True
python change_labels.py --debug=False --model.name=falcon-7b-instruct --note=random --dataset.name=mrpc --dataset.config.num_prompt_examples=20 --n_repeats=500 --dataset.config.random_labels=True
python change_labels.py --debug=False --model.name=falcon-40b --note=random --dataset.name=mrpc --dataset.config.num_prompt_examples=20 --n_repeats=500 --dataset.config.random_labels=True
python change_labels.py --debug=False --model.name=falcon-40b-instruct --note=random --dataset.name=mrpc --dataset.config.num_prompt_examples=20 --n_repeats=500 --dataset.config.random_labels=True
python change_labels.py --debug=False --model.name=llama-7b-8bit --note=random --dataset.name=mrpc --dataset.config.num_prompt_examples=20 --n_repeats=500 --dataset.config.random_labels=True
python change_labels.py --debug=False --model.name=llama-13b-8bit --note=random --dataset.name=mrpc --dataset.config.num_prompt_examples=20 --n_repeats=500 --dataset.config.random_labels=True
python change_labels.py --debug=False --model.name=llama-65b --note=random --dataset.name=mrpc --dataset.config.num_prompt_examples=20 --n_repeats=500 --dataset.config.random_labels=True
python change_labels.py --debug=False --model.name=falcon-7b --note=random --dataset.name=wnli --dataset.config.num_prompt_examples=40 --n_repeats=500 --dataset.config.random_labels=True
python change_labels.py --debug=False --model.name=falcon-7b-instruct --note=random --dataset.name=wnli --dataset.config.num_prompt_examples=40 --n_repeats=500 --dataset.config.random_labels=True
python change_labels.py --debug=False --model.name=falcon-40b --note=random --dataset.name=wnli --dataset.config.num_prompt_examples=40 --n_repeats=500 --dataset.config.random_labels=True
python change_labels.py --debug=False --model.name=falcon-40b-instruct --note=random --dataset.name=wnli --dataset.config.num_prompt_examples=40 --n_repeats=500 --dataset.config.random_labels=True
python change_labels.py --debug=False --model.name=llama-7b-8bit --note=random --dataset.name=wnli --dataset.config.num_prompt_examples=40 --n_repeats=500 --dataset.config.random_labels=True
python change_labels.py --debug=False --model.name=llama-13b-8bit --note=random --dataset.name=wnli --dataset.config.num_prompt_examples=40 --n_repeats=500 --dataset.config.random_labels=True
python change_labels.py --debug=False --model.name=llama-65b --note=random --dataset.name=wnli --dataset.config.num_prompt_examples=40 --n_repeats=500 --dataset.config.random_labels=True
python change_labels.py --debug=False --model.name=falcon-7b --note=random --dataset.name=hate_speech --dataset.config.num_prompt_examples=40 --n_repeats=500 --dataset.config.random_labels=True
python change_labels.py --debug=False --model.name=falcon-7b-instruct --note=random --dataset.name=hate_speech --dataset.config.num_prompt_examples=40 --n_repeats=500 --dataset.config.random_labels=True
python change_labels.py --debug=False --model.name=falcon-40b --note=random --dataset.name=hate_speech --dataset.config.num_prompt_examples=40 --n_repeats=500 --dataset.config.random_labels=True
python change_labels.py --debug=False --model.name=falcon-40b-instruct --note=random --dataset.name=hate_speech --dataset.config.num_prompt_examples=40 --n_repeats=500 --dataset.config.random_labels=True
python change_labels.py --debug=False --model.name=llama-7b-8bit --note=random --dataset.name=hate_speech --dataset.config.num_prompt_examples=40 --n_repeats=500 --dataset.config.random_labels=True
python change_labels.py --debug=False --model.name=llama-13b-8bit --note=random --dataset.name=hate_speech --dataset.config.num_prompt_examples=40 --n_repeats=500 --dataset.config.random_labels=True
python change_labels.py --debug=False --model.name=llama-65b --note=random --dataset.name=hate_speech --dataset.config.num_prompt_examples=40 --n_repeats=500 --dataset.config.random_labels=True
python change_labels.py --debug=False --model.name=falcon-7b --note=random --dataset.name=rte --dataset.config.num_prompt_examples=15 --n_repeats=500 --dataset.config.random_labels=True
python change_labels.py --debug=False --model.name=falcon-7b-instruct --note=random --dataset.name=rte --dataset.config.num_prompt_examples=15 --n_repeats=500 --dataset.config.random_labels=True
python change_labels.py --debug=False --model.name=falcon-40b --note=random --dataset.name=rte --dataset.config.num_prompt_examples=15 --n_repeats=500 --dataset.config.random_labels=True
python change_labels.py --debug=False --model.name=falcon-40b-instruct --note=random --dataset.name=rte --dataset.config.num_prompt_examples=15 --n_repeats=500 --dataset.config.random_labels=True
python change_labels.py --debug=False --model.name=llama-7b-8bit --note=random --dataset.name=rte --dataset.config.num_prompt_examples=15 --n_repeats=500 --dataset.config.random_labels=True
python change_labels.py --debug=False --model.name=llama-13b-8bit --note=random --dataset.name=rte --dataset.config.num_prompt_examples=15 --n_repeats=500 --dataset.config.random_labels=True
python change_labels.py --debug=False --model.name=llama-65b --note=random --dataset.name=rte --dataset.config.num_prompt_examples=15 --n_repeats=500 --dataset.config.random_labels=True
python change_labels.py --debug=False --model.name=falcon-7b --note=random --dataset.name=ag_news --dataset.config.num_prompt_examples=30 --n_repeats=500 --dataset.config.random_labels=True
python change_labels.py --debug=False --model.name=falcon-7b-instruct --note=random --dataset.name=ag_news --dataset.config.num_prompt_examples=30 --n_repeats=500 --dataset.config.random_labels=True
python change_labels.py --debug=False --model.name=falcon-40b --note=random --dataset.name=ag_news --dataset.config.num_prompt_examples=30 --n_repeats=500 --dataset.config.random_labels=True
python change_labels.py --debug=False --model.name=falcon-40b-instruct --note=random --dataset.name=ag_news --dataset.config.num_prompt_examples=30 --n_repeats=500 --dataset.config.random_labels=True
python change_labels.py --debug=False --model.name=llama-7b-8bit --note=random --dataset.name=ag_news --dataset.config.num_prompt_examples=30 --n_repeats=500 --dataset.config.random_labels=True
python change_labels.py --debug=False --model.name=llama-13b-8bit --note=random --dataset.name=ag_news --dataset.config.num_prompt_examples=30 --n_repeats=500 --dataset.config.random_labels=True
python change_labels.py --debug=False --model.name=llama-65b --note=random --dataset.name=ag_news --dataset.config.num_prompt_examples=30 --n_repeats=500 --dataset.config.random_labels=True
python change_labels.py --debug=False --model.name=Llama-2-7b-8bit --note=random --dataset.name=sst2  --dataset.config.num_prompt_examples=140 --n_repeats=500 --dataset.config.random_labels=True
python change_labels.py --debug=False --model.name=Llama-2-13b-8bit --note=random --dataset.name=sst2  --dataset.config.num_prompt_examples=140 --n_repeats=500 --dataset.config.random_labels=True
python change_labels.py --debug=False --model.name=Llama-2-70b --note=random --dataset.name=sst2  --dataset.config.num_prompt_examples=140 --n_repeats=500 --dataset.config.random_labels=True
python change_labels.py --debug=False --model.name=Llama-2-7b-8bit --note=random --dataset.name=subj --dataset.config.num_prompt_examples=80 --n_repeats=500 --dataset.config.random_labels=True
python change_labels.py --debug=False --model.name=Llama-2-13b-8bit --note=random --dataset.name=subj --dataset.config.num_prompt_examples=80 --n_repeats=500 --dataset.config.random_labels=True
python change_labels.py --debug=False --model.name=Llama-2-70b --note=random --dataset.name=subj --dataset.config.num_prompt_examples=80 --n_repeats=500 --dataset.config.random_labels=True
python change_labels.py --debug=False --model.name=Llama-2-7b-8bit --note=random --dataset.name=financial_phrasebank --dataset.config.num_prompt_examples=80 --n_repeats=500 --dataset.config.random_labels=True
python change_labels.py --debug=False --model.name=Llama-2-13b-8bit --note=random --dataset.name=financial_phrasebank --dataset.config.num_prompt_examples=80 --n_repeats=500 --dataset.config.random_labels=True
python change_labels.py --debug=False --model.name=Llama-2-70b --note=random --dataset.name=financial_phrasebank --dataset.config.num_prompt_examples=80 --n_repeats=500 --dataset.config.random_labels=True
python change_labels.py --debug=False --model.name=Llama-2-7b-8bit --note=random --dataset.name=medical_questions_pairs --dataset.config.num_prompt_examples=50 --n_repeats=500 --dataset.config.random_labels=True
python change_labels.py --debug=False --model.name=Llama-2-13b-8bit --note=random --dataset.name=medical_questions_pairs --dataset.config.num_prompt_examples=50 --n_repeats=500 --dataset.config.random_labels=True
python change_labels.py --debug=False --model.name=Llama-2-70b --note=random --dataset.name=medical_questions_pairs --dataset.config.num_prompt_examples=50 --n_repeats=500 --dataset.config.random_labels=True
python change_labels.py --debug=False --model.name=Llama-2-7b-8bit --note=random --dataset.name=mrpc --dataset.config.num_prompt_examples=40 --n_repeats=500 --dataset.config.random_labels=True
python change_labels.py --debug=False --model.name=Llama-2-13b-8bit --note=random --dataset.name=mrpc --dataset.config.num_prompt_examples=40 --n_repeats=500 --dataset.config.random_labels=True
python change_labels.py --debug=False --model.name=Llama-2-70b --note=random --dataset.name=mrpc --dataset.config.num_prompt_examples=40 --n_repeats=500 --dataset.config.random_labels=True
python change_labels.py --debug=False --model.name=Llama-2-7b-8bit --note=random --dataset.name=wnli --dataset.config.num_prompt_examples=60 --n_repeats=500 --dataset.config.random_labels=True
python change_labels.py --debug=False --model.name=Llama-2-13b-8bit --note=random --dataset.name=wnli --dataset.config.num_prompt_examples=60 --n_repeats=500 --dataset.config.random_labels=True
python change_labels.py --debug=False --model.name=Llama-2-70b --note=random --dataset.name=wnli --dataset.config.num_prompt_examples=60 --n_repeats=500 --dataset.config.random_labels=True
python change_labels.py --debug=False --model.name=Llama-2-7b-8bit --note=random --dataset.name=hate_speech --dataset.config.num_prompt_examples=80 --n_repeats=500 --dataset.config.random_labels=True
python change_labels.py --debug=False --model.name=Llama-2-13b-8bit --note=random --dataset.name=hate_speech --dataset.config.num_prompt_examples=80 --n_repeats=500 --dataset.config.random_labels=True
python change_labels.py --debug=False --model.name=Llama-2-70b --note=random --dataset.name=hate_speech --dataset.config.num_prompt_examples=80 --n_repeats=500 --dataset.config.random_labels=True
python change_labels.py --debug=False --model.name=Llama-2-7b-8bit --note=random --dataset.name=rte --dataset.config.num_prompt_examples=30 --n_repeats=500 --dataset.config.random_labels=True
python change_labels.py --debug=False --model.name=Llama-2-13b-8bit --note=random --dataset.name=rte --dataset.config.num_prompt_examples=30 --n_repeats=500 --dataset.config.random_labels=True
python change_labels.py --debug=False --model.name=Llama-2-70b --note=random --dataset.name=rte --dataset.config.num_prompt_examples=30 --n_repeats=500 --dataset.config.random_labels=True
python change_labels.py --debug=False --model.name=Llama-2-7b-8bit --note=random --dataset.name=ag_news --dataset.config.num_prompt_examples=50 --n_repeats=500 --dataset.config.random_labels=True
python change_labels.py --debug=False --model.name=Llama-2-13b-8bit --note=random --dataset.name=ag_news --dataset.config.num_prompt_examples=50 --n_repeats=500 --dataset.config.random_labels=True
python change_labels.py --debug=False --model.name=Llama-2-70b --note=random --dataset.name=ag_news --dataset.config.num_prompt_examples=50 --n_repeats=500 --dataset.config.random_labels=True
```


## Few-Shot ICL with Replacement Labels
```
export CLASS_NAMES=ALL
python change_labels.py --debug=False --model.name=falcon-7b --note=change_labels --dataset.name=sst2  --dataset.config.num_prompt_examples=70 --n_repeats=100
python change_labels.py --debug=False --model.name=falcon-7b-instruct --note=change_labels --dataset.name=sst2  --dataset.config.num_prompt_examples=70 --n_repeats=100
python change_labels.py --debug=False --model.name=falcon-40b --note=change_labels --dataset.name=sst2  --dataset.config.num_prompt_examples=70 --n_repeats=100
python change_labels.py --debug=False --model.name=falcon-40b-instruct --note=change_labels --dataset.name=sst2  --dataset.config.num_prompt_examples=70 --n_repeats=100
python change_labels.py --debug=False --model.name=llama-7b-8bit --note=change_labels --dataset.name=sst2  --dataset.config.num_prompt_examples=70 --n_repeats=100
python change_labels.py --debug=False --model.name=llama-13b-8bit --note=change_labels --dataset.name=sst2  --dataset.config.num_prompt_examples=70 --n_repeats=100
python change_labels.py --debug=False --model.name=llama-65b --note=change_labels --dataset.name=sst2  --dataset.config.num_prompt_examples=70 --n_repeats=100
python change_labels.py --debug=False --model.name=falcon-7b --note=change_labels --dataset.name=subj --dataset.config.num_prompt_examples=40 --n_repeats=100
python change_labels.py --debug=False --model.name=falcon-7b-instruct --note=change_labels --dataset.name=subj --dataset.config.num_prompt_examples=40 --n_repeats=100
python change_labels.py --debug=False --model.name=falcon-40b --note=change_labels --dataset.name=subj --dataset.config.num_prompt_examples=40 --n_repeats=100
python change_labels.py --debug=False --model.name=falcon-40b-instruct --note=change_labels --dataset.name=subj --dataset.config.num_prompt_examples=40 --n_repeats=100
python change_labels.py --debug=False --model.name=llama-7b-8bit --note=change_labels --dataset.name=subj --dataset.config.num_prompt_examples=40 --n_repeats=100
python change_labels.py --debug=False --model.name=llama-13b-8bit --note=change_labels --dataset.name=subj --dataset.config.num_prompt_examples=40 --n_repeats=100
python change_labels.py --debug=False --model.name=llama-65b --note=change_labels --dataset.name=subj --dataset.config.num_prompt_examples=40 --n_repeats=100
python change_labels.py --debug=False --model.name=falcon-7b --note=change_labels --dataset.name=financial_phrasebank --dataset.config.num_prompt_examples=45 --n_repeats=100
python change_labels.py --debug=False --model.name=falcon-7b-instruct --note=change_labels --dataset.name=financial_phrasebank --dataset.config.num_prompt_examples=45 --n_repeats=100
python change_labels.py --debug=False --model.name=falcon-40b --note=change_labels --dataset.name=financial_phrasebank --dataset.config.num_prompt_examples=45 --n_repeats=100
python change_labels.py --debug=False --model.name=falcon-40b-instruct --note=change_labels --dataset.name=financial_phrasebank --dataset.config.num_prompt_examples=45 --n_repeats=100
python change_labels.py --debug=False --model.name=llama-7b-8bit --note=change_labels --dataset.name=financial_phrasebank --dataset.config.num_prompt_examples=45 --n_repeats=100
python change_labels.py --debug=False --model.name=llama-13b-8bit --note=change_labels --dataset.name=financial_phrasebank --dataset.config.num_prompt_examples=45 --n_repeats=100
python change_labels.py --debug=False --model.name=llama-65b --note=change_labels --dataset.name=financial_phrasebank --dataset.config.num_prompt_examples=45 --n_repeats=100
python change_labels.py --debug=False --model.name=falcon-7b --note=change_labels --dataset.name=medical_questions_pairs --dataset.config.num_prompt_examples=25 --n_repeats=100
python change_labels.py --debug=False --model.name=falcon-7b-instruct --note=change_labels --dataset.name=medical_questions_pairs --dataset.config.num_prompt_examples=25 --n_repeats=100
python change_labels.py --debug=False --model.name=falcon-40b --note=change_labels --dataset.name=medical_questions_pairs --dataset.config.num_prompt_examples=25 --n_repeats=100
python change_labels.py --debug=False --model.name=falcon-40b-instruct --note=change_labels --dataset.name=medical_questions_pairs --dataset.config.num_prompt_examples=25 --n_repeats=100
python change_labels.py --debug=False --model.name=llama-7b-8bit --note=change_labels --dataset.name=medical_questions_pairs --dataset.config.num_prompt_examples=25 --n_repeats=100
python change_labels.py --debug=False --model.name=llama-13b-8bit --note=change_labels --dataset.name=medical_questions_pairs --dataset.config.num_prompt_examples=25 --n_repeats=100
python change_labels.py --debug=False --model.name=llama-65b --note=change_labels --dataset.name=medical_questions_pairs --dataset.config.num_prompt_examples=25 --n_repeats=100
python change_labels.py --debug=False --model.name=falcon-7b --note=change_labels --dataset.name=mrpc --dataset.config.num_prompt_examples=20 --n_repeats=100
python change_labels.py --debug=False --model.name=falcon-7b-instruct --note=change_labels --dataset.name=mrpc --dataset.config.num_prompt_examples=20 --n_repeats=100
python change_labels.py --debug=False --model.name=falcon-40b --note=change_labels --dataset.name=mrpc --dataset.config.num_prompt_examples=20 --n_repeats=100
python change_labels.py --debug=False --model.name=falcon-40b-instruct --note=change_labels --dataset.name=mrpc --dataset.config.num_prompt_examples=20 --n_repeats=100
python change_labels.py --debug=False --model.name=llama-7b-8bit --note=change_labels --dataset.name=mrpc --dataset.config.num_prompt_examples=20 --n_repeats=100
python change_labels.py --debug=False --model.name=llama-13b-8bit --note=change_labels --dataset.name=mrpc --dataset.config.num_prompt_examples=20 --n_repeats=100
python change_labels.py --debug=False --model.name=llama-65b --note=change_labels --dataset.name=mrpc --dataset.config.num_prompt_examples=20 --n_repeats=100
python change_labels.py --debug=False --model.name=falcon-7b --note=change_labels --dataset.name=wnli --dataset.config.num_prompt_examples=40 --n_repeats=100
python change_labels.py --debug=False --model.name=falcon-7b-instruct --note=change_labels --dataset.name=wnli --dataset.config.num_prompt_examples=40 --n_repeats=100
python change_labels.py --debug=False --model.name=falcon-40b --note=change_labels --dataset.name=wnli --dataset.config.num_prompt_examples=40 --n_repeats=100
python change_labels.py --debug=False --model.name=falcon-40b-instruct --note=change_labels --dataset.name=wnli --dataset.config.num_prompt_examples=40 --n_repeats=100
python change_labels.py --debug=False --model.name=llama-7b-8bit --note=change_labels --dataset.name=wnli --dataset.config.num_prompt_examples=40 --n_repeats=100
python change_labels.py --debug=False --model.name=llama-13b-8bit --note=change_labels --dataset.name=wnli --dataset.config.num_prompt_examples=40 --n_repeats=100
python change_labels.py --debug=False --model.name=llama-65b --note=change_labels --dataset.name=wnli --dataset.config.num_prompt_examples=40 --n_repeats=100
python change_labels.py --debug=False --model.name=falcon-7b --note=change_labels --dataset.name=hate_speech --dataset.config.num_prompt_examples=40 --n_repeats=100
python change_labels.py --debug=False --model.name=falcon-7b-instruct --note=change_labels --dataset.name=hate_speech --dataset.config.num_prompt_examples=40 --n_repeats=100
python change_labels.py --debug=False --model.name=falcon-40b --note=change_labels --dataset.name=hate_speech --dataset.config.num_prompt_examples=40 --n_repeats=100
python change_labels.py --debug=False --model.name=falcon-40b-instruct --note=change_labels --dataset.name=hate_speech --dataset.config.num_prompt_examples=40 --n_repeats=100
python change_labels.py --debug=False --model.name=llama-7b-8bit --note=change_labels --dataset.name=hate_speech --dataset.config.num_prompt_examples=40 --n_repeats=100
python change_labels.py --debug=False --model.name=llama-13b-8bit --note=change_labels --dataset.name=hate_speech --dataset.config.num_prompt_examples=40 --n_repeats=100
python change_labels.py --debug=False --model.name=llama-65b --note=change_labels --dataset.name=hate_speech --dataset.config.num_prompt_examples=40 --n_repeats=100
python change_labels.py --debug=False --model.name=falcon-7b --note=change_labels --dataset.name=rte --dataset.config.num_prompt_examples=15 --n_repeats=100
python change_labels.py --debug=False --model.name=falcon-7b-instruct --note=change_labels --dataset.name=rte --dataset.config.num_prompt_examples=15 --n_repeats=100
python change_labels.py --debug=False --model.name=falcon-40b --note=change_labels --dataset.name=rte --dataset.config.num_prompt_examples=15 --n_repeats=100
python change_labels.py --debug=False --model.name=falcon-40b-instruct --note=change_labels --dataset.name=rte --dataset.config.num_prompt_examples=15 --n_repeats=100
python change_labels.py --debug=False --model.name=llama-7b-8bit --note=change_labels --dataset.name=rte --dataset.config.num_prompt_examples=15 --n_repeats=100
python change_labels.py --debug=False --model.name=llama-13b-8bit --note=change_labels --dataset.name=rte --dataset.config.num_prompt_examples=15 --n_repeats=100
python change_labels.py --debug=False --model.name=llama-65b --note=change_labels --dataset.name=rte --dataset.config.num_prompt_examples=15 --n_repeats=100
python change_labels.py --debug=False --model.name=falcon-7b --note=change_labels --dataset.name=ag_news --dataset.config.num_prompt_examples=30 --n_repeats=100
python change_labels.py --debug=False --model.name=falcon-7b-instruct --note=change_labels --dataset.name=ag_news --dataset.config.num_prompt_examples=30 --n_repeats=100
python change_labels.py --debug=False --model.name=falcon-40b --note=change_labels --dataset.name=ag_news --dataset.config.num_prompt_examples=30 --n_repeats=100
python change_labels.py --debug=False --model.name=falcon-40b-instruct --note=change_labels --dataset.name=ag_news --dataset.config.num_prompt_examples=30 --n_repeats=100
python change_labels.py --debug=False --model.name=llama-7b-8bit --note=change_labels --dataset.name=ag_news --dataset.config.num_prompt_examples=30 --n_repeats=100
python change_labels.py --debug=False --model.name=llama-13b-8bit --note=change_labels --dataset.name=ag_news --dataset.config.num_prompt_examples=30 --n_repeats=100
python change_labels.py --debug=False --model.name=llama-65b --note=change_labels --dataset.name=ag_news --dataset.config.num_prompt_examples=30 --n_repeats=100
python change_labels.py --debug=False --model.name=Llama-2-7b-8bit --note=change_labels --dataset.name=sst2  --dataset.config.num_prompt_examples=140 --n_repeats=100
python change_labels.py --debug=False --model.name=Llama-2-13b-8bit --note=change_labels --dataset.name=sst2  --dataset.config.num_prompt_examples=140 --n_repeats=100
python change_labels.py --debug=False --model.name=Llama-2-70b --note=change_labels --dataset.name=sst2  --dataset.config.num_prompt_examples=140 --n_repeats=100
python change_labels.py --debug=False --model.name=Llama-2-7b-8bit --note=change_labels --dataset.name=subj --dataset.config.num_prompt_examples=80 --n_repeats=100
python change_labels.py --debug=False --model.name=Llama-2-13b-8bit --note=change_labels --dataset.name=subj --dataset.config.num_prompt_examples=80 --n_repeats=100
python change_labels.py --debug=False --model.name=Llama-2-70b --note=change_labels --dataset.name=subj --dataset.config.num_prompt_examples=80 --n_repeats=100
python change_labels.py --debug=False --model.name=Llama-2-7b-8bit --note=change_labels --dataset.name=financial_phrasebank --dataset.config.num_prompt_examples=80 --n_repeats=100
python change_labels.py --debug=False --model.name=Llama-2-13b-8bit --note=change_labels --dataset.name=financial_phrasebank --dataset.config.num_prompt_examples=80 --n_repeats=100
python change_labels.py --debug=False --model.name=Llama-2-70b --note=change_labels --dataset.name=financial_phrasebank --dataset.config.num_prompt_examples=80 --n_repeats=100
python change_labels.py --debug=False --model.name=Llama-2-7b-8bit --note=change_labels --dataset.name=medical_questions_pairs --dataset.config.num_prompt_examples=50 --n_repeats=100
python change_labels.py --debug=False --model.name=Llama-2-13b-8bit --note=change_labels --dataset.name=medical_questions_pairs --dataset.config.num_prompt_examples=50 --n_repeats=100
python change_labels.py --debug=False --model.name=Llama-2-70b --note=change_labels --dataset.name=medical_questions_pairs --dataset.config.num_prompt_examples=50 --n_repeats=100
python change_labels.py --debug=False --model.name=Llama-2-7b-8bit --note=change_labels --dataset.name=mrpc --dataset.config.num_prompt_examples=40 --n_repeats=100
python change_labels.py --debug=False --model.name=Llama-2-13b-8bit --note=change_labels --dataset.name=mrpc --dataset.config.num_prompt_examples=40 --n_repeats=100
python change_labels.py --debug=False --model.name=Llama-2-70b --note=change_labels --dataset.name=mrpc --dataset.config.num_prompt_examples=40 --n_repeats=100
python change_labels.py --debug=False --model.name=Llama-2-7b-8bit --note=change_labels --dataset.name=wnli --dataset.config.num_prompt_examples=60 --n_repeats=100
python change_labels.py --debug=False --model.name=Llama-2-13b-8bit --note=change_labels --dataset.name=wnli --dataset.config.num_prompt_examples=60 --n_repeats=100
python change_labels.py --debug=False --model.name=Llama-2-70b --note=change_labels --dataset.name=wnli --dataset.config.num_prompt_examples=60 --n_repeats=100
python change_labels.py --debug=False --model.name=Llama-2-7b-8bit --note=change_labels --dataset.name=hate_speech --dataset.config.num_prompt_examples=80 --n_repeats=100
python change_labels.py --debug=False --model.name=Llama-2-13b-8bit --note=change_labels --dataset.name=hate_speech --dataset.config.num_prompt_examples=80 --n_repeats=100
python change_labels.py --debug=False --model.name=Llama-2-70b --note=change_labels --dataset.name=hate_speech --dataset.config.num_prompt_examples=80 --n_repeats=100
python change_labels.py --debug=False --model.name=Llama-2-7b-8bit --note=change_labels --dataset.name=rte --dataset.config.num_prompt_examples=30 --n_repeats=100
python change_labels.py --debug=False --model.name=Llama-2-13b-8bit --note=change_labels --dataset.name=rte --dataset.config.num_prompt_examples=30 --n_repeats=100
python change_labels.py --debug=False --model.name=Llama-2-70b --note=change_labels --dataset.name=rte --dataset.config.num_prompt_examples=30 --n_repeats=100
python change_labels.py --debug=False --model.name=Llama-2-7b-8bit --note=change_labels --dataset.name=ag_news --dataset.config.num_prompt_examples=50 --n_repeats=100
python change_labels.py --debug=False --model.name=Llama-2-13b-8bit --note=change_labels --dataset.name=ag_news --dataset.config.num_prompt_examples=50 --n_repeats=100
python change_labels.py --debug=False --model.name=Llama-2-70b --note=change_labels --dataset.name=ag_news --dataset.config.num_prompt_examples=50 --n_repeats=100
```


## Few-Shot ICL with Non-Stationary Labels

### Flip After Every Observation
```
export CLASS_NAMES=DEFAULT_ONLY
python change_labels.py --debug=False --model.name=falcon-7b --note=flip_after_1 --dataset.name=sst2  --dataset.config.num_prompt_examples=70 --n_repeats=500 --dataset.config.flip_labels_after=1 --dataset.config.flip_labels_multi=True
python change_labels.py --debug=False --model.name=falcon-7b-instruct --note=flip_after_1 --dataset.name=sst2  --dataset.config.num_prompt_examples=70 --n_repeats=500 --dataset.config.flip_labels_after=1 --dataset.config.flip_labels_multi=True
python change_labels.py --debug=False --model.name=falcon-40b --note=flip_after_1 --dataset.name=sst2  --dataset.config.num_prompt_examples=70 --n_repeats=500 --dataset.config.flip_labels_after=1 --dataset.config.flip_labels_multi=True
python change_labels.py --debug=False --model.name=falcon-40b-instruct --note=flip_after_1 --dataset.name=sst2  --dataset.config.num_prompt_examples=70 --n_repeats=500 --dataset.config.flip_labels_after=1 --dataset.config.flip_labels_multi=True
python change_labels.py --debug=False --model.name=llama-7b-8bit --note=flip_after_1 --dataset.name=sst2  --dataset.config.num_prompt_examples=70 --n_repeats=500 --dataset.config.flip_labels_after=1 --dataset.config.flip_labels_multi=True
python change_labels.py --debug=False --model.name=llama-13b-8bit --note=flip_after_1 --dataset.name=sst2  --dataset.config.num_prompt_examples=70 --n_repeats=500 --dataset.config.flip_labels_after=1 --dataset.config.flip_labels_multi=True
python change_labels.py --debug=False --model.name=llama-65b --note=flip_after_1 --dataset.name=sst2  --dataset.config.num_prompt_examples=70 --n_repeats=500 --dataset.config.flip_labels_after=1 --dataset.config.flip_labels_multi=True
python change_labels.py --debug=False --model.name=falcon-7b --note=flip_after_1 --dataset.name=subj --dataset.config.num_prompt_examples=40 --n_repeats=500 --dataset.config.flip_labels_after=1 --dataset.config.flip_labels_multi=True
python change_labels.py --debug=False --model.name=falcon-7b-instruct --note=flip_after_1 --dataset.name=subj --dataset.config.num_prompt_examples=40 --n_repeats=500 --dataset.config.flip_labels_after=1 --dataset.config.flip_labels_multi=True
python change_labels.py --debug=False --model.name=falcon-40b --note=flip_after_1 --dataset.name=subj --dataset.config.num_prompt_examples=40 --n_repeats=500 --dataset.config.flip_labels_after=1 --dataset.config.flip_labels_multi=True
python change_labels.py --debug=False --model.name=falcon-40b-instruct --note=flip_after_1 --dataset.name=subj --dataset.config.num_prompt_examples=40 --n_repeats=500 --dataset.config.flip_labels_after=1 --dataset.config.flip_labels_multi=True
python change_labels.py --debug=False --model.name=llama-7b-8bit --note=flip_after_1 --dataset.name=subj --dataset.config.num_prompt_examples=40 --n_repeats=500 --dataset.config.flip_labels_after=1 --dataset.config.flip_labels_multi=True
python change_labels.py --debug=False --model.name=llama-13b-8bit --note=flip_after_1 --dataset.name=subj --dataset.config.num_prompt_examples=40 --n_repeats=500 --dataset.config.flip_labels_after=1 --dataset.config.flip_labels_multi=True
python change_labels.py --debug=False --model.name=llama-65b --note=flip_after_1 --dataset.name=subj --dataset.config.num_prompt_examples=40 --n_repeats=500 --dataset.config.flip_labels_after=1 --dataset.config.flip_labels_multi=True
python change_labels.py --debug=False --model.name=falcon-7b --note=flip_after_1 --dataset.name=financial_phrasebank --dataset.config.num_prompt_examples=45 --n_repeats=500 --dataset.config.flip_labels_after=1 --dataset.config.flip_labels_multi=True
python change_labels.py --debug=False --model.name=falcon-7b-instruct --note=flip_after_1 --dataset.name=financial_phrasebank --dataset.config.num_prompt_examples=45 --n_repeats=500 --dataset.config.flip_labels_after=1 --dataset.config.flip_labels_multi=True
python change_labels.py --debug=False --model.name=falcon-40b --note=flip_after_1 --dataset.name=financial_phrasebank --dataset.config.num_prompt_examples=45 --n_repeats=500 --dataset.config.flip_labels_after=1 --dataset.config.flip_labels_multi=True
python change_labels.py --debug=False --model.name=falcon-40b-instruct --note=flip_after_1 --dataset.name=financial_phrasebank --dataset.config.num_prompt_examples=45 --n_repeats=500 --dataset.config.flip_labels_after=1 --dataset.config.flip_labels_multi=True
python change_labels.py --debug=False --model.name=llama-7b-8bit --note=flip_after_1 --dataset.name=financial_phrasebank --dataset.config.num_prompt_examples=45 --n_repeats=500 --dataset.config.flip_labels_after=1 --dataset.config.flip_labels_multi=True
python change_labels.py --debug=False --model.name=llama-13b-8bit --note=flip_after_1 --dataset.name=financial_phrasebank --dataset.config.num_prompt_examples=45 --n_repeats=500 --dataset.config.flip_labels_after=1 --dataset.config.flip_labels_multi=True
python change_labels.py --debug=False --model.name=llama-65b --note=flip_after_1 --dataset.name=financial_phrasebank --dataset.config.num_prompt_examples=45 --n_repeats=500 --dataset.config.flip_labels_after=1 --dataset.config.flip_labels_multi=True
python change_labels.py --debug=False --model.name=falcon-7b --note=flip_after_1 --dataset.name=medical_questions_pairs --dataset.config.num_prompt_examples=25 --n_repeats=500 --dataset.config.flip_labels_after=1 --dataset.config.flip_labels_multi=True
python change_labels.py --debug=False --model.name=falcon-7b-instruct --note=flip_after_1 --dataset.name=medical_questions_pairs --dataset.config.num_prompt_examples=25 --n_repeats=500 --dataset.config.flip_labels_after=1 --dataset.config.flip_labels_multi=True
python change_labels.py --debug=False --model.name=falcon-40b --note=flip_after_1 --dataset.name=medical_questions_pairs --dataset.config.num_prompt_examples=25 --n_repeats=500 --dataset.config.flip_labels_after=1 --dataset.config.flip_labels_multi=True
python change_labels.py --debug=False --model.name=falcon-40b-instruct --note=flip_after_1 --dataset.name=medical_questions_pairs --dataset.config.num_prompt_examples=25 --n_repeats=500 --dataset.config.flip_labels_after=1 --dataset.config.flip_labels_multi=True
python change_labels.py --debug=False --model.name=llama-7b-8bit --note=flip_after_1 --dataset.name=medical_questions_pairs --dataset.config.num_prompt_examples=25 --n_repeats=500 --dataset.config.flip_labels_after=1 --dataset.config.flip_labels_multi=True
python change_labels.py --debug=False --model.name=llama-13b-8bit --note=flip_after_1 --dataset.name=medical_questions_pairs --dataset.config.num_prompt_examples=25 --n_repeats=500 --dataset.config.flip_labels_after=1 --dataset.config.flip_labels_multi=True
python change_labels.py --debug=False --model.name=llama-65b --note=flip_after_1 --dataset.name=medical_questions_pairs --dataset.config.num_prompt_examples=25 --n_repeats=500 --dataset.config.flip_labels_after=1 --dataset.config.flip_labels_multi=True
python change_labels.py --debug=False --model.name=falcon-7b --note=flip_after_1 --dataset.name=mrpc --dataset.config.num_prompt_examples=20 --n_repeats=500 --dataset.config.flip_labels_after=1 --dataset.config.flip_labels_multi=True
python change_labels.py --debug=False --model.name=falcon-7b-instruct --note=flip_after_1 --dataset.name=mrpc --dataset.config.num_prompt_examples=20 --n_repeats=500 --dataset.config.flip_labels_after=1 --dataset.config.flip_labels_multi=True
python change_labels.py --debug=False --model.name=falcon-40b --note=flip_after_1 --dataset.name=mrpc --dataset.config.num_prompt_examples=20 --n_repeats=500 --dataset.config.flip_labels_after=1 --dataset.config.flip_labels_multi=True
python change_labels.py --debug=False --model.name=falcon-40b-instruct --note=flip_after_1 --dataset.name=mrpc --dataset.config.num_prompt_examples=20 --n_repeats=500 --dataset.config.flip_labels_after=1 --dataset.config.flip_labels_multi=True
python change_labels.py --debug=False --model.name=llama-7b-8bit --note=flip_after_1 --dataset.name=mrpc --dataset.config.num_prompt_examples=20 --n_repeats=500 --dataset.config.flip_labels_after=1 --dataset.config.flip_labels_multi=True
python change_labels.py --debug=False --model.name=llama-13b-8bit --note=flip_after_1 --dataset.name=mrpc --dataset.config.num_prompt_examples=20 --n_repeats=500 --dataset.config.flip_labels_after=1 --dataset.config.flip_labels_multi=True
python change_labels.py --debug=False --model.name=llama-65b --note=flip_after_1 --dataset.name=mrpc --dataset.config.num_prompt_examples=20 --n_repeats=500 --dataset.config.flip_labels_after=1 --dataset.config.flip_labels_multi=True
python change_labels.py --debug=False --model.name=falcon-7b --note=flip_after_1 --dataset.name=wnli --dataset.config.num_prompt_examples=40 --n_repeats=500 --dataset.config.flip_labels_after=1 --dataset.config.flip_labels_multi=True
python change_labels.py --debug=False --model.name=falcon-7b-instruct --note=flip_after_1 --dataset.name=wnli --dataset.config.num_prompt_examples=40 --n_repeats=500 --dataset.config.flip_labels_after=1 --dataset.config.flip_labels_multi=True
python change_labels.py --debug=False --model.name=falcon-40b --note=flip_after_1 --dataset.name=wnli --dataset.config.num_prompt_examples=40 --n_repeats=500 --dataset.config.flip_labels_after=1 --dataset.config.flip_labels_multi=True
python change_labels.py --debug=False --model.name=falcon-40b-instruct --note=flip_after_1 --dataset.name=wnli --dataset.config.num_prompt_examples=40 --n_repeats=500 --dataset.config.flip_labels_after=1 --dataset.config.flip_labels_multi=True
python change_labels.py --debug=False --model.name=llama-7b-8bit --note=flip_after_1 --dataset.name=wnli --dataset.config.num_prompt_examples=40 --n_repeats=500 --dataset.config.flip_labels_after=1 --dataset.config.flip_labels_multi=True
python change_labels.py --debug=False --model.name=llama-13b-8bit --note=flip_after_1 --dataset.name=wnli --dataset.config.num_prompt_examples=40 --n_repeats=500 --dataset.config.flip_labels_after=1 --dataset.config.flip_labels_multi=True
python change_labels.py --debug=False --model.name=llama-65b --note=flip_after_1 --dataset.name=wnli --dataset.config.num_prompt_examples=40 --n_repeats=500 --dataset.config.flip_labels_after=1 --dataset.config.flip_labels_multi=True
python change_labels.py --debug=False --model.name=falcon-7b --note=flip_after_1 --dataset.name=hate_speech --dataset.config.num_prompt_examples=40 --n_repeats=500 --dataset.config.flip_labels_after=1 --dataset.config.flip_labels_multi=True
python change_labels.py --debug=False --model.name=falcon-7b-instruct --note=flip_after_1 --dataset.name=hate_speech --dataset.config.num_prompt_examples=40 --n_repeats=500 --dataset.config.flip_labels_after=1 --dataset.config.flip_labels_multi=True
python change_labels.py --debug=False --model.name=falcon-40b --note=flip_after_1 --dataset.name=hate_speech --dataset.config.num_prompt_examples=40 --n_repeats=500 --dataset.config.flip_labels_after=1 --dataset.config.flip_labels_multi=True
python change_labels.py --debug=False --model.name=falcon-40b-instruct --note=flip_after_1 --dataset.name=hate_speech --dataset.config.num_prompt_examples=40 --n_repeats=500 --dataset.config.flip_labels_after=1 --dataset.config.flip_labels_multi=True
python change_labels.py --debug=False --model.name=llama-7b-8bit --note=flip_after_1 --dataset.name=hate_speech --dataset.config.num_prompt_examples=40 --n_repeats=500 --dataset.config.flip_labels_after=1 --dataset.config.flip_labels_multi=True
python change_labels.py --debug=False --model.name=llama-13b-8bit --note=flip_after_1 --dataset.name=hate_speech --dataset.config.num_prompt_examples=40 --n_repeats=500 --dataset.config.flip_labels_after=1 --dataset.config.flip_labels_multi=True
python change_labels.py --debug=False --model.name=llama-65b --note=flip_after_1 --dataset.name=hate_speech --dataset.config.num_prompt_examples=40 --n_repeats=500 --dataset.config.flip_labels_after=1 --dataset.config.flip_labels_multi=True
python change_labels.py --debug=False --model.name=falcon-7b --note=flip_after_1 --dataset.name=rte --dataset.config.num_prompt_examples=15 --n_repeats=500 --dataset.config.flip_labels_after=1 --dataset.config.flip_labels_multi=True
python change_labels.py --debug=False --model.name=falcon-7b-instruct --note=flip_after_1 --dataset.name=rte --dataset.config.num_prompt_examples=15 --n_repeats=500 --dataset.config.flip_labels_after=1 --dataset.config.flip_labels_multi=True
python change_labels.py --debug=False --model.name=falcon-40b --note=flip_after_1 --dataset.name=rte --dataset.config.num_prompt_examples=15 --n_repeats=500 --dataset.config.flip_labels_after=1 --dataset.config.flip_labels_multi=True
python change_labels.py --debug=False --model.name=falcon-40b-instruct --note=flip_after_1 --dataset.name=rte --dataset.config.num_prompt_examples=15 --n_repeats=500 --dataset.config.flip_labels_after=1 --dataset.config.flip_labels_multi=True
python change_labels.py --debug=False --model.name=llama-7b-8bit --note=flip_after_1 --dataset.name=rte --dataset.config.num_prompt_examples=15 --n_repeats=500 --dataset.config.flip_labels_after=1 --dataset.config.flip_labels_multi=True
python change_labels.py --debug=False --model.name=llama-13b-8bit --note=flip_after_1 --dataset.name=rte --dataset.config.num_prompt_examples=15 --n_repeats=500 --dataset.config.flip_labels_after=1 --dataset.config.flip_labels_multi=True
python change_labels.py --debug=False --model.name=llama-65b --note=flip_after_1 --dataset.name=rte --dataset.config.num_prompt_examples=15 --n_repeats=500 --dataset.config.flip_labels_after=1 --dataset.config.flip_labels_multi=True
python change_labels.py --debug=False --model.name=falcon-7b --note=flip_after_1 --dataset.name=ag_news --dataset.config.num_prompt_examples=30 --n_repeats=500 --dataset.config.flip_labels_after=1 --dataset.config.flip_labels_multi=True
python change_labels.py --debug=False --model.name=falcon-7b-instruct --note=flip_after_1 --dataset.name=ag_news --dataset.config.num_prompt_examples=30 --n_repeats=500 --dataset.config.flip_labels_after=1 --dataset.config.flip_labels_multi=True
python change_labels.py --debug=False --model.name=falcon-40b --note=flip_after_1 --dataset.name=ag_news --dataset.config.num_prompt_examples=30 --n_repeats=500 --dataset.config.flip_labels_after=1 --dataset.config.flip_labels_multi=True
python change_labels.py --debug=False --model.name=falcon-40b-instruct --note=flip_after_1 --dataset.name=ag_news --dataset.config.num_prompt_examples=30 --n_repeats=500 --dataset.config.flip_labels_after=1 --dataset.config.flip_labels_multi=True
python change_labels.py --debug=False --model.name=llama-7b-8bit --note=flip_after_1 --dataset.name=ag_news --dataset.config.num_prompt_examples=30 --n_repeats=500 --dataset.config.flip_labels_after=1 --dataset.config.flip_labels_multi=True
python change_labels.py --debug=False --model.name=llama-13b-8bit --note=flip_after_1 --dataset.name=ag_news --dataset.config.num_prompt_examples=30 --n_repeats=500 --dataset.config.flip_labels_after=1 --dataset.config.flip_labels_multi=True
python change_labels.py --debug=False --model.name=llama-65b --note=flip_after_1 --dataset.name=ag_news --dataset.config.num_prompt_examples=30 --n_repeats=500 --dataset.config.flip_labels_after=1 --dataset.config.flip_labels_multi=True
python change_labels.py --debug=False --model.name=Llama-2-7b-8bit --note=flip_after_1 --dataset.name=sst2  --dataset.config.num_prompt_examples=140 --n_repeats=500 --dataset.config.flip_labels_after=1 --dataset.config.flip_labels_multi=True
python change_labels.py --debug=False --model.name=Llama-2-13b-8bit --note=flip_after_1 --dataset.name=sst2  --dataset.config.num_prompt_examples=140 --n_repeats=500 --dataset.config.flip_labels_after=1 --dataset.config.flip_labels_multi=True
python change_labels.py --debug=False --model.name=Llama-2-70b --note=flip_after_1 --dataset.name=sst2  --dataset.config.num_prompt_examples=140 --n_repeats=500 --dataset.config.flip_labels_after=1 --dataset.config.flip_labels_multi=True
python change_labels.py --debug=False --model.name=Llama-2-7b-8bit --note=flip_after_1 --dataset.name=subj --dataset.config.num_prompt_examples=80 --n_repeats=500 --dataset.config.flip_labels_after=1 --dataset.config.flip_labels_multi=True
python change_labels.py --debug=False --model.name=Llama-2-13b-8bit --note=flip_after_1 --dataset.name=subj --dataset.config.num_prompt_examples=80 --n_repeats=500 --dataset.config.flip_labels_after=1 --dataset.config.flip_labels_multi=True
python change_labels.py --debug=False --model.name=Llama-2-70b --note=flip_after_1 --dataset.name=subj --dataset.config.num_prompt_examples=80 --n_repeats=500 --dataset.config.flip_labels_after=1 --dataset.config.flip_labels_multi=True
python change_labels.py --debug=False --model.name=Llama-2-7b-8bit --note=flip_after_1 --dataset.name=financial_phrasebank --dataset.config.num_prompt_examples=80 --n_repeats=500 --dataset.config.flip_labels_after=1 --dataset.config.flip_labels_multi=True
python change_labels.py --debug=False --model.name=Llama-2-13b-8bit --note=flip_after_1 --dataset.name=financial_phrasebank --dataset.config.num_prompt_examples=80 --n_repeats=500 --dataset.config.flip_labels_after=1 --dataset.config.flip_labels_multi=True
python change_labels.py --debug=False --model.name=Llama-2-70b --note=flip_after_1 --dataset.name=financial_phrasebank --dataset.config.num_prompt_examples=80 --n_repeats=500 --dataset.config.flip_labels_after=1 --dataset.config.flip_labels_multi=True
python change_labels.py --debug=False --model.name=Llama-2-7b-8bit --note=flip_after_1 --dataset.name=medical_questions_pairs --dataset.config.num_prompt_examples=50 --n_repeats=500 --dataset.config.flip_labels_after=1 --dataset.config.flip_labels_multi=True
python change_labels.py --debug=False --model.name=Llama-2-13b-8bit --note=flip_after_1 --dataset.name=medical_questions_pairs --dataset.config.num_prompt_examples=50 --n_repeats=500 --dataset.config.flip_labels_after=1 --dataset.config.flip_labels_multi=True
python change_labels.py --debug=False --model.name=Llama-2-70b --note=flip_after_1 --dataset.name=medical_questions_pairs --dataset.config.num_prompt_examples=50 --n_repeats=500 --dataset.config.flip_labels_after=1 --dataset.config.flip_labels_multi=True
python change_labels.py --debug=False --model.name=Llama-2-7b-8bit --note=flip_after_1 --dataset.name=mrpc --dataset.config.num_prompt_examples=40 --n_repeats=500 --dataset.config.flip_labels_after=1 --dataset.config.flip_labels_multi=True
python change_labels.py --debug=False --model.name=Llama-2-13b-8bit --note=flip_after_1 --dataset.name=mrpc --dataset.config.num_prompt_examples=40 --n_repeats=500 --dataset.config.flip_labels_after=1 --dataset.config.flip_labels_multi=True
python change_labels.py --debug=False --model.name=Llama-2-70b --note=flip_after_1 --dataset.name=mrpc --dataset.config.num_prompt_examples=40 --n_repeats=500 --dataset.config.flip_labels_after=1 --dataset.config.flip_labels_multi=True
python change_labels.py --debug=False --model.name=Llama-2-7b-8bit --note=flip_after_1 --dataset.name=wnli --dataset.config.num_prompt_examples=60 --n_repeats=500 --dataset.config.flip_labels_after=1 --dataset.config.flip_labels_multi=True
python change_labels.py --debug=False --model.name=Llama-2-13b-8bit --note=flip_after_1 --dataset.name=wnli --dataset.config.num_prompt_examples=60 --n_repeats=500 --dataset.config.flip_labels_after=1 --dataset.config.flip_labels_multi=True
python change_labels.py --debug=False --model.name=Llama-2-70b --note=flip_after_1 --dataset.name=wnli --dataset.config.num_prompt_examples=60 --n_repeats=500 --dataset.config.flip_labels_after=1 --dataset.config.flip_labels_multi=True
python change_labels.py --debug=False --model.name=Llama-2-7b-8bit --note=flip_after_1 --dataset.name=hate_speech --dataset.config.num_prompt_examples=80 --n_repeats=500 --dataset.config.flip_labels_after=1 --dataset.config.flip_labels_multi=True
python change_labels.py --debug=False --model.name=Llama-2-13b-8bit --note=flip_after_1 --dataset.name=hate_speech --dataset.config.num_prompt_examples=80 --n_repeats=500 --dataset.config.flip_labels_after=1 --dataset.config.flip_labels_multi=True
python change_labels.py --debug=False --model.name=Llama-2-70b --note=flip_after_1 --dataset.name=hate_speech --dataset.config.num_prompt_examples=80 --n_repeats=500 --dataset.config.flip_labels_after=1 --dataset.config.flip_labels_multi=True
python change_labels.py --debug=False --model.name=Llama-2-7b-8bit --note=flip_after_1 --dataset.name=rte --dataset.config.num_prompt_examples=30 --n_repeats=500 --dataset.config.flip_labels_after=1 --dataset.config.flip_labels_multi=True
python change_labels.py --debug=False --model.name=Llama-2-13b-8bit --note=flip_after_1 --dataset.name=rte --dataset.config.num_prompt_examples=30 --n_repeats=500 --dataset.config.flip_labels_after=1 --dataset.config.flip_labels_multi=True
python change_labels.py --debug=False --model.name=Llama-2-70b --note=flip_after_1 --dataset.name=rte --dataset.config.num_prompt_examples=30 --n_repeats=500 --dataset.config.flip_labels_after=1 --dataset.config.flip_labels_multi=True
python change_labels.py --debug=False --model.name=Llama-2-7b-8bit --note=flip_after_1 --dataset.name=ag_news --dataset.config.num_prompt_examples=50 --n_repeats=500 --dataset.config.flip_labels_after=1 --dataset.config.flip_labels_multi=True
python change_labels.py --debug=False --model.name=Llama-2-13b-8bit --note=flip_after_1 --dataset.name=ag_news --dataset.config.num_prompt_examples=50 --n_repeats=500 --dataset.config.flip_labels_after=1 --dataset.config.flip_labels_multi=True
python change_labels.py --debug=False --model.name=Llama-2-70b --note=flip_after_1 --dataset.name=ag_news --dataset.config.num_prompt_examples=50 --n_repeats=500 --dataset.config.flip_labels_after=1 --dataset.config.flip_labels_multi=True
```

### Flip Halfway
```
export CLASS_NAMES=DEFAULT_AND_FLIP
python change_labels.py --debug=False --model.name=falcon-7b --note=centered_flip_labels_both_directions --dataset.name=sst2 --dataset.config.num_prompt_examples=70 --dataset.config.flip_labels_after=30 --n_repeats=500
python change_labels.py --debug=False --model.name=falcon-7b-instruct --note=centered_flip_labels_both_directions --dataset.name=sst2 --dataset.config.num_prompt_examples=70 --dataset.config.flip_labels_after=30 --n_repeats=500
python change_labels.py --debug=False --model.name=falcon-40b --note=centered_flip_labels_both_directions --dataset.name=sst2 --dataset.config.num_prompt_examples=70 --dataset.config.flip_labels_after=30 --n_repeats=500
python change_labels.py --debug=False --model.name=falcon-40b-instruct --note=centered_flip_labels_both_directions --dataset.name=sst2 --dataset.config.num_prompt_examples=70 --dataset.config.flip_labels_after=30 --n_repeats=500
python change_labels.py --debug=False --model.name=llama-7b-8bit --note=centered_flip_labels_both_directions --dataset.name=sst2 --dataset.config.num_prompt_examples=70 --dataset.config.flip_labels_after=30 --n_repeats=500
python change_labels.py --debug=False --model.name=llama-13b-8bit --note=centered_flip_labels_both_directions --dataset.name=sst2 --dataset.config.num_prompt_examples=70 --dataset.config.flip_labels_after=30 --n_repeats=500
python change_labels.py --debug=False --model.name=llama-65b --note=centered_flip_labels_both_directions --dataset.name=sst2 --dataset.config.num_prompt_examples=70 --dataset.config.flip_labels_after=30 --n_repeats=500
python change_labels.py --debug=False --model.name=falcon-7b --note=centered_flip_labels_both_directions --dataset.name=subj --dataset.config.num_prompt_examples=40 --dataset.config.flip_labels_after=15 --n_repeats=500
python change_labels.py --debug=False --model.name=falcon-7b-instruct --note=centered_flip_labels_both_directions --dataset.name=subj --dataset.config.num_prompt_examples=40 --dataset.config.flip_labels_after=15 --n_repeats=500
python change_labels.py --debug=False --model.name=falcon-40b --note=centered_flip_labels_both_directions --dataset.name=subj --dataset.config.num_prompt_examples=40 --dataset.config.flip_labels_after=15 --n_repeats=500
python change_labels.py --debug=False --model.name=falcon-40b-instruct --note=centered_flip_labels_both_directions --dataset.name=subj --dataset.config.num_prompt_examples=40 --dataset.config.flip_labels_after=15 --n_repeats=500
python change_labels.py --debug=False --model.name=llama-7b-8bit --note=centered_flip_labels_both_directions --dataset.name=subj --dataset.config.num_prompt_examples=40 --dataset.config.flip_labels_after=15 --n_repeats=500
python change_labels.py --debug=False --model.name=llama-13b-8bit --note=centered_flip_labels_both_directions --dataset.name=subj --dataset.config.num_prompt_examples=40 --dataset.config.flip_labels_after=15 --n_repeats=500
python change_labels.py --debug=False --model.name=llama-65b --note=centered_flip_labels_both_directions --dataset.name=subj --dataset.config.num_prompt_examples=40 --dataset.config.flip_labels_after=15 --n_repeats=500
python change_labels.py --debug=False --model.name=falcon-7b --note=centered_flip_labels_both_directions --dataset.name=financial_phrasebank --dataset.config.num_prompt_examples=45 --dataset.config.flip_labels_after=15 --n_repeats=500
python change_labels.py --debug=False --model.name=falcon-7b-instruct --note=centered_flip_labels_both_directions --dataset.name=financial_phrasebank --dataset.config.num_prompt_examples=45 --dataset.config.flip_labels_after=15 --n_repeats=500
python change_labels.py --debug=False --model.name=falcon-40b --note=centered_flip_labels_both_directions --dataset.name=financial_phrasebank --dataset.config.num_prompt_examples=45 --dataset.config.flip_labels_after=15 --n_repeats=500
python change_labels.py --debug=False --model.name=falcon-40b-instruct --note=centered_flip_labels_both_directions --dataset.name=financial_phrasebank --dataset.config.num_prompt_examples=45 --dataset.config.flip_labels_after=15 --n_repeats=500
python change_labels.py --debug=False --model.name=llama-7b-8bit --note=centered_flip_labels_both_directions --dataset.name=financial_phrasebank --dataset.config.num_prompt_examples=45 --dataset.config.flip_labels_after=15 --n_repeats=500
python change_labels.py --debug=False --model.name=llama-13b-8bit --note=centered_flip_labels_both_directions --dataset.name=financial_phrasebank --dataset.config.num_prompt_examples=45 --dataset.config.flip_labels_after=15 --n_repeats=500
python change_labels.py --debug=False --model.name=llama-65b --note=centered_flip_labels_both_directions --dataset.name=financial_phrasebank --dataset.config.num_prompt_examples=45 --dataset.config.flip_labels_after=15 --n_repeats=500
python change_labels.py --debug=False --model.name=falcon-7b --note=centered_flip_labels_both_directions --dataset.name=medical_questions_pairs --dataset.config.num_prompt_examples=25 --dataset.config.flip_labels_after=10 --n_repeats=500
python change_labels.py --debug=False --model.name=falcon-7b-instruct --note=centered_flip_labels_both_directions --dataset.name=medical_questions_pairs --dataset.config.num_prompt_examples=25 --dataset.config.flip_labels_after=10 --n_repeats=500
python change_labels.py --debug=False --model.name=falcon-40b --note=centered_flip_labels_both_directions --dataset.name=medical_questions_pairs --dataset.config.num_prompt_examples=25 --dataset.config.flip_labels_after=10 --n_repeats=500
python change_labels.py --debug=False --model.name=falcon-40b-instruct --note=centered_flip_labels_both_directions --dataset.name=medical_questions_pairs --dataset.config.num_prompt_examples=25 --dataset.config.flip_labels_after=10 --n_repeats=500
python change_labels.py --debug=False --model.name=llama-7b-8bit --note=centered_flip_labels_both_directions --dataset.name=medical_questions_pairs --dataset.config.num_prompt_examples=25 --dataset.config.flip_labels_after=10 --n_repeats=500
python change_labels.py --debug=False --model.name=llama-13b-8bit --note=centered_flip_labels_both_directions --dataset.name=medical_questions_pairs --dataset.config.num_prompt_examples=25 --dataset.config.flip_labels_after=10 --n_repeats=500
python change_labels.py --debug=False --model.name=llama-65b --note=centered_flip_labels_both_directions --dataset.name=medical_questions_pairs --dataset.config.num_prompt_examples=25 --dataset.config.flip_labels_after=10 --n_repeats=500
python change_labels.py --debug=False --model.name=falcon-7b --note=centered_flip_labels_both_directions --dataset.name=mrpc --dataset.config.num_prompt_examples=20 --dataset.config.flip_labels_after=10 --n_repeats=500
python change_labels.py --debug=False --model.name=falcon-7b-instruct --note=centered_flip_labels_both_directions --dataset.name=mrpc --dataset.config.num_prompt_examples=20 --dataset.config.flip_labels_after=10 --n_repeats=500
python change_labels.py --debug=False --model.name=falcon-40b --note=centered_flip_labels_both_directions --dataset.name=mrpc --dataset.config.num_prompt_examples=20 --dataset.config.flip_labels_after=10 --n_repeats=500
python change_labels.py --debug=False --model.name=falcon-40b-instruct --note=centered_flip_labels_both_directions --dataset.name=mrpc --dataset.config.num_prompt_examples=20 --dataset.config.flip_labels_after=10 --n_repeats=500
python change_labels.py --debug=False --model.name=llama-7b-8bit --note=centered_flip_labels_both_directions --dataset.name=mrpc --dataset.config.num_prompt_examples=20 --dataset.config.flip_labels_after=10 --n_repeats=500
python change_labels.py --debug=False --model.name=llama-13b-8bit --note=centered_flip_labels_both_directions --dataset.name=mrpc --dataset.config.num_prompt_examples=20 --dataset.config.flip_labels_after=10 --n_repeats=500
python change_labels.py --debug=False --model.name=llama-65b --note=centered_flip_labels_both_directions --dataset.name=mrpc --dataset.config.num_prompt_examples=20 --dataset.config.flip_labels_after=10 --n_repeats=500
python change_labels.py --debug=False --model.name=falcon-7b --note=centered_flip_labels_both_directions --dataset.name=wnli --dataset.config.num_prompt_examples=40 --dataset.config.flip_labels_after=10 --n_repeats=500
python change_labels.py --debug=False --model.name=falcon-7b-instruct --note=centered_flip_labels_both_directions --dataset.name=wnli --dataset.config.num_prompt_examples=40 --dataset.config.flip_labels_after=10 --n_repeats=500
python change_labels.py --debug=False --model.name=falcon-40b --note=centered_flip_labels_both_directions --dataset.name=wnli --dataset.config.num_prompt_examples=40 --dataset.config.flip_labels_after=10 --n_repeats=500
python change_labels.py --debug=False --model.name=falcon-40b-instruct --note=centered_flip_labels_both_directions --dataset.name=wnli --dataset.config.num_prompt_examples=40 --dataset.config.flip_labels_after=10 --n_repeats=500
python change_labels.py --debug=False --model.name=llama-7b-8bit --note=centered_flip_labels_both_directions --dataset.name=wnli --dataset.config.num_prompt_examples=40 --dataset.config.flip_labels_after=10 --n_repeats=500
python change_labels.py --debug=False --model.name=llama-13b-8bit --note=centered_flip_labels_both_directions --dataset.name=wnli --dataset.config.num_prompt_examples=40 --dataset.config.flip_labels_after=10 --n_repeats=500
python change_labels.py --debug=False --model.name=llama-65b --note=centered_flip_labels_both_directions --dataset.name=wnli --dataset.config.num_prompt_examples=40 --dataset.config.flip_labels_after=10 --n_repeats=500
python change_labels.py --debug=False --model.name=falcon-7b --note=centered_flip_labels_both_directions --dataset.name=hate_speech --dataset.config.num_prompt_examples=40 --dataset.config.flip_labels_after=10 --n_repeats=500
python change_labels.py --debug=False --model.name=falcon-7b-instruct --note=centered_flip_labels_both_directions --dataset.name=hate_speech --dataset.config.num_prompt_examples=40 --dataset.config.flip_labels_after=10 --n_repeats=500
python change_labels.py --debug=False --model.name=falcon-40b --note=centered_flip_labels_both_directions --dataset.name=hate_speech --dataset.config.num_prompt_examples=40 --dataset.config.flip_labels_after=10 --n_repeats=500
python change_labels.py --debug=False --model.name=falcon-40b-instruct --note=centered_flip_labels_both_directions --dataset.name=hate_speech --dataset.config.num_prompt_examples=40 --dataset.config.flip_labels_after=10 --n_repeats=500
python change_labels.py --debug=False --model.name=llama-7b-8bit --note=centered_flip_labels_both_directions --dataset.name=hate_speech --dataset.config.num_prompt_examples=40 --dataset.config.flip_labels_after=10 --n_repeats=500
python change_labels.py --debug=False --model.name=llama-13b-8bit --note=centered_flip_labels_both_directions --dataset.name=hate_speech --dataset.config.num_prompt_examples=40 --dataset.config.flip_labels_after=10 --n_repeats=500
python change_labels.py --debug=False --model.name=llama-65b --note=centered_flip_labels_both_directions --dataset.name=hate_speech --dataset.config.num_prompt_examples=40 --dataset.config.flip_labels_after=10 --n_repeats=500
python change_labels.py --debug=False --model.name=falcon-7b --note=centered_flip_labels_both_directions --dataset.name=rte --dataset.config.num_prompt_examples=15 --dataset.config.flip_labels_after=5 --n_repeats=500
python change_labels.py --debug=False --model.name=falcon-7b-instruct --note=centered_flip_labels_both_directions --dataset.name=rte --dataset.config.num_prompt_examples=15 --dataset.config.flip_labels_after=5 --n_repeats=500
python change_labels.py --debug=False --model.name=falcon-40b --note=centered_flip_labels_both_directions --dataset.name=rte --dataset.config.num_prompt_examples=15 --dataset.config.flip_labels_after=5 --n_repeats=500
python change_labels.py --debug=False --model.name=falcon-40b-instruct --note=centered_flip_labels_both_directions --dataset.name=rte --dataset.config.num_prompt_examples=15 --dataset.config.flip_labels_after=5 --n_repeats=500
python change_labels.py --debug=False --model.name=llama-7b-8bit --note=centered_flip_labels_both_directions --dataset.name=rte --dataset.config.num_prompt_examples=15 --dataset.config.flip_labels_after=5 --n_repeats=500
python change_labels.py --debug=False --model.name=llama-13b-8bit --note=centered_flip_labels_both_directions --dataset.name=rte --dataset.config.num_prompt_examples=15 --dataset.config.flip_labels_after=5 --n_repeats=500
python change_labels.py --debug=False --model.name=llama-65b --note=centered_flip_labels_both_directions --dataset.name=rte --dataset.config.num_prompt_examples=15 --dataset.config.flip_labels_after=5 --n_repeats=500
python change_labels.py --debug=False --model.name=falcon-7b --note=centered_flip_labels_both_directions --dataset.name=ag_news --dataset.config.num_prompt_examples=30 --dataset.config.flip_labels_after=10 --n_repeats=500
python change_labels.py --debug=False --model.name=falcon-7b-instruct --note=centered_flip_labels_both_directions --dataset.name=ag_news --dataset.config.num_prompt_examples=30 --dataset.config.flip_labels_after=10 --n_repeats=500
python change_labels.py --debug=False --model.name=falcon-40b --note=centered_flip_labels_both_directions --dataset.name=ag_news --dataset.config.num_prompt_examples=30 --dataset.config.flip_labels_after=10 --n_repeats=500
python change_labels.py --debug=False --model.name=falcon-40b-instruct --note=centered_flip_labels_both_directions --dataset.name=ag_news --dataset.config.num_prompt_examples=30 --dataset.config.flip_labels_after=10 --n_repeats=500
python change_labels.py --debug=False --model.name=llama-7b-8bit --note=centered_flip_labels_both_directions --dataset.name=ag_news --dataset.config.num_prompt_examples=30 --dataset.config.flip_labels_after=10 --n_repeats=500
python change_labels.py --debug=False --model.name=llama-13b-8bit --note=centered_flip_labels_both_directions --dataset.name=ag_news --dataset.config.num_prompt_examples=30 --dataset.config.flip_labels_after=10 --n_repeats=500
python change_labels.py --debug=False --model.name=llama-65b --note=centered_flip_labels_both_directions --dataset.name=ag_news --dataset.config.num_prompt_examples=30 --dataset.config.flip_labels_after=10 --n_repeats=500
python change_labels.py --debug=False --model.name=Llama-2-7b-8bit --note=centered_flip_labels_both_directions --dataset.name=sst2 --dataset.config.num_prompt_examples=140 --dataset.config.flip_labels_after=60 --n_repeats=500
python change_labels.py --debug=False --model.name=Llama-2-13b-8bit --note=centered_flip_labels_both_directions --dataset.name=sst2 --dataset.config.num_prompt_examples=140 --dataset.config.flip_labels_after=60 --n_repeats=500
python change_labels.py --debug=False --model.name=Llama-2-70b --note=centered_flip_labels_both_directions --dataset.name=sst2 --dataset.config.num_prompt_examples=140 --dataset.config.flip_labels_after=60 --n_repeats=500
python change_labels.py --debug=False --model.name=Llama-2-7b-8bit --note=centered_flip_labels_both_directions --dataset.name=subj --dataset.config.num_prompt_examples=80 --dataset.config.flip_labels_after=30 --n_repeats=500
python change_labels.py --debug=False --model.name=Llama-2-13b-8bit --note=centered_flip_labels_both_directions --dataset.name=subj --dataset.config.num_prompt_examples=80 --dataset.config.flip_labels_after=30 --n_repeats=500
python change_labels.py --debug=False --model.name=Llama-2-70b --note=centered_flip_labels_both_directions --dataset.name=subj --dataset.config.num_prompt_examples=80 --dataset.config.flip_labels_after=30 --n_repeats=500
python change_labels.py --debug=False --model.name=Llama-2-7b-8bit --note=centered_flip_labels_both_directions --dataset.name=financial_phrasebank --dataset.config.num_prompt_examples=80 --dataset.config.flip_labels_after=30 --n_repeats=500
python change_labels.py --debug=False --model.name=Llama-2-13b-8bit --note=centered_flip_labels_both_directions --dataset.name=financial_phrasebank --dataset.config.num_prompt_examples=80 --dataset.config.flip_labels_after=30 --n_repeats=500
python change_labels.py --debug=False --model.name=Llama-2-70b --note=centered_flip_labels_both_directions --dataset.name=financial_phrasebank --dataset.config.num_prompt_examples=80 --dataset.config.flip_labels_after=30 --n_repeats=500
python change_labels.py --debug=False --model.name=Llama-2-7b-8bit --note=centered_flip_labels_both_directions --dataset.name=medical_questions_pairs --dataset.config.num_prompt_examples=50 --dataset.config.flip_labels_after=20 --n_repeats=500
python change_labels.py --debug=False --model.name=Llama-2-13b-8bit --note=centered_flip_labels_both_directions --dataset.name=medical_questions_pairs --dataset.config.num_prompt_examples=50 --dataset.config.flip_labels_after=20 --n_repeats=500
python change_labels.py --debug=False --model.name=Llama-2-70b --note=centered_flip_labels_both_directions --dataset.name=medical_questions_pairs --dataset.config.num_prompt_examples=50 --dataset.config.flip_labels_after=20 --n_repeats=500
python change_labels.py --debug=False --model.name=Llama-2-7b-8bit --note=centered_flip_labels_both_directions --dataset.name=mrpc --dataset.config.num_prompt_examples=40 --dataset.config.flip_labels_after=17 --n_repeats=500
python change_labels.py --debug=False --model.name=Llama-2-13b-8bit --note=centered_flip_labels_both_directions --dataset.name=mrpc --dataset.config.num_prompt_examples=40 --dataset.config.flip_labels_after=17 --n_repeats=500
python change_labels.py --debug=False --model.name=Llama-2-70b --note=centered_flip_labels_both_directions --dataset.name=mrpc --dataset.config.num_prompt_examples=40 --dataset.config.flip_labels_after=17 --n_repeats=500
python change_labels.py --debug=False --model.name=Llama-2-7b-8bit --note=centered_flip_labels_both_directions --dataset.name=wnli --dataset.config.num_prompt_examples=60 --dataset.config.flip_labels_after=20 --n_repeats=500
python change_labels.py --debug=False --model.name=Llama-2-13b-8bit --note=centered_flip_labels_both_directions --dataset.name=wnli --dataset.config.num_prompt_examples=60 --dataset.config.flip_labels_after=20 --n_repeats=500
python change_labels.py --debug=False --model.name=Llama-2-70b --note=centered_flip_labels_both_directions --dataset.name=wnli --dataset.config.num_prompt_examples=60 --dataset.config.flip_labels_after=20 --n_repeats=500
python change_labels.py --debug=False --model.name=Llama-2-7b-8bit --note=centered_flip_labels_both_directions --dataset.name=hate_speech --dataset.config.num_prompt_examples=60 --dataset.config.flip_labels_after=20 --n_repeats=500
python change_labels.py --debug=False --model.name=Llama-2-13b-8bit --note=centered_flip_labels_both_directions --dataset.name=hate_speech --dataset.config.num_prompt_examples=60 --dataset.config.flip_labels_after=20 --n_repeats=500
python change_labels.py --debug=False --model.name=Llama-2-70b --note=centered_flip_labels_both_directions --dataset.name=hate_speech --dataset.config.num_prompt_examples=60 --dataset.config.flip_labels_after=20 --n_repeats=500
python change_labels.py --debug=False --model.name=Llama-2-7b-8bit --note=centered_flip_labels_both_directions --dataset.name=rte --dataset.config.num_prompt_examples=30 --dataset.config.flip_labels_after=10 --n_repeats=500
python change_labels.py --debug=False --model.name=Llama-2-13b-8bit --note=centered_flip_labels_both_directions --dataset.name=rte --dataset.config.num_prompt_examples=30 --dataset.config.flip_labels_after=10 --n_repeats=500
python change_labels.py --debug=False --model.name=Llama-2-70b --note=centered_flip_labels_both_directions --dataset.name=rte --dataset.config.num_prompt_examples=30 --dataset.config.flip_labels_after=10 --n_repeats=500
python change_labels.py --debug=False --model.name=Llama-2-7b-8bit --note=centered_flip_labels_both_directions --dataset.name=ag_news --dataset.config.num_prompt_examples=50 --dataset.config.flip_labels_after=20 --n_repeats=500
python change_labels.py --debug=False --model.name=Llama-2-13b-8bit --note=centered_flip_labels_both_directions --dataset.name=ag_news --dataset.config.num_prompt_examples=50 --dataset.config.flip_labels_after=20 --n_repeats=500
python change_labels.py --debug=False --model.name=Llama-2-70b --note=centered_flip_labels_both_directions --dataset.name=ag_news --dataset.config.num_prompt_examples=50 --dataset.config.flip_labels_after=20 --n_repeats=500
```


