## EXPLORING REASONING CAPABILITIES IN RETRIEVER-AUGMENTED LANGUAGE MODELS: ANALYZING THE INTERPLAY BETWEEN RETRIEVER AND LANGUAGE MODELS

This work is a replication study of the research paper [Can Retriever-Augmented Language Models Reason? The Blame Game Between the Retriever and the Language Model](https://arxiv.org/abs/2212.09146). by Parishad BehnamGhader, Santiago Miret, Siva Reddy


### Preparing Datasets

This section includes the code for preparing datasets for both QA and LM experiments.

#### Dependencies

- [Spacy](https://spacy.io/usage) (tested with version 3.4.3)

#### Preparation

You can use the following script to generate QA and LM json datastores for EntailmentBank and StrategyQA datasets. Note that in the experiments on StrategyQA, we have changed the question and answers to a declarative format to perform LM.
```bash
cd data
python prepare_data.py --input_file raw/entailmentbank/task_1/dev.jsonl --output_file entailmentbank_1_dev --dataset entailmentbank --qa 1 --lm 1
python prepare_data.py --input_file raw/strategyqa/strategyqa_declarative_train.json --output_file strategyqa --dataset strategyqa --split 1 --lm 1
```
If you wish to use your own data samples, you must follow the following json data format:

**QA**
```json
[
  {
    "question": "Which event occurs on a daily cycle?", 
    "answer": ["The Sun rises and sets."], %(for cases like StrategyQA, the answer would be like ["yes", "no"] with "yes" being the correct answer)
    "facts": ["The sun rising / setting occurs once per day."],
    "gold_facts": ["The sun rising / setting occurs once per day."], %(used when evaluating the models with only ground-truth facts)
    "hypothesis": "The sun rising and setting is the event that occurs once per day." %(used when evaluating the models with one single hypothesis sentence.)
  },
]
```

**LM**
```json
[
  {
    "query": "As the distance of the star to earth decreases, the [MASK] will appear brighter.",
    "target": ["star", "space"], %(the first alternative target is the ground-truth masked entity)
    "facts": ["A star produces light and heat."], 
    "gold_facts":["A star produces light and heat."] %(used when evaluating the models with only ground-truth facts)
  },
]
```
---
### Experiments

In the original paper, the authors evaluate the reasoning abilities of the following retriever-augmented language models:
1. [REALM](https://huggingface.co/docs/transformers/model_doc/realm)
2. [kNN-LM](https://github.com/urvashik/knnlm)
3. [DPR + FiD](https://github.com/facebookresearch/FiD)
4. [Contriever + ATLAS](https://github.com/facebookresearch/atlas)
5. [DPR + Flan-T5](https://huggingface.co/google/flan-t5-base)
6. [Contriever + Flan-T5/GPT-3.5 in a DSP framework](https://github.com/stanfordnlp/dspy)

In this replication study, I evaluate REALM and Contriever + ATLAS.

<details><summary>1. REALM</summary>
<p>

##### Dependencies
- python 3 (tested with 3.7)
- pytorch (tested with 1.11.0)
- transformers (tested with 4.20.1)
- numpy

You may want to use `realm/environment.yml` as well.

##### Experiments
The following scripts run all kinds of experiments
```bash
cd realm

#QA
python evaluate_reasoning.py \
  --reason_data_file <absolute address of the preprocessed json data file> \
  --reason_output_file <absolute address of a report.jsonl file> \
  --reason_task qa \
  --reason_k 5 \
  --reason_dataset <entailmentbank / strategyqa>
  
#LM (target ranking)
python evaluate_reasoning.py \
  --reason_data_file <absolute address of the preprocessed json data file> \
  --reason_output_file <absolute address of a report.jsonl file> \
  --reason_task lm \
  --reason_k 5 \
  --reason_dataset <entailmentbank / strategyqa>
```

A list of the script arguments is explained below:
- `reason_k`: number of retrieved statements
- `reason_data_file`: absolute address of the preprocessed json data file with the above-mentioned format
- `reason_output_file`: absolute address of a report.jsonl file
- `reason_task`: 'qa' | 'lm'
- `reason_lm_task`: 'target_ranking' (model preference) | 'prediction' (masked token prediction)
- `reason_fact_type`: 'facts' (default, use `facts` key) | 'gold_facts' (use `gold_facts` key) | 'single_fact' (use `hypothesis` key)
- `reason_dataset`: 'entailmentbank' | 'strategyqa'
</p></details>


<details><summary>4. Contriever + ATLAS</summary>
<p>

##### Dependencies
- python 3 (tested with 3.8)
- pytorch (tested with 1.11.0)
- transformers (tested with 4.18.0)
- faiss-gpu (tested with 1.7.2)
- numpy

You may want to use `contriever/contriever_environment.yml` as well.

##### Experiments
In order to run the ATLAS experiments, you must first download the preferred model from [ATLAS github](https://github.com/facebookresearch/atlas). In our experiments we load the `models/atlas_nq/base` ATLAS model.
The following scripts run all kinds of experiments.
```bash
cd contriever
port=$(shuf -i 15000-16000 -n 1)

#QA
python evaluate_atlas_reasoning.py \
  --generation_max_length 16 --name reason --precision fp32 --text_maxlength 512 \
  --reader_model_type google/t5-base-lm-adapt \ # architecture of Atlas
  --model_path <address to the model checkpoint - atlas_data/models/...> \
  --per_gpu_batch_size 1 --checkpoint_dir atlas_data/experiments --main_port $port \
  --reason_data_file <absolute address of the preprocessed json data file> \
  --reason_output_file <absolute address of a report.jsonl file> \
  --reason_k 5 \
  --reason_task qa \
  --reason_dataset <entailmentbank / strategyqa>
  
#LM
python evaluate_atlas_reasoning.py \
  --generation_max_length 16 --name reason --precision fp32 --text_maxlength 512\
  --reader_model_type google/t5-base-lm-adapt \ # architecture of Atlas
  --model_path <address to the model checkpoint - atlas_data/models/...> \
  --per_gpu_batch_size 1 --checkpoint_dir atlas_data/experiments --main_port $port \
  --reason_data_file <absolute address of the preprocessed json data file> \
  --reason_output_file <absolute address of a report.jsonl file> \
  --reason_k 5 \
  --reason_task lm \
  --reason_dataset <entailmentbank / strategyqa>
```

A list of the script arguments is explained below:
- `reason_k`: number of retrieved statements
- `reason_data_file`: absolute address of the preprocessed json data file with the above-mentioned format
- `reason_output_file`: absolute address of a report.jsonl file
- `reason_task`: 'qa' | 'lm'
- `reason_fact_type`: 'facts' (default, use `facts` key) | 'gold_facts' (use `gold_facts` key) | 'single_fact' (use `hypothesis` key)
- `reason_dataset`: 'strategyqa' | 'entailmentbank'
</p></details>

### Results
Here’s a sample wrong response from the QA experiment with REALM. Some of the retrieved statements are challenging to relate to the query.
```json
{
    "query": "Which of the following is visible through a reflecting telescope?",
    "retrieved_statements": [
        "A reflecting telescope is a kind of telescope.",
        "A telescope is used for observing moons by astronomers.",
        "Moons orbit planets.",
        "Jupiter is a kind of planet."
    ],
    "answer": "moons around Jupiter",
    "response": "a kind of telescope"
}
```

In the result below, the language model did not synthesize the correct response even though one of the retrieved statements had it.
```json
{
    "query": "A scientist discovered a fossil of an ocean plant in the rocks of a desert. What does the discovery of this fossil most likely tell the scientist?",
    "retrieved_statements": [
        "A fossil of an ocean plant is found in the desert.",
        "If fossils of a water animal or plant are found in a place then that place used to be covered by water in the past.",
        "An ocean plant is a kind of water plant.",
        "A desert is a kind of place."
    ],
    "answer": "The area was once covered by water.",
    "response": "found in the desert"
}
```

Here's a sample correct response from the QA experiment with REALM. Retrieved statements are relevant to context and the response is reasonable.
```json
{
    "query": "Rocks in warm and humid environments can be weathered faster than rocks in other environments. Which is most likely the next step in the rock cycle for weathered rocks?",
    "retrieved_statements": [
        "In the rock cycle , erosion follows weathering.",
        "If a step follows another step , then that step will be the next step."
    ],
    "answer": "They become eroded.",
    "response": "erosion"
}
```

Similarly, here are the equivalent results when the QA experiments were run with Contriever and ATLAS.
In this result, the retrieved statements are relevant to context but the language model, while doing a good job of returning a context relevant response, it doesn’t do a good job of answering as though it responded to the actual question. It appeared to have simply lifted one of the retrieved statements verbatim, as a response.
```json
{
    "query": "question: Which of the following is visible through a reflecting telescope? answer: <extra_id_0>",
    "retrieved_statements": [
        "A reflecting telescope is a kind of telescope.",
        "A telescope is used for observing moons by astronomers.",
        "Moons orbit planets.",
        "Jupiter is a kind of planet."
    ],
    "answer": "moons around Jupiter",
    "response": "Moons orbit planets"
}
```
In this result below, Contriever retrieves relevant statements, ATLAS synthesizes a mildly reasonable response but grammatically incorrect and would only make sense to someone who already has an idea what the external knowledge provided to the RALM looks like.

```json
{
    "query": "question: A scientist discovered a fossil of an ocean plant in the rocks of a desert. What does the discovery of this fossil most likely tell the scientist? answer: <extra_id_0>",
    "retrieved_statements": [
        "A fossil of an ocean plant is found in the desert.",
        "If fossils of a water animal or plant are found in a place then that place used to be covered by water in the past.",
        "An ocean plant is a kind of water plant.",
        "A desert is a kind of place."
    ],
    "answer": "The area was once covered by water.",
    "response": "the past used to be covered by water"
}
```

In the result below, retrieved statements and synthesized response were relevant to the context and accurate.
```json
{
    "query": "question: Rocks in warm and humid environments can be weathered faster than rocks in other environments. Which is most likely the next step in the rock cycle for weathered rocks? answer: <extra_id_0>",
    "retrieved_statements": [
        "In the rock cycle , erosion follows weathering.",
        "If a step follows another step , then that step will be the next step."
    ],
    "answer": "They become eroded.",
    "response": "erosion follows weathering."
}
```
