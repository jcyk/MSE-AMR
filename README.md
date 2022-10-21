## MSE-AMR

This repository holds the codebase of our EMNLP2022 paper: [Retrofitting Multilingual Sentence Embeddings
with Abstract Meaning Representation](https://arxiv.org/pdf/2210.09773.pdf) by [Deng Cai](https://jcyk.github.io/), Xin Li, Jackie Chun-Sing Ho, Lidong Bing and Wai Lam. 

The codebase is built on top of [SimCSE](https://github.com/princeton-nlp/SimCSE). The orignal SimCSE repo is only concerned with sentence embeddings for English, we extend the evaluation pipeline to include a set of multilingual semantic textual similarity (<b>Multilingual STS 2017</b>) tasks and a range of multilingual transfer tasks (<b>XNLI, PAWS-X, QAM, MLDoc, and MARC</b>).

Of course, the recipe of how to retrofit multilingual sentence embeddings with Abstract Meaning Representation (<b>AMR</b>) is also included.

## Requirements

1. `torch==1.7.1`

2. For evaluating mUSE embeddings, TensorFlow and TF.text are also required. Version 2.4 of those with CUDA Toolkit 11.0 were used in the testing of the scripts.

3. Run the following script to install the remaining dependencies for SimCSE,

```bash
pip install -r requirements.txt
```

## Evaluation

<b>The primary purpose of this reop is to establish a standardized evaluation protocol and provide a convenient evaluation tool for future research in multilingual sentence embeddings.</b> You can find the scripts to evaluate popular multilingual sentence embeddings in [eval_scripts/test_*.sh](./eval_scripts)

You can evaluate any `transformers`-based pre-trained models (on Huggingface) using the evaluation code. For example,
```bash
python evaluation.py \
    --model_name_or_path bert-base-multilingual-cased \
    --pooler avg \
    --task_set ml_transfer
```
which is expected to output the results in a tabular format:
```
+--------+-------+-------+-------+-------+-------+-------+-------+-------+-------+
| MLDoc  |   de  |   en  |   es  |   fr  |   it  |   ja  |   ru  |   zh  |  Avg. |
+--------+-------+-------+-------+-------+-------+-------+-------+-------+-------+
| mlearn | 92.50 | 88.72 | 90.18 | 90.42 | 80.77 | 82.62 | 83.23 | 87.38 | 86.98 |
| 0-shot | 83.73 | 89.88 | 75.75 | 83.73 | 68.25 | 71.12 | 71.08 | 79.65 | 77.90 |
+--------+-------+-------+-------+-------+-------+-------+-------+-------+-------+
+--------+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+
|  XNLI  |   ar  |   bg  |   de  |   el  |   en  |   es  |   fr  |   hi  |   ru  |   sw  |   th  |   tr  |   ur  |   vi  |   zh  |  Avg. |
+--------+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+
| 0-shot | 42.57 | 45.35 | 46.75 | 43.99 | 53.53 | 47.64 | 47.60 | 41.54 | 46.07 | 37.49 | 36.75 | 43.17 | 40.46 | 47.96 | 45.31 | 44.41 |
+--------+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+
+--------+-------+-------+-------+-------+-------+-------+-------+-------+
| PAWS-X |   de  |   en  |   es  |   fr  |   ja  |   ko  |   zh  |  Avg. |
+--------+-------+-------+-------+-------+-------+-------+-------+-------+
| 0-shot | 57.00 | 57.30 | 57.45 | 57.40 | 56.85 | 56.00 | 57.35 | 57.05 |
+--------+-------+-------+-------+-------+-------+-------+-------+-------+
+--------+-------+-------+-------+-------+-------+
|  CLS   |   de  |   en  |   fr  |   ja  |  Avg. |
+--------+-------+-------+-------+-------+-------+
| mlearn | 59.95 | 57.18 | 60.43 | 72.78 | 62.59 |
| 0-shot | 50.62 | 60.18 | 51.30 | 50.33 | 53.11 |
+--------+-------+-------+-------+-------+-------+
+--------+-------+-------+-------+-------+-------+-------+-------+
|  MARC  |   de  |   en  |   es  |   fr  |   ja  |   zh  |  Avg. |
+--------+-------+-------+-------+-------+-------+-------+-------+
| mlearn | 45.18 | 44.92 | 43.78 | 44.26 | 40.94 | 41.96 | 43.51 |
| 0-shot | 38.28 | 45.54 | 38.32 | 38.40 | 32.78 | 37.28 | 38.43 |
+--------+-------+-------+-------+-------+-------+-------+-------+
+--------+-------+-------+-------+-------+
|  QAM   |   de  |   en  |   fr  |  Avg. |
+--------+-------+-------+-------+-------+
| 0-shot | 54.21 | 56.60 | 54.94 | 55.25 |
+--------+-------+-------+-------+-------+
+-------------+-------+-------+--------+-------+-------+-------+-------+
|             | MLDoc |  XNLI | PAWS-X |  CLS  |  MARC |  QAM  |  Avg. |
+-------------+-------+-------+--------+-------+-------+-------+-------+
| mlearn Avg. | -     | -     | -      | -     | -     | -     | -     |
| 0-shot Avg. | 77.90 | 44.41 | 57.05  | 53.11 | 38.43 | 55.25 | 54.36 |
+-------------+-------+-------+--------+-------+-------+-------+-------+
```

The `--task_set` argument is used to specify what set of tasks to evaluate on, including
* `msts`: Evaluate on multilingual `STS 17` tasks.
* `ml_transfer`: Evaluate on multilingual transfer tasks.
* `na`: Manually set tasks by `--tasks`.

When the `--task_set` argument is set to `na` or not set, the `--tasks` argument can be set to specify individual task(s) to evaluate on. For example,
```bash
python evaluation.py \
    --model_name_or_path bert-base-multilingual-cased \
    --pooler avg \
    --tasks XNLI 
```

The `--pooler` argument is used to specify the pooling method used when evaluating a `transformers`-based model.
* `cls`: Use the representation of `[CLS]` token without the extra linear+activation.
* `avg`: Average embeddings of the last layer. If you use checkpoints of SBERT/SRoBERTa ([paper](https://arxiv.org/abs/1908.10084)), you should use this option.
* `simcse_sup`: Use the representation of `[CLS]` token. A linear+activation layer is applied after the representation (it's in the standard BERT implementation). If you use **supervised SimCSE**, you should use this option.

For evaluating LASER embeddings, use `--laser`. For example,
```bash
python evaluation.py \
    --laser \
    --task_set ml_transfer
```

For evaluating mUSE embeddings, use `--muse`. For example,
```bash
python evaluation.py \
    --muse \
    --task_set ml_transfer
```


<b>You can find the scripts to evaluate popular multilingual sentence embeddings in [eval_scripts/test_*.sh](./eval_scripts)</b>.


## Retrofitting multilingual sentence embeddings with AMR

TBD
