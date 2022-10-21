## SimCSE_mod

This repository contains a modified version of code originally from [SimCSE: Simple Contrastive Learning of Sentence Embeddings](https://github.com/princeton-nlp/SimCSE) with the evaluation pipeline modified to add support for the BGT model and new tasks including the STS 2017 task on an extended multilingual dataset and multiple multilingual transfer tasks.

## Requirements

First, install the correct `1.7.1` version of PyTorch corresponding to your platforms/CUDA versions. PyTorch version higher than `1.7.1` should also work. For example, if you use Linux and **CUDA11** ([how to check CUDA version](https://varhowto.com/check-cuda-version/)), install PyTorch by the following command,

```bash
pip install torch==1.7.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html
```

If you instead use **CUDA** `<11` or **CPU**, install PyTorch by the following command,

```bash
pip install torch==1.7.1
```

For evaluating mUSE embeddings, TensorFlow and TF.text are also required. Version 2.4 of those with CUDA Toolkit 11.0 were used in the testing of the scripts.

Then run the following script to install the remaining dependencies for SimCSE,

```bash
pip install -r requirements.txt
```

Also, due to the added support for the BGT model, you also need to follow the installation and setup instructions in [bilingual-generative-transformer](https://github.com/jwieting/bilingual-generative-transformer).


## Evaluation

Before evaluation, please download the original evaluation datasets by running
```bash
cd SentEval/data/downstream/
bash download_dataset.sh
```

Also download the [datasets](https://drive.google.com/file/d/1AscAYEXy2uw7ZNK-9lXW4SA852KbIR26/view?usp=sharing) for the multilingual transfer tasks and extract ml_transfer folder under SentEval/data/downstream/.

Then come back to the base directory, you can evaluate any `transformers`-based pre-trained models (on Huggingface) using the evaluation code. For example,
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
| mlearn Avg. | 86.98 |   -   |   -    | 62.59 | 43.51 |   -   | 64.36 |
| 0-shot Avg. | 77.90 | 44.41 | 57.05  | 53.11 | 38.43 | 55.25 | 54.36 |
+-------------+-------+-------+--------+-------+-------+-------+-------+
```

The `--task_set` argument is used to specify what set of tasks to evaluate on, including
* `sts`: Evaluate on STS tasks, including `STS 12~16`, `STS-B` and `SICK-R`.
* `msts`: Evaluate on multilingual `STS 17` tasks.
* `transfer`: Evaluate on transfer tasks.
* `ml_transfer`: Evaluate on multilingual transfer tasks.
* `bucc`: Evaluate on the BUCC bitext mining task.
* `tatoeba`: Evaluate on the Tatoeba test set from LASER.
* `full`: Evaluate on multilingual STS and multilingual transfer tasks (including BUCC and Tatoeba).
* `na`: Manually set tasks by `--tasks`.

When the `--task_set` argument is set to `na` or not set, the `--tasks` argument can be set to specify individual task(s) to evaluate on. For example,
```bash
python evaluation.py \
    --model_name_or_path bert-base-multilingual-cased \
    --pooler avg \
    --tasks CLS 
```

The `--pooler` argument is used to specify the pooling method used when evaluating a `transformers`-based model.
* `cls`: Use the representation of `[CLS]` token. A linear+activation layer is applied after the representation (it's in the standard BERT implementation). If you use **supervised SimCSE**, you should use this option.
* `cls_before_pooler`: Use the representation of `[CLS]` token without the extra linear+activation. If you use **unsupervised SimCSE**, you should take this option.
* `avg`: Average embeddings of the last layer. If you use checkpoints of SBERT/SRoBERTa ([paper](https://arxiv.org/abs/1908.10084)), you should use this option.
* `avg_top2`: Average embeddings of the last two layers.
* `avg_first_last`: Average embeddings of the first and last layers. If you use vanilla BERT or RoBERTa, this works the best.

The `--batch_size` argument can be used to adjust the batch size.

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

For evaluating a BGT model, evaluation script arguments from [the BGT repository](https://github.com/jwieting/bilingual-generative-transformer) are used instead along with `--bgt_folder` to specify where BGT is installed. For example,
```bash
python evaluation.py ../bgt_ml/bgt/ml/data-joint-bin \
    -s en -t ml --path ../bgt_ml/checkpoints/bgt_ml/checkpoint_best.pt \
    --tokenize 1 --sentencepiece ../bgt_ml/bgt/ml/ml-en.7m.sp.20k.model \
    --task_set ml_transfer --bgt_folder ../bgt_ml
```

A multilingual BGT model pretrained on 8 languages / 7 language pairs combined (7m examples, 20k vocab size) can be downloaded [here](https://drive.google.com/drive/folders/1PceyffZquqDe4Y_tqTEcRy4odNyrNuuf?usp=sharing). The evaluation output for `sts` and `ml_transfer` task sets is expected to be the following:

```
+-------+-------+-------+-------+-------+--------------+-----------------+-------+
| STS12 | STS13 | STS14 | STS15 | STS16 | STSBenchmark | SICKRelatedness |  Avg. |
+-------+-------+-------+-------+-------+--------------+-----------------+-------+
| 70.66 | 73.18 | 72.22 | 81.89 | 77.63 |    78.85     |      63.75      | 74.03 |
+-------+-------+-------+-------+-------+--------------+-----------------+-------+
+-------------+-------------+-------------+-------+
| STS17 en-en | STS17 es-es | STS17 ar-ar |  Avg. |
+-------------+-------------+-------------+-------+
|    84.32    |    84.70    |    74.74    | 81.25 |
+-------------+-------------+-------------+-------+
+-------------+-------------+-------------+-------------+-------------+-------------+-------------+-------+
| STS17 en-ar | STS17 en-de | STS17 en-tr | STS17 en-es | STS17 en-fr | STS17 en-it | STS17 en-nl |  Avg. |
+-------------+-------------+-------------+-------------+-------------+-------------+-------------+-------+
|    73.21    |    76.31    |    74.56    |    74.81    |    73.08    |    79.11    |    77.45    | 75.50 |
+-------------+-------------+-------------+-------------+-------------+-------------+-------------+-------+
+-------+-------+--------+-------+-------+-------+-------+
| MLDoc |  XNLI | PAWS-X |  CLS  |  MARC |  QAM  |  Avg. |
+-------+-------+--------+-------+-------+-------+-------+
| 74.58 | 38.35 | 56.07  | 59.88 | 41.20 | 55.60 | 54.28 |
+-------+-------+--------+-------+-------+-------+-------+
```

## Training Multilingual BGT

This section describes the steps used to train the multilingual BGT model mentioned above. The original individual BGT models were each trained on one language pair only. Here, we train a multilingual one on combined data in en-ar, en-de, en-tr, en-es, en-fr, en-it and en-nl (those tested in the extended multilingual STS 17 task above). 

After installing BGT from the author's [repo](https://github.com/jwieting/bilingual-generative-transformer), unbinarized data in en-ar, en-es, en-fr and en-tr are alaready available in folders ar, es, fr-os-giga and tr under bilingual-generative-transformer/bgt. For example, train and test data in en-fr are in train.en-fr and test-en-fr under bilingual-generative-transformer/bgt/fr-os-giga/. We will use these to construct our train and valid data in those languages. For the other languages, we will download and use data from OpenSubtitles 2018. Below are the steps with some example commands, but the same could be achieved by a single Python script or other means.

1. First, following the validation data size of 1000 used by original BGT models, we sample 1000 examples from the test data for each language pair. As an example, if we have already created the folder and subfolder bilingual-generative-transformer/bgt/ml_workings/t, we can use the following commands under bilingual-generative-transformer/bgt.

    ```bash
    shuf -n 1000 ar/test-en-ar > ../ml_workings/t/valid.en-ar
    shuf -n 1000 es/test-en-es > ../ml_workings/t/valid.en-es
    shuf -n 1000 fr-os-giga/test-en-fr > ../ml_workings/t/valid.en-fr
    shuf -n 1000 tr/test-en-tr > ../ml_workings/t/valid.en-tr
    ```
    
2. We will use all 1m train data for each language pair. Hence, we can just copy the train data files to our ml_workings/t folder.

3. Since each line of data is a pair delimited with a tab, we separate the data by languages. We can use the following commands.

    ```bash
    sed -e 'y/\t/\n/' bgt/ml_workings/t/valid.en-ar > bgt/ml_workings/valid.en-ar
    sed -e 'y/\t/\n/' bgt/ml_workings/t/valid.en-es > bgt/ml_workings/valid.en-es
    sed -e 'y/\t/\n/' bgt/ml_workings/t/valid.en-tr > bgt/ml_workings/valid.en-tr
    sed -e 'y/\t/\n/' bgt/ml_workings/t/valid.en-fr > bgt/ml_workings/valid.en-fr
    
    cd bgt/ml_workings
    
    awk 'NR%2' valid.en-ar > valid.en-ar.en
    awk 'NR%2' valid.en-es > valid.en-es.en
    awk 'NR%2' valid.en-fr > valid.en-fr.en
    awk 'NR%2' valid.en-tr > valid.en-tr.en

    awk 'NR%2==0' valid.en-ar > valid.en-ar.ar
    awk 'NR%2==0' valid.en-es > valid.en-es.es
    awk 'NR%2==0' valid.en-fr > valid.en-fr.fr
    awk 'NR%2==0' valid.en-tr > valid.en-tr.tr
    
    cd ../..
    
    sed -e 'y/\t/\n/' bgt/ml_workings/t/train.en-ar > bgt/ml_workings/train.en-ar
    sed -e 'y/\t/\n/' bgt/ml_workings/t/train.en-es > bgt/ml_workings/train.en-es
    sed -e 'y/\t/\n/' bgt/ml_workings/t/train.en-tr > bgt/ml_workings/train.en-tr
    sed -e 'y/\t/\n/' bgt/ml_workings/t/train.en-fr > bgt/ml_workings/train.en-fr

    cd bgt/ml_workings
    
    awk 'NR%2==0' train.en-ar > train.en-ar.en
    awk 'NR%2==0' train.en-es > train.en-es.en
    awk 'NR%2==0' train.en-fr > train.en-fr.en
    awk 'NR%2==0' train.en-tr > train.en-tr.en

    awk 'NR%2' train.en-ar > train.en-ar.ar
    awk 'NR%2' train.en-es > train.en-es.es
    awk 'NR%2' train.en-fr > train.en-fr.fr
    awk 'NR%2' train.en-tr > train.en-tr.tr
    
    cd ../..
    ```
    
4. Since the provided unbinarized data in each language pair have been encoded with the corresponding SentencePiece models, we need to decode them back into raw text. Below is an example script in Python to achieve that. Here, we assume the data files are in bgt/ml_workings/encoded folder and the resulting files are to be saved in bgt/ml_workings.

    ```python
    import sentencepiece as spm
    ff_pairs = [('fr-os-giga', 'fr'), ('es', 'es'), ('ar', 'ar'), ('tr', 'tr')]
    for folder, lang in ff_pairs:
      print(lang)
      sp = spm.SentencePieceProcessor(model_file='bgt/'+folder+'/'+lang+'-en.1m.sp.20k.model')
      file = open("bgt/ml_workings/encoded/train.en-"+lang+"."+lang, "r", encoding='utf-8')
      sents = [line.rstrip('\n').split(' ') for line in file]
      sents = sp.decode(sents)
      fo = open("bgt/ml_workings/train.en-"+lang+"."+lang, "w", encoding='utf-8')
      fo.writelines([sent+'\n' for sent in sents])
      fo.close()
      file.close()
      file = open("bgt/ml_workings/encoded/valid.en-"+lang+"."+lang, "r", encoding='utf-8')
      sents = [line.rstrip('\n').split(' ') for line in file]
      sents = sp.decode(sents)
      fo = open("bgt/ml_workings/valid.en-"+lang+"."+lang, "w", encoding='utf-8')
      fo.writelines([sent+'\n' for sent in sents])
      fo.close()
      file.close()

    ff_pairs = [('fr-os-giga', 'fr'), ('es', 'es'), ('ar', 'ar'), ('tr', 'tr')]
    for folder, lang in ff_pairs:
      print(lang)
      sp = spm.SentencePieceProcessor(model_file='bgt/'+folder+'/'+lang+'-en.1m.sp.20k.model')
      file = open("bgt/ml_workings/encoded/train.en-"+lang+".en", "r", encoding='utf-8')
      sents = [line.rstrip('\n').split(' ') for line in file]
      sents = sp.decode(sents)
      fo = open("bgt/ml_workings/train.en-"+lang+".en", "w", encoding='utf-8')
      fo.writelines([sent+'\n' for sent in sents])
      fo.close()
      file.close()
      file = open("bgt/ml_workings/encoded/valid.en-"+lang+".en", "r", encoding='utf-8')
      sents = [line.rstrip('\n').split(' ') for line in file]
      sents = sp.decode(sents)
      fo = open("bgt/ml_workings/valid.en-"+lang+".en", "w", encoding='utf-8')
      fo.writelines([sent+'\n' for sent in sents])
      fo.close()
      file.close()
    ``` 
    
5. It is time we get the raw text data in other languages from [OpenSubtitles 2018](https://opus.nlpl.eu/OpenSubtitles.php). We need to download the plain text files in [en-it](https://opus.nlpl.eu/download.php?f=OpenSubtitles/v2018/moses/en-it.txt.zip), [en-de](https://opus.nlpl.eu/download.php?f=OpenSubtitles/v2018/moses/de-en.txt.zip) and [en-nl](https://opus.nlpl.eu/download.php?f=OpenSubtitles/v2018/moses/en-nl.txt.zip). After extracting them in one place, we create train and valid data files similar to the ones made above. The Python script below can be used.

    ```python
    import random
    from sacremoses import MosesTokenizer

    entok = MosesTokenizer(lang='en')
    pairs = ['de-en.de', 'en-it.it', 'en-nl.nl']

    for pair in pairs:
        print(pair)
        with open("OpenSubtitles."+pair, "r", encoding="utf-8") as f:
            lines_ml = f.readlines()
            lines_ml = [line.rstrip() for line in lines_ml]
        with open("OpenSubtitles."+pair[:-2]+"en", "r", encoding="utf-8") as f:
            lines_en = f.readlines()
            lines_en = [line.rstrip() for line in lines_en]
        if len(lines_ml) != len(lines_en):
            print("# lines mismatched!")
            continue
        indices = []
        cnt = 0
        ind = list(range(len(lines_ml)))
        random.shuffle(ind)
        print("shuffled indices")
        for idx in ind:
            if lines_en[idx].count(' ') < 4 or lines_en[idx].count(' ') > 99 \
                    or lines_ml[idx].count(' ') < 4 or lines_ml[idx].count(' ') > 99:
                continue
            indices.append(idx)
            cnt += 1
            if cnt == 1001000:
                break
        if cnt != 1001000:
            print("not enough selected sents")
            continue
        print("selected sents")
        lang = pair[-2:]
        lines_ml_p = []
        lines_en_p = []
        for i in range(len(indices)):
            lines_ml_p.append(" ".join(entok.tokenize(lines_ml[indices[i]].lower(), escape=False))+'\n')
            if i % 10000 == 0:
                print(lang, i)
        for i in range(len(indices)):
            lines_en_p.append(" ".join(entok.tokenize(lines_en[indices[i]].lower(), escape=False))+'\n')
            if i % 10000 == 0:
                print('en', i)
        lines_ml = lines_ml_p
        lines_en = lines_en_p
        with open("train.en-"+lang+"."+lang, "w", encoding="utf-8") as f:
            f.writelines(lines_ml[:1000000])
        with open("train.en-"+lang+".en", "w", encoding="utf-8") as f:
            f.writelines(lines_en[:1000000])
        with open("valid.en-"+lang+"."+lang, "w", encoding="utf-8") as f:
            f.writelines(lines_ml[-1000:])
        with open("valid.en-"+lang+".en", "w", encoding="utf-8") as f:
            f.writelines(lines_en[-1000:])
    ```

6. Now that we have the raw text data, we need to train our own new SentencePiece model for this new language pair en-ml. Assuming we have mixed and shuffled all the train data (i.e. all decoded data files named train*) and saved it at decoded_sep/all_train_shuf (with 14m lines), we can train our SentencePiece model and save it as ml-en.7m.sp.20k.model with Python. 

    ```python
    import sentencepiece as spm
    spm.SentencePieceTrainer.train(input='decoded_sep/all_train_shuf', model_prefix='ml-en.7m.sp.20k', vocab_size=20000, character_coverage=1.0, input_sentence_size=20000000)
    ```
    
5. We also need to combine our data files in various languages into files in en-ml. The resulting files we create are train.en-ml.en (7m lines), train.en-ml.ml (7m lines), valid.en-ml.en (7k lines) and valid.en-ml.ml (7k lines).

6. Now we encode the raw text data with our new SentencePiece model. Assuming we have saved our SentencePiece model in bgt/ml folder and our 4 en-ml data files in bgt/ml/decoded, we can run the following Python script to do the encoding and save encoded data in bgt/ml folder.

    ```python
    import sentencepiece as spm
    ff_pairs = [('ml', 'ml')]
    for folder, lang in ff_pairs:
      print(lang)
      sp = spm.SentencePieceProcessor(model_file='bgt/'+folder+'/'+lang+'-en.7m.sp.20k.model')
      file = open("bgt/ml/decoded/train.en-"+lang+"."+lang, "r", encoding='utf-8')
      sents = [line.rstrip('\n') for line in file]
      sents = sp.encode(sents, out_type=str)
      fo = open("bgt/ml/train.en-"+lang+"."+lang, "w", encoding='utf-8')
      fo.writelines([' '.join(sent)+'\n' for sent in sents])
      fo.close()
      file.close()
      file = open("bgt/ml/decoded/valid.en-"+lang+"."+lang, "r", encoding='utf-8')
      sents = [line.rstrip('\n') for line in file]
      sents = sp.encode(sents, out_type=str)
      fo = open("bgt/ml/valid.en-"+lang+"."+lang, "w", encoding='utf-8')
      fo.writelines([' '.join(sent)+'\n' for sent in sents])
      fo.close()
      file.close()
      file = open("bgt/ml/decoded/train.en-"+lang+".en", "r", encoding='utf-8')
      sents = [line.rstrip('\n') for line in file]
      sents = sp.encode(sents, out_type=str)
      fo = open("bgt/ml/train.en-"+lang+".en", "w", encoding='utf-8')
      fo.writelines([' '.join(sent)+'\n' for sent in sents])
      fo.close()
      file.close()
      file = open("bgt/ml/decoded/valid.en-"+lang+".en", "r", encoding='utf-8')
      sents = [line.rstrip('\n') for line in file]
      sents = sp.encode(sents, out_type=str)
      fo = open("bgt/ml/valid.en-"+lang+".en", "w", encoding='utf-8')
      fo.writelines([' '.join(sent)+'\n' for sent in sents])
      fo.close()
      file.close()
    ```
    
7. After that, we can create our binarized train and valid data as well as the joined dictionary using the preprocessing script provided by the author under bilingual-generative-transformer folder.

    ```bash
    python -u preprocess.py --trainpref bgt/ml/train.en-ml --validpref bgt/ml/valid.en-ml --source-lang en --target-lang ml --joined-dictionary --destdir bgt/ml/data-joint-bin 
    ```
    
8. Finally, we can begin training the model. Argument `--fp16` is optional, but useful if the GPU supports fast FP16 training.

    ```bash
    python -u train.py bgt/ml/data-joint-bin -a bgt-emnlp --bgt-setting bgt --optimizer adam --lr 0.0005 -s en -t ml \
        --label-smoothing 0.1 --dropout 0.3 --max-tokens 500 --min-lr '1e-09' --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
        --criterion bgt_loss --max-epoch 20 --warmup-updates 4000 --warmup-init-lr '1e-07' --adam-betas '(0.9, 0.98)' \
        --save-dir checkpoints/bgt_ml --distributed-world-size 1 --latent-size 1024 --update-freq 50 --task bgt \
        --save-interval-updates 0 --sentencepiece bgt/ml/ml-en.7m.sp.20k.model --x0 65536 --translation-loss 1.0 \
        --sentence-avg --tokenize 1 --num-workers 0 --find-unused-parameters --fp16
    ```
