from amr.IO import read_raw_amr_data
from datasets import load_dataset
import pandas as pd

import sys
import torch
import tqdm

# device = torch.device('cuda')
# def get_mt_model(lang):
#     from transformers import MarianMTModel, MarianTokenizer
#     tokenizer = MarianTokenizer.from_pretrained(f"Helsinki-NLP/opus-mt-en-{lang}")
#     model = MarianMTModel.from_pretrained(f"Helsinki-NLP/opus-mt-en-{lang}").to(device)
#     return tokenizer, model
# def translate(tokenizer, model, sents):
#     tokenized = tokenizer(sents, return_tensors="pt", padding=True).to(device)
#     translated = model.generate(**tokenized)
#     sents = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
#     return sents

drop_parentheses=False

###the code when we generate sentences to parse###
# from datasets import load_dataset

# def gen_sents_wiki(fname):
#     datasets = load_dataset("text", data_files={'train':fname})
#     data = datasets["train"]
#     for x in data:
#         yield x['text']

def gen_sents_nli(fname):
    datasets = load_dataset(fname.split(".")[-1], data_files={'train':fname}, delimiter="\t" if "tsv" in fname else ",", encoding='utf-8')
    data = datasets["train"]
    print (len(data) * 3)
    for x in data:
        yield x['sent0']
        yield x['sent1']
        yield x['hard_neg']


# idx = 0
# with open('wiki.txt', "w") as fo:
#     for x in gen_sents_wiki('data/wiki1m_for_simcse.txt'):
#         fo.write(f"# ::id {idx}\n")
#         fo.write(f"# ::snt {x}\n")
#         fo.write(f"(e / empty)\n\n")
#         idx += 1

# idx = 0
# with open('nli.txt', "w") as fo:
#     for x in gen_sents_nli('data/nli_for_simcse.csv'):
#         fo.write(f"# ::id {idx}\n")
#         fo.write(f"# ::snt {x}\n")
#         fo.write(f"(e / empty)\n\n")
#         idx += 1

# batch_size=32
# for lang in ['zh']:
#     tokenizer, model = get_mt_model(lang)
#     idx = 0
#     print ("start!")
#     with open(f'nli_{lang}.txt', "w") as fo, tqdm.tqdm(total=826803)as pbar:
#         batch = []
#         for x in gen_sents_nli('data/nli_for_simcse.csv'):
#             batch.append(x)
#             if len(batch) == batch_size:
#                 translations = translate(tokenizer, model, batch)
#                 for translation in translations:
#                    fo.write(f"# ::id {idx}\n")
#                    fo.write(f"# ::snt {translation}\n")
#                    fo.write(f"# ::snt_lang {lang}\n")
#                    fo.write(f"(e / empty)\n\n")
#                    idx += 1
#                 pbar.update(batch_size)
#                 batch = []
#             else:
#                 continue
#         if batch:
#             translations = translate(tokenizer, model, batch)
#             for translation in translations:
#                 fo.write(f"# ::id {idx}\n")
#                 fo.write(f"# ::snt {translation}\n")
#                 fo.write(f"# ::snt_lang {lang}\n")
#                 fo.write(f"(e / empty)\n\n")
#                 idx += 1

###the code when we collect parses###
# data, snts = read_raw_amr_data("backup/wiki-test-pred.txt",
#        dereify=True,
#        remove_wiki=True,
#        use_pointer_tokens=True)

# with open("data/wiki.amr.txt", 'w') as fo:
#     for x in data:
#         fo.write(x + '\n')

# sent0 = data + snts + data + snts 
# sent1 = data + snts + snts + data
# df = pd.DataFrame({'sent0': sent0,
#                     'sent1': sent1})
# df.to_csv("data/wiki.all.csv", index=False)


# data, snts = read_raw_amr_data("backup/nli-test-pred.txt",
#         dereify=True,
#         remove_wiki=True,
#         use_pointer_tokens=True,
#         drop_parentheses=drop_parentheses)

# sent0 = data[::3]
# sent1 = data[1::3]
# hard_neg = data[2::3]
# df = pd.DataFrame({'sent0': sent0,
#                     'sent1': sent1,
#                     'hard_neg': hard_neg})
# df.to_csv("data/nli.amr.csv", index=False)

where='best_fr'
data = {}
snts = {}
for lang in ['en', 'de', 'es', 'it', 'fr', 'ar']:
    data_lang, snts_lang = read_raw_amr_data(f"backup/{where}/nli_{lang}-test-pred.txt",
        dereify=True,
        remove_wiki=True,
        use_pointer_tokens=True,
        drop_parentheses=drop_parentheses)
    data[lang] = data_lang
    snts[lang] = snts_lang


_sent0 = list(zip(snts['en'][::3], data['en'][::3], data['de'][::3], data['es'][::3], data['it'][::3], data['fr'][::3], data['ar'][::3]))
_sent1 = list(zip(snts['en'][1::3], data['en'][1::3], data['de'][1::3], data['es'][1::3], data['it'][1::3], data['fr'][1::3], data['ar'][1::3]))
_hard_neg = list(zip(snts['en'][2::3], data['en'][2::3], data['de'][2::3], data['es'][2::3], data['it'][2::3], data['fr'][2::3], data['ar'][2::3]))


import random


random.seed(19940117)
for size in range(3, 4):
    sent0 = [ random.choices(x[1:], weights=[10, 2, 2, 2, 2, 2])[0] for x in _sent0 for i in range(size)]
    sent1 = [ random.choices(x[1:], weights=[10, 2, 2, 2, 2, 2])[0] for x in _sent1 for i in range(size)]
    hard_neg = [ random.choices(x[1:], weights=[10, 2, 2, 2, 2, 2])[0] for x in _hard_neg for i in range(size)]
    df = pd.DataFrame({'sent0': sent0,
                        'sent1': sent1,
                        'hard_neg': hard_neg})
    df.to_csv(f"data/nli.{where}ar-textbalancedxrandom{size}.csv", index=False)

random.seed(19940117)
for size in range(3, 4):
    sent0 = [ random.choices(x, weights=[2, 2, 2, 2, 2, 2, 2])[0] for x in _sent0 for i in range(size)]
    sent1 = [ random.choices(x, weights=[2, 2, 2, 2, 2, 2, 2])[0] for x in _sent1 for i in range(size)]
    hard_neg = [ random.choices(x, weights=[2, 2, 2, 2, 2, 2, 2])[0] for x in _hard_neg for i in range(size)]
    df = pd.DataFrame({'sent0': sent0,
                        'sent1': sent1,
                        'hard_neg': hard_neg})
    df.to_csv(f"data/nli.{where}arxrandom{size}.csv", index=False)


