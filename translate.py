from transformers import MarianMTModel, MarianTokenizer
import tqdm
import torch

def get_langs(dataset):
    lang_1 = "en"
    lang_2 = "en"
    if dataset == "STS.input.track1.ar-ar.txt":
        lang_1 = "ar"
        lang_2 = "ar"
    elif dataset == "STS.input.track2.ar-en.txt":
        lang_1 = "en"
        lang_2 = "ar"
    elif dataset == "STS.input.track3.es-es.txt":
        lang_1 = "es"
        lang_2 = "es"
    elif dataset == "STS.input.track4a.es-en.txt":
        lang_1 = "es"
        lang_2 = "en"
    elif dataset == "STS.input.track6.tr-en.txt":
        lang_1 = "en"
        lang_2 = "tr"
    elif dataset == "STS.input.extended.de-en.txt":
        lang_1 = "en"
        lang_2 = "de"
    elif dataset == "STS.input.extended.fr-en.txt":
        lang_1 = "fr"
        lang_2 = "en"
    elif dataset == "STS.input.extended.it-en.txt":
        lang_1 = "it"
        lang_2 = "en"
    elif dataset == "STS.input.extended.nl-en.txt":
        lang_1 = "nl"
        lang_2 = "en"
    return lang_1, lang_2

datasets = ["STS.input.track1.ar-ar.txt",
 "STS.input.track2.ar-en.txt",
 "STS.input.track3.es-es.txt",
 "STS.input.track4a.es-en.txt",
 "STS.input.track5.en-en.txt",
 "STS.input.track6.tr-en.txt",
 "STS.input.extended.de-en.txt",
 "STS.input.extended.fr-en.txt",
 "STS.input.extended.it-en.txt",
 "STS.input.extended.nl-en.txt"]

for dataset in datasets:
    print (dataset)
    first = True
    lang_1, lang_2 = get_langs(dataset)

    device = torch.device("cuda")
    if lang_1 != "en":
        tokenizer1 = MarianTokenizer.from_pretrained(f"Helsinki-NLP/opus-mt-{lang_1}-en")
        model1 = MarianMTModel.from_pretrained(f"Helsinki-NLP/opus-mt-{lang_1}-en").to(device)

    if lang_2 != "en":
        if lang_2 == lang_1:
            tokenizer2 = tokenizer1
            model2 = model1
        else:
            tokenizer2 = MarianTokenizer.from_pretrained(f"Helsinki-NLP/opus-mt-{lang_2}-en")
            model2 = MarianMTModel.from_pretrained(f"Helsinki-NLP/opus-mt-{lang_2}-en").to(device)

    fo = open(f"STS/STS17-test-opus/{dataset}", 'w')
    for line in open(f"STS/STS17-test/{dataset}").readlines():
        text = line.strip().split('\t')
        if len(text) != 3:
            continue
        sent1 = text[0].strip()
        sent2 = text[1].strip()
        score = text[2]

        if first:
            print (f"{lang_1} {sent1}")
            print (f"{lang_2} {sent2}")
        if lang_1 != "en":
            translated = model1.generate(**tokenizer1([sent1], return_tensors="pt", padding=True).to(device))
            sent1 = [tokenizer1.decode(t, skip_special_tokens=True) for t in translated][0]

        if lang_2 != "en":
            translated = model2.generate(**tokenizer2([sent2], return_tensors="pt", padding=True).to(device))
            sent2 = [tokenizer2.decode(t, skip_special_tokens=True) for t in translated][0]

        if first:
            print (sent1)
            print (sent2)
            print ("-"*55)
        first = False
        fo.write(f"{sent1}\t{sent2}\t{score}\n")
    fo.close()
