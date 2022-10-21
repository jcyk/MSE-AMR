import faiss              
from laserembeddings import Laser
import numpy as np
import pycountry
from collections import OrderedDict
import logging

def run(params, batcher):
    logging.debug('\n\n***** Tatoeba *****\n\n')
    
    langs = ['ar', 'de', 'tr', 'es', 'fr', 'it', 'nl', 'ka', 'sw', 'tl', 'tt']
    
    od = OrderedDict()
    
    for lang in langs:
        lang_code = pycountry.languages.get(alpha_2=lang.upper()).alpha_3.lower()
        if lang_code == 'swa':
            lang_code = 'swh'
        
        with open('./tatoeba/data/tatoeba.{l}-eng.{l}'.format(l=lang_code), "r", encoding="utf-8") as file:
            lines = file.readlines()
            
        lines = [[line.rstrip()] for line in lines]
        
        embed_list = []
        
        for ii in range(0, len(lines), params.batch_size):
            batch = lines[ii:ii + params.batch_size]
            embed_list.append(batcher(params, batch, lang=lang))
        embeddings = np.vstack(embed_list)
        
        faiss.normalize_L2(embeddings)
        
        with open('./tatoeba/data/tatoeba.{l}-eng.eng'.format(l=lang_code), "r", encoding="utf-8") as file:
            lines = file.readlines()
        
        lines = [[line.rstrip()] for line in lines]
        
        embed_list = []
        
        for ii in range(0, len(lines), params.batch_size):
            batch = lines[ii:ii + params.batch_size]
            embed_list.append(batcher(params, batch, lang=lang))
        embeddings_en = np.vstack(embed_list)
        
        faiss.normalize_L2(embeddings_en)
        
        assert embeddings_en.shape[0] == embeddings.shape[0]
        assert embeddings_en.shape[1] == embeddings.shape[1]
        
        d = embeddings_en.shape[1]
        n = embeddings_en.shape[0]
        
        ind = np.arange(n)
        
        index = faiss.IndexFlatIP(d)  
        index.add(embeddings)                 
        _, I = index.search(embeddings_en, 1)
        I = np.squeeze(I, axis=1)
        acc_en_to_x = np.mean(I == ind)
        
        index = faiss.IndexFlatIP(d)  
        index.add(embeddings_en)                 
        _, I = index.search(embeddings, 1)
        I = np.squeeze(I, axis=1)
        acc_x_to_en = np.mean(I == ind)
        
        od['en->{}'.format(lang)] = acc_en_to_x
        od['{}->en'.format(lang)] = acc_x_to_en
        
    return od