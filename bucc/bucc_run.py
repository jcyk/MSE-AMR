import subprocess
import os
import time
import numpy as np
import torch
from . import mine_bitexts
from . import bucc as bucc_
from collections import OrderedDict

def run(params, batcher):
    cwd = os.getcwd()
    os.chdir(os.path.join(cwd, "bucc"))
    subprocess.run(['/bin/bash', './extract.sh'])
    os.chdir(cwd)
    
    # general config
    bucc="bucc2018"
    data="bucc"
    xdir=data+'/downloaded'	# tar files as distrubuted by the BUCC evaluation
    ddir=data+'/'+bucc	# raw texts of BUCC
    edir=data+'/embed'	# normalized texts and embeddings
    langs=["fr","de","ru","zh"]
    ltrg="en"		# English is always the 2nd language

    od = OrderedDict()
    
    for lsrc in langs:
        bname="{bucc}.{lsrc}-{ltrg}".format(bucc=bucc, lsrc=lsrc, ltrg=ltrg)
        part="{bname}.train".format(bname=bname)
        src_embeddings = Encode('{edir}/{part}.txt.{lsrc}'.format(edir=edir, part=part, lsrc=lsrc),
            lsrc,
            params,
            batcher)
            
        trg_embeddings = Encode('{edir}/{part}.txt.{ltrg}'.format(edir=edir, part=part, ltrg=ltrg),
            ltrg,
            params,
            batcher)
            
        torch.cuda.empty_cache()

        candidates = mine_bitexts.mine(_src='{edir}/{part}.txt.{lsrc}'.format(edir=edir, part=part, lsrc=lsrc),
            _trg='{edir}/{part}.txt.{ltrg}'.format(edir=edir, part=part, ltrg=ltrg),
            _src_lang=lsrc, _trg_lang=ltrg,
            _src_embeddings=src_embeddings, _trg_embeddings=trg_embeddings,
            _unify=True, _mode='mine', _retrieval='max', _margin='ratio',
            _neighborhood=4, _verbose=True, _gpu=True)

        f1 = bucc_.bucc(_src_lang=lsrc, _trg_lang=ltrg, 
            _bucc_texts='{edir}/{part}.txt'.format(edir=edir, part=part),
            _bucc_ids='{edir}/{part}.id'.format(edir=edir, part=part),
            _candidates=candidates,
            _gold='{ddir}/{lsrc}-{ltrg}/{lsrc}-{ltrg}.training.gold'.format(ddir=ddir, lsrc=lsrc, ltrg=ltrg),
            _verbose=True)
               
        od['{lsrc}-{ltrg}'.format(lsrc=lsrc, ltrg=ltrg)] = f1 
    
    return od
               
def buffered_read(fp, buffer_size):
    buffer = []
    for src_str in fp:
        buffer.append(src_str.strip())
        if len(buffer) >= buffer_size:
            yield buffer
            buffer = []

    if len(buffer) > 0:
        yield buffer

def EncodeTime(t):
    t = int(time.time() - t)
    if t < 1000:
        print(' in {:d}s'.format(t))
    else:
        print(' in {:d}m{:d}s'.format(t // 60, t % 60))
        
def Encodep(inp_file, lang, params, batcher, buffer_size=10000):
    n = 0
    t = time.time()
    
    embed_list = []
    for sentences in buffered_read(inp_file, buffer_size):
        sentences = [[sent] for sent in sentences]
        for ii in range(0, len(sentences), params.batch_size):
            batch = sentences[ii:ii + params.batch_size]
            embed_list.append(batcher(params, batch, lang=lang))
        n += len(sentences)
        print('\r - Encoder: {:d} sentences'.format(n), end='')
 
    embeddings = np.vstack(embed_list)
    print('\r - Encoder: {:d} sentences'.format(n), end='')
    EncodeTime(t)
    return embeddings

def Encode(inp_fname, lang, params, batcher, buffer_size=10000, inp_encoding='utf-8'):
    print(' - Encoder: {}'.
          format(os.path.basename(inp_fname)))
    fin = open(inp_fname, 'r', encoding=inp_encoding, errors='surrogateescape') if len(inp_fname) > 0 else sys.stdin
    embeddings = Encodep(fin, lang, params, batcher, buffer_size=buffer_size)
    fin.close()
    return embeddings
    