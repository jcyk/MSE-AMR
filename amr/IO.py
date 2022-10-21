from pathlib import Path
from penman import load as load_, Graph, Triple
from penman import loads as loads_
from penman import encode as encode_
from penman.model import Model
from penman.models.noop import NoOpModel
from penman.models import amr


ROOT=Path(__file__).parent
op_model = Model()
noop_model = NoOpModel()
amr_model = amr.model
DEFAULT = op_model

def _get_model(dereify):
    if dereify is None:
        return DEFAULT


    elif dereify:
        return op_model

    else:
        return noop_model

def _remove_wiki(graph):
    metadata = graph.metadata
    triples = []
    for t in graph.triples:
        v1, rel, v2 = t
        if rel == ':wiki':
            t = Triple(v1, rel, '+')
        else:
            triples.append(t)
    graph = Graph(triples)
    graph.metadata = metadata
    return graph

def load(source, dereify=None, remove_wiki=False):
    model = _get_model(dereify)
    out = load_(source=source, model=model)
    if remove_wiki:
        for i in range(len(out)):
            out[i] = _remove_wiki(out[i])
    return out

def loads(string, dereify=None, remove_wiki=False):
    model = _get_model(dereify)
    out = loads_(string=string, model=model)
    if remove_wiki:
        for i in range(len(out)):
            out[i] = _remove_wiki(out[i])
    return out

def encode(g, top=None, indent=-1, compact=False):
    model = amr_model
    return encode_(g=g, top=top, indent=indent, compact=compact, model=model)

from typing import List, Optional, Dict, Any, Set, TypeVar
import re
class AMRTokens:

    START, END = '<', '>'
    _TEMPL = START + '{}' + END

    BOS_N   = _TEMPL.format('s')
    EOS_N   = _TEMPL.format('/s')
    START_N = _TEMPL.format('start')
    STOP_N  = _TEMPL.format('stop')
    PNTR_N  = _TEMPL.format('pointer')

    LIT_START = _TEMPL.format( 'lit')
    LIT_END   = _TEMPL.format('/lit')

    BACKR_SRC_N = _TEMPL.format('backr:src:XXX')
    BACKR_TRG_N = _TEMPL.format('backr:trg:XXX')

    BOS_E   = _TEMPL.format('s')
    EOS_E   = _TEMPL.format('/s')
    START_E = _TEMPL.format('start')
    STOP_E  = _TEMPL.format('stop')

    _FIXED_SPECIAL_TOKENS_N = {
        BOS_N, EOS_N, START_N, STOP_N}
    _FIXED_SPECIAL_TOKENS_E = {
        BOS_E, EOS_E, START_E, STOP_E}
    _FIXED_SPECIAL_TOKENS = _FIXED_SPECIAL_TOKENS_N | _FIXED_SPECIAL_TOKENS_E

    # match and read backreferences
    _re_BACKR_SRC_N = re.compile(BACKR_SRC_N.replace('XXX', r'([0-9]+)'))
    _re_BACKR_TRG_N = re.compile(BACKR_TRG_N.replace('XXX', r'([0-9]+)'))
    
    pred_min = 5
    tokens = [PNTR_N, STOP_N, LIT_START, LIT_END, BACKR_SRC_N, BACKR_TRG_N]
    for line in Path(ROOT/'vocab/predicates.txt').read_text().strip().splitlines():
        tok, count = line.split()
        if int(count) >= pred_min:
            tokens.append(tok)

    for tok in Path(ROOT/'vocab/additions.txt').read_text().strip().splitlines():
        tokens.append(tok)

    for cnt in range(512):
        tokens.append(f"<pointer:{cnt}>")

    tokens = set(tokens)
    
    @classmethod
    def is_node(cls, string: str) -> bool:
        if isinstance(string, str) and string.startswith(':'):
            return False
        elif string in cls._FIXED_SPECIAL_TOKENS_E:
            return False
        return True

    @classmethod
    def read_backr(cls, string: str) -> Optional:
        m_src = cls._re_BACKR_SRC_N.search(string)
        if m_src is not None:
            return m_src
        m_trg = cls._re_BACKR_TRG_N.search(string)
        if m_trg is not None:
            return m_trg
        return None


import glob
from typing import List, Union, Iterable
from pathlib import Path
import copy
import penman
import regex as re

def _tokenize_encoded_graph(encoded):
    linearized = re.sub(r"(\".+?\")", r' \1 ', encoded)
    pieces = []
    for piece in linearized.split():
        if piece.startswith('"') and piece.endswith('"'):
            pieces.append(piece)
        else:
            piece = piece.replace('(', ' ( ')
            piece = piece.replace(')', ' ) ')
            piece = piece.replace(':', ' :')
            piece = piece.replace('/', ' / ')
            piece = piece.strip()
            pieces.append(piece)
    linearized = re.sub(r'\s+', ' ', ' '.join(pieces)).strip()
    linearized_nodes = linearized.split(' ')
    return linearized_nodes

def _get_nodes(graph, use_pointer_tokens):
    graph_ = copy.deepcopy(graph)
    graph_.metadata = {}
    linearized = penman.encode(graph_)
    linearized_nodes = _tokenize_encoded_graph(linearized)

    if use_pointer_tokens:
        remap = {}
        remapv2 = {}
        for i in range(1, len(linearized_nodes)):
            nxt = linearized_nodes[i]
            lst = linearized_nodes[i-1]
            if nxt == '/':
                remap[lst] = f'<pointer:{len(remap)}>'
                remapv2[lst] = linearized_nodes[i+1]
        i = 1
        linearized_nodes_ = [linearized_nodes[0]]
        while i < (len(linearized_nodes)):
            nxt = linearized_nodes[i]
            lst = linearized_nodes_[-1]
            if nxt in remap:
                if lst == '(' and linearized_nodes[i+1] == '/':
                    nxt = remapv2[nxt]
                    i += 2
                elif lst.startswith(':'):
                    nxt = remapv2[nxt]
            linearized_nodes_.append(nxt)
            i += 1
        linearized_nodes = linearized_nodes_
    return linearized_nodes

def linearize(graph, use_pointer_tokens, drop_parentheses):
    linearized_nodes = _get_nodes(graph, use_pointer_tokens)
    
    bpe_tokens = []
    for tokk in linearized_nodes:
        is_in_enc = tokk in AMRTokens.tokens
        is_rel = tokk.startswith(':') and len(tokk) > 1
        is_spc = tokk.startswith('<') and tokk.endswith('>')
        is_of  = tokk.startswith(':') and tokk.endswith('-of')
        is_frame = re.match(r'.+-\d\d', tokk) is not None

        if tokk.startswith('"') and tokk.endswith('"'):
            tokk = tokk[1:-1].replace('_', ' ')
            bpe_toks = [tokk]
        elif (is_rel or is_spc or is_frame or is_of):
            if is_in_enc:
                bpe_toks = [tokk]
            elif is_frame:
                bpe_toks = [tokk[:-3], tokk[-3:]]
            elif is_of:
                rel = tokk[:-3]
                if rel in AMRTokens.tokens:
                    bpe_toks = [rel, '-of']
                else:
                    bpe_toks = [':', rel[1:], '-of']
            elif is_rel:
                bpe_toks = [':', tokk[1:]]
            else:
                raise
        else:
            bpe_toks = [tokk]
        
        bpe_tokens.append(bpe_toks)
    linearized_nodes = [b for bb in bpe_tokens for b in bb]
    if drop_parentheses:
        linearized_nodes = [b for b in linearized_nodes if b!='(' and b!=')']
    return linearized_nodes

def read_raw_amr_data(
        path: Union[str, Path],
        dereify=True,
        remove_wiki=False,
        use_pointer_tokens=False,
        drop_parentheses=False
):
    path = Path(path)
    pm_graphs = load(path, dereify=dereify, remove_wiki=remove_wiki)
    graphs = [' '.join(linearize(g, use_pointer_tokens, drop_parentheses)) for g in pm_graphs]
    snts = [g.metadata['snt'] for g in pm_graphs]
    return graphs, snts
