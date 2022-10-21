from .IO import read_raw_amr_data
import logging
import torch

logging.getLogger("penman").setLevel(logging.WARNING)

class AMRParser:
    def __init__(self, fname, dereify=True, remove_wiki=False, use_pointer_tokens=False, drop_parentheses=False):
        if fname.endswith(".pt"):
            self.sent_to_amr = torch.load(fname)
            return
        data, snts = read_raw_amr_data(fname, dereify=dereify, remove_wiki=remove_wiki, use_pointer_tokens=use_pointer_tokens, drop_parentheses=drop_parentheses)
        sent_to_amr = {}
        snts = [ x[len("# ::snt"):].strip() for x in open(fname).readlines() if x.startswith("# ::snt") and not x.startswith("# ::snt_lang")] 
        for g, s in zip(data, snts):
            sent_to_amr[s] = g
        self.sent_to_amr = sent_to_amr
    
    def parse(self, sent):
        if sent not in self.sent_to_amr:
            assert False, sent
        return self.sent_to_amr[sent]
