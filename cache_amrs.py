from amr.IO import read_raw_amr_data
import torch

fname="backup/best/ml_transfer-test-pred.txt"
data, snts = read_raw_amr_data(fname,
        dereify=True,
        remove_wiki=True,
        use_pointer_tokens=True,
        drop_parentheses=False)
snts = [ x[len("# ::snt"):].strip() for x in open(fname).readlines() if x.startswith("# ::snt") and not x.startswith("# ::snt_lang")]

sent_to_amr = {}
for g, s in zip(data, snts):
    sent_to_amr[s] = g

torch.save(sent_to_amr, "tmp")

