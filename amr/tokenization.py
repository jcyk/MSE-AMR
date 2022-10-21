import torch
import collections

from .IO import AMRTokens

INIT_XLM_ROBERTA = '▁'
INIT_ROBERTA = 'Ġ'
INIT_BERT = ''
def init_amr_vocabulary(tokenizer, INIT=''):
    vocab = tokenizer.get_vocab()
    tokens = [t for t in AMRTokens.tokens if INIT+t not in vocab]
    tokens.sort()
    old_enc_size = len(tokenizer)
    tokenizer.add_tokens(tokens)
    test = tokenizer.tokenize("( cause-01 :ARG0 ( and :op1 ( begin-01 :ARG1 ( province :name ( name :op1 South :op2 Australia ) ) :ARG2 ( province :ARG0 -of ( govern-01 :ARG1 province ) ) :ARG1 -of ( resemble-01 :polarity - :ARG2 ( state :mod ( other ) :ARG1 -of ( found-01 :ARG2 ( colony ) ) ) ) ) :op2 ( attract-01 :ARG0 begin-01 :ARG1 ( person :quant ( many ) ) ) :op3 ( develop-02 :ARG1 ( and :op1 ( state :name ( name :op1 Adelaide ) ) :op2 ( state :name ( name :op1 SA ) ) ) :ARG2 ( state :ARG0 -of ( depend-01 :polarity - ) :ARG0 -of ( think-01 :ARG3 -of ( free-04 ) ) ) ) ) :ARG1 ( have-03 :ARG0 province :ARG1 ( position :mod ( unique ) :part -of ( history :poss province ) ) ) )".split(), is_split_into_words=True)
    print (test)
    return old_enc_size

def reset_model_with_tokenizer(model, tokenizer, old_enc_size, additional_tokens_smart_init=True):
    assert old_enc_size == model.get_input_embeddings().weight.data.size(0)
    model.resize_token_embeddings(len(tokenizer))

    embeddings = model.get_input_embeddings()
    if additional_tokens_smart_init:
        vocab = tokenizer.get_vocab()
        for tok, idx in vocab.items():

            if idx < old_enc_size:
                continue

            elif tok.startswith('<pointer:') and tok.endswith('>'):
                tok_split = ['pointer', str(tok.split(':')[1].strip('>'))]

            elif tok.startswith('<'):
                continue

            elif tok.startswith(':'):

                if tok.startswith(':op'):
                    tok_split = ['relation', 'operator', str(int(tok[3:]))]

                elif tok.startswith(':snt'):
                    tok_split = ['relation', 'sentence', str(int(tok[4:]))]

                elif tok.startswith(':ARG'):
                    tok_split = ['relation', 'argument', str(int(tok[4:]))]

                else:
                    tok_split = ['relation'] + tok.lstrip(':').split('-')

            else:
                tok_split = tok.split('-')
            tok_split = tokenizer.tokenize(tok_split, is_split_into_words=True)
            print ('initialize amr token: ', tok, ' => ', tok_split)
            vecs = []
            for s in tok_split:
                idx_split = vocab.get(s, -1)
                if idx_split > -1:
                    vec_split = embeddings.weight.data[idx_split].clone()
                    vecs.append(vec_split)

            if vecs:
                vec = torch.stack(vecs, 0).mean(0)
                embeddings.weight.data[idx] = vec
    model.tie_weights() 
