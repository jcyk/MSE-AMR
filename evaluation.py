import sys
import io, os
import numpy as np
import logging
import argparse
from prettytable import PrettyTable
import torch
import transformers
from transformers import AutoModel, AutoTokenizer, AutoConfig, BertModel
from tatoeba import tatoeba
from bucc import bucc_run
import collections

# Set up logger
logging.basicConfig(format='%(asctime)s %(name)s: %(message)s', level=logging.DEBUG)

# Set PATHs
PATH_TO_SENTEVAL = './SentEval'
PATH_TO_DATA = './SentEval/data'
# Import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval
from amr import AMRParser, init_amr_vocabulary, reset_model_with_tokenizer

def print_table(task_names, scores):
    tb = PrettyTable()
    tb.field_names = task_names
    try:
        collectionsAbc = collections.abc
    except AttributeError:
        collectionsAbc = collections
    if not isinstance(scores, collectionsAbc.Mapping):
        tb.add_row(scores)
    else:
        for value in scores.values():
            tb.add_row(value)
    print(tb)

def transformer_embed(model, tokenizer, sentences_or_amrs, pooler, max_length, use_amr):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if max_length is None:
        max_length = model.config.max_position_embeddings - 2 

    # Tokenization
    batch = tokenizer.batch_encode_plus(
            [x.split() for x in sentences_or_amrs],
            return_tensors='pt',
            padding=True,
            max_length=max_length,
            truncation=True,
            is_split_into_words=True
            )

    # if use_amr:
    #     seq_length = batch['input_ids'].size(1)
    #     position_ids = model.embeddings.position_ids[:, :seq_length]
    #     batch['position_ids'] = position_ids + 128
    #     batch['token_type_ids'] = batch['token_type_ids'] + 1

    # Move to the correct device
    for k in batch:
        batch[k] = batch[k].to(device)
    
    # Get raw embeddings
    with torch.no_grad():
        outputs = model(**batch, output_hidden_states=True, return_dict=True)
        last_hidden = outputs.last_hidden_state
        pooler_output = outputs.pooler_output
        hidden_states = outputs.hidden_states

    # Apply different poolers
    if pooler == 'simcse_sup':
        # this is a special setup only for simcse sup, where we use the repr after projector
        # Note that the projector is a linear+activation layer after CLS representation
        return pooler_output.cpu()
    elif pooler == 'cls':
        return last_hidden[:, 0].cpu()
    elif pooler == "avg":
        return ((last_hidden * batch['attention_mask'].unsqueeze(-1)).sum(1) / batch['attention_mask'].sum(-1).unsqueeze(-1)).cpu()
    elif pooler == "avg_first_last":
        first_hidden = hidden_states[0]
        last_hidden = hidden_states[-1]
        pooled_result = ((first_hidden + last_hidden) / 2.0 * batch['attention_mask'].unsqueeze(-1)).sum(1) / batch['attention_mask'].sum(-1).unsqueeze(-1)
        return pooled_result.cpu()
    elif pooler == "avg_top2":
        second_last_hidden = hidden_states[-2]
        last_hidden = hidden_states[-1]
        pooled_result = ((last_hidden + second_last_hidden) / 2.0 * batch['attention_mask'].unsqueeze(-1)).sum(1) / batch['attention_mask'].sum(-1).unsqueeze(-1)
        return pooled_result.cpu()
    else:
        raise NotImplementedError

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, 
            default=argparse.SUPPRESS,
            help="Transformers' model name or path")
    parser.add_argument("--batch_size", type=int, 
            default=64,
            help="Batch size")
    parser.add_argument("--laser", action='store_true', 
            default=argparse.SUPPRESS,
            help="Use LASER embeddings")
    parser.add_argument("--muse", action='store_true', 
            default=argparse.SUPPRESS,
            help="Use Multilingual Universal Sentence Encoder (mUSE)")
    parser.add_argument("--pooler", type=str, 
            #choices=['cls', 'simcse_sup', 'avg', 'avg_top2', 'avg_first_last'],
            default='cls', 
            help="Which pooler to use")
    parser.add_argument("--mode", type=str, 
            choices=['dev', 'test', 'fasttest'],
            default='test', 
            help="What evaluation mode to use (dev: fast mode, dev results; test: full mode, test results); fasttest: fast mode, test results")
    parser.add_argument("--task_set", type=str, 
            choices=['sts', 'msts', 'transfer', 'ml_transfer', 'tatoeba', 'bucc', 'full', 'na'],
            default='na',
            help="What set of tasks to evaluate on. If not 'na', this will override '--tasks'")
    parser.add_argument("--tasks", type=str, nargs='+', 
            default=['STS12', 'STS13', 'STS14', 'STS15', 'STS16',
                     'MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC',
                     'SICKRelatedness', 'STSBenchmark',
                     'MLDoc', 'XNLI', 'PAWS-X', 'MARC', 'QAM'],
            help="Tasks to evaluate on. If '--task_set' is specified, this will be overridden")
    parser.add_argument("--ml_multilingual_training", action='store_true')
    parser.add_argument("--write_sentences", type=str, default=None) 
    parser.add_argument("--use_amr", action='store_true')
    parser.add_argument("--path_to_amr", type=str, default='amr/cache.txt')
    parser.add_argument("--drop_parentheses", action='store_true')
    parser.add_argument("--normalize", action='store_true')
    parser.add_argument("--combine_method", type=str, 
            choices=['cat', 'sum'],
            default='sum', 
            help="how to combine different embeddings")
    
    args0, remaining_args = parser.parse_known_args()
    if args0.write_sentences:
        fsent = open(args0.write_sentences, "w")
        write_sentence_idx = {'x':0}
    if args0.use_amr:
        amr_parser = AMRParser(args0.path_to_amr, dereify=True, remove_wiki=True, use_pointer_tokens=True, drop_parentheses=args0.drop_parentheses)
    model_name_msg = ''
    
    if not hasattr(args0, 'model_name_or_path') and not hasattr(args0, 'laser') and not hasattr(args0, 'muse'):
        print()
        print('Either model_name_or_path or some other model must be specified!')
        exit()
    

    from argparse import Namespace

    if hasattr(args0, 'model_name_or_path') or hasattr(args0, 'laser') or hasattr(args0, 'muse'):
        args = Namespace(**vars(args0))
    else:
        args = Namespace(**vars(args0), entok=entok, sp=sp, embedder=embedder,
                         encoder=args.eval_encoder, tokenize=args.tokenize)
                     
    import bgt_evaluate_mod as bgt_eval
    
    # Load transformers' model checkpoint
    models = []
    tokenizers = []
    use_amrs = []
    if hasattr(args0, 'model_name_or_path'): 
        model_name_msg = 'Model: {}'.format(args.model_name_or_path)
        print(model_name_msg)
        for model_name_or_path in args.model_name_or_path.split(":"):
            model = AutoModel.from_pretrained(model_name_or_path)
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, add_prefix_space=True)
            if 'amr' in model_name_or_path:
                if 'xlm-roberta' in model_name_or_path:
                    INIT = '▁'
                elif 'roberta' in model_name_or_path:
                    INIT = 'Ġ'
                else:
                    INIT = ''
                init_amr_vocabulary(tokenizer)

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            model = model.to(device)

            models.append(model)
            tokenizers.append(tokenizer)
            use_amrs.append(args0.use_amr and 'amr' in model_name_or_path)

    # Load LASER
    if hasattr(args0, 'laser'):
        model_name_msg = 'Model: LASER'
        print(model_name_msg)
        from laserembeddings import Laser
        laser = Laser()
        
    # Load mUSE
    if hasattr(args0, 'muse'):
        model_name_msg = 'Model: mUSE'
        print(model_name_msg)
        model_url = "https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3"
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'        
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        import gc
        import tensorflow as tf
        import tensorflow_hub as hub
        import tensorflow_text
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            # Restrict TensorFlow to only allocate 4 GB of memory on the first GPU
            try:
                tf.config.set_logical_device_configuration(
                    gpus[0],
                    [tf.config.LogicalDeviceConfiguration(memory_limit=1024*4)])
            except RuntimeError as e:
                # Virtual devices must be set before GPUs have been initialized
                print(e)
        tf.get_logger().setLevel(logging.ERROR)
        embed = hub.load(model_url)

    # Set up the tasks
    # if args.task_set is not na, use it to overwrite args.tasks
    if args.task_set != 'na':
        args.tasks = []

    if args.task_set == 'sts':
        args.tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']
    elif args.task_set == 'transfer':
        args.tasks = ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC']
    elif args.task_set == 'ml_transfer':
        args.tasks = ['MLDoc', 'XNLI', 'PAWS-X', 'MARC', 'QAM']
    elif args.task_set == 'full':
        args.tasks = ['MLDoc', 'XNLI', 'PAWS-X', 'MARC', 'QAM']
    
    for task in ['XNLI', 'PAWS-X', 'QAM']:
        if task not in args.tasks:
            continue
        args.tasks.append(task+'_0')
        args.tasks.remove(task)

    for task in ['MLDoc', 'MARC']:
        if task not in args.tasks:
            continue
        if args0.ml_multilingual_training:
            args.tasks.append(task+'_m')
        args.tasks.append(task+'_0')
        args.tasks.remove(task)

    # Set params for SentEval
    if args.mode == 'dev' or args.mode == 'fasttest':
        # Fast mode
        params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
        params['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                         'tenacity': 3, 'epoch_size': 2}
    elif args.mode == 'test':
        # Full mode
        params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10}
        params['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                                         'tenacity': 5, 'epoch_size': 4}
    else:
        raise NotImplementedError

    params.update(vars(args))

    # SentEval prepare and batcher
    def prepare(params, samples):
        return
    
    def batcher(params, batch, lang="en", max_length=None):
        # Handle rare token encoding issues in the dataset
        if len(batch) >= 1 and len(batch[0]) >= 1 and isinstance(batch[0][0], bytes):
            batch = [[word.decode('utf-8') for word in s] for s in batch]

        sentences = [' '.join(s) for s in batch]
        if args0.write_sentences:
            for sent in sentences:
                fsent.write(f"# ::id {write_sentence_idx['x']}\n")
                fsent.write(f"# ::snt {sent}\n")
                fsent.write(f"# ::snt_lang {lang}\n")
                fsent.write(f"(e / empty)\n\n")
                write_sentence_idx['x'] =write_sentence_idx['x'] + 1
        if args0.use_amr:
            amrs = [ amr_parser.parse(s) for s in sentences]
        
        embeddings = []

        if hasattr(args0, 'muse'):
            long_ = {}
            for idx, sent in enumerate(sentences):
                if len(sent) > 400:
                    try:
                        long_[idx] = embed([sent]).numpy()
                    except:
                        with tf.device('/CPU:0'):
                            long_[idx] = embed([sent]).numpy()
            for k in long_:
                sentences[k] = sentences[k][:10]
            out_ = embed(sentences).numpy()
            for k in long_:
                out_[k] = long_[k][0]
            embeddings.append(torch.from_numpy(out_).cpu())
            
        if hasattr(args0, 'laser'):
            embeddings.append(torch.from_numpy(laser.embed_sentences(sentences, lang)).cpu())
        
        
        poolers = args.pooler.split(':')
        if len(poolers) != len(models):
            assert len(poolers) == 1
            poolers = [ args.pooler for m in models]
        for model, tokenizer, use_amr, pooler in zip(models, tokenizers, use_amrs, poolers):
            if use_amr:
                embedding = transformer_embed(model, tokenizer, amrs, pooler, max_length, use_amr)
            else:
                embedding = transformer_embed(model, tokenizer, sentences, pooler, max_length, use_amr)
            embeddings.append(embedding)
        
        if args0.normalize:
            embeddings = [ torch.nn.functional.normalize(embedding, dim=-1) for embedding in embeddings]

        if args0.combine_method == 'cat':
            return torch.cat(embeddings, -1)
        elif args0.combine_method == 'sum':
            return torch.stack(embeddings, 0).sum(0)
        else:
            raise NotImplementedError


    results = {}
    if args.task_set in ['sts', 'transfer', 'ml_transfer', 'full', 'na']:
        for task in args.tasks:
            se = senteval.engine.SE(params, batcher, prepare)
            result = se.eval(task)
            results[task] = result
    
    results_semeval17 = {}
    if args.task_set in ['msts', 'full']:
        s = bgt_eval.SemEval17('STS/STS17-test')
        s.do_prepare()
        results_semeval17 = s.run(args, batcher)
        
    results_tatoeba = {}
    if args.task_set in ['tatoeba', 'full']:
        results_tatoeba = tatoeba.run(args, batcher)
    
    results_bucc = {}
    if args.task_set in ['bucc', 'full']:
        results_bucc = bucc_run.run(args, batcher)
        
    # Print evaluation results
    if args.mode == 'dev':
        print("------ %s ------" % (args.mode))
        print(model_name_msg)
        
        task_names = []
        scores = []
        for task in ['STSBenchmark', 'SICKRelatedness']:
            task_names.append(task)
            if task in results:
                scores.append("%.2f" % (results[task]['dev']['spearman'][0] * 100))
            else:
                scores.append("0.00")
        print_table(task_names, scores)

        task_names = []
        scores = []
        for task in ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']:
            task_names.append(task)
            if task in results:
                scores.append("%.2f" % (results[task]['devacc']))    
            else:
                scores.append("0.00")
        task_names.append("Avg.")
        scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
        print_table(task_names, scores)

    elif args.mode == 'test' or args.mode == 'fasttest':
        print("------ %s ------" % (args.mode))
        print(model_name_msg)
                        
        task_names = []
        scores = []
        print_ = False
        for task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']:
            task_names.append(task)
            if task in results:
                if task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']:
                    print_ = True
                    scores.append("%.2f" % (results[task]['all']['spearman']['all'] * 100))
                else:
                    scores.append("%.2f" % (results[task]['test']['spearman'].correlation * 100))
            else:
                scores.append("0.00")
        task_names.append("Avg.")
        scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
        if print_:
            print_table(task_names, scores)

        results_semeval17_2 = {}
        if len(results_semeval17) > 0:
            for i in results_semeval17:
                if i.endswith('all'):
                    continue
                results_semeval17_2[i.split('.')[4]] = results_semeval17[i]
            task_names = ['STS17']
            scores = []
            for task in ['en-en', 'es-es', 'ar-ar']:
                task_names.append(task)
                if task in results_semeval17_2:
                    scores.append("%.2f" % (results_semeval17_2[task]['spearman'].correlation * 100))
                else:
                    scores.append("0.00")
            task_names.append("Avg.")
            scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
            scores.insert(0, '')
            print_table(task_names, scores)
            task_names = ['STS17']
            scores = []
            for task in ['ar-en', 'de-en', 'tr-en', 'es-en', 'fr-en', 'it-en', 'nl-en']:
                task_names.append(task[3:]+'-'+task[:2])
                if task in results_semeval17_2:
                    scores.append("%.2f" % (results_semeval17_2[task]['spearman'].correlation * 100))
                else:
                    scores.append("0.00")
            task_names.append("Avg.")
            scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
            scores.insert(0, '')
            print_table(task_names, scores)
        
        scores_avg = collections.OrderedDict()
        scores_avg['m'] = []
        scores_avg['0'] = []
        print_ = False
        for task_ in ['MLDoc', 'XNLI', 'PAWS-X', 'MARC', 'QAM']:
            scores = collections.OrderedDict()
            col_names = [task_]
            for setting in ['m', '0']:
                task = task_+'_'+setting
                if task not in results:
                    scores_avg[setting].append('-')
                    continue
                print_ = True
                scores_ = []
                langs__ = list(results[task]['acc'].keys())
                if len(col_names) == 1:
                    for lang in langs__:
                        col_names.append(lang)
                    col_names.append("Avg.")
                for lang in langs__:
                    scores_.append("%.2f" % (results[task]['acc'][lang]))
                avg_tmp = "%.2f" % (sum([float(score) for score in scores_]) / len(scores_))
                scores_.append(avg_tmp)
                scores_avg[setting].append(avg_tmp)
                if setting == 'm':
                    scores_.insert(0, "mlearn")
                elif setting == '0':
                    scores_.insert(0, "0-shot")
                scores[setting] = scores_
            if len(scores) > 0:
                print_table(col_names, scores)
        col_names = ['']
        col_names.extend(['MLDoc', 'XNLI', 'PAWS-X', 'MARC', 'QAM'])
        col_names.append("Avg.")
        for key in scores_avg:
            scores_ = [float(score) for score in scores_avg[key] if score != '-']
            scores_avg[key].append("%.2f" % (np.mean(scores_)) if len(scores_) > 0 else "-")
            if key == 'm':
                scores_avg[key].insert(0, "mlearn Avg.")
            elif key == '0':
                scores_avg[key].insert(0, "0-shot Avg.")
        if print_:
            print_table(col_names, scores_avg)
        
        task_names = []
        scores = []
        print_ = False
        for task in ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']:
            task_names.append(task)
            if task in results:
                print_ = True
                scores.append("%.2f" % (results[task]['acc']))    
            else:
                scores.append("0.00")
        task_names.append("Avg.")
        scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
        if print_:
            print_table(task_names, scores)

        if len(results_tatoeba) > 0:
            task_names = []
            scores = []
            for task, acc in results_tatoeba.items():
                if not task.startswith('en'):
                    continue
                task_names.append(task)
                scores.append("%.2f" % (acc * 100))
            task_names.append("Avg.")
            scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
            task_names.insert(0, 'Tatoeba en->x')
            scores.insert(0, '')
            print_table(task_names, scores)
            task_names = []
            scores = []
            for task, acc in results_tatoeba.items():
                if not task.endswith('>en'):
                    continue
                task_names.append(task)
                scores.append("%.2f" % (acc * 100))
            task_names.append("Avg.")
            scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
            task_names.insert(0, 'Tatoeba x->en')
            scores.insert(0, '')
            print_table(task_names, scores)
        
        if len(results_bucc) > 0:
            task_names = []
            scores = []
            for task, f1 in results_bucc.items():
                task_names.append(task)
                scores.append("%.2f" % (f1 * 100))
            task_names.append("Avg.")
            scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
            task_names.insert(0, 'BUCC')
            scores.insert(0, '')
            print_table(task_names, scores)
    if args0.write_sentences:
        fsent.close()

if __name__ == "__main__":
    main()
