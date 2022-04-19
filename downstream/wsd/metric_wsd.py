

import os
import json
import tqdm
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from transformers import AutoModel, AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, precision_score

np.random.seed(1001)
torch.manual_seed(1001)
random.seed(1001)


def encode_data(tokenizer, sents, starts, ends, sym='[TGT]'):
    output_sents, spans = [], []
    for sent, char_start, char_end in zip(sents, starts, ends):
        # insert target symbols
        if sym is not None:
            sent = sent[:char_start] + '{} '.format(sym) + \
                sent[char_start:char_end] + ' {}'.format(sym) + sent[char_end:]
        output_sents.append(sent)

        sent = tokenizer.encode_plus(sent, return_offsets_mapping=True)
        # transform character indices to subtoken indices
        target_start = target_end = None
        if sym is not None:
            char_start += len(sym) + 1
            char_end += len(sym) + 1
        for idx, (token_start, token_end) in enumerate(sent['offset_mapping']):
            if token_start == char_start:
                target_start = idx
            if token_end == char_end:
                target_end = idx
        if target_start is None or target_end is None:
            raise ValueError
        spans.append((target_start, target_end + 1))

    # encode sentences
    encoded = tokenizer(output_sents, return_tensors='pt', padding=True)

    return encoded, spans


class Sampler:
    def __init__(self, df,
            max_support_size=40, max_query_size=40,
            # set this to a larger value is sense_sampling_strategy is `proportional`
            sense_support_proportion=0.5,
            lemma_sampling_strategy='uniform', sense_sampling_strategy='proportional'):
        self.df = df
        self.counts = df['lemma'].value_counts()
        self.max_support_size = max_support_size
        self.max_query_size = max_query_size
        self.sense_support_proportion = sense_support_proportion
        self.lemma_sampling_strategy = lemma_sampling_strategy
        self.sense_sampling_strategy = sense_sampling_strategy

    def sample_sentences(self, subset, sampling_strategy):
        # sample support and query sets
        sense_counts = subset['depth-1'].value_counts()
        if sampling_strategy == 'uniform':
            weights = 1 / sense_counts[subset['depth-1']].values
            weights = weights / weights.sum()
        elif sampling_strategy == 'proportional':
            weights = sense_counts[subset['depth-1']].values
            weights = weights / weights.sum()
        else:
            raise ValueError(sampling_strategy)

        return subset.sample(n=min(len(subset), self.max_support_size), weights=weights)

    def sample_lemma(self, max_tries=10):
        tries = 0
        while tries < max_tries:
            data = self.sample_lemma_()
            if data is not None:
                return data
            tries += 1

        raise ValueError('Max tries reached when trying to sample')

    def sample_lemma_(self):
        # sampling uniformly
        if self.lemma_sampling_strategy == 'uniform':
            lemma = self.counts.sample(1).index[0]
        # sampling proportionally to lemma frequency
        else:
            lemma = self.counts.sample(1, weights=self.counts/self.counts.sum()).index[0]

        subset = self.df[self.df['lemma'] == lemma]

        # sample senses
        senses = subset['depth-1'].value_counts()
        # drop singleton senses if there are
        subset = subset[subset['depth-1'].isin(senses[senses > 1].index)]
        # sample according to proportion
        support, query = train_test_split(
            subset.index, 
            train_size=self.sense_support_proportion, 
            stratify=subset['depth-1'])
        support, query = subset.loc[support], subset.loc[query]
        # ensure max size in support set
        if len(support) > self.max_support_size:
            support = support.sample(n=self.max_support_size)

        # ensure no senses in query that are not in support
        query = query[query['depth-1'].isin(support['depth-1'].unique())]
        if len(query) > self.max_query_size:
            query = query.sample(n=self.max_query_size)

        # it could be, we ran out of queries
        if len(query) == 0:
            return

        # support = self.sample_sentences(subset, self.sense_sampling_strategy, )
        # subset = subset[subset['depth-1'].isin(set(support['depth-1']))]
        # # drop instances already in support
        # subset = subset.loc[np.setdiff1d(subset.index, support.index)]
        # query = self.sample_sentences(subset, self.sense_sampling_strategy)

        assert np.setdiff1d(query['depth-1'].unique(), support['depth-1'].unique()).size == 0

        return support, query


def iter_lemmas(df, random_state=1001):
    for lemma in df['lemma'].unique():
        subset = df[df['lemma']==lemma]
        counts = subset['depth-1'].value_counts()
        # remove singletons
        subset = subset[subset['depth-1'].isin(counts[counts>1].index)]
        support, query = train_test_split(
            subset, test_size=0.5, stratify=subset['depth-1'], random_state=random_state)
        yield support, query


def collate(tokenizer, support, query, **kwargs):
    sents, starts, ends = support['quote'].values, support['start'].values, support['end']
    support_input, support_spans = encode_data(tokenizer, sents, starts, ends, **kwargs)
    sents, starts, ends = query['quote'].values, query['start'].values, query['end']
    query_input, query_spans = encode_data(tokenizer, sents, starts, ends, **kwargs)
    senses = support['depth-1'].value_counts().index
    # a list of indices mapping from sense to positions
    support_targets = [np.where(support['depth-1'] == sense)[0] for sense in senses]
    query_targets = np.concatenate([np.where(sense == senses)[0] for sense in query['depth-1'].values])
    return {
        'support': support_input, 'support_spans': support_spans, 'support_targets': support_targets,
        'query': query_input, 'query_spans': query_spans, 'query_targets': query_targets}


class Model(nn.Module):
    def __init__(self, model_path, dist='dot', device='cpu'):
        self.dist = dist
        self.device = device
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_path)
        self.bert.to(device)

    def to_device(self, inputs):
        return {key: val.to(self.device) for key, val in inputs.items()}

    def encode(self, inputs, spans, max_batch_size=40):
        total = len(inputs['input_ids'])
        output = []
        if total <= max_batch_size:
            for idx, hidden in enumerate(self.bert(**self.to_device(inputs))['last_hidden_state']):
                start, end = spans[idx]
                output.append(hidden[start:end].mean(dim=0))
        else:
           for b in range(0, total, max_batch_size):
               b_inputs = {key: val[b:min(b + max_batch_size, total)] for key, val in inputs.items()}
               b_inputs = self.to_device(b_inputs)
               for idx, hidden in enumerate(self.bert(**b_inputs)['last_hidden_state']):
                   start, end = spans[idx + b]
                   output.append(hidden[start:end].mean(dim=0))

        assert total == len(output)

        return torch.stack(output)

    def forward(self, batch, max_batch_size=40):
        support = self.encode(
            batch['support'], batch['support_spans'], max_batch_size=max_batch_size)
        query = self.encode(
            batch['query'], batch['query_spans'], max_batch_size=max_batch_size)
        centroids = torch.stack([support[index].mean(0) for index in batch['support_targets']])
        # query_norm = query / torch.norm(query, dim=1)[:, None]
        # centroids_norm = centroids / torch.norm(centroids, dim=1)[:, None]
        # (batch x num_senses)
        return query @ centroids.t()

    def infer(self, batch, max_batch_size=40):
        return self(batch, max_batch_size=max_batch_size).argmax(dim=1).cpu().numpy()


def get_metrics(trues, preds, lemmas):
    f1s, r_s, p_s, support = [], [], [], []

    # micro over senses, macro over lemmas

    for lemma in np.unique(lemmas):
        index = np.where(lemmas == lemma)[0]
        f1s.append(f1_score(trues[index], preds[index], average='micro'))
        r_s.append(recall_score(trues[index], preds[index], average='micro'))
        p_s.append(precision_score(trues[index], preds[index], average='micro'))
        support.append(len(index))

    support = np.array(support)

    scores = {}
    for metric, score in {'f1': f1s, 'recall': r_s, 'precision': p_s}.items():
        score = np.array(score)
        scores['{}-micro'.format(metric)] = score.mean()
        scores['{}-macro'.format(metric)] = ((support / support.sum()) * score).sum()

    return scores


def evaluate_dev(model, tokenizer, training_data, dev_data, max_batch_size=40, **kwargs):
    trues, preds, lemmas = [], [], []
    for lemma in tqdm.tqdm(dev_data['lemma'].unique()):
        support = training_data[training_data['lemma']==lemma]
        query = dev_data[dev_data['lemma']==lemma]
        # check that no queries have senses not present in support set
        assert np.setdiff1d(query['depth-1'].unique(), support['depth-1'].unique()).size == 0
        batch = collate(tokenizer, support, query, **kwargs)
        preds.extend(model.infer(batch, max_batch_size=max_batch_size))
        trues.extend(batch['query_targets'])
        lemmas.extend(query['lemma'].to_numpy())

    return np.array(trues), np.array(preds), np.array(lemmas)


def sample_up_to_n(g, n):
    if len(g) <= n:
        return g
    return g.sample(n=n)


def evaluate_df(
        model, tokenizer, training_data, df, max_batch_size=40, 
        max_support_per_sense=np.inf, **kwargs):
    trues, preds, index = [], [], []

    for lemma in tqdm.tqdm(df['lemma'].unique()):
        support = training_data[training_data['lemma']==lemma].groupby(
            'depth-1'
        ).apply(
            lambda g: sample_up_to_n(g, max_support_per_sense)
        ).reset_index(drop=True)
        query = df[df['lemma']==lemma]
        # check that no queries have senses not present in support set
        assert np.setdiff1d(query['depth-1'].unique(), support['depth-1'].unique()).size == 0
        batch = collate(tokenizer, support, query, **kwargs)
        preds.extend(model.infer(batch, max_batch_size=max_batch_size))
        trues.extend(batch['query_targets'])
        index.extend(query.index.to_numpy())

    return np.array(trues), np.array(preds), np.array(index)


def evaluate_zero(model, tokenizer, zero_data, max_batch_size=40, **kwargs):
    preds, trues, lemmas = [], [], []
    for support, query in iter_lemmas(zero_data):
        assert np.setdiff1d(query['depth-1'].unique(), support['depth-1'].unique()).size == 0
        batch = collate(tokenizer, support, query, **kwargs)
        preds.extend(model.infer(batch, max_batch_size=max_batch_size))
        trues.extend(batch['query_targets'])
        lemmas.extend(query['lemma'].to_numpy())

    return np.array(trues), np.array(preds), np.array(lemmas)


def train_model(model, tokenizer, training_data, dev_data, zero_data,
        eval_steps=100, training_steps=10000, eval_every=1000, update_every=5,
        lr=1e-5, max_batch_size=40, max_support_size=20, max_query_size=20,
        device='cpu'):

    training_sampler = Sampler(
        training_data, 
        max_support_size=max_support_size, 
        max_query_size=max_query_size)
    dev_sampler = Sampler(dev_data)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    tloss = steps = evals = 0
    with tqdm.trange(training_steps) as pbar:
        for step in range(training_steps):
            # get data
            support, query = training_sampler.sample_lemma()
            batch = collate(tokenizer, support, query)
            targets = torch.tensor(batch['query_targets']).to(device)
            scores = model(batch, max_batch_size=max_batch_size)
            loss = F.cross_entropy(scores, targets)
            loss.backward()
            tloss += loss.item()
            steps += 1
            pbar.update()
            if step > 0 and step % update_every == 0:
                optimizer.step()
                optimizer.zero_grad()
                pbar.set_postfix(Loss=tloss / steps)

            # evaluate
            if step > 0 and step % eval_every == 0:
                # reset total loss
                tloss = steps = 0
                evals += 1
                model.eval()
                with torch.no_grad():
                    for result in evaluate_model(
                            model, tokenizer,
                            training_data, dev_data, dev_sampler, zero_data,
                            device, eval_steps, max_batch_size, evals):
                        yield dict(epoch=evals, **result)
                model.train()


def evaluate_model(
        model, tokenizer,
        training_data, dev_data, dev_sampler, zero_data, 
        device, eval_steps, max_batch_size, epoch):
    # dev loss
    dev_loss = 0
    for _ in range(eval_steps):
        support, query = dev_sampler.sample_lemma()
        batch = collate(tokenizer, support, query)
        dev_loss += F.cross_entropy(
            model(batch, max_batch_size=max_batch_size),
            torch.tensor(batch['query_targets']).to(device)
        ).item()
    dev_loss /= eval_steps
    yield dict(metric='dev-loss', score=dev_loss)
    # dev metrics
    trues, preds, lemmas = evaluate_dev(
        model, tokenizer, training_data, dev_data,
        max_batch_size=max_batch_size)
    scores = get_metrics(trues, preds, lemmas)
    yield dict(metric='dev', **scores)
    # zero-shot
    trues, preds, lemmas = evaluate_zero(
        model, tokenizer, zero_data, max_batch_size=max_batch_size)
    scores = get_metrics(trues, preds, lemmas)
    yield dict(metric='zero', **scores)


def evaluate_baseline(model_path, training_data, dev_data, zero_data, max_batch_size, device='cpu'):
    # model = Model("emanjavacas/MacBERTh")
    model = Model(model_path, device=device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # dev metrics
    trues, preds, lemmas = evaluate_dev(
        model, tokenizer, training_data, dev_data,
        max_batch_size=max_batch_size, sym=None) # don't add [TGT] tokens to baseline
    scores = get_metrics(trues, preds, lemmas)
    yield dict(metric='dev', **scores)
    # zero-shot
    trues, preds, lemmas = evaluate_zero(
        model, tokenizer, zero_data, max_batch_size=max_batch_size, sym=None)
    scores = get_metrics(trues, preds, lemmas)
    yield dict(metric='zero', **scores)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-file', required=True)
    parser.add_argument('--train-file', required=True)
    parser.add_argument('--max-train-lemmas', type=int, default=1e10,
        help="Maximum number of lemmas in training set")
    parser.add_argument('--max-train-per-sense', type=int, default=1e10,
        help="Maximum number of instances per sense in training set")
    parser.add_argument('--test-file', required=True)
    parser.add_argument('--do-predict', action='store_true')
    parser.add_argument('--zero-shot-train-file')
    parser.add_argument('--zero-shot-test-file')
    parser.add_argument('--max-support-size', type=int, default=20,
        help="Maximum number of instances in the support set")
    parser.add_argument('--max-query-size', type=int, default=20,
        help="Maximum number of instances in the query set")
    parser.add_argument('--max-batch-size', type=int, default=20)
    parser.add_argument('--dev-lemmas', type=int, default=50,
        help="Number of lemmas in the development set")
    parser.add_argument('--training-steps', type=int, default=10000)
    parser.add_argument('--eval-every', type=int, default=1000)
    parser.add_argument('--eval-steps', type=int, default=100)
    parser.add_argument('--update-every', type=int, default=5)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    if args.do_predict and not (os.path.isfile(args.zero_shot_train_file) and
            os.path.isfile(args.zero_shot_test_file)):
        raise ValueError

    # class Object(object):
    #     pass

    # args = Object()
    # args.max_train_lemmas = 100
    # args.max_train_per_sense = 2
    # args.dev_lemmas = 50

    # read data
    # source_df = pd.read_csv('/home/manjavacasema/code/macberth-eval/data/wsd/splits/oed-quotes-subset-depth-1-train.csv')
    source_df = pd.read_csv(args.train_file)
    # sample zero-shot lemmas for development
    zero_lemmas = source_df['lemma'].value_counts().sample(50, random_state=1001).index
    zero = source_df[source_df['lemma'].isin(zero_lemmas)]
    df = source_df[~source_df['lemma'].isin(zero_lemmas)]
    # sample training lemmas
    df_lemmas = df['lemma'].value_counts()
    if args.max_train_lemmas < len(df_lemmas):
        df = df[df['lemma'].isin(df_lemmas.sample(args.max_train_lemmas, random_state=1001).index)]
    # sample instances per sense for training
    if args.max_train_per_sense:
        df = df.groupby(
            ['lemma', 'depth-1']
        ).apply(
            lambda g: sample_up_to_n(g, args.max_train_per_sense)
        ).reset_index(drop=True)

    # test = pd.read_csv('/home/manjavacasema/code/macberth-eval/data/wsd/splits/oed-quotes-subset-depth-1-test.csv')
    test = pd.read_csv(args.test_file)
    # don't sample from lemmas reserved for zero-shot

    dev = test[(test['lemma'].isin(df['lemma'].unique())) & (~test['lemma'].isin(zero_lemmas))]
    if dev['lemma'].unique().size > args.dev_lemmas:
        dev = dev[dev['lemma'].isin(dev['lemma'].value_counts().sample(
            args.dev_lemmas, random_state=1991).index)]

    # tokenizer = AutoTokenizer.from_pretrained('emanjavacas/MacBERTh')
    tokenizer = AutoTokenizer.from_pretrained(args.model_file)
    tokenizer.add_special_tokens({'additional_special_tokens': ['[TGT]']})

    with torch.no_grad():
        for result in evaluate_baseline(args.model_file, df, dev, zero, 
                max_batch_size=args.max_batch_size, device=args.device):
            result['model'] = 'baseline'
            print(json.dumps(dict(result, **args.__dict__)))

    # model = Model('emanjavacas/MacBERTh')
    model = Model(args.model_file, device=args.device)
    model.bert.resize_token_embeddings(len(tokenizer))

    # dev_sampler = Sampler(dev)
    # training_sampler = Sampler(
    #     df, 
    #     max_support_size=10, 
    #     max_query_size=20)
    # for _ in range(10000):
    #     training_sampler.sample_lemma()
    # for _ in range(10000):
    #     dev_sampler.sample_lemma()

    for result in train_model(model, tokenizer, df, dev, zero, device=args.device, 
            max_batch_size=args.max_batch_size, update_every=args.update_every,
            eval_every=args.eval_every, eval_steps=args.eval_steps,
            training_steps=args.training_steps, max_support_size=args.max_support_size):
        print(json.dumps(dict(result, **args.__dict__)))

    if args.do_predict:
        def get_filename(prefix, **kwargs):
            for key, val in sorted(kwargs.items()):
                prefix += '.{key}={val}'.format(key=key, val=val)
            return prefix + ".json"

        zero_train = pd.read_csv(args.zero_shot_train_file)
        zero_test = pd.read_csv(args.zero_shot_test_file)
        baseline = Model(args.model_file, device=args.device)
        with torch.no_grad():
            model.eval()
            baseline.eval()
            for max_support in [1, 2, 5, 10, 20, np.inf]:
                # baseline on zero
                trues, preds, index = evaluate_df(baseline, tokenizer, zero_train, zero_test,
                    max_batch_size=args.max_batch_size, max_support_per_sense=max_support, sym=None)
                output_path = get_filename(
                    os.path.basename(args.model_file) + '.baseline.zero.predict',
                    max_support=max_support, max_train_lemmas=args.max_train_lemmas,
                    max_train_per_sense=args.max_train_per_sense)
                with open(output_path, 'w+') as f:
                    json.dump({
                        'trues': trues.tolist(),
                        'preds': preds.tolist(), 
                        'index': index.tolist()}, f)
                # model on zero
                trues, preds, index = evaluate_df(model, tokenizer, zero_train, zero_test,
                    max_batch_size=args.max_batch_size, max_support_per_sense=max_support)
                output_path = get_filename(
                    os.path.basename(args.model_file) + '.zero.predict',
                    max_support=max_support, max_train_lemmas=args.max_train_lemmas,
                    max_train_per_sense=args.max_train_per_sense)
                with open(output_path, 'w+') as f:
                    json.dump({
                        'trues': trues.tolist(), 
                        'preds': preds.tolist(), 
                        'index': index.tolist()}, f)
                # baseline on test
                trues, preds, index = evaluate_df(baseline, tokenizer, source_df, test,
                    max_batch_size=args.max_batch_size, max_support_per_sense=max_support, sym=None)
                output_path = get_filename(
                    os.path.basename(args.model_file) + '.baseline.predict',
                    max_support=max_support, max_train_lemmas=args.max_train_lemmas,
                    max_train_per_sense=args.max_train_per_sense)
                with open(output_path, 'w+') as f:
                    json.dump({
                        'trues': trues.tolist(),
                        'preds': preds.tolist(), 
                        'index': index.tolist()}, f)
                # model on test
                trues, preds, index = evaluate_df(model, tokenizer, source_df, test,
                    max_batch_size=args.max_batch_size, max_support_per_sense=max_support)
                output_path = get_filename(
                    os.path.basename(args.model_file) + '.predict',
                    max_support=max_support, max_train_lemmas=args.max_train_lemmas,
                    max_train_per_sense=args.max_train_per_sense)
                with open(output_path, 'w+') as f:
                    json.dump({
                        'trues': trues.tolist(), 
                        'preds': preds.tolist(), 
                        'index': index.tolist()}, f)


# type(np.array([1, 2,3]).tolist()[0])
# test = pd.read_csv('./data/wsd/splits/oed-quotes-subset-depth-1-test.zero.csv')
# train = pd.read_csv('./data/wsd/splits/oed-quotes-subset-depth-1-train.zero.csv')
# import numpy as np

# for lemma in test['lemma'].unique():
#     test_lemma = test[test['lemma'] == lemma]
#     train_lemma = train[train['lemma'] == lemma]
#     assert np.setdiff1d(test_lemma['depth-1'], train_lemma['depth-1']).size == 0
#     print(test_lemma['depth-1'].value_counts())

# training_sampler = Sampler(df)
# stats = []
# for _ in range(1000):
#     support, query = training_sampler.sample_lemma()
#     batch = collate(tokenizer, support, query)
#     assert batch['query_targets'].shape[0] == batch['query']['input_ids'].shape[0]
#     stats.append(
#         {'query': batch['query_targets'].shape[0], 
#          'query-senses': len(np.unique(batch['query_targets'])),
#          'support': batch['support']['input_ids'].shape[0], 
#          'support-senses': len(batch['support_targets'])})

# stats = pd.DataFrame.from_dict(stats)
# stats['support-ratio'] = stats['support-senses'] / stats['support']
# stats['query-ratio'] = stats['query-senses'] / stats['query']
# stats.describe()
# sents, starts, ends = support['quote'].values, support['start'].values, support['end']
# support_input, support_spans = encode_data(tokenizer, sents, starts, ends)
# start, end = support_spans[0][0], support_spans[0][1]
# tokenizer.convert_ids_to_tokens(support_input['input_ids'][0])[start:end]
# support_input, support_spans = encode_data(tokenizer, sents, starts, ends, sym=None)
# start, end = support_spans[0][0], support_spans[0][1]
# tokenizer.convert_ids_to_tokens(support_input['input_ids'][0])[start:end]
