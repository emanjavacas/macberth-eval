
import os
import tqdm
import random

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

seed = 1001
random.seed(seed)
np.random.seed(seed)


def get_pos(w):
    term, *pos = w.split(',')
    term_pos = []
    for p in pos:
        p = p.replace('and', '').replace('(', '').replace(')', '').strip()
        if ' ' in p:
            for sub_p in ' '.join(p.split()).split():
                term_pos.append(sub_p)
        else:
            term_pos.append(p)
    term_pos = [p.split('.')[0] for p in term_pos]
    return term_pos[0]


def baselines(train, test, key='depth-1'):
    # majority baseline
    true = test[key]
    counts = train[key].value_counts()
    majority = [counts.index[0] for _ in range(len(true))]
    # random baseline
    labels, pvals = zip(*dict(counts / counts.sum()).items())
    rand = np.random.multinomial(1, pvals, size=len(test))
    rand = np.where(rand)[1]
    rand = [labels[i] for i in rand]
    return majority, rand


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True) # './data/oed-quotes-subset.tsv'
    parser.add_argument('--sep', default='\t')
    parser.add_argument('--modelpaths', nargs='+', required=True)
    parser.add_argument('--output-prefix', default='./data/wsd/')
    parser.add_argument('--key', default='depth-1')
    args = parser.parse_args()

    if not os.path.isdir(args.output_prefix):
        os.makedirs(args.output_prefix)

    df_source = pd.read_csv(args.data, sep=args.sep)
    # df_source = pd.read_csv('./data/oed-quotes-subset.tsv', sep='\t')
    df_source['pos'] = df_source['word'].apply(get_pos)

    # add depths
    depths = set(df_source['numbering'].apply(lambda row: len(row.rstrip('.').split('.'))))
    for depth in depths:
        df_source['depth-{}'.format(depth)] = df_source['numbering'].apply(
            lambda row: '.'.join(row.rstrip('.').split('.')[:depth]))

    # create splits
    counts = df_source['lemma'].value_counts()
    targets = counts[counts >= 50].index
    splits = []
    for lemma, subset in df_source[df_source['lemma'].isin(targets)].groupby('lemma'):
        # drop lemmas not in target frequency
        if lemma not in targets:
            continue
        # drop senses where we can't stratify
        senses = subset[args.key].value_counts()
        subset = subset[subset[args.key].isin(senses[senses >= 2].index)]
        if len(subset) < 2:
            continue
        # drop lemmas with only one sense
        senses = subset[args.key].value_counts()
        if len(senses) == 1:
            continue
        train, test = train_test_split(subset.index, stratify=subset[args.key], test_size=0.5)
        assert set(df_source.iloc[test][args.key]).difference(
            set(df_source.iloc[train][args.key])) == set()
        splits.append((train, test))

    do_baselines = True
    results = []

    for path in args.modelpaths:
        embs = np.load(path, allow_pickle=True)
        n_subtokens = embs.item()['n_subtokens']
        embs, index = embs.item()['embs'], embs.item()['index']
        df_source['n_subtokens'] = None
        df_source['n_subtokens'].iloc[index] = n_subtokens
        assert len(embs) == len(df_source)
        # path = './data/wsd/multi_cased_L-12_H-768_A-12.embs.npy'
        model, *_ = os.path.basename(path).split('.')

        for train_index, test_index in tqdm.tqdm(splits):
            train, test = df_source.iloc[train_index], df_source.iloc[test_index]
            # drop cases if they are not in the index
            if len((set(test.index)).difference(index)) > 0:
                continue
            # compute centroids
            centroids, labels = [], []
            for sense, group in train.groupby(args.key):
                labels.append(sense)
                centroids.append(embs[group.index].mean(axis=0))
            centroids = np.array(centroids)
            # model
            sims = cosine_similarity(embs[test.index], centroids)
            pred = [labels[i] for i in np.argsort(sims)[:, -1]]

            # baselines
            majority = rand = None
            if do_baselines:
                majority, rand = baselines(train, test, key=args.key)

            for idx in range(len(test)):
                base = {'lemma': test.iloc[idx]['lemma'],
                        'year': test.iloc[idx]['year'],
                        'pos': test.iloc[idx]['pos'],
                        'true': test.iloc[idx][args.key]}

                results.append(
                    dict(model=model, 
                         pred=pred[idx], 
                         n_subtokens=test.iloc[idx]['n_subtokens'], 
                         **base))

                if do_baselines:
                    for b_pred, baseline in zip([majority, rand], ['majority', 'random']):
                        results.append(dict(model=baseline, pred=b_pred[idx], **base))
        do_baselines = False

    pd.DataFrame.from_dict(results).to_csv(
        os.path.join(args.output_prefix, 'wsd-results-{}.csv'.format(args.key)))
