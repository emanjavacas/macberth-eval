

import os
import tqdm
import random

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

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


def make_pairs(g, key, nneg=1, npos=1):
    counts = g[key].value_counts()
    groups = set(counts.index)
    for _, row in g.iterrows():
        if counts[row[key]] == 1:
            # no positive examples
            continue
        # sample negative sense
        other_sense = random.choice(list(groups.difference(set([row[key]]))))
        neg = g[g[key]==other_sense].sample(
            n=min(counts[other_sense], nneg), random_state=seed)
        for _, example in neg.iterrows():
            yield row['quoteId'], example['quoteId'], 0
        # sample positive
        rest = g[g.index != row.name]
        pos = rest[rest[key]==row[key]].sample(
            n=min(counts[row[key]] - 1, npos), random_state=seed)
        for _, example in pos.iterrows():
            yield row['quoteId'], example['quoteId'], 1


def get_df(df, depth):
    # skip lemmas with single sense
    gs = df.groupby(['lemma'], as_index=False)
    key = 'depth-{}'.format(depth)
    lemmas = [lemma for lemma, g in gs if len(g.groupby(key)) > 1]
    subset = df[df['lemma'].isin(lemmas)]

    data = []
    for lemma, g in tqdm.tqdm(subset.groupby('lemma')):
        for q1, q2, label in make_pairs(g, key):
            data.append({'q1': q1, 'q2': q2, 'label': label, 'lemma': lemma})
    # dedup
    data = [dict(id=tuple(sorted([row['q1'], row['q2']])), **row) for row in data]
    data = pd.DataFrame.from_dict(data)
    data = data[data['id'].isin((data['id'].value_counts() == 1).index)]

    return data


def export(data, df, outputpath):
    id2qid = dict(df[df['quoteId'].isin(data['q1'].value_counts().index)]['quoteId'])
    qid2id = {v: k for k, v in id2qid.items()}
    d1 = df.loc[[qid2id[qid] for qid in data['q1']]][['quote', 'keyword', 'year']]
    id2qid = dict(df[df['quoteId'].isin(data['q2'].value_counts().index)]['quoteId'])
    qid2id = {v: k for k, v in id2qid.items()}
    d2 = df.loc[[qid2id[qid] for qid in data['q2']]][['quote', 'keyword', 'year']]
    d1['id'] = np.arange(len(d1))
    d2['id'] = np.arange(len(d2))
    merge = pd.merge(d1, d2, on='id', suffixes=('_1', '_2'))
    merge['label'] = data['label']
    merge['lemma'] = data['lemma']
    merge.drop('id', axis=1, inplace=True)
    merge.to_csv(outputpath, index=False)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', required=True) # oed-quotes-subset.tsv
    parser.add_argument('--output-prefix', default='./data/word-in-context')
    parser.add_argument('--min-count', type=int, default=100)
    parser.add_argument('--min-words', type=int, default=5)
    args = parser.parse_args()

    if not os.path.isdir(args.output_prefix):
        os.makedirs(args.output_prefix)

    # df = pd.read_csv('../../Leiden/Datasets/OED/data/oed-quotes-subset.tsv', sep='\t').iloc[index]
    df = pd.read_csv(args.input_path, sep='\t')
    # drop based on quote length
    # df = df[df['nWords'] > 5]
    df = df[df['nWords'] > args.min_words]
    # drop based on frequency
    counts = df['lemma'].value_counts()
    # targets = counts[counts >= 100]
    targets = counts[counts >= args.min_count]
    df = df[df['lemma'].isin(targets.index)]
    # filter nouns, verbs and adjectives
    df['pos'] = df['word'].apply(get_pos)
    df = df[df['pos'].isin(['n', 'adj', 'v'])]

    # get available depths
    depths = set(df['numbering'].apply(lambda row: len(row.rstrip('.').split('.'))))
    depths = [1]

    # split heldout lemmas
    _, heldout = train_test_split(
        df['lemma'].value_counts().index, 
        test_size=0.1, random_state=seed)
    df_heldout = df[df['lemma'].isin(heldout)]
    df = df[~df['lemma'].isin(heldout)]

    for depth in depths:
        df['depth-{}'.format(depth)] = df['numbering'].apply(
            lambda row: '.'.join(row.rstrip('.').split('.')[:depth]))
        df_heldout['depth-{}'.format(depth)] = df_heldout['numbering'].apply(
            lambda row: '.'.join(row.rstrip('.').split('.')[:depth]))

        data = get_df(df, depth)
        export(data, df, 
            os.path.join(args.output_prefix, 'depth-{}.csv'.format(depth)))
        data_heldout = get_df(df_heldout, depth)
        export(data_heldout, df_heldout, 
            os.path.join(args.output_prefix, 'heldout-depth-{}.csv'.format(depth)))
