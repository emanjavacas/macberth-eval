
import os
import tqdm
import faiss
import pandas as pd
import numpy as np
from scipy.linalg import norm


def get_result(neighbors, labels, top_k=500):
    rows = dict()
    for item in np.unique(labels):
        rows[item] = np.where(labels == item)[0]

    retrieved, nNeighbors, ap = [], [], []
    for i in tqdm.tqdm(range(len(labels))):
        targets = rows[labels[i]]
        # average precision
        index = np.isin(neighbors[i, 1:], targets)
        positions, = np.where(index)
        p = 0
        if len(positions) > 0:
            p = np.mean([(index[:p+1].sum() / (p + 1)) for p in positions])
        ap.append(p)
        # accuracy
        retrieved.append(len(np.intersect1d(neighbors[i, 1:], targets)))
        nNeighbors.append(len(targets))

    result = pd.DataFrame(
        {'nNeighbors': nNeighbors,
         'retrieved': retrieved,
         'ap': ap})
    result['accuracy'] = result['retrieved'] / np.clip(result['nNeighbors'], 0, top_k)

    return result


def lemma_random_baseline(quotes, top_k=500):
    rows = dict()
    for item in quotes['lemma'].unique():
        rows[item] = np.array(quotes[quotes['lemma']==item]['row'])

    retrieved, nNeighbors, ap = [], [], []
    for i in tqdm.tqdm(range(len(quotes))):
        targets = rows[quotes.iloc[i]['lemma']]
        # average precision
        rnd_index = np.random.permutation(np.arange(len(quotes)))[:top_k]
        index = np.isin(rnd_index, targets)
        positions, = np.where(index)
        p = 0
        if len(positions) > 0:
            p = np.mean([(index[:p+1].sum() / (p + 1)) for p in positions])
        ap.append(p)
        # accuracy
        retrieved.append(len(np.intersect1d(rnd_index, targets)))
        nNeighbors.append(len(targets))

    result = pd.DataFrame(
        {'nNeighbors': nNeighbors,
         'retrieved': retrieved,
         'ap': ap,
         'lemma': np.array(quotes['lemma'])})
    result['accuracy'] = result['retrieved'] / np.clip(result['nNeighbors'], 0, top_k)

    return result


def sense_random_baseline(quotes, sense_field='senseId', top_k=500):
    rows, rows_candidates = dict(), dict()
    for item in quotes['lemma'].unique():
        rows_candidates[item] = np.array(quotes[quotes['lemma']==item]['row'])
    for item in quotes[sense_field].unique():
        rows[item] = np.array(quotes[quotes[sense_field]==item]['row'])

    retrieved, nNeighbors, ap = [], [], []
    for i in tqdm.tqdm(range(len(quotes))):
        targets = rows[quotes.iloc[i][sense_field]]
        candidates = rows_candidates[quotes.iloc[i]['lemma']]
        assert np.alltrue(np.isin(targets, candidates))
        # average precision
        rnd_index = np.random.permutation(candidates)[:top_k]
        index = np.isin(rnd_index, targets)
        positions, = np.where(index)
        p = 0
        if len(positions) > 0:
            p = np.mean([(index[:p+1].sum() / (p + 1)) for p in positions])
        ap.append(p)
        # accuracy
        retrieved.append(len(np.intersect1d(rnd_index, targets)))
        nNeighbors.append(len(targets))

    result = pd.DataFrame(
        {'nNeighbors': nNeighbors,
         'retrieved': retrieved,
         'ap': ap,
         'senseId': np.array(quotes['senseId'])})
    result['accuracy'] = result['retrieved'] / np.clip(result['nNeighbors'], 0, top_k)

    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', required=True,
        help="""Path to input file. Same one as passed to generate_nn_embeddings.py. 
        Output of create_nn_embedding_data.py. It should be a csv file including the following fields:
        - lemma: lemma underlying the example embedding
        - senseId: sense corresponding to the word""")
    parser.add_argument('--embeddings-path', required=True, 
        help="Path to embeddings from generate_nn_embeddings.py")
    parser.add_argument('--sense_field', default='senseId')
    parser.add_argument('--output-prefix', default='./data/nn')
    parser.add_argument('--top_k', type=int, default=500)
    args = parser.parse_args()

    embs = np.load(args.embeddings_path, allow_pickle=True)
    index = embs.item()['index']
    quotes = pd.read_csv(args.input_path).iloc[index]
    quotes['row'] = np.arange(len(quotes))
    if not os.path.isdir(args.output_prefix):
        os.makedirs(args.output_prefix)

    embs = embs.item()['embs'][index].astype(np.float32)
    embs = embs / norm(embs, axis=1)[:, None]

    sense_field = args.sense_field
    # uniquely identify senses within each lemma
    quotes[sense_field] = quotes.apply(
        lambda row: row['lemma'] + '+' + row[sense_field], axis=1)

    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    output = index.search(embs, args.top_k)
    similarity, neighbors = output
    # find clustering metric according to: words, senses
    path = '.'.join(os.path.basename(args.embeddings_path).split('.')) + '.{}.csv'
    path = os.path.join(args.output_prefix, path)
    result = get_result(neighbors, quotes['lemma'].to_numpy(), top_k=args.top_k)
    result['lemma'] = quotes['lemma'].to_list()
    result.to_csv(path.format('lemma'))
    result = get_result(neighbors, quotes[sense_field].to_numpy(), top_k=args.top_k)
    result[sense_field] = quotes[sense_field].to_list()
    result.to_csv(path.format(sense_field))
    if not os.path.isfile(os.path.join(args.output_prefix, 'nn-random.{}.csv'.format(sense_field))):
        sense_random_baseline(quotes, sense_field=sense_field).to_csv(
            os.path.join(args.output_prefix, 'nn-random.{}.csv'.format(sense_field)))
    if not os.path.isfile(os.path.join(args.output_prefix, 'nn-random.lemma.csv')):
        lemma_random_baseline(quotes).to_csv(os.path.join(args.output_prefix, 'nn-random.lemma.csv'))
