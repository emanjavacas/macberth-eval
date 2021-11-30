

import os
import collections
import tqdm
import numpy as np
import pandas as pd
from kneed import KneeLocator
from sklearn.metrics import accuracy_score


def classify(scores, background_y, threshold=None):
    knees = []
    if threshold is not None:
        gt = np.cumsum(scores[:, np.argsort(background_y)] > threshold, axis=1)
    else:
        gt = np.cumsum(scores[:, np.argsort(background_y)], axis=1)
    for row in tqdm.tqdm(np.arange(len(scores))):
        knee = KneeLocator(
            np.sort(background_y), gt[row], 
            curve='concave', interp_method='polynomial'
        ).knee
        knee = knee if knee is not None else np.nan
        knees.append(knee)
    return np.array(knees)


def find_threshold(true, scores, min_th=0, max_th=1):
    accs, ths = [], []
    for th in tqdm.tqdm(np.linspace(min_th, max_th)):
        pred = scores > th
        accs.append(accuracy_score(true.reshape(-1), pred.reshape(-1)))
        ths.append(th)
    return np.array(accs), np.array(ths)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-evals', nargs='+')
    # parser.add_argument('--model1-eval') # './bert-periodization-scores-span=25.npz'
    # parser.add_argument('--model2-eval') # './macberth-periodization-scores-span=25.npz'
    parser.add_argument('--background',
        default='./data/sentence-periodization/periodization.background.csv')
    parser.add_argument('--output-prefix', default='./data/sentence-periodization')
    parser.add_argument('--output-path', required=True)
    parser.add_argument('--threshold', action='store_true')
    args = parser.parse_args()

    if not os.path.isdir(args.output_prefix):
        os.makedirs(args.output_prefix)

    background = pd.read_csv(args.background)
    span = 50
    background['span'] = span * (background['year'] // span)

    evals = {}
    thresholds = {}
    background_y = dev_y_orig = None # these are the same for all models
    for path in args.model_evals:
        model, *_ = os.path.basename(path).split('.')
        _, *model, _, _, _, _, _ = model.split('-')
        model = '-'.join(model)
        data = np.load(path)
        if background_y is None:
            background_y, dev_y_orig = data['background_y'], data['dev_y_orig']
        scores = data['scores']
        evals[model] = scores

        true = dev_y_orig[:, None] > background_y[None, :]
        accs, ths = find_threshold(true, scores)
        thresholds[model] = ths[np.argmax(accs)]
        print(model, ths[np.argmax(accs)], np.max(accs))

    data = collections.defaultdict(list)
    for n_per_bin in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        for iteration in range(10):
            sample = background.groupby('span').sample(n_per_bin, replace=False).index
            knees = {m: classify(scores[:, sample], background_y[sample]) 
                for m, scores in evals.items()}
            knees_th = {m: classify(scores[:, sample], background_y[sample], threshold=thresholds[m]) 
                for m, scores in evals.items()}
            nans = np.where(np.isnan(knee) for knee in knees.values())
            nans = np.sort(np.unique(np.concatenate(nans)))
            mask = np.ones(len(scores))
            mask[nans] = 0
            mask = mask.astype(np.bool)
            data['iteration'].append(iteration)
            data['n_per_bin'].append(n_per_bin)
            data['n_backgrouund'].append(len(sample))
            data['n_items'].append(len(mask[mask]))
            for m, knee in knees.items():
                data[m].append(np.nanmean(np.abs(knee[mask] - dev_y_orig[mask])))
            data['nans'] = len(nans)

    pd.DataFrame.from_dict(data).to_csv(os.path.join(args.output_prefix, args.output_path))
