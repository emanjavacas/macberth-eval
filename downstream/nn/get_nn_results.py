
import os
import pandas as pd
import numpy as np


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--paths', nargs='+')
    parser.add_argument('--output-prefix', default='./data/nn')
    parser.add_argument('--output-path', default='nn-results.csv')
    args = parser.parse_args()

    result = []
    for path in args.paths:
        *_, key, _ = os.path.basename(path).split('.')
        row = {'path': path, 'key': key}
        data = pd.read_csv(path)
        macro = np.mean(data.groupby(key)['accuracy'].mean())
        micro = data['accuracy'].mean()
        row['micro-acc'] = micro
        row['macro-acc'] = macro
        macro = np.mean(data.groupby(key)['ap'].mean())
        micro = data['ap'].mean()
        row['micro-ap'] = micro
        row['macro-ap'] = macro
        result.append(row)

    if not os.path.isdir(args.output_prefix):
        os.makedirs(args.output_prefix)
    pd.DataFrame.from_dict(result).to_csv(
        os.path.join(args.output_prefix, args.output_path))