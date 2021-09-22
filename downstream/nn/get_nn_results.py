
import pandas as pd
import numpy as np

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument()


    result = []
    for path in ['./embs.macberth.npy.{}.csv', './embs.uncased_L-12_H-768_A-12.npy.{}.csv', 'random.{}.csv']:
        for key in ['senseId', 'lemma']:
            row = {'path': path, 'key': key}
            data = pd.read_csv(path.format(key))
            macro = np.mean(data.groupby(key)['accuracy'].mean())
            micro = data['accuracy'].mean()
            row['micro-acc'] = micro
            row['macro-acc'] = macro
            macro = np.mean(data.groupby(key)['ap'].mean())
            micro = data['ap'].mean()
            row['micro-ap'] = micro
            row['macro-ap'] = macro
            result.append(row)
    pd.DataFrame.from_dict(result).to_csv('nn-results.csv')