
import os

import pandas as pd
from sklearn.model_selection import train_test_split

from sentence_transformers import CrossEncoder
from run_word_in_context_bert import get_training_examples


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--split-dev', action='store_true')
    parser.add_argument('--modelpaths', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    if args.split_dev:
        _, df = train_test_split(df, random_state=1001)
    data = df.apply(get_training_examples, axis=1)
    t1, t2, _ = zip(*data.tolist())

    for path in args.modelpaths:
        name = os.path.basename(path.rstrip('/'))
        name = name.replace('word-in-context-', '')
        m = CrossEncoder(path)
        df[name] = m.predict(list(zip(t1, t2)), show_progress_bar=True)

    df.to_csv(args.output)