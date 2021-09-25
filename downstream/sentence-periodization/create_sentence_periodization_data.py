
import random
import os

random.seed(1001)
import itertools
import numpy as np
import pandas as pd


def export_pairs(df, size, output_path):
    # don't generate pairs from the same keyword
    words = df['lemma'].unique()
    pairs = list(itertools.combinations(words, 2))
    random.shuffle(pairs)
    n = 0
    with open(output_path, 'w+') as f:
        f.write('S1\tS2\tY1\tY2\n')
        while pairs and n < size:
            if n % 500 == 0:
                print(n)
            a, b = pairs.pop()
            subset_a, subset_b = df[df['lemma'] == a], df[df['lemma'] == b]
            a = subset_a.iloc[random.randint(0, len(subset_a) - 1)]
            b = subset_b.iloc[random.randint(0, len(subset_b) - 1)]
            f.write('{}\t{}\t{}\t{}\n'.format(
                a['quote'], b['quote'], str(int(a['year'])), str(int(b['year']))))
            n += 1


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', required=True, # ~/Leiden/Datasets/OED/data/oed-quotes.tsv
        help="""Path to input file. It should be a csv file including the following fields:
        - lemma: original lemma that the quote is exemplifying
        - year: int, date of the quote
        - nWords: number of words in the quote
        - quote: the actual text""")
    parser.add_argument('--sep', default='\t')
    parser.add_argument('--output-prefix', default='./data/sentence-periodization/')
    parser.add_argument('--min-words', type=int, default=5)
    parser.add_argument('--min-year', type=int, default=1450)
    parser.add_argument('--max-year', type=int, default=1950)

    args = parser.parse_args()

    if not os.path.isdir(args.output_prefix):
        os.makedirs(args.output_prefix)

    data = pd.read_csv(args.input_file, sep=args.sep) # './data/oed-quotes.tsv'
    # drop quotes with less than 5 words
    data = data[data['nWords']>=args.min_words]
    # limit to the target period
    data = data[(data['year'] > args.min_year) & (data['year'] < args.max_year)]

    # split training and test sets
    train, test, dev = np.split(
        data.sample(frac=1, random_state=42), 
        [int(.9*len(data)), int(.95*len(data))])

    # background data to produce classifications after training
    train, background = np.split(train, [int(.75*len(train))])
    background['decade'] = background['year'].apply(lambda year: 10 * (year // 10))
    background = background.groupby('decade').apply(lambda g: g.sample(n=20))
    background.to_csv(os.path.join(args.output_prefix, 'periodization.background.csv'), index=False)

    export_pairs(train, 100000, os.path.join(args.output_prefix, 'periodization.train.tsv'))
    export_pairs(test, 5000, os.path.join(args.output_prefix, 'periodization.test.tsv'))
    export_pairs(dev, 5000, os.path.join(args.output_prefix, 'periodization.dev.tsv'))