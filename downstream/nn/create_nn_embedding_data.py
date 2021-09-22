
import os
import pandas as pd


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', required=True, # ~/Leiden/Datasets/OED/data/oed-quotes-subset.tsv
        help="""Path to input file. It should be a csv file including the following fields:
        - lemma: original lemma that the quote is exemplifying
        - year: int, date of the quote
        - nWords: number of words in the quote
        - quote: the actual text""")
    parser.add_argument('--sep', default='\t')
    parser.add_argument('--output-prefix', default='./data/nn/')
    parser.add_argument('--min-count', type=int, default=200)
    args = parser.parse_args()

    quotes = pd.read_csv(args.input_file, sep=args.sep)
    if not os.path.isdir(args.output_prefix):
        os.makedirs(args.output_prefix)

    # TODO: this is all OED-specific filtering
    # - drop items where the word is not in use
    quotes = quotes[quotes.apply(lambda row: '†' not in row['word'], axis=1)]

    # - filter nouns, verbs and adjectives
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

    quotes['pos'] = quotes['word'].apply(get_pos)
    quotes['pos'].value_counts()
    quotes = quotes[quotes['pos'].isin(['n', 'adj', 'v'])]

    # select words
    counts = quotes['lemma'].value_counts()
    targets = counts[counts>args.min_count]
    # filter out function words
    targets = targets[~targets.index.isin(['the', 'much'])]

    quotes[quotes['lemma'].isin(targets.index)][['lemma', 'senseId', 'quote', 'keyword', 'year', 'nWords']].to_csv(
        os.path.join(args.output_prefix, 'nn-data.csv'), index=False)

