
import pandas as pd
from sklearn.model_selection import train_test_split
from spacy.lang.en import English

nlp = English()


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


def find_keyword(q, k):
    k_toks = nlp.tokenizer(k)
    q_toks = nlp.tokenizer(q)

    # multiword keywords
    if len(k_toks) > 1:
        start, *rest = list(map(str, k_toks))
        q_toks_str = list(map(str, q_toks))
        found = False
        acc = 0
        while not found:
            if start not in q_toks_str:
                return
            last = q_toks_str.index(start)
            if q_toks_str[last + 1: last + 1 + len(rest)] == rest:
                found = True
                last += acc
            else:
                q_toks_str = q_toks_str[last + 1:]
                acc += last + 1

        start = q_toks[last].idx
        end = q_toks[last + len(k_toks) - 1].idx + len(k_toks[-1])
        return start, end

    # uniword keywords
    else:
        for w in q_toks:
            if str(w) == k:
                start, end = w.idx, w.idx + len(w)
                return start, end


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='/home/manjavacasema/code/macberth-eval/oed-quotes-subset.tsv')
    args = parser.parse_args()

    df_source = pd.read_csv(args.input, sep='\t')
    # find keywords
    index = []
    spans = []
    for idx, row in df_source.iterrows():
        try:
            start, end = find_keyword(row['quote'], row['keyword'])
            spans.append((start, end))
            index.append(idx)
        except:
            continue
    df_source = df_source.iloc[index].reset_index()
    start, end = zip(*spans)
    df_source['start'] = start
    df_source['end'] = end

    # add pos
    df_source['pos'] = df_source['word'].apply(get_pos)
    depths = set(df_source['numbering'].apply(lambda row: len(row.rstrip('.').split('.'))))
    for depth in depths:
        df_source['depth-{}'.format(depth)] = df_source['numbering'].apply(
            lambda row: '.'.join(row.rstrip('.').split('.')[:depth]))

    # min count
    counts = df_source['lemma'].value_counts()
    targets = counts[counts >= 50].index

    # create splits
    for key in depths:
        key = 'depth-{}'.format(key)
        splits = []
        for lemma, subset in df_source[df_source['lemma'].isin(targets)].groupby('lemma'):
            # drop lemmas not in target frequency
            if lemma not in targets:
                continue
            # drop senses where we can't stratify
            senses = subset[key].value_counts()
            subset = subset[subset[key].isin(senses[senses >= 2].index)]
            if len(subset) < 2:
                continue
            # drop lemmas with only one sense
            senses = subset[key].value_counts()
            if len(senses) == 1:
                continue
            train, test = train_test_split(subset.index, stratify=subset[key], test_size=0.5)
            assert set(df_source.iloc[test][key]).difference(
                set(df_source.iloc[train][key])) == set()
            splits.append((train, test))

        *path, ext = args.input.split('.')
        pd.concat([df_source.iloc[t] for t, _ in splits]).to_csv(
            '.'.join(path) + '-' + key + '-train.csv', index=None)
        pd.concat([df_source.iloc[t] for _, t in splits]).to_csv(
            '.'.join(path) + '-' + key + '-test.csv', index=None)

