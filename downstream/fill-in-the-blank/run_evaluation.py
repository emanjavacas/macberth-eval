

import tqdm
import numpy as np
import pandas as pd
import os
from transformers import AutoModelForMaskedLM, AutoTokenizer, pipeline


def get_predictions(quotes, keywords, pipe, batch_size, mask_token):
    lengths = np.array(list(map(len, quotes)))
    order = np.argsort(lengths)
    predictions = []
    for b in tqdm.tqdm(range(0, len(order), batch_size), total=len(order) // batch_size):
        idxs = order[b: b + batch_size]
        input = [quotes[i].replace(keywords[i], mask_token, 1) for i in idxs]
        outputs = pipe(input)
        for output, keyword in zip(outputs, keywords[idxs]):
            result = [(rank, out) for rank, out in enumerate(output) if out['token_str'] == keyword]
            if result:
                (rank, out), *_ = result
                predictions.append({'rank': rank + 1, 'score': out['score'], 'keyword': keyword})
            else:
                predictions.append({'rank': -1, 'score': None, 'keyword': keyword})
    # reverse sorted order to original order
    inverse = np.arange(len(order))[np.argsort(order)]
    return pd.DataFrame.from_dict([predictions[i] for i in inverse])


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', required=True,
        help="""Path to input file. Same one as passed to generate_nn_embeddings.py. 
            Output of create_nn_embedding_data.py. It should be a csv file including the 
                following fields:
            - quote: text exemplifying the word
            - keyword: actual word being exemplified (as it appears in the quote)""")
    # ./data/nn/nn-data.csv
    parser.add_argument('--models', required=True, nargs='+')
    parser.add_argument('--output-prefix', default='./data/nn/')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    data = pd.read_csv(args.input_path)
    # has keyword
    data['has_keyword'] = np.array([(k in q) for k, q in zip(data['keyword'], data['quote'])])
    data = data[data['has_keyword']]

    print("original vocabulary: ", len(set(data['keyword'])))
    vocab = set(data['keyword'])
    for path in args.models:
        vocab = vocab.intersection(AutoTokenizer.from_pretrained(path).vocab)
    print("intersection: ", len(vocab))

    data = data[data['keyword'].isin(vocab)]
    quotes, keywords = np.array(data['quote']), np.array(data['keyword'])

    if not os.path.isdir(args.output_prefix):
        os.makedirs(args.output_prefix)

    for path in args.models:
        model = AutoModelForMaskedLM.from_pretrained(path)
        tokenizer = AutoTokenizer.from_pretrained(path)
        pipe = pipeline('fill-mask', model=model, tokenizer=tokenizer, top_k=100, 
            device=-1 if args.device == 'cpu' else 0)
        results = get_predictions(quotes, keywords, pipe, 200, tokenizer.mask_token)
        results.index = data.index
        results = pd.DataFrame.from_dict(results)
        results.index = data.index
        results.to_csv(os.path.join(args.output_prefix, os.path.basename(path)) + '.csv')