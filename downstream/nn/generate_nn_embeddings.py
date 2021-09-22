
import os
import collections

import tqdm

import pandas as pd
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer


def subwords_to_token_ids(ids, tokenizer, prefix='##'):
    # for ids, k in tqdm.tqdm(zip(tokens['input_ids'], keyword)):
    #     mapping = subwords_to_token_ids(ids, tokenizer)
    #     subwords = tokenizer.convert_ids_to_tokens(ids)
    #     for idxs in mapping[k]:
    #         out = ''.join(subwords[i].lstrip('##') for i in idxs)
    #         assert out == k, (out, k)
    output = collections.defaultdict(list)
    special = set(tokenizer.special_tokens_map.values())
    subwords = tokenizer.convert_ids_to_tokens(ids)
    ids, word = [], ''
    for idx, subword in enumerate(subwords):
        if subword in special:
            continue
        if subword.startswith(prefix):
            word += subword[len(prefix):]
            ids.append(idx)
        else:
            if word:
                output[word].append(ids)
            ids, word = [idx], subword
    if word:
        output[word].append(ids)
    return output


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', required=True,
        help="""Path to input file. Same one as passed to generate_nn_embeddings.py. 
            Output of create_nn_embedding_data.py. It should be a csv file including the following fields:
            - quote: text exemplifying the word
            - keyword: actual word being exemplified (as it appears in the quote)""")
    parser.add_argument('--model', required=True)
    parser.add_argument('--output-prefix', default='./data/nn/')
    parser.add_argument('--output-path', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--bsize', type=int, default=248)
    args = parser.parse_args()

    m = AutoModel.from_pretrained(args.model)
    m.to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    quotes = pd.read_csv(args.input_path)
    if not os.path.isdir(args.output_prefix):
        os.makedirs(args.output_prefix)
    # quotes = quotes.iloc[:10]

    text, keyword = list(quotes['quote']), list(quotes['keyword'])

    # tokenize sentences
    tokens = tokenizer(list(quotes['quote']), return_tensors='pt', padding=True)

    embs = np.zeros((len(text), m.config.hidden_size))
    index = []

    total = len(tokens['input_ids']) // args.bsize
    for i in tqdm.tqdm(range(0, len(tokens['input_ids']), args.bsize), total=total):
        with torch.no_grad():
            output = m(**{k: v[i:i + args.bsize].to(args.device) for k, v in tokens.items()}, 
                output_hidden_states=True)
            # pick last hidden layer
            output = output['hidden_states'][-1]
            for b_id, j in zip(range(0, len(output)), range(i, min(i + args.bsize, len(keyword)))):
                # target activations
                mapping = subwords_to_token_ids(tokens['input_ids'][j], tokenizer)
                if keyword[j].lower() not in mapping:
                    continue
                target = mapping[keyword[j].lower()]
                # pick first occurence of keyword and first subtoken
                target = target[0][0]
                embs[j] = output[b_id, target].cpu().numpy()
                # register id
                index.append(j)

    np.save(os.path.join(args.output_prefix, args.output_path), {'embs': embs, 'index': index})
