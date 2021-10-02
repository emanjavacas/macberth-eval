
import os
import tqdm
import torch
import numpy as np
import scipy
import pandas as pd

from sentence_transformers import CrossEncoder
from captum.attr import LayerIntegratedGradients
from captum.attr import visualization as viz


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--periodizer', required=True)
    parser.add_argument('--background',
        default='./data/sentence-periodization/periodization.background.csv')
    parser.add_argument('--output-prefix', default='./data/sentence-periodization')
    parser.add_argument('--output-path', required=True)
    parser.add_argument('--span', default=50, type=int)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    if not os.path.isdir(args.output_prefix):
        os.makedirs(args.output_prefix)

    periodizer = CrossEncoder(args.periodizer, device=args.device)
    periodizer.model.to(args.device)

    def construct_input(periodizer, a, b):
        input_example = [a, b]
        input_token = periodizer.smart_batching_collate_text_only([input_example])

        mask = torch.ones_like(input_token['input_ids'])
        mask = torch.masked_fill(
            mask, input_token['input_ids'] == periodizer.tokenizer.sep_token_id, 0)
        mask = torch.masked_fill(
            mask, input_token['input_ids'] == periodizer.tokenizer.cls_token_id, 0)
        ref_input = torch.masked_fill(
            input_token['input_ids'], mask.bool(), periodizer.tokenizer.pad_token_id)
        ref_token_type = torch.zeros_like(input_token['token_type_ids'])
        ref_token = {
            'input_ids': ref_input, 
            'token_type_ids': ref_token_type,
            # overwrite default behaviour of position_ids
            # position_ids = periodizer.model.bert.embeddings.position_ids[:, 0 : seq_length + 0]
            'position_ids': torch.zeros_like(input_token['input_ids'])}

        return input_token, ref_token

    def predict(inputs, token_type_ids, attention_mask, position_ids=None):
        output = periodizer.model(
            input_ids=inputs, token_type_ids=token_type_ids, 
            attention_mask=attention_mask, position_ids=position_ids)
        pred = periodizer.default_activation_function(output.logits)
        return pred

    background = pd.read_csv(args.background)
    background_X, background_y = zip(*background[['quote', 'year']].values)
    background_X, background_y = np.array(background_X), np.array(background_y)
    background['span'] = args.span * (background['year'] // args.span)
    background_sample = background.groupby('span').sample(
        50, replace=False, random_state=1001)
    sample_x, sample_y = [], []
    for _, g in background_sample.groupby('span'):
        x, y = np.split(g, 2)
        sample_x.append(x)
        sample_y.append(y)
    sample_x = pd.concat(sample_x)
    sample_y = pd.concat(sample_y)
    print(background_sample.head())
    print(len(sample_x), len(sample_y), len(sample_x) * len(sample_y))

    lig = LayerIntegratedGradients(predict, periodizer.model.bert.embeddings)
    vises, atts = [], []
    for a_idx in tqdm.tqdm(sample_x.index):
        for b_idx in sample_y.index:
            a, a_year = background_X[a_idx], background_y[a_idx]
            b, b_year = background_X[b_idx], background_y[b_idx]

            input_token, ref_token = construct_input(periodizer, a, b)
            # for key, val in input_token.items():
            #     print(key, val.device)

            attributions, delta = lig.attribute(
                inputs=input_token['input_ids'],
                baselines=ref_token['input_ids'],
                additional_forward_args=(input_token['token_type_ids'], input_token['attention_mask']),
                return_convergence_delta=True, n_steps=500, internal_batch_size=50)
            attributions = attributions.to('cpu').numpy()
            delta = delta.item()
            # summarise attributions
            attribution_summ = np.squeeze(attributions.sum(-1)) / scipy.linalg.norm(attributions)
            pred = predict(
                input_token['input_ids'], 
                input_token['token_type_ids'], 
                input_token['attention_mask'])
            pred = pred.squeeze(0).item()
            pred_vis = viz.VisualizationDataRecord(
                word_attributions=attribution_summ,
                pred_prob=pred,
                pred_class=delta,
                true_class=(a_year, b_year),
                attr_class=str(pred)[:6],
                attr_score=attribution_summ.sum(),
                raw_input=periodizer.tokenizer.convert_ids_to_tokens(input_token['input_ids'].squeeze(0)),
                convergence_score=delta)

            vises.append(pred_vis)
            atts.append(attributions)

    import pickle
    with open(os.path.join(args.output_prefix, args.output_path), 'wb') as f:
        pickle.dump({'viz': vises, 'attributions': atts}, f)

