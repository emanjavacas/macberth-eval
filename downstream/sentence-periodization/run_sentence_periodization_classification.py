
import os
import json
import numpy as np
from numpy.core.fromnumeric import argmax
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

from sklearn.metrics import accuracy_score
from spacecutter.models import OrdinalLogisticModel
from spacecutter.losses import cumulative_link_loss
from sentence_transformers.models import WeightedLayerPooling
from sentence_transformers import SentenceTransformer, models
from sentence_transformers.util import batch_to_device


class AttentionPooling(nn.Module):
    def __init__(self, word_embedding_dimension, use_layer_pooling=True,
                 featurewise_attention=False):
        super(AttentionPooling, self).__init__()

        self.config_keys = [
            'word_embedding_dimension',
            'featurewise_attention',
            'use_layer_pooling']

        self.featurewise_attention = featurewise_attention
        self.word_embedding_dimension = word_embedding_dimension

        self.layer_pooling = None
        if use_layer_pooling:
            self.layer_pooling = WeightedLayerPooling(word_embedding_dimension)
        if featurewise_attention:
            self.attention = nn.Linear(word_embedding_dimension * 2, word_embedding_dimension)
        else:
            self.attention = nn.Linear(word_embedding_dimension * 2, 1)

        self.pooling_output_dimension = word_embedding_dimension

    def __repr__(self):
        return "AttentionPooling({})".format(self.get_config_dict())

    def forward(self, features):
        # get input token embeddings
        first, *_, last = features['all_layer_embeddings']
        if self.layer_pooling is not None:
            inp = self.layer_pooling(features)['token_embeddings']
        else:
            inp = last
        weights = self.attention(torch.cat([first, inp], 2))
        # zero out masked tokens
        mask = (1 - features['attention_mask'].unsqueeze(-1)).bool()
        weights.masked_fill_(mask, -100000)
        weights = F.softmax(weights, 1)
        features.update({'sentence_embedding': (weights * inp).sum(1)})
        return features

    def get_sentence_embedding_dimension(self):
        return self.pooling_output_dimension

    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path):
        with open(os.path.join(output_path, 'config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

    @staticmethod
    def load(input_path):
        with open(os.path.join(input_path, 'config.json')) as fIn:
            config = json.load(fIn)

        return AttentionPooling(**config)


class FeatureExtractor(nn.Module):
    def __init__(self, path, device='cpu', 
                 use_layer_pooling=True, featurewise_attention=False, 
                 **kwargs):
        self.device = device

        super().__init__()
        model = models.Transformer(path, **kwargs)
        model.auto_model.config.output_hidden_states = True
        attention = AttentionPooling(model.get_word_embedding_dimension(), 
            use_layer_pooling=use_layer_pooling, featurewise_attention=featurewise_attention)
        self.transformer = SentenceTransformer(modules=[model, attention], device=device)
        self.output = nn.Linear(model.get_word_embedding_dimension(), 1)
        self.to(device)

    def forward(self, features):
        tokens = self.transformer.tokenize(features)
        features = self.transformer(batch_to_device(tokens, self.device))
        features = self.output(features['sentence_embedding'])
        return features


def evaluate_model(model, X, y, y_orig, mapping, batch_size):
    with torch.no_grad():
        loss = n_batches = 0
        preds, trues, orig = [], [], []
        for b in range(0, len(X), batch_size):
            n_batches += 1
            pred_y = model(X[b: b + batch_size])
            loss += cumulative_link_loss(
                pred_y, 
                torch.tensor(y[b: b + batch_size]).unsqueeze(1).to(pred_y.device)
            ).item()
            preds.extend(torch.argmax(pred_y, axis=1).tolist())
            trues.extend(y[b: b+batch_size])
            orig.extend(y_orig[b: b+batch_size])
            if b == 0:
                print(preds[:10])
                print(pred_y[:10])
                print(X[b: b+batch_size][:10])
                print(orig[:10])

        acc = accuracy_score(trues, preds)
        # transform preds to the point in between two consecutive spans
        inv_mapping = {idx: span for span, idx in mapping.items()}
        span = list(mapping.keys())[1] - list(mapping.keys())[0]
        preds = [(span / 2) + inv_mapping[pred] for pred in preds]
        mae = np.mean(np.abs(np.array(preds) - np.array(orig)))

    return {'loss': loss / n_batches, 'acc': acc, 'mae': mae}


def train_model(
        model, train_X, train_y, dev_X, dev_y, dev_y_orig, mapping,
        epochs=10, batch_size=32, dev_batch_size=None, 
        batch_report=1000, eval_report=20000,
        # ascension callback
        margin=0.0, min_val=-1.0e6):
    
    dev_batch_size = dev_batch_size or batch_size

    opt = torch.optim.Adam(model.parameters())
    for epoch in range(epochs):
        epoch_loss = n_batches = 0
        perm = np.random.permutation(np.arange(len(train_X)))
        for b in range(0, len(train_X), batch_size):
            n_batches += 1
            batch_X, batch_y = train_X[perm[b: b+batch_size]], train_y[perm[b: b+batch_size]]
            opt.zero_grad()
            pred_y = model(batch_X)
            batch_y = torch.tensor(batch_y).unsqueeze(1).to(pred_y.device)
            loss = cumulative_link_loss(pred_y, batch_y)
            loss.backward()
            opt.step()

            # # ascension callback: https://github.com/EthanRosenthal/spacecutter/blob/37a6f7367905b50e7886dc1ef2bfe1d63220347a/spacecutter/callbacks.py#L7
            cutpoints = model.link.cutpoints.data
            for i in range(cutpoints.shape[0] - 1):
                cutpoints[i].clamp_(min_val, cutpoints[i + 1] - margin)

            epoch_loss += loss.item()
            if b > 0 and b % batch_report == 0:
                print("batch: {}: loss: {:g}".format(b, epoch_loss / n_batches))
                epoch_loss = n_batches = 0

            if b > 0 and b % eval_report == 0:
                print("- epoch: {}".format(epoch + 1))
                model.eval()
                result = evaluate_model(model, dev_X, dev_y, dev_y_orig, mapping, dev_batch_size)
                model.train()
                for key, val in result.items():
                    print("  * {}={:g}".format(key, val))


def get_data(dataset, span=10):
    X, y, y_orig = [], [], []
    for _, row in dataset.iterrows():
        X.append(row['S1'])
        X.append(row['S2'])
        y.append(span * (row['Y1'] // span))
        y.append(span * (row['Y2'] // span))
        y_orig.append(row['Y1'])
        y_orig.append(row['Y2'])

    return np.array(X), np.array(y), np.array(y_orig)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', required=True)
    parser.add_argument('--dev', required=True)
    parser.add_argument('--modelpath', required=True)
    parser.add_argument('--span', type=int, default=10)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--epochs', default=2, type=int)
    parser.add_argument('--no_layer_pooling', action='store_true')
    parser.add_argument('--featurewise_attention', action='store_true')
    args = parser.parse_args()

    # test = pd.read_csv('./data/periodization/periodization.test.tsv', sep='\t')
    # dev = pd.read_csv('./data/periodization/periodization.dev.tsv', sep='\t')
    # train = pd.read_csv('./data/periodization/periodization.train.tsv', sep='\t')
    # path = './models/periodization-ckpt-1000000-2021-08-19_10-00-41/'
    # train_X, train_y, mapping = get_data(train)
    # dev_X, dev_y, _ = get_data(dev, mapping=mapping)

    train_X, train_y, _ = get_data(pd.read_csv(args.train, sep='\t'), span=args.span)
    dev_X, dev_y, dev_y_orig = get_data(pd.read_csv(args.dev, sep='\t'), span=args.span)

    mapping = {span: idx for idx, span in enumerate(set(train_y))}
    train_y = np.array([mapping[span] for span in train_y])
    dev_y = np.array([mapping[span] for span in dev_y])

    extractor = FeatureExtractor(
        args.modelpath, device=args.device,
        use_layer_pooling=not args.no_layer_pooling, 
        featurewise_attention=args.featurewise_attention)
    model = OrdinalLogisticModel(extractor, len(mapping))
    model.to(args.device)

    train_model(model, train_X, train_y, dev_X, dev_y, dev_y_orig, mapping, epochs=args.epochs)





