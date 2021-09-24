
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.linalg import norm
import pandas as pd
import faiss
from sklearn.model_selection import train_test_split

try:
    from downstream.nn.run_nn_evaluation import get_result
except:
    from run_nn_evaluation import get_result


def evaluate_embeddings(embs, labels, top_k=500):
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    output = index.search(embs, 500)
    _, neighbors = output
    return get_result(neighbors, labels, top_k=top_k)


def sample_training_examples(labels, ratio=3):
    unique = np.unique(labels)
    rows = {}
    for label in unique:
        rows[label] = np.where(labels == label)[0]

    pos, neg = [], []
    for i in range(len(labels)):
        pos.append([i, np.random.choice(rows[labels[i]])])
        for _ in range(ratio):
            candidate, candidate2, *_ = np.random.permutation(unique)
            if candidate == labels[i]:
                candidate = candidate2
            neg.append([i, np.random.choice(rows[candidate])])

    return pos, neg


def make_training_examples(pos, neg, embs):
    labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    X = np.concatenate([embs[pos], embs[neg]])
    X1, X2 = np.split(X, 2, axis=1)
    X1, X2 = np.squeeze(X1, 1), np.squeeze(X2, 1)
    return X1, X2, labels


class Encoder(nn.Module):
    def __init__(self, dim, n_layers=2, dropout=0.0):
        self.dropout = dropout
        super(Encoder, self).__init__()
        layers = []
        for layer_num in range(n_layers):
            layer = nn.Linear(dim, dim)
            self.add_module('layer_{}'.format(layer_num), layer)
            layers.append(layer)
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


def contrastive_loss(emb1, emb2, target, margin=1, size_average=True):
    distances = (emb2 - emb1).pow(2).sum(1)  # squared distances
    losses = 0.5 * (
        target.float() * distances +
        (1 + -1 * target).float() * F.relu(margin - (distances + 1e-9).sqrt()
    ).pow(2))
    return losses.mean() if size_average else losses.sum()


def batch_to_tensor(*batch, device='cpu'):
    return tuple(torch.tensor(b).to(device) for b in batch)


def evaluate_encoder(encoder, X1, X2, labels, margin, batch_size, device='cpu'):
    n_batches = loss = 0
    for b in range(0, len(X1), batch_size):
        X1_b, X2_b, labels_b = batch_to_tensor(
            X1[b:b+batch_size], X2[b:b+batch_size], labels[b:b+batch_size], 
            device=device)
        loss += contrastive_loss(
            encoder(X1_b), encoder(X2_b), labels_b, margin=margin).item()
        n_batches += 1
    return loss / n_batches


def train_epoch(encoder, X1, X2, labels,
        dev_X1, dev_X2, dev_labels,
        margin, opt, 
        batch_size=48, batch_report=100, device='cpu'):

    epoch_loss = n_batches = 0.0
    encoder.to(device)
    encoder.train()
    
    perm = np.random.permutation(np.arange(len(X1)))
    for b in range(0, len(perm), batch_size):
        n_batches += 1
        ids = perm[b:b+batch_size]
        X1_b, X2_b, labels_b = batch_to_tensor(
            X1[ids], X2[ids], labels[ids], device=device)

        opt.zero_grad()
        loss = contrastive_loss(encoder(X1_b), encoder(X2_b), labels_b, margin=margin)
        loss.backward()
        opt.step()

        epoch_loss += loss.item()

        if n_batches > 0 and n_batches % batch_report == 0:
            encoder.eval()
            with torch.no_grad():
                dev_loss = evaluate_encoder(
                    encoder, dev_X1, dev_X2, dev_labels, margin, batch_size, device=device)
            print("items: {}: loss: {:g}, dev-loss: {:g}".format(
                b, epoch_loss / n_batches, dev_loss))
            epoch_loss = n_batches = 0
            encoder.train()

    return


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', required=True)
    # './data/nn/nn-data.csv'
    parser.add_argument('--embeddings-path', required=True)
    # './data/nn/ckpt-1000000.embs.npy'
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--margin', default=1.0, type=float)
    args = parser.parse_args()

    embs = np.load(args.embeddings_path, allow_pickle=True)
    embs, index = embs.item()['embs'], embs.item()['index']
    embs = embs[index].astype(np.float32)
    embs = embs / norm(embs, axis=1)[:, None]
    quotes = pd.read_csv(args.input_path).iloc[index]
    quotes['row'] = np.arange(len(quotes))
    # subsample
    quotes = quotes.groupby('lemma').sample(n=50, random_state=100)
    embs = embs[quotes['row']]
    lemma = quotes['lemma'].to_numpy()
    # split train and test (unseen words alltogether)
    train, heldout = train_test_split(np.unique(lemma), test_size=0.1)
    index_train = np.where(np.isin(lemma, train))[0]
    index_train, index_dev = train_test_split(index_train, test_size=0.1)
    index_heldout = np.where(np.isin(lemma, heldout))[0]
    pos, neg = sample_training_examples(lemma[index_train])
    train_X1, train_X2, train_labels = make_training_examples(pos, neg, embs)
    pos, neg = sample_training_examples(lemma[index_dev])
    dev_X1, dev_X2, dev_labels = make_training_examples(pos, neg, embs)

    encoder = Encoder(embs.shape[1], dropout=0.25)
    lr, weight_decay = 1e-2, 1e-2
    opt = torch.optim.RMSprop(
        encoder.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(args.epochs):
        train_epoch(encoder, 
            train_X1, train_X2, train_labels, 
            dev_X1, dev_X2, dev_labels, 
            args.margin, opt,
            batch_size=args.batch_size, device=args.device)
        encoder.eval()
        with torch.no_grad():
            inp, = batch_to_tensor(embs[index_dev])
            inp = encoder(inp).cpu().numpy()
        result = evaluate_embeddings(inp, lemma[index_dev])
        for metric in ['ap', 'accuracy']:
            print("metric={}; result={:g}".format(
                metric, result[metric].mean()))


