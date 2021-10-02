
import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

import torch
from torch import nn
import torch.nn.functional as F
from torch import optim

from create_data import get_df, get_pos, seed


def data_to_batches(data, df, embs, batch_size, device):
    qid, row = zip(*df[['quoteId', 'row']].to_numpy())
    qid2row = dict(zip(qid, row))
    perm = np.random.permutation(np.arange(len(data)))
    for b in range(0, len(perm), batch_size):
        ids = perm[b: b+batch_size]
        batch = data.iloc[ids]
        rows1 = [qid2row[qid] for qid in batch['q1']]
        rows2 = [qid2row[qid] for qid in batch['q1']]
        emb1, emb2 = torch.tensor(embs[rows1]).to(device), torch.tensor(embs[rows2]).to(device)
        label = torch.tensor(list(batch['label'])).to(device)
        lemma = list(batch['lemma'])
        yield (emb1, emb2), label, lemma


# class Model(nn.Module):
#     def __init__(self, input_dim, lemmas):
#         self.lemma2id = {lemma: id for id, lemma in enumerate(lemmas)}
#         super().__init__()
#         self.weights = nn.Parameter(torch.FloatTensor(len(lemmas), input_dim * 2))
#         nn.init.xavier_uniform_(self.weights)

#     def device(self):
#         return next(self.parameters()).device

#     def forward(self, emb1, emb2, lemmas):
#         device = self.device()
#         inp = torch.cat([emb1, emb2], axis=1)
#         index = torch.tensor([self.lemma2id[l] for l in lemmas])
#         index = index.to(device)
#         weights = self.weights[index]
#         return torch.sigmoid(torch.sum(weights * inp, axis=1))


class Model(nn.Module):
    def __init__(self, input_dim, lemmas, dropout=0.25, layers=3):
        self.lemma2id = {lemma: id for id, lemma in enumerate(lemmas)}
        self.dropout = dropout
        super().__init__()
        self.lemma_embs = nn.Embedding(len(lemmas), input_dim * 2)
        # self.weights = nn.Linear(input_dim * 2, input_dim * 2)
        weights = []
        for _ in range(layers):
            weights.append(nn.Linear(input_dim * 2, input_dim * 2))
            weights.append(nn.Dropout(p=dropout))
        self.weights = nn.Sequential(*weights)

    def device(self):
        return next(self.parameters()).device

    def forward(self, emb1, emb2, lemmas):
        device = self.device()
        lemmas = torch.tensor([self.lemma2id[lemma] for lemma in lemmas]).to(device)
        weights = self.lemma_embs(lemmas)
        # weights = F.dropout(weights, p=self.dropout, training=self.training)
        inp = torch.cat([emb1, emb2], axis=1).float()
        scores = torch.sum(weights * inp, axis=1)
        return torch.sigmoid(scores)


def train_epoch(model, data, df, embs, batch_size, opt, device):
    tloss = n_batches = 0
    iterator = data_to_batches(data, df, embs, batch_size, device)
    for b, ((emb1, emb2), label, lemma) in tqdm.tqdm(enumerate(iterator)):
        # step
        opt.zero_grad()
        logits = model(emb1, emb2, lemma)
        loss = F.binary_cross_entropy(logits, label.float(), reduction="mean")
        loss.backward()
        opt.step()
        # report
        tloss += loss.item()
        n_batches += 1

        if b > 0 and b % 500 == 0:
            print("batch={}; loss={:g}".format(b, tloss / n_batches))
            tloss = n_batches = 0


def evaluate(model, data, df, embs, batch_size, device):
    tloss = n_batches = 0
    tprobs, tlabels = [], []
    with torch.no_grad():
        iterator = data_to_batches(data, df, embs, batch_size, device)
        for b, ((emb1, emb2), label, lemma) in tqdm.tqdm(enumerate(iterator)):
            logits = model(emb1, emb2, lemma)
            tloss += F.binary_cross_entropy(
                logits, label.float(), reduction="mean"
            ).item()
            n_batches += 1

            probs = torch.sigmoid(logits)
            tprobs.extend(probs.cpu().numpy())
            tlabels.extend(label.cpu().numpy())
    tprobs, tlabels = np.array(tprobs), np.array(tlabels)

    # find optimal threshold and report
    ths = np.linspace(0, 1, num=25)
    accs = [accuracy_score(tlabels, tprobs >= th) for th in ths]
    th = ths[np.argmax(accs)]
    acc = accuracy_score(tlabels, tprobs >= th)

    return {'acc': acc, 'th': th, 'loss': tloss / n_batches}


def train(model, train_data, val_data, df, embs, epochs, batch_size, device):
    opt = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(opt, 50, gamma=0.5, verbose=True)
    for epoch in range(epochs):
        model.train()
        train_epoch(model, train_data, df, embs, batch_size, opt, device)
        model.eval()
        results = evaluate(model, val_data, df, embs, batch_size, device)
        print('- epoch={}'.format(epoch + 1))
        for key, val in results.items():
            print('  validation - {}={:g}'.format(key, val))
        results = evaluate(model, train_data, df, embs, batch_size, device)
        for key, val in results.items():
            print('  train - {}={:g}'.format(key, val))
        scheduler.step()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', required=True) # oed-quotes-subset.tsv
    parser.add_argument('--embeddings-path', required=True, 
        help="Path to embeddings from generate_nn_embeddings.py")
    parser.add_argument('--min-count', type=int, default=200)
    parser.add_argument('--min-words', type=int, default=5)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch_size', type=int, default=48)
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()

    embs = np.load(args.embeddings_path, allow_pickle=True)
    index = embs.item()['index']
    embs = embs.item()['embs']

    df = pd.read_csv(args.input_path, sep='\t').iloc[index]
    df['row'] = np.arange(len(df))
    # drop based on quote length
    # df = df[df['nWords'] > 5]
    df = df[df['nWords'] > args.min_words]
    # drop based on frequency
    counts = df['lemma'].value_counts()
    # targets = counts[counts >= 100]
    targets = counts[counts >= args.min_count]
    df = df[df['lemma'].isin(targets.index)]
    # filter nouns, verbs and adjectives
    df['pos'] = df['word'].apply(get_pos)
    df = df[df['pos'].isin(['n', 'adj', 'v'])]

    # get available depths
    depths = set(df['numbering'].apply(lambda row: len(row.rstrip('.').split('.'))))
    depths = [1]

    for depth in depths:
        df['depth-{}'.format(depth)] = df['numbering'].apply(
            lambda row: '.'.join(row.rstrip('.').split('.')[:depth]))

        # df = df.iloc[:1000]
        data = get_df(df, depth)
        train_data, val_data = train_test_split(data, test_size=0.1, random_state=1001)
        print("- baselines: one={:g}; zero={:g}".format(
            accuracy_score(val_data['label'], np.ones(len(val_data))),
            accuracy_score(val_data['label'], np.zeros(len(val_data)))))
        # set up model and train
        model = Model(embs.shape[1], list(set(df['lemma'])))
        model.to(args.device)
        train(model, train_data, val_data, df, embs, args.epochs, args.batch_size, args.device)
