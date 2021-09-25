
import tensorflow as tf
from scipy.linalg import norm
from tensorflow import keras
import tensorflow_addons as tfa
from tensorflow.keras import layers
import numpy as np

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
        for _ in range(ratio):
            candidate, candidate2, *_ = np.random.permutation(unique)
            if candidate == labels[i]:
                candidate = candidate2
            neg.append([i, np.random.choice(rows[candidate])])
            pos.append([i, np.random.choice(rows[labels[i]])])

    return pos, neg


def make_training_examples(pos, neg, embs):
    labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    X = np.concatenate([embs[pos], embs[neg]])
    X1, X2 = np.split(X, 2, axis=1)
    X1, X2 = np.squeeze(X1, 1), np.squeeze(X2, 1)
    return X1, X2, labels


def euclidean_distance(vects):
    x, y = vects
    sum_square = tf.math.reduce_sum(tf.math.square(x - y), axis=1, keepdims=True)
    return tf.math.sqrt(tf.math.maximum(sum_square, keras.backend.epsilon()))


def contrastive_loss(margin=1):
    def contrastive_loss_(y_true, y_pred):
        """Calculates the constrastive loss.

        Arguments:
            y_true: List of labels, each label is of type float32.
            y_pred: List of predictions of same length as of y_true,
                    each label is of type float32.

        Returns:
            A tensor containing constrastive loss as floating point value.
        """
        square_pred = tf.math.square(y_pred)
        margin_square = tf.math.square(tf.math.maximum(margin - (y_pred), 0))
        return tf.math.reduce_mean(
            (1 - y_true) * square_pred + (y_true) * margin_square)

    return contrastive_loss_


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

    # - model definition
    # -- encoder
    emb_dim = embs.shape[1]
    n_layers = 2
    input = layers.Input((emb_dim,))
    # x = layers.BatchNormalization()(input)
    x = layers.Dense(
        emb_dim / 2, 
        activation="tanh",
        kernel_regularizer=keras.regularizers.l1_l2(l1=1e-5, l2=1e-4),
        bias_regularizer=keras.regularizers.l2(1e-4)
    )(input)
    if n_layers >= 2:
        for _ in range(n_layers - 1):
            x = layers.Dense(
                emb_dim / 2, 
                activation="tanh",
                kernel_regularizer=keras.regularizers.l1_l2(l1=1e-5, l2=1e-4),
                bias_regularizer=keras.regularizers.l2(1e-4)
            )(x)
    encoder = keras.Model(input, x)
    # -- siamese
    input1, input2 = layers.Input((emb_dim,)), layers.Input((emb_dim,))
    m1, m2 = encoder(input1), encoder(input2)
    merge = layers.Lambda(euclidean_distance)([m1, m2])
    # output = layers.Dense(1, activation="sigmoid")(merge)
    model = keras.Model([input1, input2], outputs=[merge])
    model.compile(loss=tfa.losses.ContrastiveLoss(margin=1), optimizer='adam')

    pos, neg = sample_training_examples(lemma[index_train])
    train_X1, train_X2, train_labels = make_training_examples(pos, neg, embs)
    pos, neg = sample_training_examples(lemma[index_dev])
    dev_X1, dev_X2, dev_labels = make_training_examples(pos, neg, embs)

    for epoch in range(10):
        model.fit(x=[train_X1, train_X2], y=train_labels, batch_size=48,
            validation_data=([dev_X1, dev_X2], dev_labels), epochs=1, validation_freq=1)
        result_baseline = evaluate_embeddings(embs[index_dev], lemma[index_dev])
        result_trained = evaluate_embeddings(encoder(embs[index_dev]).numpy(), lemma[index_dev])
        for metric in ['ap', 'accuracy']:
            print("metric={}; baseline={:g}; trained={:g}".format(
                metric, result_baseline[metric].mean(), result_trained[metric].mean()))
        print("norm {:g}".format(np.linalg.norm(encoder.layers[1].weights[1].numpy())))
