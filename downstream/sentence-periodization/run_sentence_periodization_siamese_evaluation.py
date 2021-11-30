
import itertools
import os
import tqdm
import torch
import numpy as np
import pandas as pd
from sentence_transformers import CrossEncoder
from sklearn.model_selection import train_test_split


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


def get_scores(model, X, background_X, forward=True):
    output = []
    with torch.no_grad():
        for inp in tqdm.tqdm(X):
            output.append(model.predict(
                [([inp, sent] if forward else [sent, inp]) for sent in background_X]))
    return np.array(output)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelpath', required=True)
    parser.add_argument('--background')
    parser.add_argument('--dev')
    parser.add_argument('--only-dev', action='store_true')
    parser.add_argument('--backward', action='store_true')
    parser.add_argument('--span', type=int, default=25)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output-path', required=True)
    parser.add_argument('--output-prefix', default='./data/sentence_periodization')
    args = parser.parse_args()

    model = CrossEncoder(args.modelpath, device=args.device)
    if not os.path.isdir(args.output_prefix):
        os.makedirs(args.output_prefix)

    if not args.only_dev:
        background = pd.read_csv(args.background)
        background_X, background_y = zip(*background[['quote', 'year']].values)
        background_X, background_y = np.array(background_X), np.array(background_y)

        dev_X, dev_y, dev_y_orig = get_data(pd.read_csv(args.dev, sep="\t"), span=args.span)
        scores = get_scores(model, dev_X, background_X, forward=not args.backward)
        np.savez(
            os.path.join(args.output_prefix, args.output_path) + 
                '.span={}'.format(args.span) + ('.backward' if args.backward else ''),
            scores=scores, background_y=background_y, dev_y=dev_y, dev_y_orig=dev_y_orig)

    else:
        dev = pd.read_csv(args.dev, sep="\t")
        S1, S2, Y1, Y2 = zip(*dev[['S1', 'S2', 'Y1', 'Y2']].values)
        data = pd.DataFrame({'S': list(S1) + list(S2), 'Y': list(Y1) + list(Y2)})
        data['span'] = args.span * (data['Y'] // args.span)
        index1, index2 = [], []
        for _, g in data.groupby('span').sample(200, random_state=1001).groupby('span'):
            a, b = train_test_split(g.index, test_size=0.5, random_state=1001)
            index1.extend(a)
            index2.extend(b)
        S1, Y1 = zip(*data.iloc[index1][['S', 'Y']].values)
        S2, Y2 = zip(*data.iloc[index2][['S', 'Y']].values)
        inputs = [[s1, s2] for s1 in S1 for s2 in S2]
        print(len(S1), len(S2), len(inputs))

        with torch.no_grad():
            scores = model.predict(
                [([s1, s2] if not args.backward else [s2, s1]) for s1 in S1 for s2 in S2],
                show_progress_bar=True, batch_size=128)
            scores = np.array(scores)
            print(scores.shape)
            np.savez(
                os.path.join(args.output_prefix, args.output_path) + 
                    ('.backward' if args.backward else '') + '.dev', 
                scores=scores, Y1=np.array(Y1), Y2=np.array(Y2), 
                index1=np.array(index1), index2=np.array(index2))
        

# # Generate some plots
# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_pdf import PdfPages
# pp1 = PdfPages('bert.pdf')
# pp2 = PdfPages('macberth.pdf')
# idxs = np.random.permutation(np.arange(len(scores1)))[:50]
# for idx in tqdm.tqdm(idxs):
#     knee = KneeLocator(np.sort(background_y1), gt1[idx], curve='concave', interp_method='polynomial').knee
#     plt.plot(np.sort(background_y1), gt1[idx], color='black')
#     plt.vlines(x=knee, ymin=0, ymax=gt1[idx].max(), color='green')
#     plt.vlines(x=dev_y_orig[idx], ymin=0, ymax=gt1[idx].max(), color='black')
#     pp1.savefig()
#     plt.clf()

#     knee = KneeLocator(np.sort(background_y2), gt2[idx], curve='concave', interp_method='polynomial').knee
#     plt.plot(np.sort(background_y2), gt2[idx], color='black')
#     plt.vlines(x=knee, ymin=0, ymax=gt2[idx].max(), color='green')
#     plt.vlines(x=dev_y_orig[idx], ymin=0, ymax=gt2[idx].max(), color='black')
#     pp2.savefig()
#     plt.clf()
# pp1.close()
# pp2.close()

# data = pd.read_csv('./data/periodization/periodization.evaluation.csv', index_col=0)
# data = pd.melt(
#     data, id_vars=['iteration', 'n_per_bin', 'n_backgrouund', 'n_items'], 
#     var_name='model', value_name='MAE')
# import matplotlib.pyplot as plt
# import seaborn as sns
# sns.lineplot(x='n_backgrouund', y='MAE', hue='model', data=data)
# plt.show()
# sns.lineplot(x='n_backgrouund', y='n_items', data=data)
# plt.show()