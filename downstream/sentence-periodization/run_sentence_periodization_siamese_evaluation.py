
import os
import tqdm
import torch
import numpy as np
import pandas as pd
from sentence_transformers import CrossEncoder


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
    parser.add_argument('--modelpath')
    parser.add_argument('--background', required=True)
    parser.add_argument('--dev')
    parser.add_argument('--backwward', action='store_true')
    parser.add_argument('--span', type=int, default=25)
    parser.add_argument('--n_per_bin', type=int, default=20)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output-path', required=True)
    parser.add_argument('--output-prefix', default='./data/sentence_periodization')
    parser.add_argument('--evaluate', action='store_true')
    args = parser.parse_args()

    model = CrossEncoder(args.modelpath, device=args.device)
    if not os.path.isdir(args.output_prefix):
        os.makedirs(args.output_prefix)

    background = pd.read_csv(args.background)
    background_X, background_y = zip(*background[['quote', 'year']].values)
    background_X, background_y = np.array(background_X), np.array(background_y)

    dev_X, dev_y, dev_y_orig = get_data(pd.read_csv(args.dev, sep="\t"), span=args.span)
    scores = get_scores(model, dev_X, background_X, forward=not args.backward)
    np.savez(
        os.path.join(args.output_prefix, args.output_path) + 
            '.span={}+n_per_bin={}'.format(args.span, args.n_per_bin) + 
            ('.backward' if args.backward else ''),
        scores=scores, background_y=background_y, dev_y=dev_y, dev_y_orig=dev_y_orig)
    

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