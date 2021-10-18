
import json
import os
import glob
from sklearn.model_selection import train_test_split


def generate_examples(filepaths):
    sent_id = 0
    for path in filepaths:
        with open(path) as f:
            toks, poss = [], []
            for idx, line in enumerate(f):
                line = line.strip()
                if line.startswith('<') or line.startswith('{'):
                    continue
                if not line:
                    continue
                tok, pos = line.strip().split('/')
                if pos == 'ID':
                    yield {
                        "id": tok, 
                        "tokens": toks,
                        "pos_tags": poss,
                        "source": path}
                    sent_id += 1
                    toks, poss = [], []
                else:
                    toks.append(tok)
                    poss.append(pos)


ROOT = '/Users/manjavacas/Leiden/Datasets/'
# ROOT = '/home/manjavacasema/code/downstream/pos/'
_paths = glob.glob(os.path.join(ROOT, 'PENN-CORPORA/PPCEME-RELEASE-2/corpus/pos/*pos'))

# test files from https://arxiv.org/abs/1904.02817
_test = set([
    "alhatton-e3-h.pos", "armin-e2-p1.pos", "aungier-e3-h.pos",
    "aungier-e3-p2.pos", "authold-e2-h.pos", "bacon-e2-h.pos",
    "bacon-e2-p1.pos", "blundev-e2-h.pos", "boethco-e1-p1.pos",
    "boethco-e1-p2.pos", "boethpr-e3-p2.pos", "burnetcha-e3-p1.pos",
    "burnetroc-e3-h.pos", "burnetroc-e3-p1.pos", "chaplain-e1-p2.pos",
    "clowesobs-e2-p2.pos", "cromwell-e1-p1.pos", "cromwell-e1-p2.pos",
    "delapole-e1-p1.pos", "deloney-e2-p1.pos", "drummond-e3-p1.pos",
    "edmondes-e2-h.pos", "edmondes-e2-p1.pos", "edward-e1-h.pos",
    "edward-e1-p1.pos", "eliz-1590-e2-p2.pos", "elyot-e1-p1.pos",
    "eoxinden-1650-e3-p1.pos", "fabyan-e1-p1.pos", "fabyan-e1-p2.pos",
    "farquhar-e3-h.pos", "farquhar-e3-p1.pos", "fhatton-e3-h.pos",
    "fiennes-e3-p2.pos", "fisher-e1-h.pos", "forman-diary-e2-p2.pos",
    "fryer-e3-p1.pos", "gascoigne-1510-e1-p1.pos", "gawdy-e2-h.pos",
    "gawdy-e2-p1.pos", "gawdy-e2-p2.pos", "gifford-e2-p1.pos",
    "grey-e1-p1.pos", "harley-e2-h.pos", "harley-e2-p1.pos",
    "harleyedw-e2-p2.pos", "harman-e1-h.pos", "henry-1520-e1-h.pos",
    "hooker-b-e2-h.pos", "hoole-e3-p2.pos", "hoxinden-1640-e3-p1.pos",
    "interview-e1-p2.pos", "jetaylor-e3-h.pos", "jetaylor-e3-p1.pos",
    "jopinney-e3-p1.pos", "jotaylor-e2-p1.pos", "joxinden-e2-p2.pos",
    "jubarring-e2-p1.pos", "knyvett-1630-e2-p2.pos", "koxinden-e2-p1.pos",
    "kpaston-e2-h.pos", "kpaston-e2-p1.pos", "kscrope-1530-e1-h.pos",
    "leland-e1-h.pos", "leland-e1-p2.pos", "lisle-e3-p1.pos",
    "madox-e2-h.pos", "marches-e1-p1.pos", "markham-e2-p2.pos",
    "masham-e2-p1.pos", "masham-e2-p2.pos", "memo-e3-p2.pos",
    "milton-e3-h.pos", "mroper-e1-p1.pos", "mroper-e1-p2.pos",
    "nhadd-1700-e3-h.pos", "nhadd-1700-e3-p1.pos", "penny-e3-p2.pos",
    "pepys-e3-p1.pos", "perrott-e2-p1.pos", "pettit-e2-h.pos",
    "pettit-e2-p1.pos", "proposals-e3-p2.pos", "rcecil-e2-p1.pos",
    "record-e1-h.pos", "record-e1-p1.pos", "record-e1-p2.pos",
    "rhaddjr-e3-h.pos", "rhaddsr-1670-e3-p2.pos", "rhaddsr-1700-e3-h.pos",
    "rhaddsr-1710-e3-p2.pos", "roper-e1-h.pos", "roxinden-1620-e2-h.pos",
    "rplumpt-e1-p1.pos", "rplumpt2-e1-p2.pos", "shakesp-e2-h.pos",
    "shakesp-e2-p1.pos", "somers-e3-h.pos", "stat-1540-e1-p1.pos",
    "stat-1570-e2-p1.pos", "stat-1580-e2-p2.pos", "stat-1600-e2-h.pos",
    "stat-1620-e2-p2.pos", "stat-1670-e3-p2.pos", "stevenso-e1-h.pos",
    "tillots-a-e3-h.pos", "tillots-b-e3-p1.pos", "turner-e1-h.pos",
    "tyndold-e1-p1.pos", "udall-e1-p2.pos", "underhill-e1-p2.pos",
    "vicary-e1-h.pos", "walton-e3-p1.pos", "wplumpt-1500-e1-h.pos",
    "zouch-e3-p2.pos"])

_test = [f for f in _paths if os.path.basename(f) in _test]
_paths = [f for f in _paths if f not in _test]
_train, _dev = train_test_split(_paths, test_size=0.05, random_state=1001)
for path in _test:
    assert path not in _train
    assert path not in _dev
for path in _dev:
    assert path not in _train


def write_json(examples, output):
    with open(output, 'w+') as f:
        for ex in examples:
            f.write(json.dumps(ex) + '\n')

for size in [50, 100, 150, 200, None]:
    target_dir = os.path.join(ROOT, 'ppceme-{}'.format(size or 'full'))
    if not os.path.isdir(target_dir):
        os.makedirs(target_dir)
    write_json(list(generate_examples(_test)), os.path.join(target_dir, 'test.json'))
    write_json(list(generate_examples(_dev)), os.path.join(target_dir, 'dev.json'))
    write_json(list(generate_examples(_train[:size])), os.path.join(target_dir, 'train.json'))
    # pd.DataFrame.from_dict(list(generate_examples(_test))).to_csv(
    #     os.path.join(target_dir, 'test.csv'))
    # pd.DataFrame.from_dict(list(generate_examples(_dev))).to_csv(
    #     os.path.join(target_dir, 'dev.csv'))
    # pd.DataFrame.from_dict(list(generate_examples(_train[:size]))).to_csv(
    #     os.path.join(target_dir, 'train.csv'))
