
from collections import defaultdict
import math
import os
from datetime import datetime
import pandas as pd
from torch.utils.data import DataLoader
from sentence_transformers.readers import InputExample
from sentence_transformers import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator


def load_dataset(path):
    df = pd.read_csv(path, sep='\t')
    examples = []
    for _, row in df.iterrows():
        row['S1']
        examples.append(InputExample(
            texts=[row['S1'], row['S2']],
            label=int(row['Y1'] > row['Y2'])))
    return examples


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', required=True)
    parser.add_argument('--dev', required=True)
    # parser.add_argument('--test', required=True)
    parser.add_argument('--modelpath', required=True)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=2)
    parser.add_argument('--warmup_steps', type=float, default=0.1)
    parser.add_argument('--output-prefix', default='./data/periodization/')
    args = parser.parse_args()

    # train = load_dataset('./data/periodization/periodization.train.tsv')
    # dev = load_dataset('./data/periodization/periodization.dev.tsv')
    # test = load_dataset('./data/periodization/periodization.test.tsv')
    train = load_dataset(args.train)
    dev = load_dataset(args.dev)
    # test = load_dataset(args.test)

    # Configuration
    model_save_path = 'periodization-' + os.path.basename(os.path.dirname(args.modelpath))
    model_save_path += '-' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_save_path = os.path.join(args.output_prefix, model_save_path)
    if not os.path.isdir(os.path.dirname(model_save_path)):
        os.makedirs(os.path.dirname(model_save_path))

    model = CrossEncoder(args.modelpath, num_labels=1)

    # We wrap train_samples (which is a List[InputExample]) into a pytorch DataLoader
    train_dataloader = DataLoader(train, shuffle=True, batch_size=args.batch_size)

    # We add an evaluator, which evaluates the performance during training
    evaluator = CEBinaryClassificationEvaluator.from_input_examples(dev, name='periodization-dev')

    # Configure the training: 10% of train data for warm-up
    warmup_steps = math.ceil(len(train_dataloader) * args.num_epochs * args.warmup_steps)

    # Train the model
    model.fit(
        train_dataloader=train_dataloader,
        evaluator=evaluator,
        epochs=args.num_epochs,
        evaluation_steps=1000,
        warmup_steps=warmup_steps,
        output_path=model_save_path)