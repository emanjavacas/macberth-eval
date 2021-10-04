
import math
import os
import shutil
from datetime import datetime

import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from sentence_transformers.readers import InputExample
from sentence_transformers import CrossEncoder
from transformers import AutoTokenizer, AutoModel
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator

TGT = '[TGT]'

def get_training_examples(row, target_token=TGT):
    def get_quote(num):
        pos = row['quote_' + num].find(row['keyword_' + num])
        middle = '{} {} {}'.format(target_token, row['keyword_' + num], target_token)
        pre = row['quote_' + num][:pos] 
        post = row['quote_' + num][pos + len(row['keyword_' + num]):]
        return pre + middle + post

    return get_quote('1'), get_quote('2'), row['label']


def load_dataset(path):
    df = pd.read_csv(path)
    examples = []
    for q1, q2, label in df.apply(get_training_examples, axis=1).tolist():
        examples.append(InputExample(texts=[q1, q2], label=label))
    return examples


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--modelpath', required=True)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--warmup_steps', type=float, default=0.1)
    parser.add_argument('--output-prefix', default='./models/periodization/')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    train = load_dataset(args.data)
    # train = load_dataset('./data/word-in-context/depth-1.csv')
    train, dev = train_test_split(train, random_state=1001)

    # Configuration
    model_save_path = 'word-in-context-' + os.path.basename(os.path.dirname(args.modelpath))
    model_save_path += '-' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_save_path = os.path.join(args.output_prefix, model_save_path)
    if not os.path.isdir(os.path.dirname(model_save_path)):
        os.makedirs(os.path.dirname(model_save_path))

    tokenizer = AutoTokenizer.from_pretrained(args.modelpath)
    tokenizer.add_special_tokens({'additional_special_tokens': ['[TGT]']})
    model = AutoModel.from_pretrained(args.modelpath)
    model.resize_token_embeddings(len(tokenizer))
    # save resized model to be able to open it with CrossEncoder
    modelpath = args.modelpath.rstrip('/') + '-wic'
    model.save_pretrained(modelpath)
    tokenizer.save_pretrained(modelpath)
    model = CrossEncoder(modelpath, num_labels=1, device=args.device)
    shutil.rmtree(modelpath)

    # We wrap train_samples (which is a List[InputExample]) into a pytorch DataLoader
    train_dataloader = DataLoader(train, shuffle=True, batch_size=args.batch_size)
    train_dataloader.collate_fn = model.smart_batching_collate
    # We add an evaluator, which evaluates the performance during training
    evaluator = CEBinaryClassificationEvaluator.from_input_examples(dev, name='wic-dev')

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
