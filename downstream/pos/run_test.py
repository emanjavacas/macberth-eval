
import tqdm
import datasets
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification)
import torch
from torch.utils.data import DataLoader


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-path', required=True)
    parser.add_argument('--model-path', required=True)
    args = parser.parse_args()

    # test_path = '/Users/manjavacas/Leiden/Datasets/ppceme-50/test.json'
    # model_path = './data/pos/bert_1760_1900-ppceme-100/'

    m = AutoModelForTokenClassification.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    label2id = m.config.label2id
    text_column_name = 'tokens'
    label_column_name = 'pos_tags'
    padding = False
    max_seq_length = 512
    label_all_tokens = False

    # Tokenize all texts and align the labels with them.
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples[text_column_name],
            padding=padding,
            truncation=True,
            max_length=max_seq_length,
            is_split_into_words=True)
        labels = []
        for i, label in enumerate(examples[label_column_name]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label2id.get(label[word_idx], -101))
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(label2id[label[word_idx]] if label_all_tokens else -100)
                previous_word_idx = word_idx

            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    dataset = datasets.load_dataset(
        'json', data_files={'test': args.test_path}
    )['test'].map(tokenize_and_align_labels, batched=True)

    collator = DataCollatorForTokenClassification(tokenizer)
    data_loader = DataLoader(
        dataset.remove_columns(['id', 'pos_tags', 'source', 'tokens']),
        batch_size=48, collate_fn=collator)

    idx = 0
    data = []
    for input in tqdm.tqdm(data_loader, total=len(data_loader)):
        with torch.no_grad():
            output = m(**{k:v for k, v in input.items() if k != 'labels'})
        has_unk = False
        for labels, preds in zip(input['labels'].numpy(), output.logits.argmax(2).numpy()):
            instance = dataset[idx]
            has_unk = -101 in labels
            id_labels = [m.config.id2label.get(id) for id in labels[labels != -100]]
            pred_labels = [m.config.id2label.get(id) for id in preds[labels != -100]]
            data.append({
                'id': instance['id'],
                'n_tokens': len(instance['tokens']),
                'n_labels': len(list(filter(None, id_labels))),
                'n_true': sum(1 if l==t else 0 for l, t in zip(pred_labels, id_labels) if l is not None)
            })
            if len(labels) < 512 and not has_unk:
                true_labels = instance['pos_tags']
                assert true_labels == id_labels
            has_unk = False
            idx += 1
