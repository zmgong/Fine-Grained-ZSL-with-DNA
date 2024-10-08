import argparse
import os
import random

import numpy as np
import scipy.io as sio
import torch
from tqdm import tqdm
from sklearn.model_selection import StratifiedShuffleSplit

from model import load_model
from bert_extract_dna_feature import extract_clean_barcode_list_for_aligned

from torch.utils.data import DataLoader, Dataset


device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
random.seed(10)


import torch


"""
This script is created for fine-tuning the BarcodeBERT model on the INSECT dataset.
Then extract the class-level features for downstream tasks.
"""


class DNADataset(Dataset):
    def __init__(self, barcodes, labels, tokenizer, pre_tokenize=False):
        # Vocabulary
        self.barcodes = barcodes
        self.labels = labels
        self.pre_tokenize = pre_tokenize
        self.tokenizer = tokenizer

        self.tokenized = tokenizer(self.barcodes.tolist()) if self.pre_tokenize else None

    def __len__(self):
        return len(self.barcodes)

    def __getitem__(self, idx):
        tokens = self.tokenized[idx] if self.pre_tokenize else self.tokenizer(self.barcodes[idx])
        if not isinstance(tokens, torch.Tensor):
            processed_barcode = torch.tensor(tokens, dtype=torch.int64)
        else:
            processed_barcode = tokens.clone().detach().to(dtype=torch.int64)
        return processed_barcode, self.labels[idx]


def load_data(args):
    x = sio.loadmat(args.input_path)

    # if args.using_aligned_barcode:
    #     barcodes = extract_clean_barcode_list_for_aligned(x["nucleotides_aligned"])
    # else:
    #     barcodes = extract_clean_barcode_list(x["nucleotides"])
    barcodes = extract_clean_barcode_list_for_aligned(x["nucleotides_aligned"])

    labels = x["labels"].squeeze() - 1

    stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_index = None
    val_index = None
    for train_split, val_split in stratified_split.split(barcodes, labels):
        train_index = train_split
        val_index = val_split

    x_train = np.array([barcodes[i] for i in train_index])
    x_val = np.array([barcodes[i] for i in val_index])
    y_train = np.array([labels[i] for i in train_index])
    y_val = np.array([labels[i] for i in val_index])

    number_of_classes = np.unique(labels).shape[0]

    return x_train, y_train, x_val, y_val, barcodes, labels, number_of_classes


def extract_and_save_class_level_feature(args, model, sequence_pipeline, barcodes, labels):
    all_label = np.unique(labels)
    all_label.sort()
    dict_emb = {}

    with torch.no_grad():
        # model.eval()
        pbar = tqdm(enumerate(labels), total=len(labels))
        for i, label in pbar:
            pbar.set_description("Extracting features: ")
            _barcode = barcodes[i]
            if args.model == "dnabert2":
                x = sequence_pipeline(_barcode).to(device)
                x = model(x)[0]
                # x = torch.mean(x[0], dim=0)  # mean pooling
                x = torch.max(x[0], dim=0)[0]  # max pooling
            else:
                x = torch.tensor(sequence_pipeline(_barcode), dtype=torch.int64).unsqueeze(0).to(device)
                _, x = model(x)
                x = x.squeeze()

            x = x.cpu().numpy()

            if str(label) not in dict_emb.keys():
                dict_emb[str(label)] = []
            dict_emb[str(label)].append(x)

    class_embed = []
    for i in all_label:
        class_embed.append(np.sum(dict_emb[str(i)], axis=0) / len(dict_emb[str(i)]))
    class_embed = np.array(class_embed, dtype=object)
    class_embed = class_embed.T.squeeze()

    # save results
    os.makedirs(args.output_dir, exist_ok=True)

    if args.using_aligned_barcode:
        np.savetxt(
            os.path.join(args.output_dir, "dna_embedding_supervised_aligned.csv"),
            class_embed,
            delimiter=",",
        )
    else:
        np.savetxt(
            os.path.join(args.output_dir, "dna_embedding_supervised.csv"),
            class_embed,
            delimiter=",",
        )

    print("DNA embeddings is saved.")


def construct_dataloader(X_train, X_val, y_train, y_val, batch_size, tokenizer, pre_tokenize):
    train_dataset = DNADataset(X_train, y_train, tokenizer, pre_tokenize)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=12)

    val_dataset = DNADataset(X_val, y_val, tokenizer, pre_tokenize)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=12)
    return train_dataloader, val_dataloader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", default="../data/INSECT/res101.mat", type=str)
    parser.add_argument("--model", choices=["bioscanbert", "barcodebert", "dnabert", "dnabert2", "new_barcode_bert"], default="new_barcode_bert")
    parser.add_argument("--checkpoint", default="../checkpoints/BarcodeBERT_before_tuning/(CANADA-1.5M)-BEST_k4_4_4_w1_m0_r0_wd.pt", type=str)
    parser.add_argument("--output_dir", type=str, default="../data/INSECT/")
    # parser.add_argument("--using_aligned_barcode", default=False, action="store_true")
    parser.add_argument("--n_epoch", default=100, type=int)
    parser.add_argument("-k", "--kmer", default=6, type=int, dest="k", help="k-mer value for tokenization")
    parser.add_argument(
        "--batch-size", default=128, type=int, dest="batch_size", help="batch size for supervised training"
    )
    parser.add_argument(
        "--model-output", default=None, type=str, dest="model_out", help="path to save model after training"
    )

    args = parser.parse_args()

    x_train, y_train, x_val, y_val, barcodes, labels, num_classes = load_data(args)

    model, sequence_pipeline = load_model(
        args, k=args.k, padding=True, classification_head=True, num_classes=num_classes
    )

    train_loader, val_loader = construct_dataloader(
        x_train,
        x_val,
        y_train,
        y_val,
        args.batch_size,
        sequence_pipeline,
        pre_tokenize=args.model in {"dnabert", "dnabert2"},
    )

    train_and_eval(model, train_loader, val_loader, device=device, n_epoch=args.n_epoch, num_classes=num_classes)

    extract_and_save_class_level_feature(args, model, sequence_pipeline, barcodes, labels)

    if args.model_out:
        torch.save(model.bert_model.state_dict(), args.model_out)
