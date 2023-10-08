import argparse
import os
import random
from itertools import product

import numpy as np
import scipy.io as sio
import torch
from torchtext.vocab import build_vocab_from_iterator
from transformers import AutoModel, AutoTokenizer, BertConfig, BertForMaskedLM
from tqdm import tqdm
from sklearn.model_selection import StratifiedShuffleSplit

from util import KmerTokenizer
from bert_extract_dna_feature import extract_clean_barcode_list, extract_clean_barcode_list_for_aligned
from pablo_bert_with_prediction_head import Bert_With_Prediction_Head, train_and_eval
from torch.utils.data import DataLoader, Dataset


device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
random.seed(10)




class DNADataset(Dataset):
    def __init__(self, barcodes, labels, k_mer=4, stride=4, max_len=256):
        self.k_mer = k_mer
        self.stride = stride
        self.max_len = max_len

        # Vocabulary
        kmer_iter = ([''.join(kmer)] for kmer in product('ACGT', repeat=self.k_mer))
        self.vocab = build_vocab_from_iterator(kmer_iter, specials=["<MASK>", "<CLS>", "<UNK>"])
        self.vocab.set_default_index(self.vocab["<UNK>"])
        self.vocab_size = len(self.vocab)

        self.tokenizer = KmerTokenizer(self.k_mer, self.vocab, stride=self.stride, padding=True, max_len=self.max_len)


        self.barcodes = barcodes
        self.labels = labels

    def __len__(self):
        return len(self.barcodes)

    def __getitem__(self, idx):
        processed_barcode = torch.tensor(self.tokenizer(self.barcodes[idx]), dtype=torch.int64)
        return processed_barcode, self.labels[idx]


class kmer_tokenizer(object):
    def __init__(self, k, stride=1):
        self.k = k
        self.stride = stride

    def __call__(self, dna_sequence):
        tokens = []
        for i in range(0, len(dna_sequence) - self.k + 1, self.stride):
            k_mer = dna_sequence[i: i + self.k]
            tokens.append(k_mer)
        return tokens


class PadSequence(object):
    def __init__(self, max_len):
        self.max_len = max_len

    def __call__(self, dna_sequence):
        if len(dna_sequence) > self.max_len:
            return dna_sequence[: self.max_len]
        else:
            return dna_sequence + "N" * (self.max_len - len(dna_sequence))

        # return new_sequence


def remove_extra_pre_fix(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[7:]  # 去除 'module.' 前缀
        new_state_dict[key] = value
    return new_state_dict


def load_model(args, number_of_classes):
    k = args.k
    kmer_iter = (["".join(kmer)] for kmer in product("ACGT", repeat=k))
    vocab = build_vocab_from_iterator(kmer_iter, specials=["<MASK>", "<CLS>", "<UNK>"])
    vocab.set_default_index(vocab["<UNK>"])
    vocab_size = len(vocab)
    max_len = 660
    pad = PadSequence(max_len)

    print("Initializing the model . . .")

    tokenizer = kmer_tokenizer(k, stride=k)
    sequence_pipeline = lambda x: [0, *vocab(tokenizer(pad(x)))]
    configuration = BertConfig(vocab_size=vocab_size, output_hidden_states=True)
    bert_model = BertForMaskedLM(configuration)

    state_dict = torch.load(args.checkpoint, map_location=torch.device("cpu"))
    state_dict = remove_extra_pre_fix(state_dict)
    bert_model.load_state_dict(state_dict)

    model = Bert_With_Prediction_Head(out_feature=number_of_classes, bert_model=bert_model)
    model.to(device)

    print("The model has been succesfully loaded . . .")
    return model, sequence_pipeline


def load_data(args):
    x = sio.loadmat(args.input_path)

    if args.using_aligned_barcode:
        barcodes = extract_clean_barcode_list_for_aligned(x["nucleotides_aligned"])
    else:
        barcodes = extract_clean_barcode_list(x["nucleotides"])
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
        pbar = tqdm(enumerate(labels), total=len(labels))
        for i, label in pbar:
            pbar.set_description("Extracting features: ")
            _barcode = barcodes[i]
            if args.model == "dnabert2":
                x = sequence_pipeline(_barcode).to(device)
                x = model(x)[-1]
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

    if args.using_aligned_barcode:
        np.savetxt(os.path.join(args.output_dir, "dna_embedding_supervised_fine_tuned_pablo_bert_5_mer_ep_40_aligned.csv"), class_embed, delimiter=",")
    else:
        np.savetxt(os.path.join(args.output_dir, "dna_embedding_supervised_fine_tuned_pablo_bert_5_mer_ep_40.csv"),
                   class_embed, delimiter=",")


    print("DNA embeddings is saved.")

def construct_dataloader(X_train, X_val, y_train, y_val, batch_size, k=5, max_len=660):

    train_dataset = DNADataset(X_train, y_train, k_mer=k, stride=k, max_len=max_len)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    val_dataset = DNADataset(X_val, y_val, k_mer=k, stride=k, max_len=max_len)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return train_dataloader, val_dataloader

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", default="../data/INSECT/res101.mat", type=str)
    parser.add_argument("--model", choices=["bioscanbert", "dnabert", "dnabert2"], default="bioscanbert")
    parser.add_argument("--checkpoint", default="bert_checkpoint/5-mer/model_41.pth", type=str)
    parser.add_argument("--output_dir", type=str, default="../data/INSECT/")
    parser.add_argument("--using_aligned_barcode", default=False, action="store_true")
    parser.add_argument("--n_epoch", default=12, type=int)
    parser.add_argument("--k", default=5, type=int)

    args = parser.parse_args()

    x_train, y_train, x_val, y_val, barcodes, labels, number_of_classes = load_data(args)

    model, sequence_pipeline = load_model(args, number_of_classes)

    train_loader, val_loader = construct_dataloader(x_train, x_val, y_train, y_val, 32, k=args.k)

    train_and_eval(model, train_loader, val_loader, device=device, n_epoch=args.n_epoch)

    extract_and_save_class_level_feature(args, model, sequence_pipeline, barcodes, labels)


