import argparse
import random
from itertools import product

import numpy as np
import scipy.io as sio
import torch
from torchtext.vocab import build_vocab_from_iterator
from transformers import BertConfig
from transformers import BertForMaskedLM
from tqdm import tqdm

device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
random.seed(10)


class kmer_tokenizer(object):

    def __init__(self, k, stride=1):
        self.k = k
        self.stride = stride

    def __call__(self, dna_sequence):
        tokens = []
        for i in range(0, len(dna_sequence) - self.k + 1, self.stride):
            k_mer = dna_sequence[i:i + self.k]
            tokens.append(k_mer)
        return tokens


class pad_sequence(object):

    def __init__(self, max_len):
        self.max_len = max_len

    def __call__(self, dna_sequence):
        if len(dna_sequence) > self.max_len:
            return dna_sequence[:self.max_len]
        else:
            return dna_sequence + 'N' * (self.max_len - len(dna_sequence))

        # return new_sequence


def remove_extra_pre_fix(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            key = key[7:]  # 去除 'module.' 前缀
        new_state_dict[key] = value
    return new_state_dict


def load_model(args):
    k = 6
    kmer_iter = ([''.join(kmer)] for kmer in product('ACGT', repeat=k))
    tokenizer = kmer_tokenizer(k, stride=k)
    vocab = build_vocab_from_iterator(kmer_iter, specials=["<MASK>", "<CLS>", "<UNK>"])
    vocab.set_default_index(vocab["<UNK>"])
    vocab_size = len(vocab)

    max_len = 660
    PAD = pad_sequence(max_len)
    sequence_pipeline = lambda x: vocab(tokenizer(PAD(x)))

    print("Initializing the model . . .")

    configuration = BertConfig(vocab_size=vocab_size, output_hidden_states=True)

    model = BertForMaskedLM(configuration)
    state_dict = torch.load(args.checkpoint)
    state_dict = remove_extra_pre_fix(state_dict)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("The model has been succesfully loaded . . .")
    return model, sequence_pipeline


def extract_clean_barcode_list(barcodes):
    barcode_list = []

    for i in barcodes:
        barcode_list.append(str(i[0][0]))

    return barcode_list


def extract_clean_barcode_list_for_aligned(barcodes):
    barcodes = barcodes.squeeze().T
    barcode_list = []
    for i in barcodes:
        barcode_list.append(str(i[0]))

    return barcode_list


def load_data(args):
    x = sio.loadmat(args.input_path)
    if args.using_aligned_barcode:
        barcodes = extract_clean_barcode_list_for_aligned(x['nucleotides_aligned'])
    else:
        barcodes = extract_clean_barcode_list(x['nucleotides'])
    print(len(barcodes))
    print(barcodes)
    exit()
    labels = x['labels'].squeeze()

    return barcodes, labels


def extract_and_save_class_level_feature(args, model, sequence_pipeline, barcodes, labels):
    all_label = np.unique(labels)
    all_label.sort()
    dict_emb = {}

    with torch.no_grad():
        pbar = tqdm(enumerate(labels), total=len(labels))
        for i, label in pbar:
            pbar.set_description("Extracting features: ")
            _barcode = barcodes[i]
            x = torch.tensor([0] + sequence_pipeline(_barcode), dtype=torch.int64).unsqueeze(0).to(device)
            x = model(x).hidden_states[-1]
            x = x.mean(1)  # Global Average Pooling excluding CLS token
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
        np.savetxt("../data/INSECT/dna_embedding_using_bert_of_pablo_team.csv", class_embed, delimiter=",")
    else:
        np.savetxt("../data/INSECT/dna_embedding_using_bert_of_pablo_team_no_alignment.csv", class_embed, delimiter=",")
    print('DNA embeddings is saved.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', default="../data/INSECT/res101.mat", type=str)
    parser.add_argument('--checkpoint', default="bert_checkpoint/model_44.pth", type=str)
    parser.add_argument('--output_dir', type=str, default="../data/INSECT")
    parser.add_argument('--using_aligned_barcode', default=False, action="store_true")
    args = parser.parse_args()

    model, sequence_pipeline = load_model(args)

    barcodes, labels = load_data(args)

    extract_and_save_class_level_feature(args, model, sequence_pipeline, barcodes, labels)