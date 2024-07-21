import argparse
import os
import pickle
import random
import sys
from itertools import product

import numpy as np
import scipy.io as sio
import torch
from torch import optim
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
    def __init__(self, barcodes, k_mer=4, stride=4, max_len=256):
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

    def __len__(self):
        return len(self.barcodes)

    def __getitem__(self, idx):
        processed_barcode = torch.tensor(self.tokenizer(self.barcodes[idx]), dtype=torch.int64)
        return processed_barcode


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
    k = 6
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

    model = BertForMaskedLM(configuration)
    state_dict = torch.load(args.checkpoint)
    state_dict = remove_extra_pre_fix(state_dict)
    model.load_state_dict(state_dict)
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


    number_of_classes = np.unique(labels).shape[0]

    return barcodes, labels, number_of_classes





def construct_dataloader(barcodes, batch_size, k=6, max_len=660):
    dataset = DNADataset(barcodes, k_mer=k, stride=k, max_len=max_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    return dataloader


def train(args, dataloader, device, model, optimizer, scheduler):
    model.train()
    epoch_loss_list = []
    training_epoch = 100
    continue_epoch = 0

    if not os.path.isdir(args.ckpt_output_dir):
        os.makedirs(args.ckpt_output_dir, exist_ok=True)

    print("Training is started:\n")

    for epoch in range(args.n_epoch):
        epoch_loss = 0
        pbar = tqdm(enumerate(dataloader), total=len(dataloader))
        for i, batch in pbar:

            optimizer.zero_grad()

            # Build the masking on the fly every time something different
            batch = batch.to(device)
            masked_input = batch.clone()
            random_mask = torch.rand(masked_input.shape).to(device)  # I can only do this for non-overlapping
            random_mask = (random_mask < 0.5) * (masked_input != 2)  # Cannot mask the [<UNK>] token
            mask_idx = (random_mask.flatten() == True).nonzero().view(-1)
            masked_input = masked_input.flatten()
            masked_input[mask_idx] = 1
            masked_input = masked_input.view(batch.size())


            out = model(masked_input, labels=batch)
            loss = out.loss
            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()
            pbar.set_description("loss: " + str(loss.item()))

        epoch_loss = epoch_loss / len(dataloader)
        epoch_loss_list.append(epoch_loss)
        before_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        after_lr = optimizer.param_groups[0]["lr"]
        print("Epoch %d: lr %f -> %f" % (epoch, before_lr, after_lr))
        print("Epoch loss: " + str(epoch_loss))

    torch.save(model.state_dict(), args.ckpt_output_dir + "/model_" + str(args.n_epoch) + '.pth')
    torch.save(optimizer.state_dict(), args.ckpt_output_dir + "/optimizer_" + str(args.n_epoch) + '.pth')
    torch.save(scheduler.state_dict(), args.ckpt_output_dir + "/scheduler_" + str(args.n_epoch) + '.pth')

    a_file = open(args.ckpt_output_dir + f"/loss_{device}.pkl", "wb")
    pickle.dump(epoch_loss_list, a_file)
    a_file.close()

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
                x = model(x).hidden_states[-1].mean(dim=1)
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
        np.savetxt(os.path.join(args.output_dir, "dna_embedding_mlm_fine_tuned_pablo_bert_aligned.csv"), class_embed, delimiter=",")
    else:
        np.savetxt(os.path.join(args.output_dir, "dna_embedding_mlm_fine_tuned_pablo_bert.csv"),
                   class_embed, delimiter=",")


    print("DNA embeddings is saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", default="../data/INSECT/res101.mat", type=str)
    parser.add_argument("--model", choices=["bioscanbert", "dnabert", "dnabert2"], default="bioscanbert")
    parser.add_argument("--checkpoint", default="bert_checkpoint/model_44.pth", type=str)
    parser.add_argument("--output_dir", type=str, default="../data/INSECT/")
    parser.add_argument("--using_aligned_barcode", default=False, action="store_true")
    parser.add_argument("--ckpt_output_dir", type=str, default="../checkpoints/mlm_fine_tuned_pablo_model")
    parser.add_argument('--lr', action='store', type=float, default=1e-4)
    parser.add_argument('--weight_decay', action='store', type=float, default=1e-05)
    parser.add_argument("--n_epoch", default=20, type=int)

    args = parser.parse_args()

    barcodes, labels, number_of_classes = load_data(args)

    model, sequence_pipeline = load_model(args, number_of_classes)

    dataloader = construct_dataloader(barcodes, 32)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-2, total_iters=5)

    train(args, dataloader, device, model, optimizer, scheduler)

    extract_and_save_class_level_feature(args, model, sequence_pipeline, barcodes, labels)
