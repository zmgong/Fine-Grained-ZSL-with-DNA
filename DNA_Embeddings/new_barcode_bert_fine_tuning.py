# -*- coding: utf-8 -*-
import argparse
import os
import pickle
import sys
from itertools import product

import numpy as np
import pandas as pd
import scipy.io as sio
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchtext.vocab import build_vocab_from_iterator
from tqdm import tqdm
from transformers import BertForMaskedLM, BertConfig, BertForTokenClassification
from transformers.modeling_outputs import TokenClassifierOutput
from torchtext.vocab import vocab as build_vocab_from_dict
from bert_extract_dna_feature import extract_clean_barcode_list_for_aligned
import torch.nn.functional as F

"""# Load data and tokenize """


class ClassificationModel(nn.Module):
    def __init__(self, base_model, num_labels):
        super(ClassificationModel, self).__init__()
        self.num_labels = num_labels
        self.base_model = base_model
        self.hidden_size = self.base_model.config.hidden_size
        self.classifier = nn.Linear(self.hidden_size, self.num_labels)

    def forward(self, input_ids=None, mask=None, labels=None):
        # Getting the embedding
        outputs = self.base_model(input_ids=input_ids, attention_mask=mask)
        embeddings = outputs.hidden_states[-1]
        GAP_embeddings = embeddings.mean(1)
        # calculate losses
        logits = self.classifier(GAP_embeddings.view(-1, self.hidden_size))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1))
        return TokenClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states)

def extract_and_save_class_level_feature(args, model, dataloader_all_data, device="cuda"):
    all_label = []
    dict_emb = {}


    with torch.no_grad():
        # model.eval()
        pbar = tqdm(enumerate(dataloader_all_data), total=len(dataloader_all_data))
        for _, batch in pbar:
            pbar.set_description("Extracting features: ")
            sequences = batch[0]
            att_mask = batch[1]
            labels = batch[2]
            sequences = sequences.to(device)
            att_mask = att_mask.to(device)
            sequences = model(sequences, att_mask).hidden_states[-1]

            sum_embeddings = (sequences * att_mask.unsqueeze(-1)).sum(1)
            sum_mask = att_mask.sum(1, keepdim=True)
            feature = sum_embeddings / sum_mask

            for idx, label in enumerate(labels):
                if str(label) not in dict_emb.keys():
                    dict_emb[str(label.item())] = []
                dict_emb[str(label.item())].append(feature[idx].cpu().numpy())
                all_label.append(label.item())

    all_label = np.unique(all_label)
    all_label.sort()
    class_embed = []
    for label in all_label:
        class_embed.append(np.sum(dict_emb[str(label)], axis=0) / len(dict_emb[str(label)]))
    class_embed = np.array(class_embed, dtype=object)
    class_embed = class_embed.T.squeeze()

    # save results

    if args['extract_feature_without_fine_tuning']:
        args['output_dir'] = os.path.join("embedding_extract_from_new_BarcodeBERT", f"pre_trained_on{args['pre_trained_on']}", "without_fine_tuning")
        os.makedirs(args['output_dir'], exist_ok=True)
        embedding_path = os.path.join(args['output_dir'],
                                        f"dna_embedding_from_barcode_bert_pre_trained_on_{args['pre_trained_on']}_without_fine_tuning.csv")
        np.savetxt(
            embedding_path,
            class_embed,
            delimiter=",",
        )
        print(f"DNA embeddings is saved in {embedding_path}.")
    else:
        args['output_dir'] = os.path.join("..", "data", "INSECT", "embedding_extract_from_new_BarcodeBERT")
        os.makedirs(args['output_dir'], exist_ok=True)
        embedding_path = os.path.join(args['output_dir'],
                                        f"dna_embedding_from_barcode_bert_pre_trained_on_{args['pre_trained_on']}_with_fine_tuning.csv")

        if os.path.exists(embedding_path):
            os.remove(embedding_path)

        np.savetxt(
            embedding_path,
            class_embed,
            delimiter=",",
        )
        print(f"DNA embeddings is saved in {embedding_path}.")


class DNADataset(Dataset):
    def __init__(self, x, y, k_mer=4, stride=4, max_len=256, randomize_offset=False, tokenizer="kmer",
                 tokenize_n_nucleotide=False):
        self.k_mer = k_mer
        self.stride = k_mer if stride is None else stride
        self.max_len = max_len
        self.randomize_offset = randomize_offset

        # Vocabulary
        base_pairs = "ACGT"
        self.special_tokens = ["[MASK]", "[UNK]"]  # ["[MASK]", "[CLS]", "[SEP]", "[PAD]", "[EOS]", "[UNK]"]
        UNK_TOKEN = "[UNK]"

        if tokenize_n_nucleotide:
            # Encode kmers which contain N differently depending on where it is
            base_pairs += "N"

        if tokenizer == "kmer":
            kmers = ["".join(kmer) for kmer in product(base_pairs, repeat=self.k_mer)]

            # Separate between good (idx < 4**k) and bad k-mers (idx > 4**k) for prediction
            if tokenize_n_nucleotide:
                prediction_kmers = []
                other_kmers = []
                for kmer in kmers:
                    if "N" in kmer:
                        other_kmers.append(kmer)
                    else:
                        prediction_kmers.append(kmer)

                kmers = prediction_kmers + other_kmers

            kmer_dict = dict.fromkeys(kmers, 1)
            self.vocab = build_vocab_from_dict(kmer_dict, specials=self.special_tokens)
            self.vocab.set_default_index(self.vocab[UNK_TOKEN])
            self.vocab_size = len(self.vocab)
            self.tokenizer = KmerTokenizer(
                self.k_mer, self.vocab, stride=self.stride, padding=True, max_len=self.max_len
            )
        else:
            raise ValueError(f'Tokenizer "{tokenizer}" not recognized.')

        self.barcodes = x
        self.labels = y
        self.num_labels = len(np.unique(self.labels))

    def __len__(self):
        return len(self.barcodes)

    def __getitem__(self, idx):
        if self.randomize_offset:
            offset = torch.randint(self.k_mer, (1,)).item()
        else:
            offset = 0
        processed_barcode, att_mask = self.tokenizer(self.barcodes[idx], offset=offset)
        label = torch.tensor(self.labels[idx], dtype=torch.int64)
        return processed_barcode, att_mask, label

class KmerTokenizer(object):
    def __init__(self, k, vocabulary_mapper, stride=1, padding=False, max_len=660):
        self.k = k
        self.stride = stride
        self.padding = padding
        self.max_len = max_len
        self.vocabulary_mapper = vocabulary_mapper

    def __call__(self, dna_sequence, offset=0):
        tokens = []
        att_mask = [1] * (self.max_len // self.stride)
        x = dna_sequence[offset:]
        if self.padding:
            if len(x) > self.max_len:
                x = x[: self.max_len]
            else:
                att_mask[len(x) // self.stride :] = [0] * (len(att_mask) - len(x) // self.stride)
                x = x + "N" * (self.max_len - len(x))
        for i in range(0, len(x) - self.k + 1, self.stride):
            k_mer = x[i : i + self.k]
            tokens.append(k_mer)

        tokens = torch.tensor(self.vocabulary_mapper(tokens), dtype=torch.int64)
        att_mask = torch.tensor(att_mask, dtype=torch.int32)

        return tokens, att_mask


def __init__(self, x, y, k_mer=4, stride=4, max_len=256, randomize_offset=False, tokenizer="kmer",
             tokenize_n_nucleotide=False):
    self.k_mer = k_mer
    self.stride = k_mer if stride is None else stride
    self.max_len = max_len
    self.randomize_offset = randomize_offset

    # Vocabulary
    base_pairs = "ACGT"
    self.special_tokens = ["[MASK]", "[UNK]"]  # ["[MASK]", "[CLS]", "[SEP]", "[PAD]", "[EOS]", "[UNK]"]
    UNK_TOKEN = "[UNK]"

    if tokenize_n_nucleotide:
        # Encode kmers which contain N differently depending on where it is
        base_pairs += "N"

    if tokenizer == "kmer":
        kmers = ["".join(kmer) for kmer in product(base_pairs, repeat=self.k_mer)]

        # Separate between good (idx < 4**k) and bad k-mers (idx > 4**k) for prediction
        if tokenize_n_nucleotide:
            prediction_kmers = []
            other_kmers = []
            for kmer in kmers:
                if "N" in kmer:
                    other_kmers.append(kmer)
                else:
                    prediction_kmers.append(kmer)

            kmers = prediction_kmers + other_kmers

        kmer_dict = dict.fromkeys(kmers, 1)
        self.vocab = build_vocab_from_dict(kmer_dict, specials=self.special_tokens)
        self.vocab.set_default_index(self.vocab[UNK_TOKEN])
        self.vocab_size = len(self.vocab)
        self.tokenizer = KmerTokenizer(
            self.k_mer, self.vocab, stride=self.stride, padding=True, max_len=self.max_len
        )
    else:
        raise ValueError(f'Tokenizer "{tokenizer}" not recognized.')

    self.barcodes = x
    self.labels = y
    self.num_labels = len(np.unique(self.labels))

def __len__(self):
    return len(self.barcodes)

def __getitem__(self, idx):
    if self.randomize_offset:
        offset = torch.randint(self.k_mer, (1,)).item()
    else:
        offset = 0
    processed_barcode, att_mask = self.tokenizer(self.barcodes[idx], offset=offset)
    label = torch.tensor(self.labels[idx], dtype=torch.int64)
    return processed_barcode, att_mask, label


class ClassificationModel(nn.Module):
    def __init__(self, base_model, num_labels):
        super(ClassificationModel, self).__init__()
        self.num_labels = num_labels
        self.base_model = base_model
        self.hidden_size = self.base_model.config.hidden_size
        self.classifier = nn.Linear(self.hidden_size, self.num_labels)

    def forward(self, input_ids=None, mask=None, labels=None):
        # Getting the embedding
        outputs = self.base_model(input_ids=input_ids, attention_mask=mask)
        embeddings = outputs.hidden_states[-1]
        GAP_embeddings = embeddings.mean(1)
        # calculate losses
        logits = self.classifier(GAP_embeddings.view(-1, self.hidden_size))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1))
        return TokenClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states)


"""# Train """


def train(args, dataloader, device, model, optimizer, scheduler):
    epoch_loss_list = []
    eval_epoch_loss_list = []
    eval_acc_list = []
    training_epoch = args["epoch"]
    continue_epoch = 0
    dataloader_train, dataloader_dev = dataloader[0], dataloader[1]

    saving_path = os.path.join(args["save_checkpoint_path"], f"pre_trained_with{args['pre_trained_on']}", "fine_tuning_checkpoints/")
    if not os.path.isdir(saving_path):
        os.makedirs(saving_path, exist_ok=True)

    # if args['checkpoint']:
    #     continue_epoch = 4
    #     model.load_state_dict(torch.load(saving_path + f'model_{continue_epoch}.pth'))
    #     optimizer.load_state_dict(torch.load(saving_path + f"optimizer_{continue_epoch}.pth"))
    #     scheduler.load_state_dict(torch.load(saving_path + f"scheduler_{continue_epoch}.pth"))
    #     a_file = open(saving_path + f"loss_{device}.pkl", "rb")
    #     epoch_loss_list = pickle.load(a_file)
    #     print("Training is continued...")

    sys.stdout.write("Training is started:\n")

    for epoch in range(continue_epoch + 1, training_epoch + 1):
        epoch_loss = 0
        eval_loss = 0
        acc = 0
        dataloader_train.sampler.set_epoch(epoch)
        model.train()
        pbar = tqdm(enumerate(dataloader_train), total=len(dataloader_train))
        for i, batch in pbar:

            optimizer.zero_grad()
            sequences = batch[0]
            att_mask = batch[1]
            labels = batch[2]

            sequences = sequences.to(device)
            labels = labels.to(device)

            sequences = sequences.clone()
            labels = labels.clone()

            out = model(sequences, labels=labels)
            loss = out.loss
            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()
            pbar.set_description(f"Epoch {epoch}, Device {device}: Loss is {loss.item()} || lr: {scheduler.get_last_lr()}")
        epoch_loss = epoch_loss / len(dataloader_train)
        epoch_loss_list.append(epoch_loss)

        before_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        after_lr = optimizer.param_groups[0]["lr"]
        sys.stdout.write("Epoch %d: lr %f -> %f" % (epoch, before_lr, after_lr))

        sys.stdout.write(f"Epoch {epoch}, Device {device}: Loss is {epoch_loss}\n")

        # torch.save(model.state_dict(), saving_path + "model_" + str(epoch) + '.pth')
        # torch.save(optimizer.state_dict(), saving_path + "optimizer_" + str(epoch) + '.pth')
        # torch.save(scheduler.state_dict(), saving_path + "scheduler_" + str(epoch) + '.pth')

        model.eval()
        for i, batch in enumerate(tqdm(dataloader_dev)):
            eval_sequences = batch[0]
            eval_att_mask = batch[1]
            eval_labels = batch[2]
            with torch.no_grad():
                outputs = model(eval_sequences, eval_att_mask, labels=eval_labels)

            eval_loss += outputs.loss.item()
            eval_logits = outputs.logits
            predictions = torch.argmax(eval_logits, dim=-1)
            acc += accuracy_score(predictions.cpu(), eval_labels)

        eval_acc = acc / len(dataloader_dev)
        eval_acc_list.append(eval_acc)

        eval_loss = eval_loss / len(dataloader_dev)
        eval_epoch_loss_list.append(eval_loss)

        sys.stdout.write("validation set loss: %f \n" % eval_loss)
        sys.stdout.write("validation set accuracy: %f \n" % eval_acc)

        sys.stdout.write("--------------------------------------------")
        a_file = open(saving_path + f"loss_{device}.pkl", "wb")
        l = [epoch_loss_list, eval_epoch_loss_list, eval_acc_list]
        pickle.dump(l, a_file)

        a_file.close()

    return model

def test(dataloader_test, device, model):
    acc = 0

    model.eval()
    for i, batch in enumerate(tqdm(dataloader_test)):
        eval_sequences = batch[0]
        att_mask = batch[1]
        eval_labels = batch[2]
        eval_sequences = eval_sequences.to(device)
        eval_labels = eval_labels.to(device)
        with torch.no_grad():
            outputs = model(eval_sequences)

        eval_logits = outputs.logits
        predictions = torch.argmax(eval_logits, dim=-1)
        acc += accuracy_score(predictions.cpu(), eval_labels.cpu())

    eval_acc = acc / len(dataloader_test)

    sys.stdout.write("test set accuracy: %f" % eval_acc)


def ddp_setup(rank: int, world_size: int):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '5000'
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def prepare(dataset, rank, world_size, batch_size=32, pin_memory=False, num_workers=0):
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)

    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers,
                            drop_last=False, shuffle=False, sampler=sampler)

    return dataloader

def load_data(args):
    x = sio.loadmat(args["input_path"])

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

def remove_extra_pre_fix(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[7:]
        new_state_dict[key] = value
    return new_state_dict

def load_pretrained_model(checkpoint_path, device=None):
    """
    Load a pretrained model from a checkpoint file.

    Parameters
    ----------
    checkpoint_path : str
        Path to the pretrained checkpoint file.

    Returns
    -------
    model : torch.nn.Module
        The pretrained model.
    ckpt : dict
        The contents of the checkpoint file.
    """
    print(f"\nLoading model from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    model_ckpt = remove_extra_pre_fix(ckpt["model"])

    assert "bert_config" in ckpt  # You may be trying to load an old checkpoint

    bert_config = BertConfig(**ckpt["bert_config"])
    model = BertForTokenClassification(bert_config)
    if "bert.embeddings.position_ids" in model_ckpt:
        model_ckpt.pop("bert.embeddings.position_ids")

    model.load_state_dict(model_ckpt)
    print(f"Loaded model from {checkpoint_path}")
    return model, ckpt

def main(rank: int, world_size: int, args):
    ddp_setup(rank, world_size)
    # Loading data
    sys.stdout.write("Loading the dataset is started.\n")
    x_train, y_train, x_val, y_val, barcodes, labels, num_classes = load_data(args)

    dataset_train = DNADataset(x=x_train, y=y_train, k_mer=args['k_mer'],
                               stride=args['stride'],
                               max_len=args['max_len'])
    dataloader_train = prepare(dataset_train, rank, world_size=world_size, batch_size=args['batch_size'])
    dataset_dev = DNADataset(x=x_val, y=y_val, k_mer=args['k_mer'], stride=args['stride'],
                             max_len=args['max_len'])
    dataloader_dev = prepare(dataset_dev, rank, world_size=world_size, batch_size=args['batch_size'])

    dataset_for_all_data = DNADataset(x=barcodes, y=labels, k_mer=args['k_mer'], stride=args['stride'],
                                      max_len=args['max_len'])
    dataloader_all_data = prepare(dataset_for_all_data, rank, world_size=world_size, batch_size=args['batch_size'])

    # loading model
    checkpoint_path = args["Pretrained_checkpoint_path"]
    num_labels = dataset_train.num_labels
    vocab_size = dataset_train.vocab_size

    sys.stdout.write("Initializing the model ...\n")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pre_model, pre_checkpoint = load_pretrained_model(checkpoint_path)

    model = ClassificationModel(pre_model, dataset_train.num_labels)
    model.to(rank)

    sys.stdout.write("The model has been successfully initialized ...\n")

    if args['extract_feature_without_fine_tuning']:
        pass

    elif args['use_saved_model'] and os.path.exists(
            os.path.join(args['save_checkpoint_path'],
                         f"BarcodeBERT_pre_trained_with_{args.pre_trained_on}_after_tuning.pth")):
        fine_tuned_checkpoint_path = os.path.join(args['save_checkpoint_path'],
                                                  f"BarcodeBERT_pre_trained_with_{args.pre_trained_on}_after_tuning.pth")
        model.load_state_dict(torch.load(fine_tuned_checkpoint_path))
    else:
        learning_rate = args['lr'] * args['batch_size'] / 128
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            [p["lr"] for p in optimizer.param_groups],
            epochs=args['epoch'],
            steps_per_epoch=len(dataloader_train),
        )
        print(type(model))
        model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
        train(args, [dataloader_train, dataloader_dev], rank, model, optimizer, scheduler)
        destroy_process_group()
        model = model.module

    # save model
    if rank == 0:
        os.makedirs(args['save_checkpoint_path'], exist_ok=True)
        torch.save(model.state_dict(), os.path.join(args['save_checkpoint_path'],
                                                    f"BarcodeBERT_pre_trained_with_{args['pre_trained_on']}_after_tuning.pth"))
    model = model.base_model

    if rank == 0:
        extract_and_save_class_level_feature(args, model, dataloader_all_data)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", default="../data/INSECT/res101.mat", type=str)
    parser.add_argument("--pre_trained_on", type=str, default="CANADA-1.5M")
    parser.add_argument('--Pretrained_checkpoint_path', action='store', default="../checkpoints/BarcodeBERT_before_tuning",
                        type=str)
    parser.add_argument('--checkpoint', action='store', type=bool, default=False)
    parser.add_argument('--k_mer', action='store', type=int, default=4)
    parser.add_argument('--stride', action='store', type=int, default=4)
    parser.add_argument('--max_len', action='store', type=int, default=660)
    parser.add_argument('--batch_size', action='store', type=int, default=256)
    parser.add_argument('--lr', action='store', type=float, default=1e-4)
    parser.add_argument('--epoch', action='store', type=int, default=80)
    parser.add_argument('--weight_decay', action='store', type=float, default=1e-05)
    parser.add_argument('--extract_feature_without_fine_tuning', action='store_true', default=False)
    parser.add_argument('--save_checkpoint_path', action='store', type=str, default="../checkpoints/BarcodeBERT_fine_tuned_with_INSECT")
    parser.add_argument('--use_saved_model', action='store', type=bool, default=False)

    args = vars(parser.parse_args())
    if args['pre_trained_on'] == "CANADA-1.5M":
        args['Pretrained_checkpoint_path'] = os.path.join(args['Pretrained_checkpoint_path'], "(CANADA-1.5M)-BEST_k4_4_4_w1_m0_r0_wd.pt")
    elif args['pre_trained_on'] == "BIOSCAN-5M":
        args['Pretrained_checkpoint_path'] = os.path.join(args['Pretrained_checkpoint_path'], "(BIOSCAN-5M)-BEST_k4_6_6_w1_m0_r0.pt")
    else:
        raise ValueError("The model is not trained on the given dataset.")

    sys.stdout.write("\nTraining Parameters:\n")
    for key in args:
        sys.stdout.write(f'{key} \t -> {args[key]}\n')

    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, args), nprocs=world_size)