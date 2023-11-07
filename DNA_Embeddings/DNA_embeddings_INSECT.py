import os

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
from util import construct_dataloader
from CNN import Model, train_and_eval

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data(aligned=True):
    datapath = r"../../data"
    dataset = "INSECT"
    x = sio.loadmat(os.path.join(datapath, dataset, "res101.mat"))
    x2 = sio.loadmat(os.path.join(datapath, dataset, "att_splits.mat"))
    if aligned:
        barcodes = x["nucleotides_aligned"].T
    else:
        barcodes = x["nucleotides"]
    breakpoint()
    species = x["labels"]
    train_loc = x2["trainval_loc"]

    # Number of training samples and entire data
    N = len(barcodes)

    bcodes = []
    labels = []
    for i in range(N):
        if len(barcodes[i][0][0]) > 0:
            bcodes.append(barcodes[i][0][0])
            labels.append(species[i][0])
    # Can be optimize, write function to avoid using tensorflow.
    tokenizer = Tokenizer(char_level=True)
    tokenizer.fit_on_texts(bcodes)
    sequence_of_int = tokenizer.texts_to_sequences(bcodes)

    cnt = 0

    ll = np.zeros((N, 1))
    for i in range(N):
        ll[i] = len(sequence_of_int[i])
        if ll[i] == 658:
            # print(i)
            cnt += 1

    # Calculating sequence count for each species
    cl = len(np.unique(species))
    seq_cnt = np.zeros(cl)
    for i in range(N):
        k = labels[i] - 1  # Accounting for Matlab labeling
        seq_cnt[k] += 1
        # seq_len[k]+=len(sequence_of_int[k])

    # Converting all the data into one-hot encoding. Note that this data is not used during training.
    # We are getting it ready for the prediction time to get the final DNA embeddings after training is done
    sl = 658
    allX = np.zeros((N, sl, 5))
    for i in range(N):
        for j in range(sl):
            if len(sequence_of_int[i]) > j:
                k = sequence_of_int[i][j] - 1
                if k > 4:
                    k = 4
                allX[i][j][k] = 1.0

    # Initialize the training matrix and labels
    trainX = np.zeros((N, sl, 5))
    trainY = np.zeros((N, cl))
    labelY = np.zeros(N)

    Nc = -1
    class_cnt = np.zeros(cl)
    for i in range(N):
        k = labels[i] - 1
        class_cnt[k] += 1
        itl = i + 1
        if (
            seq_cnt[k] >= 10 and class_cnt[k] <= 50 and itl in train_loc[0]
        ):  # Note that samples from training set are only used
            Nc = Nc + 1
            for j in range(sl):
                if len(sequence_of_int[i]) > j:
                    k = sequence_of_int[i][j] - 1
                    if k > 4:
                        k = 4
                    trainX[Nc][j][k] = 1.0

            k = labels[i] - 1
            trainY[Nc][k] = 1.0
            labelY[Nc] = k

    trainX = trainX[0:Nc]
    trainY = trainY[0:Nc]
    all_Y = labelY
    labelY = labelY[0:Nc]

    # To make sure the training data does not include any unseen class nucleotides
    label_us = species[x2["test_unseen_loc"][0] - 1]
    np.intersect1d(np.unique(labelY), np.unique(label_us) - 1)

    # Cleaning empty classes
    idx = np.argwhere(np.all(trainY[..., :] == 0, axis=0))
    trainY = np.delete(trainY, idx, axis=1)

    # Expanding the training set shape for CNN
    trainX = np.expand_dims(trainX, axis=3)
    allX = np.expand_dims(allX, axis=3)

    X_train, X_test, y_train, y_test = train_test_split(trainX, labelY, test_size=0.2, random_state=42)

    total_number_of_classes = len(np.unique(labels))
    return X_train, X_test, y_train, y_test, torch.Tensor(allX), species, total_number_of_classes


def get_embedding(model, all_X):
    embedding = None
    with torch.no_grad():
        for inputs in all_X:
            _, feature = model(torch.unsqueeze(inputs.to(device), 0))
            if embedding is None:
                embedding = feature
            else:
                embedding = torch.cat((embedding, feature), 0)
    embedding = embedding.to("cpu")
    return embedding


if __name__ == "__main__":
    # NOTE: it didn't feel worth implementing a proper argparse for this since we barely run this script,
    # but if we do use it in the future, it would be worth adding
    aligned = True
    extract_class_embeddings = True

    X_train, X_test, y_train, y_test, all_X, species, total_number_of_classes = load_data(aligned=aligned)
    trainloader, testloader = construct_dataloader(X_train, X_test, y_train, y_test, 32)
    model = Model(1, total_number_of_classes).to(device)
    train_and_eval(model, trainloader, testloader, device=device)
    dna_embeddings = get_embedding(model, all_X)
    dict_emb = {}

    if extract_class_embeddings:
        for index, label in enumerate(species):
            if str(label[0]) not in dict_emb.keys():
                dict_emb[str(label[0])] = []
            dict_emb[str(label[0])].append(np.array(dna_embeddings[index]))
        all_label = np.unique(species)
        class_embed = []
        for i in all_label:
            class_embed.append(np.sum(dict_emb[str(i)], axis=0) / len(dict_emb[str(i)]))
        class_embed = np.array(class_embed, dtype=object)
        final_embeddings = class_embed.T
    else:
        final_embeddings = dna_embeddings

    np.savetxt("../../data/INSECT/dna_embeddings.csv", final_embeddings, delimiter=",")
    print("DNA embeddings is saved.")
