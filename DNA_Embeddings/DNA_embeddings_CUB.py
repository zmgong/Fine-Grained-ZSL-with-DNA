import os

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

from CNN import Model, train_and_eval

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data():
    datapath = r'../data'
    dataset = 'CUB'
    x2 = sio.loadmat(os.path.join(datapath, dataset, 'CUB_DNA.mat'))
    x = sio.loadmat(os.path.join(datapath, dataset, 'Bird_DNA.mat'))
    us_cls = sio.loadmat(os.path.join(datapath, dataset, 'CUB_unseen_classes_sci_name.mat'))
    barcodes = x['nucleotides_aligned'][0]
    unseen_cls = us_cls['us_classes']
    bird_species = x['species'][0]
    cub_barcodes = x2['nucleotides_aligned'][0]
    c_labels = x2['species'][0]

    # Number of training samples and entire data
    N = len(barcodes)

    bcodes = []
    # labels=[]
    b_species = []
    for i in range(N):
        if len(barcodes[i][0]) > 0:
            bcodes.append(barcodes[i][0])
            b_species.append(bird_species[i][0])

    b_species = np.asarray(b_species)
    bcodes = np.asarray(bcodes)

    cub_bcodes = []
    c_labels_ = []
    for i in range(len(cub_barcodes)):
        cub_bcodes.append(cub_barcodes[i][0])
        c_labels_.append(c_labels[i][0])

    unseen_classes = []
    for i in range(len(unseen_cls)):
        unseen_classes.append(unseen_cls[i][0][0])
    idx = np.in1d(b_species, unseen_classes)

    b_species = b_species[~idx]
    bcodes = bcodes[~idx]

    uy = np.unique(b_species)
    labels = np.zeros(len(b_species), dtype='int32')
    for i in range(len(uy)):
        idx = b_species == uy[i]
        labels[idx.ravel()] = i + 1  # Matlab indexing

    uy = np.unique(c_labels_)
    c_labels_ = np.asarray(c_labels_)
    c_labels = np.zeros(len(c_labels_), dtype='int32')
    for i in range(len(uy)):
        idx = c_labels_ == uy[i]
        c_labels[idx] = i + 1  # Matlab indexing

    tokenizer = Tokenizer(char_level=True)
    tokenizer.fit_on_texts(bcodes)
    sequence_of_int = tokenizer.texts_to_sequences(bcodes)

    ### For CUB data barcodes
    tokenizer = Tokenizer(char_level=True)
    tokenizer.fit_on_texts(cub_bcodes)
    sequence_of_int_cub = tokenizer.texts_to_sequences(cub_bcodes)

    # Bird data barcode lengths have a high variance and not as in INSECT data majority has seq length of 658
    # These barcodes are aligned ones as in INSECT data
    cnt = 0
    ll = np.zeros((len(bcodes), 1))
    for i in range(len(bcodes)):
        ll[i] = len(sequence_of_int[i])
        if ll[i] == 658:
            cnt += 1

    # Calculating sequence count for each species
    cl = len(np.unique(b_species))
    N = len(b_species)
    seq_cnt = np.zeros(cl)
    for i in range(N):
        k = labels[i] - 1
        seq_cnt[k] += 1
        # seq_len[k]+=len(sequence_of_int[k])

    sl = 1500  # Fixed to the max sequnce length. Note that, changing it to median value, 1548, does have an infitesimal effect
    N_cub = len(cub_bcodes)
    allX = np.zeros((N_cub, sl, 5))
    for i in range(N_cub):
        for j in range(sl):
            if len(sequence_of_int_cub[i]) > j:
                k = sequence_of_int_cub[i][j] - 1
                if k > 4:
                    k = 4
                allX[i][j][k] = 1.0
    trainX = np.zeros((N, sl, 5))
    trainY = np.zeros((N, cl))
    labelY = np.zeros(N)

    Nc = -1
    clas_cnt = np.zeros((cl))
    for i in range(N):
        k = labels[i] - 1
        clas_cnt[k] += 1
        if seq_cnt[k] >= 10 and clas_cnt[k] <= 50:
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
    labelY = labelY[0:Nc]

    idx = np.argwhere(np.all(trainY[..., :] == 0, axis=0))

    trainX = np.expand_dims(trainX, axis=3)
    allX = np.expand_dims(allX, axis=3)

    X_train, X_test, y_train, y_test = train_test_split(trainX, labelY, test_size=0.2, random_state=42)
    total_number_of_classes = 1213
    return X_train, X_test, y_train, y_test, torch.Tensor(allX), c_labels_, total_number_of_classes


def construct_dataloader(X_train, X_test, y_train, y_test, batch_size):
    X_train = torch.Tensor(X_train)
    X_test = torch.Tensor(X_test)
    y_train = torch.Tensor(y_train)
    y_test = torch.Tensor(y_test)

    train_dataset = TensorDataset(X_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    test_dataset = TensorDataset(X_test, y_test)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    return train_dataloader, test_dataloader


def get_embedding(model, all_X):
    embedding = None
    with torch.no_grad():
        for inputs in all_X:
            _, feature = model(torch.unsqueeze(inputs.to(device), 0))
            if embedding is None:
                embedding = feature
            else:
                embedding = torch.cat((embedding, feature), 0)
    embedding = embedding.to('cpu')
    return embedding


if __name__ == '__main__':
    X_train, X_test, y_train, y_test, all_X, species, total_number_of_classes = load_data()
    trainloader, testloader = construct_dataloader(X_train, X_test, y_train, y_test, 32)
    model = Model(1, total_number_of_classes, 4320, embedding_dim=400).to(device)
    train_and_eval(model, trainloader, testloader, device=device, lr=0.001, n_epoch=10)
    dna_embeddings = get_embedding(model, all_X)

    dict_emb = {}
    for index, label in enumerate(species):
        if str(label) not in dict_emb.keys():
            dict_emb[str(label)] = []
        dict_emb[str(label)].append(np.array(dna_embeddings[index]))
    all_label = np.unique(species)
    class_embed = []
    for i in all_label:
        class_embed.append(np.sum(dict_emb[str(i)], axis=0) / len(dict_emb[str(i)]))
    class_embed = np.array(class_embed, dtype=object)
    class_embed = class_embed.T

    np.savetxt("../data/CUB/dna_embedding.csv", class_embed, delimiter=",")
