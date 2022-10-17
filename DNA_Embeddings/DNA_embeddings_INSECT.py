import scipy.io as sio
import os
import torch
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from CNN import Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data():
    datapath = r'../data'
    dataset = 'INSECT'
    x = sio.loadmat(os.path.join(datapath, dataset, 'res101.mat'))
    x2 = sio.loadmat(os.path.join(datapath, dataset, 'att_splits.mat'))
    barcodes = x['nucleotides_aligned'].T
    species = x['labels']
    train_loc = x2['trainval_loc']

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
    seq_len = np.zeros((cl))
    seq_cnt = np.zeros((cl))
    for i in range(N):
        k = labels[i] - 1  # Accounting for Matlab labeling
        seq_cnt[k] += 1
        # seq_len[k]+=len(sequence_of_int[k])

    # Converting all the data into one-hot encoding. Note that this data is not used during training.
    # We are getting it ready for the prediction time to get the final DNA embeddings after training is done
    sl = 658
    allX = np.zeros((N, sl, 5))
    for i in range(N):
        Nt = len(sequence_of_int[i])

        for j in range(sl):
            if (len(sequence_of_int[i]) > j):
                k = sequence_of_int[i][j] - 1
                if (k > 4):
                    k = 4
                allX[i][j][k] = 1.0

    # Initialize the training matrix and labels
    trainX = np.zeros((N, sl, 5))
    trainY = np.zeros((N, cl))
    labelY = np.zeros(N)

    Nc = -1
    clas_cnt = np.zeros((cl))
    for i in range(N):
        Nt = len(sequence_of_int[i])
        k = labels[i] - 1
        clas_cnt[k] += 1
        itl = i + 1
        if (seq_cnt[k] >= 10 and clas_cnt[k] <= 50 and itl in train_loc[
            0]):  # Note that samples from training set are only used
            Nc = Nc + 1
            for j in range(sl):
                if (len(sequence_of_int[i]) > j):
                    k = sequence_of_int[i][j] - 1
                    if (k > 4):
                        k = 4
                    trainX[Nc][j][k] = 1.0

            k = labels[i] - 1
            trainY[Nc][k] = 1.0
            labelY[Nc] = k

    trainX = trainX[0:Nc]
    trainY = trainY[0:Nc]
    labelY = labelY[0:Nc]

    # To make sure the training data does not include any unseen class nucleotides
    label_us = species[x2['test_unseen_loc'][0] - 1]
    np.intersect1d(np.unique(labelY), np.unique(label_us) - 1)

    # Cleaning empty classes
    idx = np.argwhere(np.all(trainY[..., :] == 0, axis=0))
    trainY = np.delete(trainY, idx, axis=1)

    # Expanding the training set shape for CNN
    trainX = np.expand_dims(trainX, axis=3)
    allX = np.expand_dims(allX, axis=3)

    X_train, X_test, y_train, y_test = train_test_split(trainX, labelY, test_size=0.2, random_state=42)


    total_number_of_classes = len(np.unique(labels))
    return X_train, X_test, y_train, y_test, torch.Tensor(allX), total_number_of_classes


def train_and_eval(model, trainloader, testloader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    print('start training')
    for epoch in range(3):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].type(torch.LongTensor).to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 0:
                correct = 0
                total = 0
                with torch.no_grad():
                    for data in testloader:
                        inputs, labels = data[0].to(device), data[1].type(torch.LongTensor).to(device)
                        labels = labels.int()
                        # calculate outputs by running images through the network
                        outputs, _ = model(inputs)
                        # the class with the highest energy is what we choose as prediction
                        _, predicted = torch.max(outputs.data, 1)
                        predicted = predicted.int()
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                print("Epoch: " + str(epoch) + " ||Iteration: " + str(i) + "|| loss: " + str(running_loss / 100) + "|| Accuracy: " + str(correct/total))
                running_loss = 0

    print('Finished Training')


def construct_dataloader(X_train, X_test, y_train, y_test, batch_size):
    X_train = torch.Tensor(X_train)
    X_test = torch.Tensor(X_test)
    y_train = torch.Tensor(y_train)
    y_test = torch.Tensor(y_test)

    train_dataset = TensorDataset(X_train,y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    test_dataset = TensorDataset(X_test,y_test)
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
    X_train, X_test, y_train, y_test, all_X, total_number_of_classes = load_data()
    trainloader, testloader = construct_dataloader(X_train, X_test, y_train, y_test, 32)

    model = Model(1, total_number_of_classes).to(device)
    train_and_eval(model, trainloader, testloader)
    dna_embeddings = get_embedding(model, all_X)
    np.savetxt("../data/INSECT/insect_dna_embedding.csv", dna_embeddings, delimiter=",")
