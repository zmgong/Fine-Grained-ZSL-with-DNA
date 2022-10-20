import copy
import math

import scipy.io as sio
import os
import numpy as np
from Bio import AlignIO
from Bio import SeqIO
from Bio import Align
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment


def seqneighjoin(D):
    new_D = [[D[1], D[2], D[1]], [D[0], D[1], D[2]], [D[1], D[2], D[1]]]
    D = np.array(new_D, dtype='object')
    N = len(D)
    Z = np.zeros((N - 1, 2))
    bd = np.zeros((N * 2 - 1, 1))
    p = np.array(list(range(N)), dtype=object)
    bc = 0
    for n in range(N, 2, -1):
        R = np.sum(D, axis=0) / (n - 2)
        R_a = np.array([list(R), list(R), list(R)], dtype=object)
        R_b = np.array([[R[0], R[0], R[0]], [R[1], R[1], R[1]], [R[2], R[2], R[2]]], dtype=object)
        inf_matrix = np.array([[np.inf, 0, 0], [0, np.inf, 0], [0, 0, np.inf]], dtype=object)
        Q = D - R_a - R_b + inf_matrix
        row_i, col_j = np.argmin(np.min(Q)), np.argmin(Q)
        if row_i > col_j:
            k = row_i
            row_i = col_j
            col_j = k
        pp = np.array([row_i, col_j])
        flat_R = R.flatten()
        partR = np.zeros((1, 2))
        partR[0][0] = flat_R[row_i]
        partR[0][1] = flat_R[col_j]
        # print(partR.shape)
        # print(np.array([[1, -1], [-1, 1]]).shape)
        # exit()

        bl = np.matmul(partR, np.array([[1, -1], [-1, 1]])) + np.array([D[col_j][row_i], D[col_j][row_i]])
        bl = np.divide(bl, 2)
        bl = bl[0]
        for ind, val in enumerate(bl):
            if val < 0:
                bl[ind] = 0
            bd[pp[ind]] = bl[ind]
        Z[bc] = pp
        h = list(range(n))
        h.remove(row_i)
        h.remove(col_j)

        d = (sum(np.hstack((D[h][:, row_i], D[h][:, col_j]))) - D[col_j][
            row_i]) / 2  # mat contain bug when the dimension of input changes
        # but should work fine in this script
        if d < 0:
            d = 0
        D = np.array([[D[h[0]][h[0]], d], [d, 0]], dtype=object)
        p = np.append(np.take(p, h), N + bc)
        bc = bc + 1
    Z[bc, :] = p
    bd[p] = D[1]/2



    # print(Z)


def seqprofile(P):
    profile = np.zeros((21, 658))
    alphabet = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    for i in range(len(P[0])):
        dict = {}
        sum = 0
        for letter in alphabet:
            dict[letter] = 0
        for seq in P:
            dict[seq[i]] += 1
            sum += 1
        for index, ch in enumerate(alphabet):
            dict[ch] = dict[ch] / sum
            profile[index][i] = dict[ch]
    return profile


def seqconsensus(P):
    # altered from matlab bioinformatics toolbox seqconsensus, with 'gaps','noflanks'
    isAAalpha = True
    alpha = 'aa'
    gaps = 'none'
    ambiguous = 'ignore'
    # limits = [0, math.inf]
    numericSMProvided = False
    predefSMProvided = False
    gapScoresIncluded = False
    gaps = 'noflanks'
    pname = 'gaps'
    ScoringMatrix = [[5, -2, -1, -2, -1, -1, -1, 0, -2, -1, -2, -1, -1, -3, -1, 1, 0, -3, -2, 0, -2, -1, -1, -5],
                     [-2, 7, -1, -2, -4, 1, 0, -3, 0, -4, -3, 3, -2, -3, -3, -1, -1, -3, -1, -3, -1, 0, -1, -5],
                     [-1, -1, 7, 2, -2, 0, 0, 0, 1, -3, -4, 0, -2, -4, -2, 1, 0, -4, -2, -3, 4, 0, -1, -5],
                     [-2, -2, 2, 8, -4, 0, 2, -1, -1, -4, -4, -1, -4, -5, -1, 0, -1, -5, -3, -4, 5, 1, -1, - 5],
                     [-1, -4, -2, -4, 13, -3, -3, -3, -3, -2, -2, -3, -2, -2, -4, -1, -1, -5, -3, -1, -3, -3, -2, -5],
                     [-1, 1, 0, 0, -3, 7, 2, -2, 1, -3, -2, 2, 0, -4, -1, 0, -1, -1, -1, -3, 0, 4, -1, -5],
                     [-1, 0, 0, 2, -3, 2, 6, -3, 0, -4, -3, 1, -2, -3, -1, -1, -1, -3, -2, -3, 1, 5, -1, -5],
                     [0, -3, 0, -1, -3, -2, -3, 8, -2, -4, -4, -2, -3, -4, -2, 0, -2, -3, -3, -4, -1, -2, -2, -5],
                     [-2, 0, 1, -1, -3, 1, 0, -2, 10, -4, -3, 0, -1, -1, -2, -1, -2, -3, 2, -4, 0, 0, -1, -5],
                     [-1, -4, -3, -4, -2, -3, -4, -4, -4, 5, 2, -3, 2, 0, -3, -3, -1, -3, -1, 4, -4, -3, -1, -5],
                     [-2, -3, -4, -4, -2, -2, -3, -4, -3, 2, 5, -3, 3, 1, -4, -3, -1, -2, -1, 1, -4, -3, -1, -5],
                     [-1, 3, 0, -1, -3, 2, 1, -2, 0, -3, -3, 6, -2, -4, -1, 0, -1, -3, -2, -3, 0, 1, -1, -5],
                     [-1, -2, -2, -4, -2, 0, -2, -3, -1, 2, 3, -2, 7, 0, -3, -2, -1, -1, 0, 1, -3, -1, -1, -5],
                     [-3, -3, -4, -5, -2, -4, -3, -4, -1, 0, 1, -4, 0, 8, -4, -3, -2, 1, 4, -1, -4, -4, -2, -5],
                     [-1, -3, -2, -1, -4, -1, -1, -2, -2, -3, -4, -1, -3, -4, 10, -1, -1, -4, -3, -3, -2, -1, -2, -5],
                     [1, -1, 1, 0, -1, 0, -1, 0, -1, -3, -3, 0, -2, -3, -1, 5, 2, -4, -2, -2, 0, 0, -1, -5],
                     [0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 2, 5, -3, -2, 0, 0, -1, 0, -5],
                     [-3, -3, -4, -5, -5, -1, -3, -3, -3, -3, -2, -3, -1, 1, -4, -4, -3, 15, 2, -3, -5, -2, -3, -5],
                     [-2, -1, -2, -3, -3, -1, -2, -3, 2, -1, -1, -2, 0, 4, -3, -2, -2, 2, 8, -1, -3, -2, -1, -5],
                     [0, -3, -3, - 4, -1, -3, -3, -4, -4, 4, 1, -3, 1, -1, -3, -2, 0, -3, -1, 5, -4, -3, -1, -5],
                     [-2, -1, 4, 5, -3, 0, 1, -1, 0, -4, -4, 0, -3, -4, -2, 0, 0, -5, -3, -4, 5, 2, -1, -5],
                     [-1, 0, 0, 1, -3, 4, 5, -2, 0, -3, -3, 1, -1, -4, -1, 0, -1, -2, -2, -3, 2, 5, -1, -5, ],
                     [-1, -1, -1, -1, -2, -1, -1, -2, -1, -1, -1, -1, -1, -2, -2, -1, 0, -3, -1, -1, -1, -1, -1, -5],
                     [-5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, 1]]
    tmp = []
    for i in ScoringMatrix:
        tmp.append(i[0:20])

    ScoringMatrix = tmp[0:20]
    P = seqprofile(P)
    ScoringMatrix = np.asarray(ScoringMatrix, dtype="object")
    SMs = len(ScoringMatrix[0])
    if P.shape[0] == 5 or P.shape[0] == 21:
        sum_of_each_row = [sum(x) for x in ScoringMatrix]
        diag = ScoringMatrix.diagonal()
        temp = (sum_of_each_row - diag) / (SMs - 1)
        new_ScoringMatrix = np.zeros((21, 21))
        new_ScoringMatrix[0:SMs, 0:SMs] = ScoringMatrix
        new_ScoringMatrix[SMs, SMs] = np.mean(ScoringMatrix.diagonal())
        ScoringMatrix = new_ScoringMatrix
        for i in range(SMs):
            ScoringMatrix[i][SMs] = temp[i]
            ScoringMatrix[SMs][i] = temp[i]
        SMs = SMs + 1
        # TODO
        pass
    X = np.matmul(ScoringMatrix, P)
    X = np.argmax(X, axis=0)
    alphabet = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    proto = ""
    for x in X:
        proto += alphabet[x]
    return proto


def multialign(iseqs):
    tr = "existingGapAdjust"
    d = [p_distance(iseqs[0], iseqs[1]), p_distance(iseqs[1], iseqs[1]), p_distance(iseqs[2], iseqs[1])]

    tr = seqneighjoin(d)
    exit()
    pass


def p_distance(seq_1, seq_2):
    # p-distance (Nucleotide)
    count = 0

    for i in range(len(seq_1)):
        if seq_1[i] != seq_2[i]:
            count += 1
    return count / len(seq_1)


def main():
    datapath = '../data'
    dataset = 'INSECT'

    data_mat = sio.loadmat(os.path.join(datapath, dataset, 'res101.mat'))
    splits_mat = sio.loadmat(os.path.join(datapath, dataset, 'att_splits.mat'))

    sio.loadmat('nucleotides_aligned')
    nucleotides = data_mat['nucleotides']
    trainval_loc = splits_mat['trainval_loc'][0]
    tmp_nucleotides = []
    for i in nucleotides:
        tmp_nucleotides.append(str(i[0][0]))
    nucleotides = tmp_nucleotides

    for index, nuc in enumerate(nucleotides):
        c = 0
        tmp = copy.copy(nuc)
        for i in nuc:
            if i is 'N' or i is '-':
                tmp = tmp[1:]
            else:
                break
        c = len(tmp) - 1
        tmpi = tmp[c]
        while tmpi is 'N' or tmpi is '-':
            tmp = tmp[0:-1]
            c -= 1
            tmpi = tmp[c]
        nucleotides[index] = tmp
    # lnt = []
    # for i in nucleotides:
    #     lnt.append(len(i))

    xtrain = [nucleotides[i - 1] for i in trainval_loc]
    lnt = np.zeros((len(xtrain), 1))
    lntd = np.zeros((len(xtrain), 1))
    filtered = []
    for index, seq in enumerate(xtrain):
        lnt[index] = seq.count('A') + seq.count('G') + seq.count('C') + seq.count('T')
        lntd[index] = seq.count('-') + seq.count('N')
        if lnt[index] == 658 and lntd[index] == 0:
            filtered.append(seq)

    proto = seqconsensus(filtered)

    for i, nucleotide in enumerate(nucleotides):
        tmp = [proto, nucleotide, proto]
        alg = multialign(tmp)


if __name__ == '__main__':
    main()
