import copy

import scipy.io as sio
import os
import numpy as np
from Bio import AlignIO
from Bio import SeqIO
from Bio import Align
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord



def main():
    datapath = '../data'
    dataset = 'INSECT'
    data_mat = sio.loadmat(os.path.join(datapath, dataset, 'res101.mat'))
    splits_mat = sio.loadmat(os.path.join(datapath, dataset, 'att_splits.mat'))
    nucleotides = data_mat['nucleotides']
    trainval_loc = splits_mat['trainval_loc'][0]
    tmp_list = []
    for i in nucleotides:
        tmp_list.append(str(i[0][0]))
    nucleotides = tmp_list

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
    lnt = []
    for i in nucleotides:
        lnt.append(len(i))

    xtrain = [nucleotides[i - 1] for i in trainval_loc]
    lnt = np.zeros((len(xtrain), 1))
    lntd = np.zeros((len(xtrain), 1))
    filtered = []
    for index, seq in enumerate(xtrain):
        lnt[index] = seq.count('A') + seq.count('G') + seq.count('C') + seq.count('T')
        lntd[index] = seq.count('-') + seq.count('N')
        if lnt[index] == 658 and lntd[index] == 0:
            filtered.append(seq)

    # records = (SeqRecord(Seq(seq, generic_dna), str(index)) for index,seq in enumerate(sequence_set) )

    my_seqs = []
    for id, seq in enumerate(filtered):
        my_seqs.append(SeqRecord(Seq(seq), id = str(id)))

    SeqIO.write(my_seqs, "tmp.fasta", "fasta")

    alignments = AlignIO.parse(open('tmp.fasta', 'r'), 'fasta')
    for alignment in alignments:
        print(alignment)


if __name__ == '__main__':
    main()
