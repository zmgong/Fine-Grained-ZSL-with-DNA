import math
import random
import json
import numpy as np
import scipy.io as sio
from scipy.linalg import eigh
import os
from sklearn.model_selection import train_test_split

"""Data loading part"""


class data_loader(object):
    def __init__(
        self, datapath, dataset, side_info="original", tuning=False, alignment=True, embeddings=None, use_genus=False, bioscan_clip_image_feature_not_fine_tuned_on_insect=False, bioscan_clip_image_feature=False, using_fine_turned_vit_feature=False, using_freeze_vit_feature = False
    ):
        print("The current working directory is")
        print(os.getcwd())
        self.datapath = os.path.join(datapath, dataset)  # '../data/'
        self.dataset = dataset
        self.side_info_source = side_info
        self.tuning = tuning
        self.alignment = alignment
        self.embeddings = embeddings
        self.use_genus = use_genus
        self.label_to_genus = None
        self.bioscan_clip_image_feature = bioscan_clip_image_feature
        self.bioscan_clip_image_feature_not_fine_tune_on_INSECT = bioscan_clip_image_feature_not_fine_tuned_on_insect
        self.using_fine_turned_vit_feature = using_fine_turned_vit_feature
        self.using_freeze_vit_feature = using_freeze_vit_feature
        self.read_matdata()

    def get_embeddings_path(self, embeddings: str, splits_mat):
        if embeddings:
            return embeddings

        if self.side_info_source == "dna_bioscan_clip":
            if self.alignment is True:
                print("Aligned")
                return os.path.join(self.datapath, "embeddings_from_bioscan_clip/dna_embedding_from_bioscan_clip.csv")
            else:
                exit("Not available: not aligned barcodes' feature from BioScan-CLIP")

        if self.side_info_source == "dna_bioscan_clip_no_fine_turned_on_INSECT":
            return os.path.join(self.datapath, "embeddings_from_bioscan_clip/dna_embedding_from_bioscan_clip_no_fine_tuned_on_INSECT.csv")

        if self.side_info_source == "dna_bioscan_clip_fine_turned_on_INSECT":
            return os.path.join(self.datapath, "embeddings_from_bioscan_clip/dna_embedding_from_bioscan_clip.csv")

        if self.side_info_source == "dna":
            if self.alignment is True:
                print("Aligned")
                return os.path.join(self.datapath, "dna_embedding.csv")
            else:
                print("Not aligned")
                return os.path.join(self.datapath, "dna_embedding_no_alignment.csv")
        elif self.side_info_source == "dna_pablo_bert":
            if self.alignment is True:
                print("Aligned")
                return os.path.join(self.datapath, "dna_embedding_using_bert_of_pablo_team.csv")
            else:
                print("Not aligned")
                return os.path.join(self.datapath, "dna_embedding_using_bert_of_pablo_team_no_alignment.csv")
        elif self.side_info_source == "dna_dnabert":
            if self.alignment is True:
                print("Aligned")
                return os.path.join(self.datapath, "dna_embedding_insect_dnabert_aligned.csv")
            else:
                print("Not aligned")
                return os.path.join(self.datapath, "dna_embedding_insect_dnabert.csv")
        elif self.side_info_source == "dna_dnabert2":
            if self.alignment is True:
                print("Aligned")
                return os.path.join(self.datapath, "dna_embedding_insect_dnabert2_aligned.csv")
            else:
                print("Not aligned")
                return os.path.join(self.datapath, "dna_embedding_insect_dnabert2.csv")
        elif self.side_info_source == "dna_pablo_bert_tuned":
            if self.alignment is True:
                print("Aligned")
                return os.path.join(self.datapath, "dna_embedding_supervised_fine_tuned_pablo_bert_aligned.csv")
            else:
                print("Not aligned")
                return os.path.join(self.datapath, "dna_embedding_supervised_fine_tuned_pablo_bert.csv")
        elif self.side_info_source == "dna_pablo_bert_mlm_tuned":
            if self.alignment is True:
                print("Aligned")
                return os.path.join(self.datapath, "dna_embedding_mlm_fine_tuned_pablo_bert_aligned.csv")
            else:
                print("Not aligned")
                return os.path.join(self.datapath, "dna_embedding_mlm_fine_tuned_pablo_bert.csv")

        elif self.side_info_source == "dna_pablo_bert_tuned_5_mer":
            if self.alignment is True:
                print("Aligned")
                return os.path.join(
                    self.datapath,

                    "dna_embedding_supervised_fine_tuned_pablo_bert_5_mer_ep_40_aligned.csv",
                )
            else:
                print("Not aligned")
                return os.path.join(self.datapath, "dna_embedding_supervised_fine_tuned_pablo_bert_5_mer_ep_40.csv")

        elif self.side_info_source == "dna_pablo_bert_pre_trained_on_5m_with_fine_tuning":
            path = os.path.join(self.datapath,"embedding_extract_from_new_BarcodeBERT", "dna_embedding_from_barcode_bert_pre_trained_on_BIOSCAN-5M_with_fine_tuning.csv")
            print(f"Using embeddings from {path}")
            return os.path.join(
                self.datapath,
                "embedding_extract_from_new_BarcodeBERT",
                "dna_embedding_from_barcode_bert_pre_trained_on_BIOSCAN-5M_with_fine_tuning.csv"
            )

        elif self.side_info_source == "dna_pablo_bert_pre_trained_on_5m_without_fine_tuning":
            return os.path.join(
                self.datapath,
                "embedding_extract_from_new_BarcodeBERT",
                "dna_embedding_from_barcode_bert_pre_trained_on_BIOSCAN-5M_without_fine_tuning.csv"
            )

        elif self.side_info_source == "dna_pablo_bert_pre_trained_on_canada_1_5M_with_fine_tuning":
            return os.path.join(
                self.datapath,
                "embedding_extract_from_new_BarcodeBERT",
                "dna_embedding_from_barcode_bert_pre_trained_on_CANADA-1.5M_with_fine_tuning.csv"
            )

        elif self.side_info_source == "dna_pablo_bert_pre_trained_on_canada_1_5M_without_fine_tuning":
            return os.path.join(
                self.datapath,
                "embedding_extract_from_new_BarcodeBERT",
                "dna_embedding_from_barcode_bert_pre_trained_on_CANADA-1.5M_without_fine_tuning.csv"
            )

        if self.side_info_source == "w2v":
            return splits_mat["att_w2v"]

    def read_matdata(self):
        self.label_to_species_dict = {}
        if self.dataset == "BIOSCAN_1M":
            path_to_meta = os.path.join(self.datapath, "bioscan_1m_data_in_insect_format.json")
            path_to_image_feature = os.path.join(self.datapath, "image_embed.csv")
            path_to_dna_feature = os.path.join(self.datapath, "dna_embed.csv")

            self.features = np.loadtxt(path_to_image_feature, delimiter=',')

            with open(path_to_meta, 'r') as f:
                metadata = json.load(f)
            self.labels = metadata['labels']
            self.num_species = max(self.labels) + 1
            self.train_loc = metadata['train_loc']
            self.val_seen_loc = metadata['val_seen_loc']
            self.val_unseen_loc = metadata['val_unseen_loc']
            self.trainval_loc = self.train_loc + self.val_seen_loc + self.val_unseen_loc
            self.test_seen_loc = metadata['test_seen_loc']
            self.test_unseen_loc = metadata['test_unseen_loc']

            self.labels = np.array(self.labels)
            self.train_loc = np.array(self.train_loc)
            self.val_seen_loc = np.array(self.val_seen_loc)
            self.val_unseen_loc = np.array(self.val_unseen_loc)
            self.trainval_loc = np.array(self.trainval_loc)
            self.test_seen_loc = np.array(self.test_seen_loc)
            self.test_unseen_loc = np.array(self.test_unseen_loc)
            self.side_info = np.loadtxt(path_to_dna_feature, delimiter=",")


            self.species = metadata['species']

            for idx, label in enumerate(self.labels):
                if label not in self.label_to_species_dict:
                    self.label_to_species_dict[label] = self.species[idx]

        else:
            path = os.path.join(self.datapath, "res101.mat")
            data_mat = sio.loadmat(path)
            if self.using_fine_turned_vit_feature is True:
                # Overwrite here to use fine-tuned ViT-B feature
                self.features = np.loadtxt(
                    os.path.join(self.datapath,
                                 "image_embedding_from_vit_fine_tuned_on_insect.csv"),
                    delimiter=',').T
            elif self.using_freeze_vit_feature is True:
                # Overwrite here to use fine-tuned ViT-B feature
                self.features = np.loadtxt(
                    os.path.join(self.datapath,
                                 "image_embedding_from_freeze_vit.csv"),
                    delimiter=',').T

            elif self.bioscan_clip_image_feature is True:
                # Overwrite here to use BioScan-CLIP feature
                self.features = np.loadtxt(
                    os.path.join(self.datapath, "embeddings_from_bioscan_clip", "image_embedding_from_bioscan_clip.csv"),
                    delimiter=',').T
            elif self.bioscan_clip_image_feature_not_fine_tune_on_INSECT is True:
                # Overwrite here to use BioScan-CLIP feature
                self.features = np.loadtxt(
                    os.path.join(self.datapath, "embeddings_from_bioscan_clip", "image_embedding_from_bioscan_clip_no_fine_tuned_on_INSECT.csv"),
                    delimiter=',').T
            else:
                if "features" in data_mat:
                    self.features = data_mat["features"].T
                else:
                    self.features = data_mat["embeddings_img"]

            # get labels
            self.labels = data_mat["labels"].ravel() - 1
            self.num_species = np.max(self.labels) + 1



            att_splits_path = os.path.join(self.datapath, "att_splits.mat")
            splits_mat = sio.loadmat(att_splits_path)

            self.trainval_loc = splits_mat["trainval_loc"].ravel() - 1
            self.train_loc = splits_mat["train_loc"].ravel() - 1
            self.val_unseen_loc = splits_mat.get("val_loc", splits_mat.get("val_unseen_loc")).ravel() - 1
            self.test_seen_loc = splits_mat["test_seen_loc"].ravel() - 1
            self.test_unseen_loc = splits_mat["test_unseen_loc"].ravel() - 1
            self.side_info = np.genfromtxt(self.get_embeddings_path(self.embeddings, splits_mat), delimiter=",")



            if self.use_genus:  # generate mapping of species classes to genera
                # find genus labels per sample
                # genus labels will start at max_label + 1 (e.g. for 1213 species classes, genus labels start at 1213)
                genera = [species[0][0].split()[0] for species in data_mat["species"]]
                unique_genera = np.unique(genera)
                genus_to_idx = dict(zip(unique_genera, self.num_species + np.arange(len(genera))))
                self.genus_labels = np.array(list(map(lambda g: genus_to_idx[g], genera)))

                # build mapping of species label to genus label
                self.label_to_genus = {}
                for label, genus in zip(self.labels, self.genus_labels):
                    if label not in self.label_to_genus:
                        self.label_to_genus[label] = genus
                    else:
                        assert (
                            self.label_to_genus[label] == genus
                        ), f"Found label which has multiple genera: {label=}, genera=[{self.label_to_genus[label]}, {genus}]"

            else:
                self.label_to_genus = {idx: idx for idx in range(self.num_species)}

    def get_label_to_species_dict(self):
        return self.label_to_species_dict
    def data_split(self):
        if self.tuning:
            if self.dataset != "BIOSCAN_1M":
                train_idx, test_seen_idx = crossvalind_holdout(self.labels, self.train_loc, 0.2)
            else:
                train_idx = self.train_loc
                test_seen_idx = self.val_seen_loc
            test_unseen_idx = self.val_unseen_loc
        else:
            train_idx = self.train_loc
            test_seen_idx = self.test_seen_loc
            test_unseen_idx = self.test_unseen_loc

        xtrain = self.features[train_idx]
        ytrain = self.labels[train_idx]
        xtest_seen = self.features[test_seen_idx]
        ytest_seen = self.labels[test_seen_idx]
        xtest_unseen = self.features[test_unseen_idx]
        ytest_unseen = self.labels[test_unseen_idx]

        self.seenclasses = np.unique(ytrain)
        self.unseenclasses = np.unique(ytest_unseen)

        # print(self.side_info.shape)

        # revise labels to use mix of genus and species
        if self.use_genus:
            assert self.label_to_genus is not None
            ytest_unseen = np.array([self.label_to_genus[x] for x in ytest_unseen])

        return xtrain, ytrain, xtest_seen, ytest_seen, xtest_unseen, ytest_unseen

    def load_tuned_params(self):
        if self.dataset not in ["INSECT", "CUB", "BIOSCAN_1M"]:
            raise Exception('The provided dataset is not in the gallery. Please use one of these 2 datsets to load tuned params: ["INSECT", "CUB"]')


        dim = 500

        if self.dataset == "INSECT":
            hyperparams = [0.1, 10, 12500, 10, 1]

        if self.dataset == "CUB":
            if self.side_info_source == "original":
                hyperparams = [1, 25, 500 * dim, 10, 3]
            elif self.side_info_source == "w2v":
                hyperparams = [0.1, 25, 5 * dim, 5, 2]
            elif self.side_info_source == "dna":
                hyperparams = [0.1, 25, 25 * dim, 5, 3]

        if self.dataset == "BIOSCAN_1M":
            hyperparams = [1.0, 10, 50000, 5, 2]



        return self.side_info, *hyperparams


### Seen, Unseen class and Harmonic mean claculation ###
def perf_calc_acc(y_ts_s, y_ts_us, ypred_s, ypred_us, label_to_genus=None):
    if label_to_genus:
        # we only need to do this for unseen, since seen classes will always be at species level
        ypred_us = np.array([[label_to_genus[int(x[0])]] for x in ypred_us])

    seen_cls = np.unique(y_ts_s)
    unseen_cls = np.unique(y_ts_us)



    # Performance calculation
    acc_per_cls_s = np.zeros((len(seen_cls), 1))
    acc_per_cls_us = np.zeros((len(unseen_cls), 1))

    for i in range(len(seen_cls)):
        lb = seen_cls[i]
        idx = y_ts_s == lb
        acc_per_cls_s[i] = np.sum(ypred_s[idx.ravel()] == lb) / np.sum(idx)

    for i in range(len(unseen_cls)):
        lb = unseen_cls[i]
        idx = y_ts_us == lb
        acc_per_cls_us[i] = np.sum(ypred_us[idx.ravel()] == lb) / np.sum(idx)

    ave_s = np.mean(acc_per_cls_s)
    ave_us = np.mean(acc_per_cls_us)
    H = 2 * ave_s * ave_us / (ave_s + ave_us)

    return acc_per_cls_s, acc_per_cls_us, ave_s, ave_us, H


def apply_pca(x_tr, x_ts_s, x_ts_us, pca_dim):
    # Dimentionality reduction using PCA
    _, eig_vec = eigh(np.cov(x_tr.T))
    x_tr = np.dot(x_tr, eig_vec[:, -pca_dim:])
    x_ts_s = np.dot(x_ts_s, eig_vec[:, -pca_dim:])
    x_ts_us = np.dot(x_ts_us, eig_vec[:, -pca_dim:])

    return x_tr, x_ts_s, x_ts_us


def crossvalind_holdout(labels, idxs, ratio=0.2):
    # Shuffle the index
    random.seed(42)
    random.shuffle(idxs)
    Y = labels[idxs]
    unique_Y, count = np.unique(Y, return_counts=True)
    count = dict(zip(unique_Y, count))
    train = []
    test = []
    for cls in unique_Y:
        threshold = math.ceil(ratio * count[cls])
        current_count = 0
        for idx, i in enumerate(Y):
            if i == cls:
                if current_count < threshold:
                    test.append(idxs[idx])
                else:
                    train.append(idxs[idx])
                current_count += 1
    return train, test


def split_loc_by_ratio(loc_index, ratio=0.2):
    n = int(len(loc_index) * ratio)
    np.random.shuffle(loc_index)
    test_loc = loc_index[0:n]
    train_loc = loc_index[n:]
    return np.sort(train_loc), np.sort(test_loc)
