from bayesian_model import Model
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--side_info", default=None, type=str)
parser.add_argument("--image_feature_path", default=None, type=str)
parser.add_argument("--pca_dim", default=500, type=int)
parser.add_argument("--datapath", default="../data/", type=str)
parser.add_argument("--dataset", default="INSECT", type=str) # BIOSCAN_1M
parser.add_argument("--tuning", default=False, action="store_true")
parser.add_argument("--alignment", default=False, action="store_true")
parser.add_argument("--using_bioscan_clip_image_feature", default=False, action="store_true")
parser.add_argument("--bioscan_clip_image_feature_not_fine_tuned_on_insect", default=False, action="store_true")
parser.add_argument("--using_fine_turned_vit_feature", default=False, action="store_true")
parser.add_argument("--using_freeze_vit_feature", default=False, action="store_true")
parser.add_argument("--embeddings", default=None, type=str)

parser.add_argument(
    "--k0",
    dest="k_0",
    default=None,
    type=float,
    help="scaling constant for dispersion of centers of metaclasses around mu_0",
)
parser.add_argument(
    "--k1",
    dest="k_1",
    default=None,
    type=float,
    help="scaling constant for dispersion of actual class means around corresponding metaclass means",
)
parser.add_argument(
    "-m",
    dest="m",
    default=None,
    type=int,
    help="defines dimension of Wishart distribution for sampling covariance matrices of metaclasses",
)
parser.add_argument("-s", dest="s", default=None, type=float, help="scalar for mean of class covariances")
parser.add_argument("-K", dest="K", default=None, type=int, help="number of most similar seen classes to find in BZSL")
parser.add_argument("--genus", action="store_true", help="if True, use genus labels for unseen species")
parser.add_argument("--output", default=None, type=str, dest="output", help="path to save final results after tuning")


"""
You may alter the model hyperparameters -- k_0, k_1, m, s, K -- inside train_and_eval() function from Model class.
This setting will reproduce the results presented in the paper 
"""

args = parser.parse_args()

model = Model(args)
model.train_and_eval()
