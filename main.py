import argparse
import cv2
import numpy as np
import os
import pickle

from sklearn.metrics import confusion_matrix
from sklearn.metrics.pairwise import chi2_kernel
from sklearn.multiclass import OneVsRestClassifier as one_vs_all
from sklearn.svm import SVC as svm
from utilities import classifiers
from utilities.leung_malik import LeungMalik as LM
from utilities.codebooks import make_bot, get_cluster_centers


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_dir", type=str, default=None)
    parser.add_argument("--test_data_dir", type=str, default=None)
    parser.add_argument("--save_codebooks_to", type=str, default=None)
    parser.add_argument('--train_arrays', nargs=2, type=str, default=None)
    parser.add_argument('--test_arrays', nargs=2, type=str, default=None)
    parser.add_argument("--textons_model", type=str, default=None)
    parser.add_argument("--classifier", type=str, default=None)
    return parser.parse_args(args)


def main(args):
    lm = LM()

    if args.train_data_dir is not None:
        train_image_paths = [os.path.join(args.train_data_dir, f) for f in os.listdir(args.train_data_dir) if not f.startswith(".")]
        train_image_paths.sort()

    if args.textons_model is not None and os.path.isfile(args.textons_model) and args.textons_model.endswith(".p"):
        textons = pickle.load(open(args.textons_model, "rb"))
    else:
        print("No valid textons model path provided. Creating model from training data.")
        if args.train_data_dir is None:
            raise ValueError("No train data provided.")
        textons = get_cluster_centers(train_image_paths, 15, (lm.filter_image,))

    trained_clf = classifiers.train(args, textons, lm.filter_image, svm(C=700, kernel=chi2_kernel))
    conf_mat = classifiers.test(args, textons, lm.filter_image, trained_clf, plot=True, save_plot_to=os.getcwd())


if __name__ == "__main__":
    args = parse_args()
    main(args)
