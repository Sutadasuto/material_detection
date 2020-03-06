import argparse
import cv2
import numpy as np
import os
import pickle

from sklearn.metrics.pairwise import chi2_kernel
from sklearn.multiclass import OneVsRestClassifier as one_vs_all
from sklearn.svm import SVC as svm
from utilities.leung_malik import LeungMalik as LM
from utilities.codebooks import make_bot, get_cluster_centers


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_dir", type=str, default=None)
    parser.add_argument("--test_data_dir", type=str, default=None)
    parser.add_argument("--save_codebooks_to", type=str, default=None)
    parser.add_argument('--train_arrays', nargs=2, type=str)
    parser.add_argument('--test_arrays', nargs=2, type=str)
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

    train = False
    if args.train_data_dir is not None and args.train_arrays is None:
        train_image_paths = [os.path.join(args.train_data_dir, f) for f in os.listdir(args.train_data_dir) if
                             not f.startswith(".")]
        train_image_paths.sort()
        print("Extracting codebooks from training images.")
        train_data, train_labels = make_bot(train_image_paths, textons, lm.filter_image, save_to=args.save_codebooks_to)
        train = True
    elif args.train_arrays is not None:
        print("Loading train arrays.")
        train_data = np.load(args.train_arrays[0])
        train_labels = np.load(args.train_arrays[1])
        train = True

    if train:
        if args.classifier is None:
            clf = one_vs_all(svm(C=700, kernel=chi2_kernel))
        else:
            clf = pickle.load(args.classifier)
        print("Training one-vs-all classifier.")
        clf.fit(train_data, train_labels)
        pickle.dump(clf, open(os.path.join("models_and_data", "one_vs_all_model.p"), "wb"))
    else:
        if args.classifier is None:
            raise ValueError("Cannot test model if no trained model nor training data are provided.")
        clf = pickle.load(args.classifier)

    test = False
    if args.test_data_dir is not None and args.test_arrays is None:
        test_image_paths = [os.path.join(args.test_data_dir, f) for f in os.listdir(args.test_data_dir) if
                             not f.startswith(".")]
        test_image_paths.sort()
        print("Extracting codebooks from test images.")
        test_data, test_labels = make_bot(test_image_paths, textons, lm.filter_image, save_to=args.save_codebooks_to)
        test = True
    elif args.train_arrays is not None:
        print("Loading test arrays.")
        test_data = np.load(args.test_arrays[0])
        test_labels = np.load(args.test_arrays[1])
        test = True


if __name__ == "__main__":
    args = parse_args()
    main(args)
