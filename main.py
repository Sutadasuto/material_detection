import argparse
import cv2
import numpy as np
import os
import pickle
import yaml

from utilities.classifiers import create_arguments
from sklearn.metrics.pairwise import chi2_kernel
from sklearn.svm import SVC as svm
from utilities import classifiers
from utilities.leung_malik import LeungMalik as LM
from utilities.codebooks import make_bot, get_cluster_centers

callables_dict = {"chi2_kernel": chi2_kernel}
kwargs_kmeans, kwargs_filters, kwargs_classifier = create_arguments(callables_dict)


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_dir", type=str, default=None)
    parser.add_argument("--test_data_dir", type=str, default=None)
    parser.add_argument("--n_clusters", type=int, default=15)
    parser.add_argument("--cluster_train_data_dir", type=str, default="same")
    parser.add_argument("--save_codebooks_to", type=str, default=None)
    parser.add_argument('--train_arrays', nargs=2, type=str, default=None)
    parser.add_argument('--test_arrays', nargs=2, type=str, default=None)
    parser.add_argument("--textons_model", type=str, default=None)
    parser.add_argument("--classifier", type=str, default=None)
    return parser.parse_args(args)


def main(args):
    lm = LM()

    if args.textons_model is not None and os.path.isfile(args.textons_model) and args.textons_model.endswith(".p"):
        textons = pickle.load(open(args.textons_model, "rb"))
    else:
        print("No valid textons model path provided. Creating model from training data.")
        if args.cluster_train_data_dir == "same":
            if args.train_data_dir is None:
                raise ValueError("No directory providing training images was provided.")
            train_image_paths = [os.path.join(args.train_data_dir, f) for f in os.listdir(args.train_data_dir) if
                                 not f.startswith(".")]
            train_image_paths.sort()
            textons = get_cluster_centers(train_image_paths, args.n_clusters, (lm.filter_image,),
                                          kwargs_kmeans=kwargs_kmeans, kwargs_filters=kwargs_filters)
        else:
            cluster_train_image_paths = [os.path.join(args.cluster_train_data_dir, f) for f in
                                         os.listdir(args.cluster_train_data_dir) if
                                         not f.startswith(".")]
            cluster_train_image_paths.sort()
            textons = get_cluster_centers(cluster_train_image_paths, args.n_clusters, (lm.filter_image,),
                                          kwargs_kmeans=kwargs_kmeans, kwargs_filters=kwargs_filters)

    trained_clf = classifiers.train(args, textons, lm.filter_image, svm(**kwargs_classifier))
    conf_mat = classifiers.test(args, textons, lm.filter_image, trained_clf, plot=True, save_plot_to=os.getcwd())


if __name__ == "__main__":
    args = parse_args()
    main(args)
