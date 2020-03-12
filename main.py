import argparse
import cv2
import numpy as np
import os
import pickle
import yaml

from sklearn.svm import SVC as SVM
from utilities.miscellaneous import create_arguments, change_color_space
from sklearn.metrics.pairwise import chi2_kernel
from utilities import classifiers
from utilities.leung_malik import LeungMalik as LM
from utilities.codebooks import fit_texton_instances

callables_dict = {
    "chi2_kernel": chi2_kernel,
    "leung_malik": LM().lm_responses,
    "change_color_space": change_color_space,
    "BGR2HSV": cv2.COLOR_BGR2HSV
}
kwargs_kmeans, filters, kwargs_filters, kwargs_classifier = create_arguments(callables_dict)


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_dir", type=str, default=None)
    parser.add_argument("--test_data_dir", type=str, default=None)
    parser.add_argument("--n_clusters", nargs="+", type=int, default=[15])
    parser.add_argument("--cluster_train_data_dir", type=str, default="same")
    parser.add_argument("--save_codebooks_to", type=str, default=None)
    parser.add_argument('--train_arrays', nargs=3, type=str, default=None)
    parser.add_argument('--test_arrays', nargs=3, type=str, default=None)
    parser.add_argument("--textons_models", type=str, default=None)
    parser.add_argument("--classifiers", type=str, default=None)
    parser.add_argument("--concatenate_features", type=bool, default=False)
    args_dict = parser.parse_args(args)
    if len(args_dict.n_clusters) == 1:
        args_dict.n_clusters = args_dict.n_clusters[0]
    return args_dict


def main(args):

    textons = fit_texton_instances(args, kwargs_kmeans, filters, kwargs_filters)

    trained_clf = classifiers.train(args, textons, filters, kwargs_filters, SVM(**kwargs_classifier), normalize=True,
                                    save_filter_outputs=True)
    conf_mat = classifiers.test(args, textons, filters, kwargs_filters, trained_clf, plot=True,
                                save_plot_to=os.getcwd(), normalize=True, save_filter_outputs=True)


if __name__ == "__main__":
    args = parse_args()
    main(args)
