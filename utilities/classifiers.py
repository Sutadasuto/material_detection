import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.multiclass import OneVsRestClassifier as one_vs_all
from utilities.codebooks import make_bot, get_cluster_centers


def plot_conf_mat(classifier, x_test, y_test, title="Confusion matrix", normalization=True):
    normalization = "true" if normalization else None
    disp = plot_confusion_matrix(classifier, x_test, y_test, xticks_rotation="vertical", values_format="0.2f", cmap=plt.cm.Greys, normalize=normalization)
    disp.ax_.set_title(title)
    return disp


def train(args, textons, filter, base_classifier):
    train = False
    if args.train_data_dir is not None and args.train_arrays is None:
        train_image_paths = [os.path.join(args.train_data_dir, f) for f in os.listdir(args.train_data_dir) if
                             not f.startswith(".")]
        train_image_paths.sort()
        print("Extracting codebooks from training images.")
        train_data, train_labels = make_bot(train_image_paths, textons, filter, save_to=args.save_codebooks_to)
        train = True
    elif args.train_arrays is not None:
        print("Loading train arrays.")
        train_data = np.load(args.train_arrays[0])
        train_labels = np.load(args.train_arrays[1])
        train = True

    if train:
        if args.classifier is None:
            clf = one_vs_all(base_classifier)
        else:
            clf = pickle.load(args.classifier)
        print("Training one-vs-all classifier.")
        clf.fit(train_data, train_labels)
        if not os.path.exists("models"):
            os.makedirs("models")
        pickle.dump(clf, open(os.path.join("models", "one_vs_all_model.p"), "wb"))
    else:
        if args.classifier is None:
            raise ValueError("Cannot test model if no trained model nor training data are provided.")
        clf = pickle.load(args.classifier)
    return clf


def test(args, textons, filter, classifier, plot=False, save_plot_to=None):
    test = False
    if args.test_data_dir is not None and args.test_arrays is None:
        test_image_paths = [os.path.join(args.test_data_dir, f) for f in os.listdir(args.test_data_dir) if
                             not f.startswith(".")]
        test_image_paths.sort()
        print("Extracting codebooks from test images.")
        test_data, test_labels = make_bot(test_image_paths, textons, filter, save_to=args.save_codebooks_to)
        test = True
    elif args.train_arrays is not None:
        print("Loading test arrays.")
        test_data = np.load(args.test_arrays[0])
        test_labels = np.load(args.test_arrays[1])
        test = True
    if test:
        predicted_labels = classifier.predict(test_data)
        conf_mat = confusion_matrix(test_labels, predicted_labels)
        if plot or save_plot_to:
            ax = plot_conf_mat(classifier, test_data, test_labels)
        if save_plot_to is not None:
            if not os.path.exists(save_plot_to):
                os.makedirs(save_plot_to)
            plt.savefig(os.path.join(save_plot_to, "confusion_matrix.png"))
        if plot:
            plt.show()
        return conf_mat
    return None
