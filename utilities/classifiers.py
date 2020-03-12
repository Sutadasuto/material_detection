import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.multiclass import OneVsRestClassifier as one_vs_all
from utilities.codebooks import make_bot


def plot_conf_mat(classifier, x_test, y_test, title="Confusion matrix", normalization=True):
    normalization = "true" if normalization else None
    disp = plot_confusion_matrix(classifier, x_test, y_test, xticks_rotation="vertical", values_format="0.2f",
                                 cmap=plt.cm.Greys, normalize=normalization)
    disp.ax_.set_title(title)
    return disp


def train(args, textons, filters, filters_args, base_classifier, normalize=False, save_filter_outputs=False):
    if len(filters) != len(textons) and not args.concatenate_features:
        raise ValueError("The number of provided fitlers is different than the number of textos models.")
    train = False
    if args.train_data_dir is not None and args.train_arrays is None:
        train_image_paths = [os.path.join(args.train_data_dir, f) for f in os.listdir(args.train_data_dir) if
                             not f.startswith(".")]
        train_image_paths.sort()
        print("Extracting codebooks from training images.")
        train_data, train_labels, applied_filters = make_bot(train_image_paths, textons, filters, filters_args,
                                                             concatenate=args.concatenate_features,
                                                             save_to=args.save_codebooks_to,
                                                             normalize=normalize,
                                                             save_filter_outputs=save_filter_outputs)
        train = True
    elif args.train_arrays is not None:
        print("Loading train arrays.")
        train_data = np.load(args.train_arrays[0])
        train_labels = np.load(args.train_arrays[1])
        applied_filters = np.load(args.train_arrays[2])
        train = True

    if train:
        n_filters, n_samples, n_features = train_data.shape
        if args.classifiers is None:
            clfs = []
        else:
            clfs = pickle.load(args.classifiers)

        for filter in range(n_filters):
            if args.classifiers is None:
                clfs.append([one_vs_all(base_classifier), applied_filters[filter]])
            print("Training one-vs-all classifier number %s." % filter)
            clfs[filter][0].fit(train_data[filter], train_labels)

        if not os.path.exists("models"):
            os.makedirs("models")
        pickle.dump(clfs, open(os.path.join("models", "one_vs_all_models.p"), "wb"))
        print("Models saved to '%s'" % os.path.join("models", "one_vs_all_models.p"))
    else:
        if args.classifiers is None:
            raise ValueError("Cannot test model if no trained models nor training data are provided.")
        clfs = pickle.load(open(args.classifiers, "rb"))
    return clfs


def test(args, textons, filters, filters_args, classifiers, plot=False, save_plot_to=None, normalize=False,
         save_filter_outputs=False):
    test = False
    if args.test_data_dir is not None and args.test_arrays is None:
        test_image_paths = [os.path.join(args.test_data_dir, f) for f in os.listdir(args.test_data_dir) if
                            not f.startswith(".")]
        test_image_paths.sort()
        print("Extracting codebooks from test images.")
        test_data, test_labels, applied_filters = make_bot(test_image_paths, textons, filters, filters_args,
                                                           concatenate=args.concatenate_features,
                                                           save_to=args.save_codebooks_to,
                                                           normalize=normalize, save_filter_outputs=save_filter_outputs)
        test = True
    elif args.test_arrays is not None:
        print("Loading test arrays.")
        test_data = np.load(args.test_arrays[0])
        test_labels = np.load(args.test_arrays[1])
        test = True
    if test:
        conf_mats = []
        for filter, classifier in enumerate(classifiers):
            predicted_labels = classifier[0].predict(test_data[filter])
            print("Calculating confusion matrix.")
            conf_mat = confusion_matrix(test_labels, predicted_labels)
            if plot or save_plot_to:
                print("Plotting confusion matrix.")
                ax = plot_conf_mat(classifier[0], test_data[filter], test_labels)
            if save_plot_to is not None:
                print("Saving plot to '%s'" % os.path.join(save_plot_to, "confusion_matrix_%s.png" % classifier[1]))
                if not os.path.exists(save_plot_to):
                    os.makedirs(save_plot_to)
                plt.savefig(os.path.join(save_plot_to, "confusion_matrix_%s.png" % classifier[1]), bbox_inches='tight')
            if plot:
                plt.show()
            conf_mats.append([conf_mat, classifier[1]])
        return conf_mats
    return None
