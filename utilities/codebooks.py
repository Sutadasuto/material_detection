import cv2
import numpy as np
import os
import pickle

from collections.abc import Iterable
from sklearn.cluster import MiniBatchKMeans


class BotNormalizer(object):
    def __init__(self):
        self.ready = True

    def normalize(self, data):
        n_samples, n_features = data.shape
        data = data.astype(np.float32)
        for sample in range(n_samples):
            n_terms = np.sum(data[sample, :])
            for feature in range(n_features):
                data[sample, feature] = data[sample, feature] / n_terms
        return data


class Textons(MiniBatchKMeans):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.normalize = None

    def multiple_fit(self, samples):
        self.fit(np.concatenate(samples, -1))

    def unroll(self, features_3d):
        h, w, d = features_3d.shape
        return np.reshape(features_3d, (h * w, d))

    def predict(self, filtered_image):
        X = self.unroll(filtered_image)
        return super().predict(X)


def create_texton_instances(n_clusters, filters, concatenate, **kwargs):
    if not isinstance(n_clusters, Iterable):
        if type(n_clusters) is int:
            if n_clusters > 0:
                if concatenate is False:
                    if len(filters) == 1:
                        textons = [Textons(n_clusters=n_clusters, **kwargs)]
                    else:
                        textons = [Textons(n_clusters=n_clusters, **kwargs) for i in range(len(filters))]
                else:
                    textons = [Textons(n_clusters=n_clusters, **kwargs)]
            else:
                raise ValueError("Number of clusters must be a positive integer.")
        else:
            raise TypeError
    else:
        if concatenate:
            raise ValueError(
                "As filter responses will be concatenated only a Texton object will be generated. No more than 1 number of clusters can be provided.")
        if len(n_clusters) != len(filters):
            raise ValueError(
                "You should provide a single number of clusters for each filter (or a single int to be shatred among "
                "filters). No more, no less.")
        textons = []
        for n in n_clusters:
            if type(n) is not int:
                raise TypeError("Number of clusters must be a positive integer.")
            if n <= 0:
                raise ValueError("Number of clusters must be a positive integer.")
            textons.append(Textons(n_clusters=n, **kwargs))
    return textons


def fit_texton_instances(args, kwargs_kmeans, filters, kwargs_filters):
    if args.textons_models is not None and os.path.isfile(args.textons_models) and args.textons_models.endswith(".p"):
        textons = pickle.load(open(args.textons_models, "rb"))
    else:
        print("No valid textons model path provided. Creating model from training data.")
        if args.cluster_train_data_dir == "same":
            if args.train_data_dir is None:
                raise ValueError("No directory providing training images was provided.")
            train_image_paths = [os.path.join(args.train_data_dir, f) for f in os.listdir(args.train_data_dir) if
                                 not f.startswith(".")]
            train_image_paths.sort()
            textons = get_cluster_centers(train_image_paths, args.n_clusters, filters, args.concatenate_features,
                                          kwargs_kmeans,
                                          kwargs_filters)
        else:
            cluster_train_image_paths = [os.path.join(args.cluster_train_data_dir, f) for f in
                                         os.listdir(args.cluster_train_data_dir) if
                                         not f.startswith(".")]
            cluster_train_image_paths.sort()
            textons = get_cluster_centers(cluster_train_image_paths, args.n_clusters, filters,
                                          args.concatenate_features,
                                          kwargs_kmeans, kwargs_filters)
    return textons


def get_cluster_centers(image_paths, n_clusters, filters, concatenate=False, kwargs_kmeans={}, kwargs_filters=({},)):
    if not os.path.exists("models"):
        os.makedirs("models")
    print("Filter responses are being concatenated." if concatenate else "Filter responses are not being concatenated.")
    print("Creating textons.")
    textons = create_texton_instances(n_clusters, filters, concatenate, **kwargs_kmeans)

    if len(filters) == 1:
        train_data = []
        multiple_textons = False
    elif not concatenate:
        train_data = [[] for i in range(len(filters))]
        multiple_textons = True
    else:
        train_data = []
        multiple_textons = False
    n_i = 0
    total_n_i = len(image_paths)
    for image_path in image_paths:
        processed_image = os.path.split(image_path)[-1]
        img = cv2.imread(image_path)
        if len(filters) == 1:
            train_data.append(np.ascontiguousarray(textons[0].unroll(filters[0](img, **kwargs_filters[0]))))
        else:
            if not concatenate:
                for idx, filter in enumerate(filters):
                    train_data[idx].append(
                        np.ascontiguousarray(textons[idx].unroll(filter(img, **kwargs_filters[idx]))))
            else:
                feature_sets = []
                for idx, filter in enumerate(filters):
                    feature_sets.append(filter(img, **kwargs_filters[idx]))
                train_data.append(np.ascontiguousarray(textons[0].unroll(np.concatenate(feature_sets, -1))))
        n_i += 1
        print("Last image processed ({}/{}): {}".format(n_i, total_n_i, processed_image))

    if not multiple_textons:
        print("Calculating %s-D cluster centers." % train_data[0].shape[-1])
        textons[0].fit(np.ascontiguousarray(np.concatenate(train_data, 0)))
        pickle.dump(textons, open(os.path.join("models", "texton_model.p"), "wb"))
        print("K-textons model saved to disk.")
    else:
        for texton_model in range(len(train_data)):
            print("Calculating %s-D cluster centers for textons model %s." % (train_data[texton_model][0].shape[-1], (texton_model + 1)))
            textons[texton_model].fit(np.ascontiguousarray(np.concatenate(train_data[texton_model], 0)))
        pickle.dump(textons, open(os.path.join("models", "texton_models.p"), "wb"))
        print("K-textons models saved to disk.")
    return textons


def make_bot(image_paths, texton_models, filter_functions, filters_arguments, concatenate=False, normalize=False,
             save_to=None,
             save_filter_outputs=False):
    if save_to is not None and not os.path.exists(save_to):
        os.makedirs(save_to)
    data_dir = os.path.split(image_paths[0])[0]
    data = []
    filters = []
    for idx, texton_model in enumerate(texton_models):
        coded_vectors = []
        labels = []
        filter_name = filter_functions[idx].__name__
        for image_path in image_paths:
            vector = np.zeros((1, texton_model.n_clusters))
            processed_image = os.path.split(image_path)[-1]
            img = cv2.imread(image_path)
            if concatenate:
                if save_filter_outputs:
                    containing_dir = os.path.split(image_path)[0]
                    for filter in range(len(filter_functions)):
                        filter_name = filter_functions[filter].__name__
                        if not os.path.exists(containing_dir + "_" + filter_name):
                            os.makedirs(containing_dir + "_" + filter_name)
                        save_path = image_path.replace(containing_dir, containing_dir + "_" + filter_name)
                        filters_arguments[filter]["save_activations_to"] = save_path

                coded_image = texton_model.predict(np.concatenate(
                    [filter_functions[filter](img, **filters_arguments[filter]) for filter in
                     range(len(filter_functions))], axis=-1))
            else:
                if save_filter_outputs:
                    containing_dir = os.path.split(image_path)[0]
                    if not os.path.exists(containing_dir + "_" + filter_name):
                        os.makedirs(containing_dir + "_" + filter_name)
                    save_path = image_path.replace(containing_dir, containing_dir + "_" + filter_name)
                    filters_arguments[idx]["save_activations_to"] = save_path
                coded_image = texton_model.predict(filter_functions[idx](img, **filters_arguments[idx]))
            unique_elements, counts_elements = np.unique(coded_image, return_counts=True)
            for element in range(unique_elements.shape[0]):
                vector[0, unique_elements[element]] = counts_elements[element]
            label = processed_image.split("_")[0].lower()
            coded_vectors.append(vector.astype(np.uint16))
            labels.append(np.array([[label]]))

        coded_vectors = np.concatenate(coded_vectors, 0)
        if normalize:
            coded_vectors = BotNormalizer().normalize(coded_vectors)
        else:
            coded_vectors = coded_vectors.astype(np.uint16)
        data.append(np.expand_dims(coded_vectors, axis=0))
        filters.append([filter_name])

    labels = np.concatenate(labels, 0)
    data = np.concatenate(data, 0)
    if concatenate:
        filters = [" + ".join([filter_functions[filter].__name__ for filter in range(len(filter_functions))])]
    filters = np.array(filters)
    if save_to is not None:
        dir_name = os.path.split(data_dir)[-1]
        np.save(os.path.join(save_to, "%s_bots.npy" % dir_name), data)
        np.save(os.path.join(save_to, "%s_filter_names.npy" % dir_name), filters)
        np.save(os.path.join(save_to, "%s_labels.npy" % dir_name), labels)
        print("Data and label arrays saved to '%s' and '%s', respectively." % (
            os.path.join(save_to, "%s_bot.npy" % dir_name), os.path.join(save_to, "%s_labels.npy" % dir_name)))

    return data, labels, filters
