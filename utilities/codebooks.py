import cv2
import numpy as np
import os
import pickle

from collections.abc import Iterable
from sklearn.cluster import KMeans


class Textons(KMeans):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def multiple_fit(self, samples):
        self.fit(np.concatenate(samples, -1))

    def unroll(self, features_3d):
        h, w, d = features_3d.shape
        return np.reshape(features_3d, (h*w, d))

    def predict(self, filtered_image):
        X = self.unroll(filtered_image)
        return super().predict(X)


def change_color_space(image, color_transformer=None):
    if color_transformer is None:
        return image
    return cv2.cvtColor(image, color_transformer)


def create_texton_instances(n_clusters, filters, concatenate, verbose, **kwargs):
    if not isinstance(n_clusters, Iterable):
        if type(n_clusters) is int:
            if n_clusters > 0:
                if concatenate is False:
                    if len(filters) == 1:
                        textons = Textons(n_clusters=n_clusters, verbose=verbose, **kwargs)
                    else:
                        textons = [Textons(n_clusters=n_clusters, verbose=verbose, **kwargs) for i in range(len(filters))]
                else:
                    textons = Textons(n_clusters=n_clusters, verbose=verbose, **kwargs)
            else:
                raise ValueError("Number of clusters must be a positive integer.")
        else:
            raise TypeError
    else:
        if concatenate:
            raise ValueError("As filter responses will be concatenated only a Texton object will be generated. No more than 1 number of clusters can be provided.")
        if len(n_clusters) != len(filters):
            raise ValueError("You should provide a single number of clusters for each filter (or a single int to be shatred among filters). No more, no less.")
        textons = []
        for n in n_clusters:
            if type(n) is not int:
                raise TypeError("Number of clusters must be a positive integer.")
            if n <= 0:
                raise ValueError("Number of clusters must be a positive integer.")
            textons.append(Textons(n_clusters=n, verbose=verbose, **kwargs))
    return textons


def get_cluster_centers(image_paths, n_clusters, filters, concatenate=False, verbose=1, kwargs_kmeans={}, kwargs_filters=({},)):
    if verbose == 1:
        print("Filter responses are being concatenated." if concatenate else "Filter responses are not being concatenated.")
        print("Creating textons.")
    textons = create_texton_instances(n_clusters, filters, concatenate, verbose, **kwargs_kmeans)

    train_data = []
    n_i = 0
    total_n_i = len(image_paths)
    for image_path in image_paths:
        processed_image = os.path.split(image_path)[-1]
        img = cv2.imread(image_path)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        if len(filters) == 1:
            train_data.append(textons.unroll(filters[0](img)))
        if verbose == 1:
            n_i += 1
            print("Last image processed ({}/{}): {}".format(n_i, total_n_i, processed_image))

    if verbose == 1:
        print("Calculating %s-D cluster centers." % train_data[0].shape[-1])
    textons.fit(np.concatenate(train_data, 0))
    pickle.dump(textons, open("kmeans_model.p", "wb"))
    if verbose == 1:
        print("K-means model saved to disk.")
    del train_data
    return textons


def make_bot(image_paths, texton_model, filter_function, save_to=None, **kwargs):
    data_dir = os.path.split(image_paths[0])[0]
    coded_vectors = []
    labels = []
    for image_path in image_paths:
        vector = np.zeros((1, texton_model.n_clusters))
        processed_image = os.path.split(image_path)[-1]
        img = cv2.imread(image_path)
        coded_image = texton_model.predict(filter_function(img, **kwargs))
        unique_elements, counts_elements = np.unique(coded_image, return_counts=True)
        for element in range(unique_elements.shape[0]):
            vector[0, unique_elements[element]] = counts_elements[element]
        label = processed_image.split("_")[0].lower()
        coded_vectors.append(vector.astype(np.uint16))
        labels.append(np.array([[label]]))

    if save_to is not None:
        dir_name = os.path.split(data_dir)[-1]
        coded_vectors = np.concatenate(coded_vectors, 0).astype(np.uint16)
        labels = np.concatenate(labels, 0)
        np.save("%s_bot.npy" % dir_name, coded_vectors)
        np.save("%s_labels.npy" % dir_name, labels)
    return coded_vectors, labels