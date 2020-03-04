import cv2
import numpy as np
import os
import pickle

from utilities.leung_malik import LeungMalik as LM
from utilities.codebooks import Textons

lm = LM()
textons = Textons(n_clusters=10)
train_data_dir = "/media/winbuntu/databases/GeoMat_just_im/train"

image_paths = [os.path.join(train_data_dir, f) for f in os.listdir(train_data_dir) if not f.startswith(".")]
image_paths.sort()

if not os.path.isfile("kmeans_model.p"):
    train_data = []
    for image_path in image_paths:
        processed_image = os.path.split(image_path)[-1]
        img = cv2.imread(image_path)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        train_data.append(textons.unroll(lm.filter_image(img, True)))
        print("Last processed: {}".format(processed_image), end='\r')

    textons.fit(np.concatenate(train_data, 0))
    pickle.dump(textons, open("kmeans_model.p", "wb"))
    del train_data

else:
    textons = pickle.load(open("kmeans_model.p", "rb"))

coded_vectors = []
labels = []
for image_path in image_paths:
    vector = np.zeros((1, textons.n_clusters))
    processed_image = os.path.split(image_path)[-1]
    img = cv2.imread(image_path)
    coded_image = textons.predict(lm.filter_image(img))
    unique_elements, counts_elements = np.unique(coded_image, return_counts=True)
    for element in range(unique_elements.shape[0]):
        vector[0, unique_elements[element]] = counts_elements[element]
    label = processed_image.split("_")[0].lower()
    coded_vectors.append(vector.astype(np.uint16))
    labels.append(np.array([[label]]))

dir_name = os.path.split(train_data_dir)[-1]
coded_vectors = np.concatenate(coded_vectors, 0).astype(np.uint16)
labels = np.concatenate(labels, 0)
np.save("%s_textons.npy" % dir_name, coded_vectors)
np.save("%s_labels.npy" % dir_name, labels)
