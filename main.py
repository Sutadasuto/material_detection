import cv2
import numpy as np
import os

from utilities.leung_malik import LeungMalik as LM
from utilities.codebooks import Textons
from utilities.codebooks import make_bot, get_cluster_centers

lm = LM()
train_data_dir = "/media/winbuntu/databases/GeoMat_just_im/dummy"

image_paths = [os.path.join(train_data_dir, f) for f in os.listdir(train_data_dir) if not f.startswith(".")]
image_paths.sort()

if not os.path.isfile("kmeans_model.p"):
    textons = get_cluster_centers(image_paths, 10, (lm.filter_image,))

# else:
#     textons = pickle.load(open("kmeans_model.p", "rb"))

make_bot(image_paths, textons, lm.filter_image, save_to=os.getcwd())
