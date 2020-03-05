import argparse
import cv2
import os

from utilities.leung_malik import LeungMalik as LM
from utilities.codebooks import make_bot, get_cluster_centers


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str)
    return parser.parse_args(args)


def main(args):
    lm = LM()

    image_paths = [os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir) if not f.startswith(".")]
    image_paths.sort()

    if not os.path.isfile("kmeans_model.p"):
        textons = get_cluster_centers(image_paths, 10, (lm.filter_image,))
    # else:
    #     textons = pickle.load(open("kmeans_model.p", "rb"))

    make_bot(image_paths, textons, lm.filter_image, save_to=os.getcwd())


if __name__ == "__main__":
    args = parse_args()
    main(args)
