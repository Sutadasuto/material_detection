import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml


def change_color_space(image, color_transformer=None, normalize_output=True, show_activations=False,
                       save_activations_to=None):
    if color_transformer is not None:
        image = cv2.cvtColor(image, color_transformer)

    if normalize_output:
        image = (image - image.mean()) / image.std()

    if show_activations:
        height, width, channels = image.shape
        cols = channels
        canvas = np.ones((height, cols * width + (cols - 1) * int(width / 10)))
        for channel in range(channels):
            x_0 = channel * (int(width / 10) + width)
            y_0 = 0
            filter = cv2.normalize(image[..., channel], None, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            canvas[y_0:y_0 + height, x_0:x_0 + width] = filter
        plt.imshow(canvas, cmap='gray')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    if save_activations_to is not None:
        height, width, channels = image.shape
        cols = channels
        canvas = np.ones((height, cols * width + (cols - 1) * int(width / 10)))
        for channel in range(channels):
            x_0 = channel * (int(width / 10) + width)
            y_0 = 0
            filter = cv2.normalize(image[..., channel], None, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            canvas[y_0:y_0 + height, x_0:x_0 + width] = filter
        plt.imshow(canvas, cmap='gray')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(save_activations_to, bbox_inches='tight')
        plt.close()

    return image


def create_arguments(callables_dict):
    yaml_file = open("models_arguments.yaml", "r")
    args_dict = yaml.load(yaml_file)

    kmeans_args = {} if args_dict['kmeans'] is None else args_dict['kmeans']
    for key in kmeans_args.keys():
        if kmeans_args[key] in callables_dict:
            kmeans_args[key] = callables_dict[kmeans_args[key]]

    filters_args = args_dict['filters']
    filters_args = [filters_args[filter] for filter in filters_args.keys() if filters_args[filter] is not None]
    filter_functions = []
    for filter_args in filters_args:
        for key in filter_args.keys():
            if key == "name":
                filter_functions.append(callables_dict[filter_args[key]])
            elif filter_args[key] in callables_dict:
                filter_args[key] = callables_dict[filter_args[key]]
        filter_args.pop("name")

    classifier_args = {} if args_dict['classifier'] is None else args_dict['classifier']
    for key in classifier_args.keys():
        if classifier_args[key] in callables_dict:
            classifier_args[key] = callables_dict[classifier_args[key]]

    return kmeans_args, tuple(filter_functions), tuple(filters_args), classifier_args
