import numpy as np
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



