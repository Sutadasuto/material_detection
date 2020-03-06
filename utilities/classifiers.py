from sklearn.svm import SVC as SVM


class CodebookSVM(SVM):

    def __init__(self, texton_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.texton_model = texton_model
