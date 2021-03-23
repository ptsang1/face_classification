class Classification:
    def __init__(self, name='base_class'):
        """
        :param name: the class name
        """
        self._name = name
        self._data = None
        self._labels = None

    def predict(self, test):
        """
        :param test: the data need(s) to be predicted
        :return: the corresponding label(s)
        """
        pass

    def pre_processing(self, data):
        """
        :param data: the data need(s) to be preprocessed
        :return: the transformed data
        """
        pass

    def training(self, data, labels, input_shape, **kwargs):
        """
        :param data: The training data
        :param labels: The corresponding labels
        :param input_shape: The shape of input (The shape of the transformed data)
        :return:
        """
        pass
