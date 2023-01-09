from keras import backend as K


def f1(y_true, y_pred):
    def recall():
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall_ = true_positives / (possible_positives + K.epsilon())
        return recall_

    def precision():
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision_ = true_positives / (predicted_positives + K.epsilon())
        return precision_

    precision = precision()
    recall = recall()
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))
