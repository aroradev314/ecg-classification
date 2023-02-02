from keras import backend as K

def f1(y_true, y_pred):
    def recall():
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall_ = true_positives / (possible_positives + K.epsilon())
        return recall_

    def precision():
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision_ = true_positives / (predicted_positives + K.epsilon())
        return precision_

    precision = precision()
    recall = recall()
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

