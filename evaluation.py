from sklearn import metrics
import numpy as np

def evaluation(model, X_test, Y_test_num):

    # make a set of predictions for the test data
    pred = np.argmax(model.predict(X_test), axis=-1)

    # print performance details
    print(metrics.classification_report(Y_test_num, pred))