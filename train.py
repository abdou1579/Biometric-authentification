import keras
import keras.backend as K
from keras import Model
from keras.models import Sequential
from keras.layers import Input, Softmax, Conv1D, Dense, Dropout, ReLU, MaxPooling1D, Flatten
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import LearningRateScheduler
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt


def modelling(X_train,Y_train_wide):
    input_shape = (X_train.shape[1], X_train.shape[2])
    inputs = Input(shape=input_shape)
    x = Conv1D(16, 7)(inputs)
    x = ReLU()(x)
    x = MaxPooling1D(pool_size=3, strides=2)(x)

    x = Conv1D(32 ,5)(x)
    x = ReLU()(x)
    x = MaxPooling1D(pool_size=3, strides=2)(x)

    x = Conv1D(64, 5)(x)
    x = ReLU()(x)
    x = MaxPooling1D(pool_size=3, strides=2)(x)

    x = Conv1D(128, 7)(x)
    x = ReLU()(x)

    x = Conv1D(128, 7)(x)
    x = ReLU()(x)

    x = Conv1D(128, 8)(x)
    x = ReLU()(x)
    x = Flatten()(x)
    x = Dense(num_classes)(x)

    predictions = Softmax()(x)

    model = Model(inputs=inputs, outputs=predictions)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()


    batch_size = 32
    epochs = 10

    best_weights_filepath = './best_weights.hdf5'
    mcp = ModelCheckpoint(best_weights_filepath, monitor="val_accuracy",
                        save_best_only=True, save_weights_only=False)

    history = model.fit(X_train,Y_train_wide,
            batch_size=batch_size,
            epochs=epochs,
            verbose = 1,
            validation_split = 0.2,
            shuffle=True,
            callbacks=[mcp])

    model.load_weights(best_weights_filepath)

    model.save('model.h5')

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(loss, 'blue', label='Training Loss')
    plt.plot(val_loss, 'green', label='Validation Loss')
    plt.xticks(range(0,epochs)[0::2])
    plt.legend()
    plt.show()

    return model