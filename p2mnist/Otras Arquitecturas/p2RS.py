from builtins import str
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Input, Add
from keras.models import Model
from keras.optimizers import Adam
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer

def MNIST_ResNet(EPOCHS, BATCH):
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    X_train = X_train.reshape((X_train.shape[0], 28 * 28 * 1))
    X_test = X_test.reshape((X_test.shape[0], 28 * 28 * 1))
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    label_binarizer = LabelBinarizer()
    y_train = label_binarizer.fit_transform(y_train)
    y_test = label_binarizer.fit_transform(y_test)
    
    # Capa de entrada
    input_layer = Input(shape = (28 * 28 * 1,))

    # Bloque residual
    x = Dense(784, activation = 'relu')(input_layer)
    x = Dense(784, activation = 'relu')(x)
    residual = x

    # Añadir la suma residual a la entrada
    x = Add()([input_layer, residual])
    x = Dense(10, activation = 'softmax')(x)

    # Modelo
    model = Model(inputs = input_layer, outputs = x)

    adam = Adam()
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH, validation_split=0.1, validation_data=(X_test, y_test))
    evaluacion = model.evaluate(X_test, y_test, batch_size=BATCH)
    print("test loss, test acc:", evaluacion)

if __name__ == "__main__":
    MNIST_ResNet(15, 128)