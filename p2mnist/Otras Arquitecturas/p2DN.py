from keras.models import Model
from keras.layers import Input, Dense, concatenate, BatchNormalization
from keras.datasets import mnist
from keras.optimizers import Adam
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer

def MNIST(EPOCHS, BATCH):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape((X_train.shape[0], 28 * 28 * 1))
    X_test = X_test.reshape((X_test.shape[0], 28 * 28 * 1))
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    label_binarizer = LabelBinarizer()
    y_train = label_binarizer.fit_transform(y_train)
    y_test = label_binarizer.fit_transform(y_test)

    input_layer = Input(shape = (28 * 28 * 1,))

    x = Dense(128, activation = 'relu')(input_layer)
    x = BatchNormalization()(x)
    x = Dense(128, activation = 'relu')(x)

    x = concatenate([input_layer, x])
    output_layer = Dense(10, activation = 'softmax')(x)

    model = Model(inputs = input_layer, outputs = output_layer)

    adam = Adam()
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH, validation_split=0.1, validation_data=(X_test, y_test))
    evaluacion = model.evaluate(X_test, y_test, batch_size=BATCH)
    print("test loss, test acc:", evaluacion)

if __name__ == "__main__":
    MNIST(15, 128)
