from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.datasets import mnist
from keras.optimizers import Adam
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer

def MNIST_CNN(EPOCHS, BATCH):

    # Accedemos a MNIST mediante Keras. 
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Redimensionamos los datos.
    X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))
    X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))

    # Normalizamos los datos para evitar problemas. Los convertimos en floats de 32-bits.
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    # Convertimos las etiquetas de su versión numérica a una versión one-hot encoded.
    label_binarizer = LabelBinarizer()
    y_train = label_binarizer.fit_transform(y_train)
    y_test = label_binarizer.fit_transform(y_test)

    model = Sequential()
    model.add(Conv2D(32, kernel_size = (3, 3), activation = 'relu', input_shape = (28, 28, 1)))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation = 'relu'))
    model.add(Dense(10, activation = 'softmax'))

    adam = Adam()
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH, validation_split=0.1, validation_data=(X_test, y_test))

    predictions = model.predict(X_test)
    print()
    print(classification_report(y_test.argmax(axis=1), predictions.argmax(axis=1), target_names=[str(x) for x in label_binarizer.classes_]))
    print()
    evaluacion = model.evaluate(X_test, y_test, batch_size=BATCH)
    print()
    print("test loss, test acc:", evaluacion)
    print()

if __name__ == "__main__":
    MNIST_CNN(15, 128)