"""Plot module."""
import matplotlib.pyplot as plt
from keras.callbacks import History


def plot_keras_history(history: History):
    """Plot history."""
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'y', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()


def plot_pytorch_history(history: dict):
    train_acc = history['acc']['train']
    val_acc = history['acc']['val']
    train_loss = history['loss']['train']
    val_loss = history['loss']['val']

    epochs = range(1, len(train_acc) + 1)

    plt.plot(epochs, train_acc, 'bo-', label='training acc')
    plt.plot(epochs, val_acc, 'y', label='val acc')
    plt.title('training and val acc')
    plt.legend()

    plt.figure()

    plt.plot(epochs, train_loss, 'bo-', label='training loss')
    plt.plot(epochs, val_loss, 'y', label='val loss')
    plt.title('training and val loss')
    plt.legend()

    plt.show()