import os
import struct
import numpy as np
import matplotlib.pyplot as plt

def load_mnist(path, kind='train'):
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte' %kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte' %kind)

    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))

        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
        images = ((images / 255.) - .5) * 2

    return images, labels

def viewer():
    fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True)
    ax = ax.flatten()
    for i in range(10):
        img = X_train[y_train == i][0].reshape(28, 28)
        ax[i].imshow(img, cmap='Greys')

    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()

    plt.show()

    fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True)
    ax = ax.flatten()
    for i in range(25):
        img = X_train[y_train == 7][i].reshape(28,28)
        ax[i].imshow(img, cmap='Greys')
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()

def save_n_load(X_train, y_train, X_test, y_test):
    x_tr = X_train[:55000]
    x_val = X_train[55000:]
    y_tr = y_train[:55000]
    y_val = y_train[55000:]
    x_ts = X_test
    y_ts = y_test
    np.savez_compressed('mnist_scaled.npz', X_train=x_tr, y_train=y_tr, X_valid=x_val, y_valid=y_val, X_test=x_ts, y_test=y_ts)

    mnist = np.load('mnist_scaled.npz')
    print(mnist.files)
    X_train, y_train, X_valid, y_valid, X_test, y_test = [mnist[f] for f in mnist.files]
    print(X_train.shape, y_train.shape, X_valid.shape, y_valid.shape, X_test.shape, y_test.shape)


if __name__ == "__main__":
    X_train, y_train = load_mnist('', kind='train')
    print('행: %d, 열: %d' %(X_train.shape[0], X_train.shape[1]))
    x_train = X_train[:55000]
    x_valid = X_train[55000:]
    print(x_train.shape, x_valid.shape)
    X_test, y_test = load_mnist('', kind='t10k')
    print('행: %d, 열: %d' %(X_test.shape[0], X_test.shape[1]))

    # viewer()
    save_n_load(X_train, y_train, X_test, y_test)