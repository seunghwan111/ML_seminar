from MLP import *
from data_load import *
import matplotlib.pyplot as plt


if __name__ == '__main__':

    mnist = np.load('mnist_scaled.npz')
    # print(mnist.files)
    y_train, X_test, y_valid, y_test, X_train, X_valid = [mnist[f] for f in mnist.files]

    # print(X_train.shape)
    # print(X_train.shape, y_train.shape, X_valid.shape, y_valid.shape, X_test.shape, y_test.shape)


    nn = NeuralNetMLP(n_hidden=100, l2=0.1, epochs=200, eta=0.0005, minibatch_size=100, shuffle=True, seed=1)

    nn.fit(X_train=X_train, y_train=y_train, X_valid=X_valid, y_valid=y_valid)

    plt.plot(range(nn.epochs), nn.eval_['cost'])
    plt.ylabel('Cost')
    plt.xlabel('Epochs')
    plt.show()

    plt.plot(range(nn.epochs), nn.eval_['train_acc'], label='training')
    plt.plot(range(nn.epochs), nn.eval_['valid_acc'], label='validation', linestyle='--')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()

    y_test_pred = nn.predict(X_test)
    acc = (np.sum(y_test == y_test_pred).astype(np.float) / X_test.shape[0])
    print('테스트 정확도: %.2f%%' %(acc * 100))

    miscl_img = X_test[y_test != y_test_pred][:25]
    correct_lab = y_test[y_test != y_test_pred][:25]
    miscl_lab = y_test_pred[y_test != y_test_pred][:25]

    fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True)
    ax = ax.flatten()
    for i in range(25):
        img = miscl_img[i].reshape(28,28)
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')
        ax[i].set_title('%d) t: %d p: %d' %(i+1, correct_lab[i], miscl_lab[i]))

    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()