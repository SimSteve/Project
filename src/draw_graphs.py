import matplotlib.pyplot as plt
import yaml


def plot_graphs(filename, title):
    results = []
    for result in open(filename):
        results.append(yaml.safe_load(result))

    epochs = results[0]['epochs']
    epochs = [e + 1 for e in epochs]

    _, ax = plt.subplots()
    for i, result in enumerate(results):
        ax.plot(epochs, [r * 100 for r in result['acc']], label=result['name'])
    ax.set_xticks(epochs)
    ax.set_ylabel('accuracy (%)')
    ax.set_xlabel('epoch')
    ax.grid(True)
    ax.set_title(title)
    ax.legend(loc='lower right')
    plt.draw()


if __name__ == "__main__":
    plot_graphs("../json/unencrypted.json", "UNENCRYPTED MODELS")
    plot_graphs("../json/permutated.json", "ENCRYPTED BY PERMUTATION MODELS")


    '''
    plot_graphs("../json/unencrypted_mnist_models.json", "UNENCRYPTED MNIST MODELS")
    plot_graphs("../json/unencrypted_fashion_mnist_models.json", "UNENCRYPTED FASHION_MNIST MODELS")

    '''
    plt.show()
