import matplotlib.pyplot as plt


def visualize(true, pred, title, low, high):
    plt.plot([low, high], [low, high], color='gray', ls='--')
    plt.scatter(true, pred)
    plt.title(title, fontsize=16)
    plt.xlabel('Истинные значения')
    plt.ylabel('Предсказанные значения')
    plt.show()
