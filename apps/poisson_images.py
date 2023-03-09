import numpy as np
import os
import matplotlib.pyplot as plt
from stgp_poisson import triang, apps_path


def get_poisson_images():
    # correct path
    path = apps_path[:apps_path.rfind("apps")]
    # load saved vectors
    train_fit_history = np.load(os.path.join(path, "train_fit_history.npy"))
    val_fit_history = np.load(os.path.join(path, "val_fit_history.npy"))
    best_sol_test_0 = np.load(os.path.join(path, "best_sol_test_0.npy"))
    best_sol_test_1 = np.load(os.path.join(path, "best_sol_test_1.npy"))
    best_sol_test_2 = np.load(os.path.join(path, "best_sol_test_2.npy"))
    true_sol_test_0 = np.load(os.path.join(path, "true_sol_test_0.npy"))
    true_sol_test_1 = np.load(os.path.join(path, "true_sol_test_1.npy"))
    true_sol_test_2 = np.load(os.path.join(path, "true_sol_test_2.npy"))
    best_sol_test = [best_sol_test_0, best_sol_test_1, best_sol_test_2]
    true_sol_test = [true_sol_test_0, true_sol_test_1, true_sol_test_2]

    # make images
    x = range(1, len(train_fit_history) + 1)
    plt.plot(x, train_fit_history, 'b', label="Training Fitness")
    plt.plot(x, val_fit_history, 'r', label="Validation Fitness")
    plt.xticks(np.arange(min(x), max(x)+1, 1.0))
    plt.legend(loc='upper right')
    plt.xlabel("Generation #")
    plt.ylabel("Best Fitness")
    plt.show()

    _, axes = plt.subplots(2, 3, num=10)
    plt.figure(10, figsize=(8, 4))
    fig = plt.gcf()
    for i in range(0, 3):
        axes[0, i].tricontourf(triang, best_sol_test[i], cmap='RdBu', levels=20)
        pltobj = axes[1, i].tricontourf(
            triang, true_sol_test[i], cmap='RdBu', levels=20)
        axes[0, i].set_box_aspect(1)
        axes[1, i].set_box_aspect(1)
    plt.colorbar(pltobj, ax=axes)
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.show()


if __name__ == "__main__":
    get_poisson_images()
