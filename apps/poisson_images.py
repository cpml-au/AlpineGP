import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from stgp_poisson import apps_path
from alpine.data import poisson_dataset as d


def get_poisson_images():
    # correct path
    _, _, triang = d.generate_complex("test3.msh")
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

    # make and save images
    plt.figure(figsize=(8, 4), dpi=300)
    x = range(1, len(train_fit_history) + 1)
    plt.plot(x, train_fit_history, 'b', label="Training Fitness")
    plt.plot(x, val_fit_history, 'r', label="Validation Fitness")
    plt.tick_params(axis='both', which='major', labelsize=5)
    plt.tick_params(axis='both', which='minor', labelsize=5)
    plt.xticks(np.arange(min(x), max(x)+1, 1.0))
    #plt.legend(loc='upper right')
    plt.xlabel("Generation #")
    plt.ylabel("Best Fitness")
    plt.savefig("fitness.png", dpi=300)
    plt.show()

    _, axes = plt.subplots(2, 3, figsize=(8, 4), num=10)
    fig = plt.gcf()
    for i in range(0, 3):
        axes[0, i].tricontourf(triang, best_sol_test[i], cmap='RdBu', levels=20)
        pltobj = axes[1, i].tricontourf(
            triang, true_sol_test[i], cmap='RdBu', levels=20)
        axes[0, i].set_box_aspect(1)
        axes[1, i].set_box_aspect(1)
        # set the size of tick labels
        axes[0, i].tick_params(axis='both', which='major', labelsize=5)
        axes[0, i].tick_params(axis='both', which='minor', labelsize=5)
        axes[1, i].tick_params(axis='both', which='major', labelsize=5)
        axes[1, i].tick_params(axis='both', which='minor', labelsize=5)

    cb = plt.colorbar(pltobj, ax=axes)
    cb.ax.tick_params(labelsize='small')
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.savefig("test_performance.png", dpi=300)
    plt.show()

    '''
    fig = plt.figure(figsize=(10, 5), constrained_layout=True)
    grid = gridspec.GridSpec(2, 5, wspace=0.2, hspace=0.2)
    fit_axis = fig.add_subplot(grid[:, 0])
    x = range(1, len(train_fit_history) + 1)
    fit_axis.plot(x, train_fit_history, 'b', label="Training Fitness")
    fit_axis.plot(x, val_fit_history, 'r', label="Validation Fitness")
    fit_axis.tick_params(axis='both', which='major', labelsize=5)
    fit_axis.tick_params(axis='both', which='minor', labelsize=5)
    fit_axis.set_xticks(np.arange(min(x), max(x)+1, 1.0))
    fit_axis.set_xlabel("Generation #")
    fit_axis.set_ylabel("Best Fitness")
    for i in range(3):
        test_computed_axis = fig.add_subplot(grid[0, i+1])
        test_true_axis = fig.add_subplot(grid[1, i+1])
        computed_plot = test_computed_axis.tricontourf(
            triang, best_sol_test[i], cmap='RdBu', levels=20)
        true_plot = test_true_axis.tricontourf(
            triang, true_sol_test[i], cmap='RdBu', levels=20)
        #plt.colorbar(computed_plot, shrink=0.5)
        #plt.colorbar(true_plot, shrink=0.5)
        test_computed_axis.set_box_aspect(1)
        test_true_axis.set_box_aspect(1)
        # set the size of tick labels
        test_computed_axis.tick_params(axis='both', which='major', labelsize=5)
        test_computed_axis.tick_params(axis='both', which='minor', labelsize=5)
        test_true_axis.tick_params(axis='both', which='major', labelsize=5)
        test_true_axis.tick_params(axis='both', which='minor', labelsize=5)

    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.show()
    '''


if __name__ == "__main__":
    get_poisson_images()
