import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np
from scipy.interpolate import Rbf
from scipy.spatial.distance import cdist
from scipy.special import xlogy


"""
Using Scipy to fit and evaluate a RBF, including thin-plate spline.
Our own implementation using the Scipy code and fitted weights.

Basically fits a weight per input (train) point.
I can probably learn a similar thing in Pytorch. Most of the scipy functions used below have equivalent pytorch functions.
"""


def thin_plate_spline_fn(r, epsilon=None):
    return xlogy(r ** 2, r)  # xlogy => x * log(y). torch.xlogy


def gaussian_fn(r, epsilon):
    return np.exp(-((r / epsilon) ** 2))


def evaluate_rbf(xi, yi, weights, x, y, norm, basis_fn, epsilon=None):
    # Points used to fit the RBF.
    xy_train = np.asarray([x, y], dtype=np.float_)
    # Test points.
    xy_test = np.asarray([xi.flatten(), yi.flatten()], dtype=np.float_)
    # Distance between each test point and each train point.  torch.cdist
    r = cdist(xy_test.T, xy_train.T, norm)
    # Weighted sum of the result of applying the TPS function to the distances.
    if basis_fn == 'gaussian':
        basis_fn = gaussian_fn
    elif basis_fn == 'thin_plate':
        basis_fn = thin_plate_spline_fn
    else:
        print(f'Unknown basis fn {basis_fn}')
    results = np.dot(basis_fn(r, epsilon), weights)
    # Reshape to the input dimensions. These results should be the same as the di results above.
    results = results.reshape(xi.shape)

    return results


def plot_surface_points(surface_x, surface_y, surface_d, points_x, points_y, points_d):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(surface_x, surface_y, surface_d, cmap=cm.coolwarm, linewidth=0, antialiased=False, alpha=0.5)
    ax.scatter(points_x, points_y, points_d, c='orange')

    # Customize the z axis.
    #ax.set_zlim(-0.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter('{x:.02f}')

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)


def main():
    x = [0.25, 0.75, 0.75, 0.25]
    y = [0.25, 0.25, 0.75, 0.75]
    d = [0.5, 0.2, 0.8, 0.1]

    xi = np.arange(0, 1, 0.1)
    yi = np.arange(0, 1, 0.1)
    xi, yi = np.meshgrid(xi, yi)

    norm = 'euclidean'
    basis_fn = 'gaussian'
    epsilon = 0.25

    rbfi = Rbf(x, y, d, function=basis_fn, norm=norm)  # radial basis function interpolator instance
    di = rbfi(xi, yi)  # interpolated values

    if 1:
        results = evaluate_rbf(xi, yi, rbfi.nodes, x, y, norm, basis_fn=basis_fn, epsilon=epsilon)
    elif 1:
        # Our own RBF evaluation using the scipy parameters.
        # Code taken from the scipy Rbf code.

        # The learned weights. One value per input (training) point.
        weights = rbfi.nodes
        # Points used to fit the RBF.
        xy_train = np.asarray([x, y], dtype=np.float_)
        # Test points.
        xy_test = np.asarray([xi.flatten(), yi.flatten()], dtype=np.float_)
        # Distance between each test point and each train point.  torch.cdist
        r = cdist(xy_test.T, xy_train.T, norm)
        # Weighted sum of the result of applying the TPS function to the distances.
        results = np.dot(thin_plate_spline_fn(r), weights)
        # Reshape to the input dimensions. These results should be the same as the di results above.
        results = results.reshape(xi.shape)

    plot_surface_points(xi, yi, di, x, y, d)

    plt.show()

if __name__ == '__main__':
    main()
