#  Copyright 2019 Martin Haesemeyer. All rights reserved.
#
# Licensed under the MIT license

"""
Module with general utility functions
"""


try:
    import Tkinter
    import tkFileDialog
except ImportError:
    import tkinter as Tkinter
    import tkinter.filedialog as tkFileDialog

import numpy as np


def ui_get_file(filetypes=None, multiple=False):
    """
    Shows a file selection dialog and returns the path to the selected file(s)
    """
    if filetypes is None:
        filetypes = [('Tiff stack', '.tif;.tiff')]
    options = {'filetypes': filetypes, 'multiple': multiple}
    Tkinter.Tk().withdraw()  # Close the root window
    return tkFileDialog.askopenfilename(**options)


def get_component_coordinates(matrix_a):
    """
    For each component in the sparse cnmf.estimates.A matrix, returns the x-y pixel coordinates
    of constituent pixels
    :param matrix_a: Sparse spatial component matrix of estimates object
    :return:
        [0]: List of length n_components of n_pixel*2 x[column]-y[row] coordinates
        [1]: List of coordinate weights
    """
    n_comp = matrix_a.shape[1]
    coordinates = []
    weights = []
    for i in range(n_comp):
        # re-transform sparse F flattened representation into image matrix
        im = matrix_a[:, i].toarray().reshape(512, 512, order='F')
        y, x = np.where(im > 0)
        w = im[y, x]
        coordinates.append(np.c_[x[:, None], y[:, None]])
        weights.append(w)
    return coordinates, weights


def get_component_centroids(matrix_a):
    """
    For each component in the sparse cnmf.estimates.A matrix, returns the x-y coordinates
    of the weighted centroid
    :param matrix_a: Sparse spatial component matrix of estimates object
    :return:  Array of n_components x 2 x/y centroid coordinates
    """
    n_comp = matrix_a.shape[1]
    centroids = np.full((n_comp, 2), np.nan)
    coords, weights = get_component_coordinates(matrix_a)
    for i, (c, w) in enumerate(zip(coords, weights)):
        centroids[i, :] = np.sum(c * w[:, None], 0) / w.sum()
    return centroids


def transform_pixel_coordinates(coords, z_plane, dxy_um, dz_um,):
    """
    Transforms pixel based 2-D coordinates into um based 3-D coordinates
    :param coords: n-components x 2 2D pixel coordinates
    :param z_plane: The corresponding stack z-plane
    :param dxy_um: Tuple of x and y resolution or scalar of combined resolution (um / pixel)
    :param dz_um: z-resolution in um distance between planes
    :return: n-components x 3 [x,y,z] 3D stack coordinates
    """
    if type(dxy_um) is tuple:
        dx = dxy_um[0]
        dy = dxy_um[1]
    else:
        dx = dxy_um
        dy = dxy_um
    z = np.full(coords.shape[0], z_plane*dz_um)
    dxy = np.array([dx, dy])[None, :]
    return np.c_[coords*dxy, z]
