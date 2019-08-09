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
import shutil
import tempfile
import subprocess as sp
from os import path


def ui_get_file(filetypes=None, multiple=False):
    """
    Shows a file selection dialog and returns the path to the selected file(s)
    """
    if filetypes is None:
        filetypes = [('Tiff stack', '.tif;.tiff')]
    options = {'filetypes': filetypes, 'multiple': multiple}
    Tkinter.Tk().withdraw()  # Close the root window
    return tkFileDialog.askopenfilename(**options)


def get_component_coordinates(matrix_a, im_dim_0: int, im_dim_1: int):
    """
    For each component in the sparse cnmf.estimates.A matrix, returns the x-y pixel coordinates
    of constituent pixels
    :param matrix_a: Sparse spatial component matrix of estimates object
    :param im_dim_0: The size of the first dimension of the original image stack
    :param im_dim_1: The size of the second dimension of the original image stack
    :return:
        [0]: List of length n_components of n_pixel*2 x[column]-y[row] coordinates
        [1]: List of coordinate weights
    """
    n_comp = matrix_a.shape[1]
    coordinates = []
    weights = []
    for i in range(n_comp):
        # re-transform sparse F flattened representation into image matrix
        im = matrix_a[:, i].toarray().reshape(im_dim_0, im_dim_1, order='F')
        y, x = np.where(im > 0)
        w = im[y, x]
        coordinates.append(np.c_[x[:, None], y[:, None]])
        weights.append(w)
    return coordinates, weights


def get_component_centroids(matrix_a, im_dim_0: int, im_dim_1: int):
    """
    For each component in the sparse cnmf.estimates.A matrix, returns the x-y coordinates
    of the weighted centroid
    :param matrix_a: Sparse spatial component matrix of estimates object
    :param im_dim_0: The size of the first dimension of the original image stack
    :param im_dim_1: The size of the second dimension of the original image stack
    :return:  Array of n_components x 2 x/y centroid coordinates
    """
    n_comp = matrix_a.shape[1]
    centroids = np.full((n_comp, 2), np.nan)
    coords, weights = get_component_coordinates(matrix_a, im_dim_0, im_dim_1)
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


def trial_average(activity_matrix: np.ndarray, n_trials: int, sum_it=False, rem_nan=False) -> np.ndarray:
    """
    Compute trial average for each trace in activity_matrix
    :param activity_matrix: n_cells x n_timepoints matrix of traces
    :param n_trials: The number of trials across which to average
    :param sum_it: If true, summing instead of averaging will be performed
    :param rem_nan: If true, nan-mean or nan-sum will be used across trials
    :return: n_cells x (n_timepoints//n_trials) matrix of trial averaged activity traces
    """
    if activity_matrix.shape[1] % n_trials != 0:
        raise ValueError(f"Axis 1 of activity_matrix has {activity_matrix.shape[1]} timepoints which is not divisible"
                         f"into the requested number of {n_trials} trials")
    if activity_matrix.ndim == 2:
        m_t = np.reshape(activity_matrix, (activity_matrix.shape[0], n_trials, activity_matrix.shape[1] // n_trials))
    elif activity_matrix.ndim == 1:
        m_t = np.reshape(activity_matrix, (1, n_trials, activity_matrix.shape[0] // n_trials))
    else:
        raise ValueError(f"Activity matrix has to be a vector or 2D matrix but is {activity_matrix.ndim} dimensional")
    if sum_it:
        return np.nansum(m_t, 1) if rem_nan else np.sum(m_t, 1)
    else:
        return np.nanmean(m_t, 1) if rem_nan else np.mean(m_t, 1)


def test_cmtk_install():
    """
    Tries to determine if CMTK is installed and whether individual binaries are directly accessible
    or are accessible via the cmtk script call
    :return: -1: No cmtk install detected, 0: Direct call, 1: call via cmtk script
    """
    if shutil.which("warp") is not None:
        return 0
    if shutil.which("cmtk") is not None:
        return 1
    return -1


def cmtk_transform_3d_coordinates(coords: np.ndarray, transform_file: str) -> np.ndarray:
    """
    Uses cmtk and the indicated transform to map nx3 3D [x,y,z] coordinates into a reference space
    In this case x: Columns in an image stack, left to right; y: Rows in an image stack, top to bottom; z: d-v
    This is the same convention used in swc files and in extracting component coordinates above
    :param coords: The nx3 matrix of 3D coordinates
    :param transform_file: The transformation file path to use for mapping
    :return: nx3 matrix of transformed coordinates
    """
    cmtki = test_cmtk_install()
    if cmtki == -1:
        raise OSError("cmtk installation not found")
    if cmtki == 0:
        prog = "streamxform"
    else:
        prog = "cmtk streamxform"
    # set up temporary directory for transformation input and output
    with tempfile.TemporaryDirectory() as tmpdirname:
        infile = path.join(tmpdirname, "input.txt")
        outfile = path.join(tmpdirname, "output.txt")
        np.savetxt(infile, coords, fmt="%.1f", delimiter=' ')
        # NOTE: The following might have to be replaced with: 'cat {infile} | {prog} ... > {outfile}' on windows
        command = f'{prog} -- --inverse "{transform_file}" < "{infile}" > "{outfile}"'
        sp.run(command, shell=True)
        coords_out = np.genfromtxt(outfile, delimiter=' ')[:, :3]
        # for every transformed coordinate with at least one NaN replace all values with NaN
        has_nan = np.sum(np.isnan(coords_out), 1) > 0
        coords_out[has_nan, :] = np.nan
    return coords_out
