#  Copyright 2019 Martin Haesemeyer. All rights reserved.
#
# Licensed under the MIT license

"""
Script for testing caiman - cai_demo.py implementation with slight adjustments
"""


import numpy as np
from utilities import ui_get_file, trial_average
import matplotlib.pyplot as pl
import seaborn as sns
from experiment import Experiment2P
from os import path
import logging
import warnings
from sklearn.exceptions import ConvergenceWarning


# Shut down some noise clogging the interpreter
logging.basicConfig(level=logging.ERROR)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)


if __name__ == "__main__":
    # Shut down some noise clogging the interpreter
    logging.basicConfig(level=logging.ERROR)
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=DeprecationWarning)
    warnings.simplefilter(action='once', category=ConvergenceWarning)

    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"

    info_file = ui_get_file(filetypes=[('Experiment info', '*.info')], multiple=False)
    if type(info_file) == list:
        info_file = info_file[0]
    exp = Experiment2P().analyze_experiment(info_file, "OSU 2P", "", {"indicator_decay_time": 3.0})
    all_c = [a for b in exp.all_c for a in b]
    regressors = np.load("rh56_regs.npy")
    regressors = np.r_[regressors, regressors, regressors]
    r_times = np.arange(regressors.shape[0]) / 5
    r_mat = np.full((len(all_c), regressors.shape[1]), np.nan)
    interp_c = np.zeros((len(all_c), r_times.size))
    for j, trace in enumerate(all_c):
        data_times = np.arange(trace.size) * exp.info_data["frame_duration"]
        i_trace = np.interp(r_times, data_times, trace)
        interp_c[j, :] = i_trace
    tavg_interp_c = trial_average(interp_c, 3)
    for i, reg in enumerate(regressors.T):
        for j, i_trace in enumerate(tavg_interp_c):
            r_mat[j, i] = np.corrcoef(i_trace, reg[:reg.size//3])[0, 1]

    fig, (ax1, ax2) = pl.subplots(1, 2)
    sns.heatmap(r_mat, vmin=-1, vmax=1, center=0, ax=ax1)
    r_mat[r_mat < 0.6] = 0
    sns.heatmap(r_mat, vmin=0, vmax=1, ax=ax2)
    fig.tight_layout()
    membership = np.full(r_mat.shape[0], -1)
    membership[np.max(r_mat, 1) > 0] = np.argmax(r_mat, 1)[np.max(r_mat, 1) > 0]

    colors = ["C3", "C1", "C2", "C0", "C7"]
    labels = ["Fast ON", "Slow ON", "Fast OFF", "Slow OFF", "Dld. OFF"]
    trial_times = r_times[:r_times.size // 3]
    if np.sum(membership > -1) > 0:
        fig, ax = pl.subplots()
        for cid in range(5):
            if np.sum(membership == cid) > 0:
                ax.plot(trial_times, np.mean(tavg_interp_c[membership == cid, :], 0), color=colors[cid],
                        label=f"{labels[cid]}: {np.sum(membership==cid)}")
        ax.set_title(f"{membership.size} units total. {np.round(np.sum(membership > -1)/membership.size*100, 1)}"
                     f" % heat sensitive")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Trial average C")
        ax.legend()
        sns.despine(fig, ax)

    fig, axes = pl.subplots(ncols=int(np.sqrt(exp.n_planes))+1, nrows=int(np.sqrt(exp.n_planes)))
    axes = axes.ravel()
    for i in range(exp.n_planes):
        if i >= axes.size:
            break
        axes[i].imshow(exp.projections[i], vmax=np.percentile(exp.projections[i], 90))
        cents = exp.all_centroids[i]
        axes[i].scatter(cents[:, 0], cents[:, 1], s=2, color='C1')
    fig.tight_layout()

    exp.save_experiment(f"{path.join(exp.original_path, exp.experiment_name)}.hdf5")
