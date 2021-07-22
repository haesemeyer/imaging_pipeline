#  Copyright 2019 Martin Haesemeyer. All rights reserved.
#
# Licensed under the MIT license

"""
Script for testing caiman - cai_demo.py implementation with slight adjustments
"""


import numpy as np
from utilities import ui_get_file
import matplotlib.pyplot as pl
from experiment import analyze_experiment
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
    exp = analyze_experiment(info_file, "OSU 2P", "", {"indicator_decay_time": 3.0})
    acb_func = exp.avg_component_brightness(False)

    fig, axes = pl.subplots(ncols=int(np.sqrt(exp.n_planes))+1, nrows=int(np.sqrt(exp.n_planes)))
    axes = axes.ravel()
    for i in range(exp.n_planes):
        if i >= axes.size:
            break
        axes[i].imshow(exp.projections[i], vmax=np.percentile(exp.projections[i], 90))
        cents = exp.all_centroids[i]
        avg_brightness = acb_func[i]
        likely_background = avg_brightness < 0.1
        likely_foreground = np.logical_not(likely_background)
        axes[i].scatter(cents[likely_foreground, 0], cents[likely_foreground, 1], s=2, color='C1')
        axes[i].scatter(cents[likely_background, 0], cents[likely_background, 1], s=2, color='w')
    fig.tight_layout()

    exp.save_experiment(f"{path.join(exp.original_path, exp.experiment_name)}.hdf5")
