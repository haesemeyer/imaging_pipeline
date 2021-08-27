#  Copyright (c) 2021. Martin Haesemeyer. All rights reserved.
#
#  Licensced under the MIT license. See LICENSE

"""
Functions for Caiman based analysis into experiment classes
"""

from experiment import Experiment2P
from utilities import get_component_centroids, get_component_coordinates, TailData
from experiment_parser import ExperimentParser
from cai_wrapper import CaImAn
import os
from os import path
import numpy as np

import logging
from sklearn.exceptions import ConvergenceWarning
import matplotlib.pyplot as pl
from utilities import ui_get_file
import warnings

import argparse
from typing import Any


class CheckArgs(argparse.Action):
    """
    Check our command line arguments for validity
    """
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super().__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values: Any, option_string=None):
        if self.dest == 'file':
            if not os.path.exists(values):
                raise argparse.ArgumentError(self, "Experiment .info file does not exist")
            if ".info" not in values:
                raise argparse.ArgumentError(self, "Provided file is not a .info file")
            setattr(namespace, self.dest, values)
        elif self.dest == 'decay_time':
            if values <= 0:
                raise argparse.ArgumentError(self, "Decay time must be larger than 0")
            setattr(namespace, self.dest, values)
        elif self.dest == 'func_channel':
            if values != 0 and values != 1:
                raise argparse.ArgumentError(self, "Functional channel identifier has to be 0 or 1")
            setattr(namespace, self.dest, values)
        else:
            raise Exception("Parser was asked to check unknown argument")


def analyze_experiment(info_file_name: str, scope_name: str, comment: str, cai_params: dict, func_channel=0,
                       tail_frame_rate=250) -> Experiment2P:
    """
    Performs the intial analysis (file collection and segmentation) on the experiment identified by the info file
    :param info_file_name: The path and name of the info-file identifying the experiment
    :param scope_name: Identifier of the microscope used for the experiment
    :param comment: Comment associated with the experiment
    :param cai_params: Dictionary of caiman parameters - fov and time-per-frame will be replaced with exp. data
    :param func_channel: The channel (either 0 or 1) containing the functional imaging data
    :param tail_frame_rate: The frame-rate of the tail camera
    :return: Experiment object with all relevant data
    """
    if "indicator_decay_time" not in cai_params:
        raise ValueError("At least 'indicator_decay_time' has to be specified in cai_params")
    if func_channel < 0 or func_channel > 1:
        raise ValueError(f'func_channel {func_channel} is not valid. Has to be 0 ("green"") or 1 ("red"")')
    exp = Experiment2P()
    exp.scope_name = scope_name
    exp.comment = comment
    # copy acquisition information extracted from experiment files
    eparser = ExperimentParser(info_file_name)
    if not eparser.is_dual_channel and func_channel != 0:
        raise ValueError(f"Experiment is single channel but func_channel was set to {func_channel}")
    exp.experiment_name = eparser.experiment_name
    exp.original_path = eparser.original_path
    exp.info_data = eparser.info_data
    exp.scanner_data = eparser.scanner_data
    # collect data in tail files if applicable
    try:
        if eparser.has_tail_data:
            for tf in eparser.tail_files:
                exp.tail_data.append(np.genfromtxt(path.join(exp.original_path, tf), delimiter='\t'))
                # Since we only keep the bout calls, the time constant passed below is arbitrary
                td = TailData.load_tail_data(path.join(exp.original_path, tf), 3.0, tail_frame_rate,
                                             eparser.info_data["frame_duration"])
                exp.bout_data.append(td.bouts)
                exp.tail_frame_times.append(td.frame_time)
                exp.tail_frame_rate = td.frame_rate
    except (IOError, OSError) as e:
        print(f".tail files are present but at least one file failed to load. Not attaching any tail data.")
        print(e)
        exp.tail_data = []
        exp.bout_data = []
    # collect data in laser files if applicable
    try:
        if eparser.has_laser_data:
            for lf in eparser.laser_files:
                exp.laser_data.append(np.genfromtxt(path.join(exp.original_path, lf)))
    except (IOError, OSError) as e:
        print(f".laser files are present but at least one file failed to load. Not attaching any laser data.")
        print(e)
        exp.laser_data = []
    # use caiman to extract units and calcium data
    if eparser.is_dual_channel:
        print(f"This experiment has dual channel data. Ch{func_channel} is being processed as functional channel."
              f" Other, anatomy channel, is co-aligned.")
    data_files = eparser.ch_0_files if func_channel == 0 else eparser.ch_1_files
    if eparser.is_dual_channel:
        co_files = eparser.ch_1_files if func_channel == 0 else eparser.ch_0_files
    else:
        co_files = None
    for i, ifl in enumerate(data_files):
        cai_params["time_per_frame"] = exp.info_data["frame_duration"]
        cai_params["fov_um"] = exp.scanner_data[i]["fov"]
        cai_wrapper = CaImAn(**cai_params)
        ifile = path.join(exp.original_path, ifl)
        if eparser.is_dual_channel:
            cofile = path.join(exp.original_path, co_files[i])
        else:
            cofile = None
        print(f"Now analyzing: {ifile}")
        images, params, co_images = cai_wrapper.motion_correct(ifile, cofile)

        exp.mcorr_dicts.append(params["Motion Correction"])
        exp.projections.append(np.mean(images, 0))
        stack = np.array(images)
        stack -= np.min(stack)
        stack[stack > 255] = 255
        stack = stack.astype(np.uint8)
        exp.func_stacks.append(stack)
        if eparser.is_dual_channel:
            exp.anat_projections.append(np.mean(co_images, 0))
        print("Motion correction completed")
        cnm2, params = cai_wrapper.extract_components(images, ifile)[1:]
        exp.cnmf_extract_dicts.append(params["CNMF"])
        exp.cnmf_val_dicts.append(params["Validation"])
        print("Source extraction completed")
        exp.all_c.append(cnm2.estimates.C.copy())
        exp.all_dff.append(cnm2.estimates.F_dff.copy())
        exp.all_centroids.append(get_component_centroids(cnm2.estimates.A, images.shape[1], images.shape[2]))
        coords, weights = get_component_coordinates(cnm2.estimates.A, images.shape[1], images.shape[2])
        exp.all_sizes.append(np.array([w.size for w in weights]))
        spatial_footprints = []
        for c_ix, (comp_coords, comp_weights) in enumerate(zip(coords, weights)):
            ix = np.full(comp_coords.shape[0], c_ix)[:, None]
            spat = np.c_[ix, comp_weights[:, None], comp_coords]
            spatial_footprints.append(spat)
        exp.all_spatial.append(np.vstack(spatial_footprints))
    exp.populated = True
    return exp


def main(ca_decay: float, exp_info_file: str, func_channel: int) -> None:
    if exp_info_file == "":
        info_file = ui_get_file(filetypes=[('Experiment info', '*.info')], multiple=False)
        if type(info_file) == list:
            info_file = info_file[0]
    else:
        info_file = exp_info_file

    exp = analyze_experiment(info_file, "OSU 2P", "", {"indicator_decay_time": ca_decay}, func_channel=func_channel)
    exp.save_experiment(f"{path.join(exp.original_path, exp.experiment_name)}.hdf5")
    acb_func = exp.avg_component_brightness(False)

    fig, axes = pl.subplots(ncols=int(np.sqrt(exp.n_planes)) + 1, nrows=int(np.sqrt(exp.n_planes)))
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
    fig.savefig(f"{path.join(exp.original_path, exp.experiment_name)}_extraction.png", dpi=600)


if __name__ == "__main__":

    a_parser = argparse.ArgumentParser(prog="imaging_pipeline.analysis",
                                       description="Aggregates all experiment information into one hdf5 file and"
                                                   " extracts calcium signals and corresponding units using CAIMAN."
                                                   " Also performs rudimentary swim bout identification.")
    a_parser.add_argument("decay_time", help="Estimated calcium indicator decay time", type=float, action=CheckArgs)
    a_parser.add_argument("-f", "--file", help="File name and path of experiment's .info file", type=str, default="",
                          action=CheckArgs)
    a_parser.add_argument("-fc", "--func_channel", help="The index of the functional channel", type=int,
                          action=CheckArgs, default=0)

    args = a_parser.parse_args()

    if_name = args.file
    decay_time = args.decay_time
    acq_channel = args.func_channel

    # Shut down some noise clogging the interpreter
    logging.basicConfig(level=logging.ERROR)
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=DeprecationWarning)
    warnings.simplefilter(action='once', category=ConvergenceWarning)

    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"

    main(decay_time, if_name, acq_channel)
