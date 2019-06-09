#  Copyright (c) 2019. Martin Haesemeyer. All rights reserved.
#
#  Licensced under the MIT license. See LICENSE

"""
Class wrapper of experiments with support of serialization to/from HDF5 files
"""


import h5py
from cai_wrapper import CaImAn
from experiment_parser import ExperimentParser
import numpy as np
from utilities import get_component_centroids, get_component_coordinates
from datetime import datetime


class Experiment2P:
    """
    Represents a 2-photon imaging experiment on which cells have been segmented
    """
    def __init__(self):
        self.info_data = {}  # data from the experiment's info data
        self.experiment_name = ""  # name of the experiment
        self.original_path = ""  # the original path when the experiment was analyzed
        self.scope_name = ""  # the name assigned to the microscope for informational purposes
        self.comment = ""  # general comment associated with the experiment
        self.scanner_data = []  # for each experimental plane the associated scanner data (resolution, etc.)
        self.tail_data = []  # for each experimental plane the tail data if applicable
        self.all_c = []  # for each experimental plane the inferred calcium of each extracted unit
        self.all_dff = []  # for each experimental plane the dF/F of each extracted unit
        self.all_centroids = []  # for each experimental plane the unit centroid coordinates as (x [col]/y [row]) pairs
        self.all_sizes = []  # for each experimental plane the size of each unit in pixels (not weighted)
        self.projections = []  # list of 32 bit plane projections after motion correction
        self.mcorr_dict = {}  # the motion correction parameters used on each plane
        self.cnmf_extract_dict = {}  # the cnmf source extraction parameters used on each plane
        self.cnmf_val_dict = {}  # the cnmf validation parameters used on each plane
        self.version = "unstable"  # version ID for future-proofing
        self.populated = False  # indicates if class contains experimental data through analysis or loading

    @staticmethod
    def analyze_experiment(info_file_name: str, scope_name: str, comment: str, cai_wrapper: CaImAn, func_channel=0):
        """
        Performs the intial analysis (file collection and segmentation) on the experiment identified by the info file
        :param info_file_name: The path and name of the info-file identifying the experiment
        :param scope_name: Identifier of the microscope used for the experiment
        :param comment: Comment associated with the experiment
        :param cai_wrapper: CaImAn wrapper object used to extract calcium data
        :param func_channel: The channel (either 0 or 1) containing the functional imaging data
        :return: Experiment object with all relevant data
        """
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
                    exp.tail_data.append(np.genfromtxt(tf, delimiter='\t'))
        except (IOError, OSError) as e:
            print(f".tail files are present but at least one file failed to load. Not attaching any tail data.")
            print(e)
            exp.tail_data = []
        # use caiman to extract units and calcium data
        if eparser.is_dual_channel:
            print(f"This experiment has dual channel data. Ch{func_channel} is being processed as functional channel."
                  f" Other, anatomy channel, is currently being ignored")
        data_files = eparser.ch_0_files if func_channel == 0 else eparser.ch_1_files
        for ifile in data_files:
            print(f"Now analyzing: {ifile}")
            images, exp.mcorr_dict = cai_wrapper.motion_correct(ifile)
            exp.projections.append(np.sum(images, 0))
            print("Motion correction completed")
            cnm2, params = cai_wrapper.extract_components(images, ifile)[1:]
            exp.cnmf_extract_dict = params["CNMF"]
            exp.cnmf_val_dict = params["Validation"]
            print("Source extraction completed")
            exp.all_c.append(cnm2.estimates.C)
            exp.all_dff.append(cnm2.estimates.F_dff)
            exp.all_centroids.append(get_component_centroids(cnm2.estimates.A))
            coords, weights = get_component_coordinates(cnm2.estimates.A)
            exp.all_sizes.append(np.array([w.size for w in weights]))
            # TODO: Add spatial unit composition to experiment class
        exp.populated = True
        return exp

    @staticmethod
    def load_experiment(file_name: str):
        """
        Loads an experiment from a serialization in an hdf5 file
        :param file_name: The name of the hdf5 file storing the experiment
        :return: Experiment object with all relevant data
        """
        exp = Experiment2P()
        with h5py.File(file_name, 'r') as dfile:
            n_planes = dfile["n_planes"][()]  # inferrred property of class but used here for loading plane data
            exp.experiment_name = dfile["experiment_name"][()]
            pass
        exp.populated = True
        return exp

    @staticmethod
    def _save_dictionary(d: dict, dict_name: str, file: h5py.File):
        """
        Saves a dictionary to hdf5 file. Note: Does not work for general dictionaries!
        :param d: The dictionary to save
        :param dict_name: The name of the dictionary
        :param file: The hdf5 file to which the dictionary will be added
        """
        g = file.create_group(dict_name)
        for k in d:
            if "finish_time" in k or "start_time" in k:
                # need to encode datetime object as string
                date_time_string = d[k].strftime("%m/%d/%Y %I:%M:%S %p")
                g.create_dataset(k, data=date_time_string)
            else:
                g.create_dataset(k, data=d[k])

    @staticmethod
    def _load_dictionary(dict_name: str, file: h5py.File):
        """
        Loads a experiment related dictionary from file
        :param dict_name: The name of the dictionary
        :param file: The hdf5 file containing the dictionary
        :return: The populated dictionary
        """
        d = {}
        g = file[dict_name]
        for k in g:
            if "finish_time" in k or "start_time" in k:
                # need to decode string into datetime object
                date_time_string = g[k][()]
                d[k] = datetime.strptime(date_time_string, "%m/%d/%Y %I:%M:%S %p")
            else:
                d[k] = g[k][()]
        return d

    def save_experiment(self, file_name: str, ovr_if_exists=False):
        """
        Saves the experiment to the indicated file in hdf5 format
        :param file_name: The name of the file to save to
        :param ovr_if_exists: If set to true and file exists it will be overwritten otherwise exception will be raised
        """
        if not self.populated:
            raise ValueError("Empty experiment class cannot be saved. Load or analyze experiment first.")
        if ovr_if_exists:
            dfile = h5py.File(file_name, "w")
        else:
            dfile = h5py.File(file_name, "x")
        try:
            # save general experiment data
            dfile.create_dataset("experiment_name", data=self.experiment_name)
            dfile.create_dataset("original_path", data=self.original_path)
            dfile.create_dataset("scope_name", data=self.scope_name)
            dfile.create_dataset("comment", data=self.comment)
            dfile.create_dataset("n_planes", data=self.n_planes)
            # save parameter dictionaries
            # save per-plane data
        finally:
            dfile.close()

    @property
    def n_planes(self):
        return len(self.scanner_data)
