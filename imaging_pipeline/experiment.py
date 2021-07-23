#  Copyright (c) 2019. Martin Haesemeyer. All rights reserved.
#
#  Licensced under the MIT license. See LICENSE

"""
Class wrapper of experiments with support of serialization to/from HDF5 files
"""
import warnings

import h5py
from cai_wrapper import CaImAn
from experiment_parser import ExperimentParser
import numpy as np
from utilities import get_component_centroids, get_component_coordinates, ExperimentException, TailData, ui_get_file
from datetime import datetime
from os import path
import json
import logging
from sklearn.exceptions import ConvergenceWarning
import matplotlib.pyplot as pl
import sys


class Experiment2P:
    """
    Represents a 2-photon imaging experiment on which cells have been segmented
    """
    def __init__(self, lazy_load_filename=""):
        """
        Creates a new Experiment2P object
        :param lazy_load_filename: If given, will attach this instance to an existing hdf5 store at the filename
        """
        self.info_data = {}  # data from the experiment's info data
        self.experiment_name = ""  # name of the experiment
        self.original_path = ""  # the original path when the experiment was analyzed
        self.scope_name = ""  # the name assigned to the microscope for informational purposes
        self.comment = ""  # general comment associated with the experiment
        self.tail_frame_rate = 0  # the frame-rate of the tail camera
        self.scanner_data = []  # for each experimental plane the associated scanner data (resolution, etc.)
        self.__tail_data = []  # for each experimental plane the tail data if applicable
        self.__laser_data = []  # for each experimental plane 20Hz vector of laser command voltages if applicable
        self.bout_data = []  # for each experimental plane, matrix of extracted swim bouts
        self.tail_frame_times = []  # for each experimental plane, the scan relative time of each tail cam frame
        self.__all_c = []  # for each experimental plane the inferred calcium of each extracted unit
        self.__all_dff = []  # for each experimental plane the dF/F of each extracted unit
        self.all_centroids = []  # for each experimental plane the unit centroid coordinates as (x [col]/y [row]) pairs
        self.all_sizes = []  # for each experimental plane the size of each unit in pixels (not weighted)
        self.all_spatial = []  # for each experimental plane n_comp x 4 array <component-ix, weight, x-coord, y-coord>
        self.projections = []  # list of 32 bit plane projections after motion correction
        self.anat_projections = []  # for dual-channel experiments, list of 32 bit plane projections of anatomy channel
        self.__func_stacks = []  # for each plane the realigned 8-bit functional stack
        self.mcorr_dicts = []  # the motion correction parameters used on each plane
        self.cnmf_extract_dicts = []  # the cnmf source extraction parameters used on each plane
        self.cnmf_val_dicts = []  # the cnmf validation parameters used on each plane
        self.version = "1"  # version ID for future-proofing
        self.populated = False  # indicates if class contains experimental data through analysis or loading
        self.lazy = False  # indicates that we have lazy-loaded and attached to hdf5 file
        self.__hdf5_store = None  # type: h5py.File
        if lazy_load_filename != "" and not path.exists(lazy_load_filename):
            raise ValueError(f"The file {lazy_load_filename} does not exist. Cannot attach to store.")
        if lazy_load_filename != "":
            self.__lazy_load(lazy_load_filename)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.__hdf5_store is not None:
            self.__hdf5_store.close()
            self.__hdf5_store = None
            self.lazy = False

    def __lazy_load(self, lazy_load_filename: str):
        """
        Performs "lazy loading" loading some properties and making others available through an hdf5 store backend
        :param lazy_load_filename: The name of the experiment hdf5 file
        """
        self.__hdf5_store = h5py.File(lazy_load_filename, 'r')
        self.version = self.__hdf5_store["version"][()]
        try:
            if self.version == b"unstable" or self.version == "unstable":
                warnings.warn("Experiment file was created with development version of analysis code. Trying to "
                              "load as version 1")
            elif int(self.version) > 1:
                self.__hdf5_store.close()
                self.__hdf5_store = None
                raise IOError(f"File version {self.version} is larger than highest recognized version '1'")
        except ValueError:
            self.__hdf5_store.close()
            self.__hdf5_store = None
            raise IOError(f"File version {self.version} not recognized")
        # load general experiment data
        n_planes = self.__hdf5_store["n_planes"][()]
        self.experiment_name = self.__hdf5_store["experiment_name"][()]
        self.original_path = self.__hdf5_store["original_path"][()]
        self.scope_name = self.__hdf5_store["scope_name"][()]
        self.comment = self.__hdf5_store["comment"][()]
        self.tail_frame_rate = self.__hdf5_store["tail_frame_rate"][()]
        # load singular parameter dictionary
        self.info_data = self._load_dictionary("info_data", self.__hdf5_store)
        # load some per-plane data leave larger objects unloaded
        for i in range(n_planes):
            plane_group = self.__hdf5_store[str(i)]
            self.scanner_data.append(self._load_dictionary("scanner_data", plane_group))
            self.projections.append(plane_group["projection"][()])
            if "anat_projection" in plane_group:  # test if this experiment was dual-channel
                self.anat_projections.append(plane_group["anat_projection"][()])
            if "tail_data" in plane_group:  # test if this experiment had tail data (for all planes)
                self.bout_data.append(plane_group["bout_data"][()])
                self.tail_frame_times.append(plane_group["tail_frame_time"][()])
            self.all_centroids.append(plane_group["centroids"][()])
            self.all_sizes.append(plane_group["sizes"][()])
            self.all_spatial.append(plane_group["spatial"][()])
            ps = plane_group["mcorr_dict"][()]
            self.mcorr_dicts.append(json.loads(ps))
            ps = plane_group["cnmf_extract_dict"][()]
            self.cnmf_extract_dicts.append(json.loads(ps))
            ps = plane_group["cnmf_val_dict"][()]
            self.cnmf_val_dicts.append(json.loads(ps))
        self.lazy = True

    @property
    def tail_data(self):
        if not self.lazy:
            return self.__tail_data
        else:
            if self.__hdf5_store is None:
                raise ExperimentException("Hdf5 store not properly attached")
            rval = []
            for i in range(self.n_planes):
                plane_group = self.__hdf5_store[str(i)]
                if "tail_data" in plane_group:  # test if this experiment had tail data (for all planes)
                    rval.append(plane_group["tail_data"][()])
            return rval

    @tail_data.setter
    def tail_data(self, value):
        self.__tail_data = value

    @property
    def laser_data(self):
        if not self.lazy:
            return self.__laser_data
        else:
            if self.__hdf5_store is None:
                raise ExperimentException("Hdf5 store not properly attached")
            rval = []
            for i in range(self.n_planes):
                plane_group = self.__hdf5_store[str(i)]
                if "laser_data" in plane_group:  # test if this experiment had laser data
                    rval.append(plane_group["laser_data"][()])
            return rval

    @laser_data.setter
    def laser_data(self, value):
        self.__laser_data = value

    @property
    def all_c(self):
        if not self.lazy:
            return self.__all_c
        else:
            if self.__hdf5_store is None:
                raise ExperimentException("Hdf5 store not properly attached")
            rval = []
            for i in range(self.n_planes):
                plane_group = self.__hdf5_store[str(i)]
                rval.append(plane_group["C"][()])
            return rval

    @all_c.setter
    def all_c(self, value):
        self.__all_c = value

    @property
    def all_dff(self):
        if not self.lazy:
            return self.__all_dff
        else:
            if self.__hdf5_store is None:
                raise ExperimentException("Hdf5 store not properly attached")
            rval = []
            for i in range(self.n_planes):
                plane_group = self.__hdf5_store[str(i)]
                rval.append(plane_group["dff"][()])
            return rval

    @all_dff.setter
    def all_dff(self, value):
        self.all_dff = value

    @property
    def func_stacks(self):
        if not self.lazy:
            return self.__func_stacks
        else:
            if self.__hdf5_store is None:
                raise ExperimentException("Hdf5 store not properly attached")
            rval = []
            for i in range(self.n_planes):
                plane_group = self.__hdf5_store[str(i)]
                if "func_stack" in plane_group:
                    rval.append(plane_group["func_stack"][()])
            return rval

    @func_stacks.setter
    def func_stacks(self, value):
        self.func_stacks = value

    @staticmethod
    def load_experiment(file_name: str):
        """
        Loads an experiment from a serialization in an hdf5 file
        :param file_name: The name of the hdf5 file storing the experiment
        :return: Experiment object with all relevant data
        """
        exp = Experiment2P()
        with h5py.File(file_name, 'r') as dfile:
            exp.version = dfile["version"][()]  # in future allows for version specific loading
            try:
                if exp.version == b"unstable" or exp.version == "unstable":
                    warnings.warn("Experiment file was created with development version of analysis code. Trying to "
                                  "load as version 1")
                elif int(exp.version) > 1:
                    raise IOError(f"File version {exp.version} is larger than highest recognized version '1'")
            except ValueError:
                raise IOError(f"File version {exp.version} not recognized")
            # load general experiment data
            n_planes = dfile["n_planes"][()]  # inferrred property of class but used here for loading plane data
            exp.experiment_name = dfile["experiment_name"][()]
            exp.original_path = dfile["original_path"][()]
            exp.scope_name = dfile["scope_name"][()]
            exp.comment = dfile["comment"][()]
            exp.tail_frame_rate = dfile["tail_frame_rate"][()]
            # load singular parameter dictionary
            exp.info_data = exp._load_dictionary("info_data", dfile)
            # load per-plane data
            for i in range(n_planes):
                plane_group = dfile[str(i)]
                exp.scanner_data.append(exp._load_dictionary("scanner_data", plane_group))
                exp.tail_data.append(plane_group["tail_data"][()])
                exp.projections.append(plane_group["projection"][()])
                if "func_stack" in plane_group:
                    exp.func_stacks.append(plane_group["func_stack"][()])
                if "anat_projection" in plane_group:  # test if this experiment was dual-channel
                    exp.anat_projections.append(plane_group["anat_projection"][()])
                if "tail_data" in plane_group:  # test if this experiment had tail data (for all planes)
                    exp.tail_data.append(plane_group["tail_data"][()])
                    exp.bout_data.append(plane_group["bout_data"][()])
                    exp.tail_frame_times.append(plane_group["tail_frame_time"][()])
                if "laser_data" in plane_group:  # test if this experiment had laser data
                    exp.laser_data.append(plane_group["laser_data"][()])
                exp.all_c.append(plane_group["C"][()])
                exp.all_dff.append(plane_group["dff"][()])
                exp.all_centroids.append(plane_group["centroids"][()])
                exp.all_sizes.append(plane_group["sizes"][()])
                exp.all_spatial.append(plane_group["spatial"][()])
                ps = plane_group["mcorr_dict"][()]
                exp.mcorr_dicts.append(json.loads(ps))
                ps = plane_group["cnmf_extract_dict"][()]
                exp.cnmf_extract_dicts.append(json.loads(ps))
                ps = plane_group["cnmf_val_dict"][()]
                exp.cnmf_val_dicts.append(json.loads(ps))
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
                # need to decode byte-string into datetime object
                try:
                    date_time_string = g[k][()].decode('UTF-8')  # convert bye string to string
                except AttributeError:
                    # it is already a unicode string
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
            dfile.create_dataset("version", data=self.version)  # for later backwards compatibility
            # save general experiment data
            dfile.create_dataset("experiment_name", data=self.experiment_name)
            dfile.create_dataset("original_path", data=self.original_path)
            dfile.create_dataset("scope_name", data=self.scope_name)
            dfile.create_dataset("comment", data=self.comment)
            dfile.create_dataset("n_planes", data=self.n_planes)
            dfile.create_dataset("tail_frame_rate", data=self.tail_frame_rate)
            # save singular parameter dictionary
            self._save_dictionary(self.info_data, "info_data", dfile)
            # save per-plane data
            for i in range(self.n_planes):
                plane_group = dfile.create_group(str(i))
                self._save_dictionary(self.scanner_data[i], "scanner_data", plane_group)
                if len(self.tail_data) > 0:
                    plane_group.create_dataset("tail_data", data=self.tail_data[i], compression="gzip",
                                               compression_opts=5)
                    if self.bout_data[i] is not None:
                        plane_group.create_dataset("bout_data", data=self.bout_data[i], compression="gzip",
                                                   compression_opts=5)
                    else:
                        # no bouts were found, save dummy array of one line of np.nan
                        bd = np.full((1, 8), np.nan)
                        plane_group.create_dataset("bout_data", data=bd, compression="gzip", compression_opts=5)
                    plane_group.create_dataset("tail_frame_time", data=self.tail_frame_times[i])
                if len(self.laser_data) > 0:
                    plane_group.create_dataset("laser_data", data=self.laser_data[i], compression="gzip",
                                               compression_opts=5)
                plane_group.create_dataset("projection", data=self.projections[i], compression="gzip",
                                           compression_opts=5)
                plane_group.create_dataset("func_stack", data=self.func_stacks[i], compression="gzip",
                                           compression_opts=5)
                if len(self.anat_projections) > 0:  # this is a dual-channel experiment
                    plane_group.create_dataset("anat_projection", data=self.anat_projections[i], compression="gzip",
                                               compression_opts=5)
                plane_group.create_dataset("C", data=self.all_c[i], compression="gzip", compression_opts=5)
                plane_group.create_dataset("dff", data=self.all_dff[i], compression="gzip", compression_opts=5)
                plane_group.create_dataset("centroids", data=self.all_centroids[i], compression="gzip",
                                           compression_opts=5)
                plane_group.create_dataset("sizes", data=self.all_sizes[i], compression="gzip", compression_opts=5)
                plane_group.create_dataset("spatial", data=self.all_spatial[i], compression="gzip", compression_opts=5)
                # due to mixed python types in caiman parameter dictionaries these currently get pickled
                ps = json.dumps(self.mcorr_dicts[i])
                plane_group.create_dataset("mcorr_dict", data=ps)
                ps = json.dumps(self.cnmf_extract_dicts[i])
                plane_group.create_dataset("cnmf_extract_dict", data=ps)
                ps = json.dumps(self.cnmf_val_dicts[i])
                plane_group.create_dataset("cnmf_val_dict", data=ps)
        finally:
            dfile.close()

    def avg_component_brightness(self, use_anat):
        """
        Computes the brightness of each identified component on the functional or anatomical channel
        :param use_anat: If True returns the average brightness of each component on anatomy not functional channel
        :return: n_planes long list of vectors with the time-average brightness of each identified component
        """
        if not self.populated:
            raise ExperimentException("Experiment does not have data. Use Analyze or Load first.")
        if use_anat and not self.is_dual_channel:
            raise ValueError("Experiment does not have anatomy channel")
        p = self.anat_projections if use_anat else self.projections
        acb = []
        for i in range(self.n_planes):
            # <component-ix, weight, x-coord, y-coord>
            n_components = int(np.max(self.all_spatial[i][:, 0]) + 1)  # component indices are 0-based
            br = np.zeros(n_components, dtype=np.float32)
            for j in range(n_components):
                this_component = self.all_spatial[i][:, 0].astype(int) == j
                spatial_x = self.all_spatial[i][this_component, 2].astype(int)
                spatial_y = self.all_spatial[i][this_component, 3].astype(int)
                br[j] = np.mean(p[i][spatial_y, spatial_x])
            acb.append(br)
        return acb

    @property
    def n_planes(self):
        return len(self.scanner_data)

    @property
    def is_dual_channel(self):
        return len(self.anat_projections) > 0


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


def main(ca_decay: float, exp_info_file: str, func_channel: int):
    if exp_info_file == "":
        info_file = ui_get_file(filetypes=[('Experiment info', '*.info')], multiple=False)
        if type(info_file) == list:
            info_file = info_file[0]
    else:
        info_file = exp_info_file

    exp = analyze_experiment(info_file, "OSU 2P", "", {"indicator_decay_time": ca_decay}, func_channel=func_channel)
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


if __name__ == "__main__":

    if_name = ""
    decay_time = -1
    acq_channel = -1

    def parse(arg):
        global if_name
        global decay_time
        global acq_channel
        if arg.upper() == "CH0":
            acq_channel = 0
            return
        if arg.upper() == "CH1":
            acq_channel = 1
            return
        if path.exists(arg) and ".info" in arg:
            if_name = arg
            return
        try:
            dt = float(arg)
        except ValueError:
            return
        decay_time = dt

    if len(sys.argv) > 1:
        for a in sys.argv[1:]:
            parse(a)

    if decay_time < 0:
        decay_str = input("Please enter the calcium indicator decay time in seconds:")
        try:
            decay_time = float(decay_str)
        except ValueError:
            print(f"{decay_str} is not a valid number")
            raise

    if acq_channel < 0:
        acq_str = input("Please indicate functional channel [Ch0/Ch1]")
        if acq_str.upper() == "CH0":
            acq_channel = 0
        elif acq_str.upper() == "CH1":
            acq_channel = 1
        else:
            raise ValueError(f"{acq_str} is not a recognized channel indicator. Has to be Ch0 or Ch1")

    # Shut down some noise clogging the interpreter
    logging.basicConfig(level=logging.ERROR)
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=DeprecationWarning)
    warnings.simplefilter(action='once', category=ConvergenceWarning)

    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"

    main(decay_time, if_name, acq_channel)
