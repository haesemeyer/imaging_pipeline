#  Copyright 2019 Martin Haesemeyer. All rights reserved.
#
# Licensed under the MIT license

"""
Module to parse experimental file structure extracting relevant acquisition information
"""


from datetime import datetime
from os import path, listdir


class LineParser:
    def __init__(self, file_name: str, type_string: str):
        if type_string not in file_name:
            raise ValueError(f"The file {file_name} is not a {type_string} file")
        info_file = open(file_name, 'rt')
        self.all_contents = info_file.readlines()
        info_file.close()
        self.info = {"full_text": "\n".join(self.all_contents), "original_file": file_name}

    @staticmethod
    def _parse(line: str) -> (str, object):
        raise NotImplementedError()


class InfoFile(LineParser):
    """
    Parser for .info file (one such file per experiment)
    """
    def __init__(self, file_name: str):
        """
        Parses an info file - currently experiment specific data is not parsed
        as this should be handled by downstream analysis code
        :param file_name: The file path and name of a .info file
        """
        super().__init__(file_name, '.info')
        for ln in self.all_contents:
            k, v = self._parse(ln)
            if k is not None:
                self.info[k] = v
        if "stable_z" not in self.info:
            self.info["stable_z"] = False

    @staticmethod
    def _parse(line: str) -> (str, object):
        """
        Processes a line of text and if it contains experimental parameters
        returns a corresponding key and value or None, None otherwise
        :param line: A line of the info file
        :return: If general experiment information was present on the line a corresponding key, value pair
        """
        if "Experiment type:" in line:
            return "experiment_type", line.split(':')[-1].strip()
        if "Z-Step" in line:
            v = line.split(':')[-1].strip()
            return "z_step", float(v)
        if "Frame duration" in line:
            v = line.split(':')[-1].strip()
            return "frame_duration", float(v)
        if "Start time" in line:
            v = ':'.join(line.split(':')[1:]).strip()
            t = datetime.strptime(v, "%m/%d/%Y %I:%M:%S %p")
            return "start_time", t
        if "Finish time" in line:
            v = ':'.join(line.split(':')[1:]).strip()
            t = datetime.strptime(v, "%m/%d/%Y %I:%M:%S %p")
            return "finish_time", t
        if "turn-around" in line:
            v = line.split(':')[-1].strip()
            return "turn_around_px", int(v)
        if "Stable Z enabled" in line:
            return "stable_z", True
        return None, None


class ImageScannerFixed(LineParser):
    """
    Parser for plane-specific ImageScannerFixed.txt file
    """

    def __init__(self, file_name: str):
        """
        Parses information in ...ImageScannerFixed.txt files of each imaging plane
        :param file_name: The file path and name of a .txt file
        """
        super().__init__(file_name, 'ImageScannerFixed.txt')
        for ln in self.all_contents:
            k, v = self._parse(ln)
            if k is not None:
                self.info[k] = v

    @staticmethod
    def _parse(line: str) -> (str, object):
        """
        Processes a line of text and if it contains experimental parameters
        returns a corresponding key and value or None, None otherwise
        :param line: A line of the info file
        :return: If general experiment information was present on the line a corresponding key, value pair
        """
        if "Zoom" in line:
            v = line.split("\t")[-1].strip()
            return "fov", 500/float(v)
        if "DwellTimeIn" in line:
            v = line.split("\t")[-1].strip()
            return "dwell_time_us", float(v)
        if "Pixels" in line:
            v = line.split("\t")[-1].strip()
            return "pixels_per_line", int(v)
        if "LinesPer" in line:
            v = line.split("\t")[-1].strip()
            return "lines_per_image", int(v)
        if "Reps" in line:
            v = line.split("\t")[-1].strip()
            return "reps_per_line", int(v)
        if "Power" in line:
            v = line.split("\t")[-1].strip()
            return "power_stage", float(v)
        if "X" in line:
            v = line.split("\t")[-1].strip()
            return "x_stage", float(v)
        if "Y" in line:
            v = line.split("\t")[-1].strip()
            return "y_stage", float(v)
        if "Z" in line:
            v = line.split("\t")[-1].strip()
            return "z_stage", float(v)


class ExperimentParser:
    """
    Class to extract whole experimental file structure based on .info file
    location as well as experiment specific acquisition information
    """
    def __init__(self, info_file_name: str):
        """
        Create new experiment parser
        :param info_file_name: Path and name of .info file identifying the experiment
        """
        self.info_data = InfoFile(info_file_name).info
        # Extract experiment path and experiment name
        exp_path = path.dirname(info_file_name)
        self.experiment_name = '.'.join(path.split(info_file_name)[1].split('.')[:-1])
        self.original_path = exp_path
        # Get all file objects in the experiment's directory
        all_files = [f for f in listdir(exp_path) if path.isfile(path.join(exp_path, f))]
        # Obtain all files that belong to the experiment in question
        exp_files = [f for f in all_files if self.experiment_name in f]
        # Seperately collect different file types
        self.tail_files = [f for f in exp_files if f.split('.')[-1] == 'tail']
        self.tail_files.sort(key=self._file_sort_key)
        self.scanner_fixed_files = [f for f in exp_files if ("ImageScannerFixed.txt" in f and "_stableZ_" not in f)]
        self.scanner_fixed_files.sort(key=self._file_sort_key)
        self.ch_0_files = [f for f in exp_files if ("_0.tif" in f and "_stableZ_" not in f and "Z_0.tif" not in f)]
        self.ch_0_files.sort(key=self._file_sort_key)
        self.ch_1_files = [f for f in exp_files if ("_1.tif" in f and "_stableZ_" not in f and "Z_1.tif" not in f)]
        self.ch_1_files.sort(key=self._file_sort_key)
        n_ch_1 = len(self.ch_1_files)
        n_tail = len(self.tail_files)
        self.is_dual_channel = n_ch_1 > 0
        self.has_tail_data = n_tail > 0
        # check consistency of experimental files
        n_scan = len(self.scanner_fixed_files)
        n_ch_0 = len(self.ch_0_files)
        if n_ch_0 != n_scan:
            raise ValueError(f"Inconsistent experiment file structure. {n_scan} scanner files and {n_ch_0}"
                             f"Ch0 imaging files.")
        if self.is_dual_channel and n_ch_0 != n_ch_1:
            raise ValueError(f"Ch0 has {n_ch_0} tif stacks but Ch1 has {n_ch_1} tif stacks")
        if self.has_tail_data and n_ch_0 != n_tail:
            raise ValueError(f"Ch0 has {n_ch_0} tif stacks but there are {n_tail} taildata files")
        self.scanner_data = [ImageScannerFixed(path.join(self.original_path, scfile)).info
                             for scfile in self.scanner_fixed_files]

    @staticmethod
    def _file_sort_key(name: str):
        """
        Finds the imaging plane part of a filename for proper sort order of file names
        :param name: The name of the file
        :return: Imaging plane
        """
        plane_start = name.find("_Z_") + 3
        plane_end = name[plane_start:].find("_")
        if plane_end == -1:
            # tail file
            plane_end = name[plane_start:].find(".tail")
        if plane_end == -1:
            raise ValueError("Can't find plane identification in file name")
        return int(name[plane_start:plane_start + plane_end])
