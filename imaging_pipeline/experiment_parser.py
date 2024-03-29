#  Copyright 2019 Martin Haesemeyer. All rights reserved.
#
# Licensed under the MIT license

"""
Module to parse experimental file structure extracting relevant acquisition information
"""


from datetime import datetime
from os import path, listdir
from typing import Any, Dict, List, Tuple, Optional


class LineParser:
    def __init__(self, file_name: str, type_string: str):
        if type_string not in file_name:
            raise ValueError(f"The file {file_name} is not a {type_string} file")
        info_file = open(file_name, 'rt')
        self.all_contents = info_file.readlines()
        info_file.close()
        self.info: Dict[str, Any] = {"full_text": "\n".join(self.all_contents), "original_file": file_name}

    @staticmethod
    def _parse(line: str) -> Tuple[Optional[str], Any]:
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
    def _parse(line: str) -> Tuple[Optional[str], Any]:
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
    def _parse(line: str) -> Tuple[Optional[str], Any]:
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
        self.info_data: Dict[str, Any] = InfoFile(info_file_name).info
        # Extract experiment path and experiment name
        exp_path = path.dirname(info_file_name)
        self.experiment_name: str = '.'.join(path.split(info_file_name)[1].split('.')[:-1])
        self.original_path: str = exp_path
        # Get all file objects in the experiment's directory
        all_files = [f for f in listdir(exp_path) if path.isfile(path.join(exp_path, f))]
        # Obtain all files that belong to the experiment in question
        exp_files = [f for f in all_files if self.experiment_name in f]
        # Seperately collect different file types - note we collect everything here - later code has to decide
        # which of these files are relevant
        self.laser_files: List[str] = [f for f in exp_files if f.split('.')[-1] == 'laser']
        self.laser_files.sort(key=self._file_sort_key)
        self.fish_temp_files = [f for f in exp_files if f.split('.')[-1] == "temp"]
        self.fish_temp_files.sort(key=self._file_sort_key)
        self.control_temp_files = [f for f in exp_files if f.split('.')[-1] == "c_temp"]
        self.control_temp_files.sort(key=self._file_sort_key)
        self.tail_files: List[str] = [f for f in exp_files if f.split('.')[-1] == 'tail']
        self.tail_files.sort(key=self._file_sort_key)
        self.scanner_fixed_files: List[str] = [f for f in exp_files if ("ImageScannerFixed.txt" in f
                                                                        and "_stableZ_" not in f)]
        self.scanner_fixed_files.sort(key=self._file_sort_key)
        self.ch_0_files: List[str] = [f for f in exp_files if ("_0.tif" in f and "_stableZ_"
                                                               not in f and "Z_0.tif" not in f)]
        self.ch_0_files.sort(key=self._file_sort_key)
        self.ch_1_files: List[str] = [f for f in exp_files if ("_1.tif" in f and "_stableZ_" not in f and "tailImage"
                                                               not in f and "Z_1.tif" not in f)]
        self.ch_1_files.sort(key=self._file_sort_key)
        n_ch_1 = len(self.ch_1_files)
        n_tail = len(self.tail_files)
        n_laser = len(self.laser_files)
        n_ftemp = len(self.fish_temp_files)
        n_ctemp = len(self.control_temp_files)
        self.is_dual_channel: bool = n_ch_1 > 0
        self.has_tail_data: bool = n_tail > 0
        self.has_laser_data: bool = n_laser > 0
        self.has_fish_temp = n_ftemp > 0
        self.has_control_temp = n_ctemp > 0
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
        if self.has_laser_data and n_laser != n_ch_0:
            raise ValueError(f"Ch0 has {n_ch_0} tif stacks but there are {n_laser} laser data files")
        if self.has_fish_temp and n_ftemp != n_ch_0:
            raise ValueError(f"Ch0 has {n_ch_0} tif stacks but there are {n_ftemp} fish temperature files")
        if self.has_control_temp and n_ctemp != n_ch_0:
            raise ValueError(f"Ch0 has {n_ch_0} tif stacks but there are {n_ctemp} control temperature files")
        self.scanner_data: List[Dict[str, Any]] = [ImageScannerFixed(path.join(self.original_path, scfile)).info
                                                   for scfile in self.scanner_fixed_files]

    @staticmethod
    def _file_sort_key(name: str) -> int:
        """
        Finds the imaging plane part of a filename for proper sort order of file names
        :param name: The name of the file
        :return: Imaging plane
        """
        f_name = path.splitext(name)[0]
        plane_start = f_name.find("_Z_") + 3
        plane_end = f_name[plane_start:].find("_")
        if plane_end == -1:
            plane_end = len(f_name)
        try:
            plane_number = int(f_name[plane_start:plane_start + plane_end])
        except ValueError:
            print(f"Could not convert plane identification for {name}")
            raise
        print(plane_number)
        return plane_number
