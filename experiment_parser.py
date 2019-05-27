#  Copyright 2019 Martin Haesemeyer. All rights reserved.
#
# Licensed under the MIT license

"""
Module to parse experimental file structure extracting relevant acquisition information
"""


from datetime import datetime


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
            v = line.split(':')[-1].strip()
            t = datetime.strptime(v, "%m/%d/%Y %I:%M:%S %p")
            return "start_time", t
        if "Finish time" in line:
            v = line.split(':')[-1].strip()
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
    pass
