#  Copyright 2019 Martin Haesemeyer. All rights reserved.
#
# Licensed under the MIT license

"""
Module with general utility functions and classes
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
from collections import Counter
import numba
from numba import njit
from scipy.signal.signaltools import lfilter
import matplotlib.pyplot as pl
import seaborn as sns
from peakfinder import peakdet


class ExperimentException(Exception):
    """
    Exception to signal that invalid operation was performed on Experiment
    """
    def __init__(self, message: str):
        super().__init__(message)


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


def filtfilt(x, window_len):
    """
    Performs zero-phase digital filtering with a boxcar filter. Data is first
    passed through the filter in the forward and then backward direction. This
    approach preserves peak location.
    """
    if x.ndim != 1:
        raise ValueError("filtfilt only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    # pad signals with reflected versions on both ends
    s = np.r_[x[window_len - 1:0:-1], x, x[-1:-window_len:-1]]
    # create our smoothing window
    w = np.ones(window_len, 'd')
    # convolve forwards
    y = np.convolve(w / w.sum(), s, mode='valid')
    # convolve backwards
    yret = np.convolve(w / w.sum(), y[-1::-1], mode='valid')
    return yret[-1::-1]


@njit(numba.float64[:](numba.float64[:], numba.int32))
def comp_vigor(cum_angle, winlen=10):
    """
    Computes the swim vigor based on a cumulative angle trace
    as the windowed standard deviation of the cumAngles
    """
    s = cum_angle.size
    vig = np.zeros(s)
    for i in range(winlen, s):
        vig[i] = np.std(cum_angle[i - winlen + 1:i + 1])
    return vig


def detect_tail_bouts(cum_angles, vigor, threshold=10, min_frames=20, frame_rate=250):
    """
    Detects swim-bouts in a tail cumulative angle trace or vigor trace (sliding sdev of cum_angle) and collects their
    characteristics.
    :param cum_angles: The cumulative angle trace
    :param vigor: The swim vigor as a sliding standard deviation of the cum angle trace
    :param threshold: This threshold has to be crossed in the vigor trace
    :param min_frames: The minimal length of a bout in frames
    :param frame_rate: The acquisition frame rate
    :return:
        Tail-bout array with nBout rows and 8 columns:
        start - end - meanAmpl. - bias - duration(s) - beatFreq. - power - vigor
    """
    above_t = vigor > threshold
    crossings = np.r_[0, np.diff(above_t.astype(int))]
    # find first and last frames of bouts
    s = np.nonzero(crossings > 0)[0]
    e = np.nonzero(crossings < 0)[0] - 1

    if len(s) == 0 or len(e) == 0:
        return None

    # trim in case our recording started or ended in the middle of a bout
    if s[0] > e[0]:
        # our first start is AFTER our first end. Remove the first bout end
        # as it belongs to a bout that was ongoing upon experiment start
        e = e[1::]

    if s.size > e.size:
        # the last bout starts bout doesn't end before the experiment ended
        # remove its start
        s = s[0:-1]

    ln = e - s + 1  # length of all putative bouts
    # remove episodes that are too short
    e = e[ln >= min_frames]
    s = s[ln >= min_frames]
    # remove episodes

    if e.size < 1:
        return None

    # pre-allocate bout matrix based on number of putative bouts
    bouts = np.zeros((e.size, 8))

    # loop over putative bouts and collect all data. Bouts that don't satisfy
    # criteria will be marked for removal by setting the start (column 0) to
    # -1
    for i in range(e.size):
        btrace = cum_angles[s[i]:e[i] + 1]
        # some bouts have a long tail (lead maybe?) during which there wasn't
        # any actual tail undulation and some detected events only consist of
        # slow angle drifts. To trim these periods off (and remove non-bouts)
        # we look at the angular speed (diff(cumAngles)) and trim off frames
        # after which the speed trace does not go above 10 degrees (or below -10)
        # we also remove periods that don't have any frame with at least a
        # 20 degree cumAngle
        ang_speed = np.diff(btrace)
        absang_speed = np.abs(ang_speed)
        speed_above = np.nonzero(absang_speed > 5)[0]
        if speed_above.size == 0 or np.abs(btrace).max() < 20:
            bouts[i, 0] = -1
            continue
        # do relaxed trim. Keep up to 5 frames before and after the first/last
        # threshold crossing respectively
        if speed_above[0] > 6:  # need to trim head
            s[i] += speed_above[0] - 6
        if speed_above[-1] < btrace.size - 5:  # need to trim tail
            e[i] = e[i] - btrace.size + speed_above[-1] + 5

        # pruned bout trace
        btrace = cum_angles[s[i]:e[i] + 1]

        # detect peaks, mark bout for removal if we can't find any
        maxima, minima = peakdet(btrace)
        n_peaks = maxima.shape[0] + minima.shape[0]
        if n_peaks == 0:
            bouts[i, 0] = -1
            continue
        if minima.ndim == 1:
            minima = np.zeros((1, 2))
        if maxima.ndim == 1:
            maxima = np.zeros((1, 2))
        bouts[i, 0] = s[i]
        bouts[i, 1] = e[i]
        # mean amplitude
        bouts[i, 2] = (maxima[:, 1].sum() - minima[:, 1].sum()) / n_peaks
        # bias
        bouts[i, 3] = maxima[:, 1].sum() + minima[:, 1].sum()
        # duration
        bouts[i, 4] = (e[i] - s[i] + 1) / frame_rate
        # beat frequency
        bouts[i, 5] = n_peaks / bouts[i, 4]
        # power
        bouts[i, 6] = maxima[:, 1].sum() - minima[:, 1].sum()
        # vigor
        bouts[i, 7] = bouts[i, 2] * bouts[i, 5]

    # remove bouts marked as such
    starts = bouts[:, 0]
    return bouts[starts != -1, :]


class TailData:
    # class representing tail track data of one experimental plane
    def __init__(self, file_data, ca_timeconstant, frame_rate, scan_frame_length=None):
        """
        Creates a new TailData object
        Args:
            file_data: Matrix loaded from tailfile
            ca_timeconstant: Timeconstant of calcium indicator used during experiments
            frame_rate: The tail camera acquisition framerate
            scan_frame_length: For more accurate alignment the time it took to acquire each 2P scan frame
        """
        self.scanning = file_data[:, 0] == 1
        self.scan_frame = file_data[:, 1].astype(int)
        self.scan_frame[np.logical_not(self.scanning)] = -1
        # after the last frame is scanned, the scanImageIndex will be incremented further
        # and the isScanning indicator will not immediately switch off. Therefore, if
        # the highest index frame has less than 75% of the average per-index frame-number
        # set it to -1 as well
        c = Counter(self.scan_frame[self.scan_frame != -1])
        avg_count = np.median(list(c.values()))
        max_frame = np.max(self.scan_frame)
        if np.sum(self.scan_frame == max_frame) < 0.75*avg_count:
            self.scan_frame[self.scan_frame == max_frame] = -1
        self.cumAngles = np.rad2deg(file_data[:, 2])
        self.ca_original = self.cumAngles.copy()
        self.remove_track_errors()
        self.vigor = comp_vigor(self.cumAngles, 80//(1000/frame_rate))
        # compute vigor bout threshold
        t = np.mean(self.vigor[self.vigor < 25]) + 3*np.std(self.vigor[self.vigor < 25])
        print(f"Vigor threshold = {t}")
        self.bouts = detect_tail_bouts(self.cumAngles, self.vigor, min_frames=80//(1000/frame_rate), threshold=t,
                                       frame_rate=frame_rate)
        if self.bouts is not None and self.bouts.size == 0:
            self.bouts = None
        if self.bouts is not None:
            bs = self.bouts[:, 0].astype(int)
            self.boutFrames = self.scan_frame[bs]
        else:
            self.boutFrames = []
        self.ca_kernel = TailData.ca_kernel(ca_timeconstant, frame_rate)
        self.ca_timeconstant = ca_timeconstant
        # try to compute frame-rate from timestamps in the tail-data if they are present
        # use heuristics to determine if the 4th column (index 3) contains timestamp data
        putative_ts = file_data[:, 3]
        if np.all(np.diff(putative_ts) > 0) and np.mean(putative_ts) > 1e9:
            frame_time_ms = np.median(np.diff(putative_ts)) / 1_000_000  # timestamp in ns
            print(f"Found timestamp information in tail file. Median time between frames is {int(frame_time_ms)} ms")
            self.frame_rate = int(1000 / frame_time_ms)
            print(f"Set tail camera frame-rate to {self.frame_rate} Hz")
        else:
            print("Did not find timestamp information in tail file.")
            self.frame_rate = frame_rate
        # compute tail velocities based on 10-window filtered cumulative angle trace
        fca = lfilter(np.ones(10)/10, 1, self.cumAngles)
        self.velocity = np.hstack((0, np.diff(fca)))
        self.velcty_noise = np.nanstd(self.velocity[self.velocity < 4])
        # compute a time-trace assuming a constant frame-rate which starts at 0
        # for the likely first camera frame during the first acquisition frame
        # we infer this frame by going back avgCount frames from the first frame
        # of the second (!) scan frame (i.e. this should then be the first frame
        # of the first scan frame)
        frames = np.arange(self.cumAngles.size)
        first_frame = np.min(frames[self.scan_frame == 1]) - avg_count.astype(int)
        # remove overhang from frame 0 call
        self.scan_frame[:first_frame] = -1
        if scan_frame_length is not None:
            # build frame-time tied to the scan-frame clock in case camera acquisition does not follow the
            # intended frame rate: For the camera frame which is in the middle of a given scan-frame set
            # its time to the middle time of scan acquisition of that frame. Then interpolate times in between
            ix_key_frame = []
            key_frame_times = []
            # add very first frame and its approximated time purely based on camera time
            ix_key_frame.append(0)
            key_frame_times.append((frames[0] - first_frame) / frame_rate)
            for i in range(self.scan_frame.max()):
                ix_key = int(np.mean(frames[self.scan_frame == i]))
                key_time = scan_frame_length / 2 + scan_frame_length * i
                ix_key_frame.append(ix_key)
                key_frame_times.append(key_time)
            # use linear interpolation to create times for each frame
            self.frame_time = np.interp(frames, np.array(ix_key_frame), np.array(key_frame_times), right=np.nan)
            # self.frameTime = self.frameTime[np.logical_not(np.isnan(self.frameTime))]
        else:
            frames -= first_frame
            self.frame_time = (frames / frame_rate).astype(np.float32)
        # create bout-start trace at original frame-rate
        self.starting = np.zeros_like(self.frame_time)
        if self.bouts is not None:
            bout_starts = self.bouts[:, 0].astype(int)
            # since we potentially clip our starting trace to the last valid frame-time (experiment end)
            # we also only include bout-starts that occured up to that index
            bout_starts = bout_starts[bout_starts < self.frame_time.size]
            self.starting[bout_starts] = 1

    def remove_track_errors(self):
        """
        If part of the agarose gel boundary is visible in the frame
        the tracker will occasionally latch onto it for single frames.
        Tries to detect these instances and corrects them
        """
        vicinity = np.array([-2, -1, 1, 2])
        for i in range(2, self.cumAngles.size-2):
            d_pre = self.cumAngles[i] - self.cumAngles[i-1]
            d_post = self.cumAngles[i+1] - self.cumAngles[i]
            # the current point is surrounded by two similar cumulative angles
            # that are both 45 degrees away in the same direction
            if (d_pre > 30 and d_post < -30) or (d_pre < -30 and d_post > 30):
                # the angles in the vicinity of the current point are similar
                if np.ptp(self.cumAngles[vicinity+i]) < 10:
                    self.cumAngles[i] = (self.cumAngles[i-1] + self.cumAngles[i+1])/2

    def plot_bouts(self):
        bs, be = self.bout_starts_ends
        with sns.axes_style('white'):
            pl.figure()
            pl.plot(self.cumAngles, label='Angle trace')
            if bs is not None:
                pl.plot(bs, self.cumAngles[bs], 'r*', label='Starts')
                pl.plot(be, self.cumAngles[be], 'k*', label='Ends')
            pl.ylabel('Cumulative tail angle')
            pl.xlabel('Frames')
            sns.despine()

    @property
    def per_frame_vigor(self):
        """
        For each scan frame returns the average
        swim vigor
        """
        sf = np.unique(self.scan_frame)
        sf = sf[sf != -1]
        sf = np.sort(sf)
        conv_vigor = np.convolve(self.vigor, self.ca_kernel, mode='full')[:self.vigor.size]
        pfv = np.zeros(sf.size)
        for i, s in enumerate(sf):
            pfv[i] = np.mean(conv_vigor[self.scan_frame == s])
        return pfv

    @property
    def bout_starts_ends(self):
        if self.bouts is None:
            return None, None
        else:
            return self.bouts[:, 0].astype(int), self.bouts[:, 1].astype(int)

    @property
    def convolved_starting(self):
        """
        Returns a convolved version of the camera frame-rate
        bout start trace
        """
        return np.convolve(self.starting, self.ca_kernel, mode='full')[:self.starting.size]

    @property
    def frame_bout_starts(self):
        """
        Returns a convolved per-frame bout-start trace
            image_freq: Imaging frequency
        """
        if self.bouts is None:
            sf = np.unique(self.scan_frame)
            sf = sf[sf != -1]
            return np.zeros(sf.size)
        conv_starting = self.convolved_starting
        # collect all valid scan-frames
        sf = np.unique(self.scan_frame)
        sf = sf[sf != -1]
        sf = np.sort(sf)
        per_f_s = np.zeros(sf.size)
        for i, s in enumerate(sf):
            per_f_s[i] = np.mean(conv_starting[self.scan_frame == s])
        return per_f_s

    @staticmethod
    def load_tail_data(filename, ca_time_constant, frame_rate=100, scan_frame_length=None):
        try:
            data = np.genfromtxt(filename, delimiter='\t')
        except (IOError, OSError):
            return None
        return TailData(data, ca_time_constant, frame_rate, scan_frame_length)

    @staticmethod
    def ca_kernel(tau, frame_rate):
        """
        Creates a calcium decay kernel for the given frameRate
        with the given half-life in seconds
        """
        fold_length = 5  # make kernel length equal to 5 half-times (decay to 3%)
        klen = int(fold_length * tau * frame_rate)
        tk = np.linspace(0, fold_length*tau, klen, endpoint=False)
        k = 2**(-1*tk/tau)
        k = k / k.sum()
        return k
