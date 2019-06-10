#  Copyright 2019 Martin Haesemeyer. All rights reserved.
#
# Licensed under the MIT license

"""
Class wrapper of caiman motion correction and unit extraction
"""


import numpy as np
from tifffile import imsave, imread
from os import path, makedirs, remove
import caiman as cm
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.source_extraction.cnmf import params as params


class CaImAn:
    def __init__(self, indicator_decay_time: float, fov_um: float, time_per_frame: float, **kwargs):
        """
        Creates new CaImAn instance
        :param indicator_decay_time: Decay time of the calcium indicator in seconds
        :param fov_um: The size of the imaged fov in um
        :param time_per_frame: The time of each frame acquisition in seconds
        """
        # NOTE: There are still hidden and non-settable parameters in "extract_components"
        self.decay_time = indicator_decay_time
        self.fov_um = fov_um
        self.time_per_frame = time_per_frame
        # directory for saving motion correction results
        if"ana_dir" in kwargs:
            self.ana_dir = kwargs["ana_dir"]
        else:
            self.ana_dir = "analysis"
        # quantile for dff detrending
        if"detrend_dff_quantile_min" in kwargs:
            self.detrend_dff_quantile_min = kwargs["detrend_dff_quantile_min"]
        else:
            self.detrend_dff_quantile_min = 8
        # time window for detrending quantile measurement in seconds
        if"detrend_dff_time_window" in kwargs:
            self.detrend_dff_time_window = kwargs["detrend_dff_time_window"]
        else:
            self.detrend_dff_time_window = 180
        # Indicates whether to use non-rigid motion correction
        if"pw_rigid" in kwargs:
            self.pw_rigid = kwargs["pw_rigid"]
        else:
            self.pw_rigid = True
        # The expected radius of neurons in the dataset in um
        if"neuron_radius" in kwargs:
            self.neuron_radius = kwargs["neuron_radius"]
        else:
            self.neuron_radius = 3.0
        # The size of patches for non-rigid motion correction in um
        if"patch_motion_um" in kwargs:
            self.patch_motion_um = kwargs["patch_motion_um"]
        else:
            self.patch_motion_um = (50.0, 50.0)  # patch size for non-rigid motion correction
        # Whether to save a motion corrected projection through the stack
        if"save_projection" in kwargs:
            self.save_projection = kwargs["save_projection"]
        else:
            self.save_projection = True

    @property
    def detrend_dff_params(self):
        return {"quantileMin": self.detrend_dff_quantile_min,
                "frames_window": int(self.detrend_dff_time_window/self.time_per_frame)}

    def motion_correct(self, fname: str) -> (np.ndarray, dict):
        """
        Uses caiman non-rigid motion correction to remove/reduce motion artefacts
        Note: ALways saves an intermediate mem-map representation in order C of the corrected 32-bit stack
        :param fname: The filename of the source file
        :return:
            [0]: Corrected stack as a memmap
            [1]: Wrapped CaImAn parameter dictionary
        """
        cont_folder = path.dirname(fname)  # the containing folder
        stack_name = path.split(fname)[1]
        save_dir = cont_folder + f"/{self.ana_dir}"
        if not path.exists(save_dir):
            makedirs(save_dir)
            print("Created analysis directory", flush=True)
        out_name = save_dir + '/' + stack_name
        test_image = imread(fname, key=0)  # load first frame of stack to compute resolution
        assert test_image.shape[0] == test_image.shape[1]
        resolution = self.fov_um / test_image.shape[0]

        fr = 1 / self.time_per_frame  # frame-rate
        dxy = (resolution, resolution)  # spatial resolution in um/pixel
        max_shift_um = (self.neuron_radius*4, self.neuron_radius*4)  # maximally allow shift by ~2 cell diameters
        # maximum allowed rigid shift in pixels
        max_shifts = [int(a / b) for a, b in zip(max_shift_um, dxy)]
        # start a new patch for pw-rigid motion correction every x pixels
        strides = tuple([int(a / b) for a, b in zip(self.patch_motion_um, dxy)])
        # overlap between patches (size of patch in pixels: strides+overlaps)
        overlaps = (24, 24)
        # maximum deviation allowed for patch with respect to rigid shifts (unit unclear - likely pixels as it is int)
        max_deviation_rigid = 3
        # create parameter dictionary
        mc_dict = {
            'fnames': [fname],
            'fr': fr,
            'decay_time': self.decay_time,
            'dxy': dxy,
            'pw_rigid': self.pw_rigid,
            'max_shifts': max_shifts,
            'strides': strides,
            'overlaps': overlaps,
            'max_deviation_rigid': max_deviation_rigid,
            'border_nan': 'copy'
        }
        opts = params.CNMFParams(params_dict=mc_dict)
        # start a cluster for parallel processing
        c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=None, single_thread=False)
        try:
            # motion correction
            mc = MotionCorrect(fname, dview=dview, **opts.get_group('motion'))
            # Run (piecewise-rigid motion) correction using NoRMCorre
            # if we don't save the movie into a memmap, there doesn't seem to be
            # any possibility to get at the corrected data later???
            mc.motion_correct(save_movie=True)
            # memory mapping
            border_to_0 = 0 if mc.border_nan is 'copy' else mc.border_to_0
            # memory map the file in order 'C'
            fname_new = cm.save_memmap(mc.mmap_file, base_name=out_name, order='C', border_to_0=border_to_0)
            # delete the original mem-map
            if mc.fname_tot_els is None:
                [remove(fn) for fn in mc.fname_tot_rig]
            else:
                [remove(fn) for fn in mc.fname_tot_els]
            # now load the new file and transform output into [z,y,x] stack
            yr, dims, n_t = cm.load_memmap(fname_new)
            images = np.reshape(yr.T, [n_t] + list(dims), order='F')
            if self.save_projection:
                # save anatomical projection as 16bit tif
                anat_projection = images.copy()
                anat_projection = np.sum(anat_projection, 0)
                anat_projection -= anat_projection.min()
                anat_projection /= anat_projection.max()
                anat_projection *= (2 ** 16 - 1)
                anat_projection[anat_projection < 0] = 0
                anat_projection[anat_projection > (2 ** 16 - 1)] = (2 ** 16 - 1)
                anat_projection = anat_projection.astype(np.uint16)
                imsave(out_name, anat_projection, imagej=True, resolution=(1 / dxy[0], 1 / dxy[1]),
                       metadata={'axes': 'YX', 'unit': 'um'})
        finally:
            cm.stop_server(dview=dview)
        return images, {"Motion Correction": mc_dict}

    def extract_components(self, images, fname) -> (cnmf.CNMF, cnmf.CNMF, dict):
        """
        Uses constrained NNMF to extract spatial and temporal components, performs deconvolution and validates
        extracted components
        :param images: The tyx stack from which components should be extracted
        :param fname: The name of the original file from which <images> originated
        :return:
            [0]: Cnmf object after initial extraction
            [1]: Cnmf object after deconvolution and subsequent component validation
            [2]: Wrapped CaImAn parameter dictionary
        """
        resolution = self.fov_um / images.shape[1]
        fr = 1 / self.time_per_frame  # frame-rate
        dxy = (resolution, resolution)  # spatial resolution in um/pixel
        c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=None, single_thread=False)
        try:
            # set up parameters for source extraction
            p = 1  # order of the autoregressive system
            gnb = 2  # number of global background components
            merge_thr = 0.85  # merging threshold, max correlation allowed
            rf = 15  # half-size of the patches in pixels. e.g., if rf=25, patches are 50x50
            stride_cnmf = 6  # amount of overlap between the patches in pixels
            patch_area_um = (rf * 2 * dxy[0]) ** 2  # we use this to calculate expected number of components per patch
            neur_area_um = np.pi * self.neuron_radius ** 2
            K = int(patch_area_um / neur_area_um)  # number of components (~neurons) per patch
            # expected half size of neurons in pixels
            gSig = [int(self.neuron_radius / dxy[0]), int(self.neuron_radius / dxy[1])]
            method_init = 'greedy_roi'  # initialization method (if analyzing dendritic data using 'sparse_nmf')
            ssub = 2  # spatial subsampling during initialization
            tsub = 1  # temporal subsampling during intialization
            # parameters for component evaluation
            opts_dict = {'fnames': [fname],  # NOTE: This parameter seems only necessary to allow contour extraction
                         'fr': fr,
                         'decay_time': self.decay_time,
                         'dxy': dxy,
                         'nb': gnb,
                         'rf': rf,
                         'K': K,
                         'gSig': gSig,
                         'stride': stride_cnmf,
                         'method_init': method_init,
                         'rolling_sum': True,
                         'merge_thr': merge_thr,
                         'n_processes': n_processes,
                         'only_init': True,
                         'ssub': ssub,
                         'tsub': tsub}
            opts = params.CNMFParams(params_dict=opts_dict)
            # First extract spatial and temporal components on patches and combine them
            # for this step deconvolution is turned off (p=0)
            opts.change_params({'p': 0})
            cnm = cnmf.CNMF(n_processes, params=opts, dview=dview)
            cnm = cnm.fit(images)
            # rerun seeded CNMF on accepted patches to refine and perform deconvolution
            cnm.params.change_params({'p': p})
            cnm2 = cnm.refit(images, dview=dview)
            # component evaluations
            # components are evaluated in three ways:
            #   a) the shape of each component must be correlated with the data
            #   b) a minimum peak SNR is required over the length of a transient
            #   c) each shape passes a CNN based classifier
            min_SNR = 2  # signal to noise ratio for accepting a component
            rval_thr = 0.8  # space correlation threshold for accepting a component
            cnn_thr = 0.99  # threshold for CNN based classifier
            cnn_lowest = 0.1  # neurons with cnn probability lower than this value are rejected
            val_dict = {
                'decay_time': self.decay_time,
                'min_SNR': min_SNR,
                'rval_thr': rval_thr,
                'use_cnn': True,
                'min_cnn_thr': cnn_thr,
                'cnn_lowest': cnn_lowest}
            cnm2.params.set('quality', val_dict)
            cnm2.estimates.evaluate_components(images, cnm2.params, dview=dview)
            # update object with selected components
            cnm2.estimates.select_components(use_object=True)
            # extract DF/F values
            cnm2.estimates.detrend_df_f(**self.detrend_dff_params)
        finally:
            cm.stop_server(dview=dview)
        return cnm, cnm2, {"CNMF": opts_dict, "Validation": val_dict}
