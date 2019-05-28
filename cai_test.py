#  Copyright 2019 Martin Haesemeyer. All rights reserved.
#
# Licensed under the MIT license

"""
Script for testing caiman - cai_demo.py implementation with slight adjustments
"""


import numpy as np
from tifffile import imsave
from os import path, makedirs, remove
from utilities import ui_get_file
import matplotlib.pyplot as pl
import seaborn as sns
import logging
import caiman as cm
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.source_extraction.cnmf import params as params
from cai_wrapper import CaImAn
from experiment_parser import ExperimentParser

logging.basicConfig(format=
                    "%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s]"\
                    "[%(process)d] %(message)s", level=logging.INFO)


def main(zoom_level, t_per_frame, filenames):
    f = filenames[0]
    cont_folder = path.dirname(f)  # the containing folder
    stack_name = path.split(f)[1]
    save_dir = cont_folder + "/analysis"
    if not path.exists(save_dir):
        makedirs(save_dir)
        print("Created analysis directory", flush=True)
    out_name = save_dir+'/'+stack_name

    fr = 1 / t_per_frame  # frame-rate
    decay_time = 4.0  # decay time approximation of nuclear 6s
    dxy = (500/512/zoom_level, 500/512/zoom_level)  # spatial resolution in um/pixel
    max_shift_um = (12.0, 12.0)  # maximally allow shift by ~2 cell diameters
    patch_motion_um = (50.0, 50.0)  # patch size for non-rigid motion correction

    # motion correction parameters
    pw_rigid = True  # use non-rigid motion correction
    # maximum allowed rigid shift in pixels
    max_shifts = [int(a/b) for a, b in zip(max_shift_um, dxy)]
    # start a new patch for pw-rigid motion correction every x pixels
    strides = tuple([int(a/b) for a, b in zip(patch_motion_um, dxy)])
    # overlap between patches (size of patch in pixels: strides+overlaps)
    overlaps = (24, 24)
    # maximum deviation allowed for patch with respect to rigid shifts
    max_deviation_rigid = 3

    mc_dict = {
        'fnames': [f],
        'fr': fr,
        'decay_time': decay_time,
        'dxy': dxy,
        'pw_rigid': pw_rigid,
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
        mc = MotionCorrect(f, dview=dview, **opts.get_group('motion'))

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
        del mc  # this object is now likely invalid
        # now load the new file - why do we need the intermediate? Because of order vs reshape???
        yr, dims, n_t = cm.load_memmap(fname_new)
        images = np.reshape(yr.T, [n_t] + list(dims), order='F')
        # save anatomical projection as 16bit tif
        anat_projection = images.copy()
        anat_projection = np.sum(anat_projection, 0)
        anat_projection -= anat_projection.min()
        anat_projection /= anat_projection.max()
        anat_projection *= (2**16 - 1)
        anat_projection[anat_projection < 0] = 0
        anat_projection[anat_projection > (2**16 - 1)] = (2**16 - 1)
        anat_projection = anat_projection.astype(np.uint16)
        imsave(out_name, anat_projection, imagej=True, resolution=(1/dxy[0], 1/dxy[1]),
               metadata={'axes': 'YX', 'unit': 'um'})
        # restart cluster
        cm.stop_server(dview=dview)
        c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=None, single_thread=False)
        print(f"Starting component extraction on {n_processes} parallel processes")
        # set up parameters for source extraction
        p = 1  # order of the autoregressive system
        gnb = 2  # number of global background components
        merge_thr = 0.85  # merging threshold, max correlation allowed
        rf = 15  # half-size of the patches in pixels. e.g., if rf=25, patches are 50x50
        stride_cnmf = 6  # amount of overlap between the patches in pixels
        patch_area_um = (rf*2*dxy[0])**2  # we use this to calculate expected number of components per patch
        neur_area_um = np.pi * 6**2
        K = int(patch_area_um/neur_area_um)  # number of components (~neurons) per patch
        gSig = [int(3/dxy[0]), int(3/dxy[1])]  # expected half size of neurons in pixels
        method_init = 'greedy_roi'  # initialization method (if analyzing dendritic data using 'sparse_nmf')
        ssub = 2  # spatial subsampling during initialization
        tsub = 1  # temporal subsampling during intialization
        # parameters for component evaluation
        opts_dict = {'fnames': [f],
                     'fr': fr,
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
        opts.change_params(params_dict=opts_dict)
        # First extract spatial and temporal components on patches and combine them
        # for this step deconvolution is turned off (p=0)
        opts.change_params({'p': 0})
        cnm = cnmf.CNMF(n_processes, params=opts, dview=dview)
        cnm = cnm.fit(images)
        # plot contours of found components on local correlation image
        Cn = cm.local_correlations(images, swap_dim=False)
        Cn[np.isnan(Cn)] = 0
        cnm.estimates.plot_contours(img=Cn)
        pl.title('Contour plots of found components')
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
        cnm2.params.set('quality', {'decay_time': decay_time,
                                    'min_SNR': min_SNR,
                                    'rval_thr': rval_thr,
                                    'use_cnn': True,
                                    'min_cnn_thr': cnn_thr,
                                    'cnn_lowest': cnn_lowest})
        cnm2.estimates.evaluate_components(images, cnm2.params, dview=dview)
        # plot components
        cnm2.estimates.plot_contours(img=anat_projection/anat_projection.max(), idx=cnm2.estimates.idx_components)
        # update object with selected components
        cnm2.estimates.select_components(use_object=True)
        # extract DF/F values
        cnm2.estimates.detrend_df_f(quantileMin=8, frames_window=int(60/t_per_frame))
        # show final traces
        cnm2.estimates.view_components(img=Cn)
    finally:
        cm.stop_server(dview=dview)
    return cnm, cnm2


def main2(fov, t_per_frame, filename):
    cai = CaImAn(4.0, fov, t_per_frame)
    im = cai.motion_correct(filename)[0]
    return cai.extract_components(im, filename)[1]


if __name__ == "__main__":
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"

    info_file = ui_get_file(filetypes=[('Experiment info', '*.info')], multiple=False)
    if type(info_file) == list:
        info_file = info_file[0]
    eparser = ExperimentParser(info_file)
    fnames = [path.join(eparser.original_path, ch0) for ch0 in eparser.ch_0_files]
    frame_duration = eparser.info_data["frame_duration"]
    all_c = []
    for i, fn in enumerate(fnames):
        fit_cnm2 = main2(eparser.scanner_data[i]["fov"], frame_duration, fn)
        all_c.append(fit_cnm2.estimates.C)
    all_c = [a for b in all_c for a in b]
    regressors = np.load("rh56_regs.npy")
    regressors = np.r_[regressors, regressors, regressors]
    r_times = np.arange(regressors.shape[0]) / 5
    # data_times = np.arange(all_c.shape[1])*frame_duration
    r_mat = np.full((len(all_c), regressors.shape[1]), np.nan)
    interp_c = np.zeros((len(all_c), r_times.size))
    for i, reg in enumerate(regressors.T):
        for j, trace in enumerate(all_c):
            data_times = np.arange(trace.size) * frame_duration
            i_trace = np.interp(r_times, data_times, trace)
            interp_c[j, :] = i_trace
            r_mat[j, i] = np.corrcoef(i_trace, reg)[0, 1]
    fig, (ax1, ax2) = pl.subplots(1, 2)
    sns.heatmap(r_mat, vmin=-1, vmax=1, center=0, ax=ax1)
    r_mat[r_mat < 0.5] = 0
    sns.heatmap(r_mat, vmin=0, vmax=1, ax=ax2)
    fig.tight_layout()
    membership = np.full(r_mat.shape[0], -1)
    membership[np.max(r_mat, 1) > 0] = np.argmax(r_mat, 1)[np.max(r_mat, 1) > 0]
