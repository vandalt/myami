import inspect
import pickle
from pathlib import Path
from typing import List, Optional, Tuple, Union

import amical
import matplotlib.pyplot as plt
import numpy as np
from amical.mf_pipeline.idl_function import dist
from astropy.io import fits
from jwst.datamodels import dqflags
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
from munch import Munch


def amical_save_raw_bs(
    bs: Munch,
    fprefix: str,
    out_dir: str,
    overwrite: bool = False,
    save_pkl: bool = True,
    verbose: bool = False,
):
    """
    Save raw AMICAL bispectrum to oifits and pickle file. Oifits can be more
    convenient, but pkl preserves all information of the bispectrum object.

    :param bs: AMICAL bispectrum
    :type bs: Munch
    :param fprefix: Prefix for file names (without extension)
    :type fprefix: str
    :param out_dir: Output directory
    :type out_dir: str
    :param overwrite: Overwrite existing files if True, defaults to False
    :type overwrite: bool, optional
    :param save_pkl: Save as a pickle file in addition to oifits
    :type save_pkl: bool, optional
    :param verbose: Show extra information if True, defaults to False
    :type verbose: bool, optional
    """
    oifits_fname = fprefix + ".oifits"
    if (Path(out_dir) / oifits_fname).is_file() and not overwrite:
        print(f"File {oifits_fname} exists. Nothing will be saved on disk.")
    else:
        # Save oifits for convenience
        amical.save(
            bs,
            oifits_file=oifits_fname,
            datadir=out_dir,
            verbose=verbose,
        )

    if save_pkl:
        pkl_fname = Path(out_dir) / (fprefix + ".pkl")
        if Path(pkl_fname).is_file() and not overwrite:
            print("NOT SAVING")
            print(f"File {pkl_fname} exists. Nothing will be saved on disk.")
        else:
            # Save pickle to re-use in calibration if needed
            with open(pkl_fname, "wb") as f:
                pickle.dump(bs, f)


def get_bad_mask(
    fpath: str,
    dq_ext: str = "DQ",
    bpix_flags: Union[str, List[str]] = "DO_NOT_USE",
) -> np.ndarray:
    """
    Get bad pixel mask from NIRISS data

    :param fpath: Path to the fits file
    :type fpath: str
    :param dq_ext: Extension with the bad pixel map, defaults to "DQ"
    :type dq_ext: str, optional
    :param bpix_flags: Bad pixel flag(s), defaults to "DO_NOT_USE"
    :type bpix_flags: Union[str, List[str]], optional
    :return: Bad pixel mask for the input file
    :rtype: np.ndarray
    :raises ValueError: ValueError is raised if bpix_flags is unknown
    """

    dqarr = fits.getdata(fpath, dq_ext)

    # Handle "all" special case where any non-zero value is flagged
    if isinstance(bpix_flags, str):
        if bpix_flags.upper() == "ALL":
            return dqarr > 0
        elif bpix_flags in dqflags.pixel:
            bpix_flags = [bpix_flags]
        else:
            raise ValueError(
                "bpix_flags should be 'all', a bad pixel flag (e.g. 'DO_NOT_USE'),"
                " or a list of bad pixel flag strings"
            )

    # Iterate flags and use element-wise OR each time to update map
    bmap = np.zeros(dqarr.shape, dtype=bool)
    for flag_name in bpix_flags:
        bmap = bmap | np.bitwise_and(dqarr, dqflags.pixel[flag_name]).astype(bool)

    return bmap


def amical_cleaning(
    in_file: str,
    cpar: dict,
    ihdu: str = "SCI",
    display: bool = False,
):
    """
    Clean data cube AMICAL

    :param in_file: Path to fits file
    :type in_file: str
    :param cpar: Paramters for AMICAL cleaning (can contain parameters for
                 `select_clean_data` and `show_clean_params`)
    :type cpar: dict
    :param ihdu: Fits extension to use, defaults to "SCI"
    :type ihdu: str, optional
    :param display: Show AMICAL plots if True, defaults to False
    :type display: bool, optional
    """
    # Use copy to avoid modifing existing dict
    cpar = cpar.copy()
    # Get bad pixel map
    if "bad_map" not in cpar:
        bad_map = None
    elif cpar["bad_map"] == "DQ":
        if "bpix_flags" not in cpar:
            cpar["bpix_flags"] = "DO_NOT_USE"
        bad_map = get_bad_mask(in_file, bpix_flags=cpar["bpix_flags"])
    else:
        raise ValueError("Unexpected value for bad_map AMICAL parameter")

    # FIXME: This forces 2d bad pixel map common to all frames.
    # Known limitation in AMICAL, waiting for PR to get merged, then we can remove
    # this. If too long, should be relatively simple to do per-frame correction here
    # with amical.data_processing.fix_bad_pixels() and a loop.
    # AMICAL PR: https://github.com/SydneyAstrophotonicInstrumentationLab/AMICAL/pull/83
    if bad_map is not None and bad_map.ndim == 3:
        bad_map = np.any(bad_map, axis=0)

    cpar["bad_map"] = bad_map

    # TODO: Call only select_clean_data with display=True once supported
    # Ref: https://github.com/SydneyAstrophotonicInstrumentationLab/AMICAL/pull/78
    if display:
        # HACK: see TODO above
        show_clean_args = inspect.signature(amical.show_clean_params).parameters
        show_arg_dict = {k: cpar[k] for k in cpar if k in show_clean_args}
        amical.show_clean_params(in_file, **show_arg_dict, ihdu=ihdu)
        plt.show(block=True)
    # HACK: see TODO above
    select_clean_args = inspect.signature(amical.select_clean_data).parameters
    clean_arg_dict = {k: cpar[k] for k in cpar if k in select_clean_args}
    cube = amical.select_clean_data(
        in_file,
        **clean_arg_dict,
        display=display,
        ihdu=ihdu,
    )
    if display:
        plt.show(block=True)

    return cube


def mask_splodges(
    npix: int, x_splodge: np.ndarray, y_splodge: np.ndarray, splodge_size: float
) -> np.ndarray:
    """
    Get npix by npix boolean mask that is True everywhere except in splodges

    :param npix: Length of the sides in pixels
    :type npix: int
    :param x_splodge: x coordinates of the splodges
    :type x_splodge: np.ndarray
    :param y_splodge: y coordinates of the splodges
    :type y_splodge: np.ndarray
    :param splodge_size: Size of the splodges
    :type splodge_size: float
    :return: Boolean mask that is True only outside of the splodges
    :rtype: np.ndarray
    """

    x_splodge = np.atleast_1d(x_splodge)
    y_splodge = np.atleast_1d(y_splodge)

    npix_arr = np.arange(npix)
    x, y = np.meshgrid(npix_arr, npix_arr)

    # Get x and y difference for each pixel and for all baselines
    # results have shape (nbase, npix, npix)
    xdiff = x_splodge[:, None, None] - x  # x diffs for each baseline
    ydiff = y_splodge[:, None, None] - y  # y diffs for each baseline
    rdiff = np.sqrt(xdiff ** 2 + ydiff ** 2)  # dist from expected peak for each pix

    splodge_mask = np.ones((npix, npix), dtype=bool)
    scond = np.any(rdiff < splodge_size, axis=0)  # unmask true values for any baseline
    splodge_mask[scond] = False  # This mask is false where we look for peaks

    return splodge_mask


def mask_central_splodge(
    npix: int, pixelsize: float, filt_wl: np.ndarray, center: bool = True
) -> np.ndarray:
    """
    Get npix by npix boolean mask that is True in the central splodge

    :param npix: Lenght of the sides in pixel (npix by npix array)
    :type npix: int
    :param pixelsize: Pixel size in rad
    :type pixelsize: float
    :param filt_wl: Length 2 array with filter wavelength in m
    :type filt_wl: np.ndarray
    :param center: Wheter the central splodge is centred, defaults to True
    :type center: bool, optional
    :return: Boolean mask where the central splodge pixels are True
    :rtype: np.ndarray
    """

    # Get index (np.where tuple) and bool mask for central splodge (don't want this peak)
    central_mask = np.zeros((npix, npix), dtype=bool)
    central_ind = tuple(find_central_splodge(npix, pixelsize, filt_wl, center=center))
    central_mask[central_ind] = True

    return central_mask


def find_central_splodge(
    npix: int,
    pixelsize: float,
    filt: np.ndarray,
    hole_diam: float = 0.8,
    center: bool = False,
) -> np.ndarray:
    """
    Find index of pixels that are in central splodge

    Taken from AMICAL with edits:
        - Return either centered or not
        - Remove *0.6 factor because it did not properly match the center in our FTs
        - Fix meshgrid call to use vectors and not integers as arguments

    :param npix: Number of pixels in image
    :type npix: int
    :param pixelsize: Size of a pixel in radians
    :type pixelsize: float
    :param filt: Length 2 array with filter wavelength and error
    :type filt: np.ndarray
    :param hole_diam: Diameter of a single aperture, defaults to 0.8 (NIRISS value)
    :type hole_diam: float, optional
    :param center: Whether the FT 0-frequency is centered, defaults to False
    :type center: bool, optional
    :return: (2, N) array with X and Y vector of indices
    :rtype: np.ndarray
    """

    tmp = dist(npix)
    innerpix = np.array(
        np.array(
            np.where(
                tmp < get_splodge_size(npix, pixelsize, filt, hole_diam=hole_diam) * 0.9
            )
        ),
        dtype=int,
    )

    if center:
        x, y = np.meshgrid(np.arange(npix), np.arange(npix))
        dist_c = np.sqrt((x - npix // 2) ** 2 + (y - npix // 2) ** 2)
        inner_pos = np.array(
            np.where(dist_c < (hole_diam / filt[0] * pixelsize * npix) * 0.9)
        )
        innerpix = np.array(inner_pos, dtype=int)

    return innerpix


def get_splodge_size(
    npix: int, pixelsize: float, filt: np.ndarray, hole_diam: float = 0.8
) -> float:
    """
    Compute splodge size based on pixelsize, number of pixels in image, filter
    and hole diameter

    :param npix: Number of pixels (side length)
    :type npix: int
    :param pixelsize: Size of pixels in radians
    :type pixelsize: float
    :param filt: Filter wavelength in m
    :type filt: np.ndarray
    :param hole_diam: Diameter of a single aperture, defaults to 0.8 (value of NIRISS AMI)
    :type hole_diam: float, optional
    :return: Splodge size
    :rtype: float
    """
    return hole_diam / filt[0] * pixelsize * npix


def show_complex_ps(
    ft_arr: np.ndarray, i_frame: int = 0, savepath: Optional[Union[str, Path]] = None
) -> Figure:
    """
    Show the complex fft image (real and imaginary) and power spectrum (abs(fft)) of the first frame
    to check the applied correction on the cube.

    Taken from AMICAL
    """
    fig = plt.figure(figsize=(16, 6))
    ax1 = plt.subplot(1, 3, 1)
    plt.title("Real part")
    plt.imshow(ft_arr[i_frame].real, cmap="gist_stern", origin="lower")
    plt.subplot(1, 3, 2, sharex=ax1, sharey=ax1)
    plt.title("Imaginary part")
    plt.imshow(ft_arr[i_frame].imag, cmap="gist_stern", origin="lower")
    plt.subplot(1, 3, 3)
    plt.title("Power spectrum (centred)")
    plt.imshow(np.fft.fftshift(abs(ft_arr[i_frame])), cmap="gist_stern", origin="lower")
    plt.tight_layout()

    if savepath is not None:
        plt.savefig(savepath)
    return fig


def construct_ft_arr(cube: np.ndarray, pad: int = 0) -> Tuple[np.ndarray, int, int]:
    """
    Open the data cube and perform a series of roll (both axis) to avoid grid artefact
    (negative fft values). Remove the last row/column in case of odd array.

    Taken from AMICAL

    Parameters:
    -----------
    `cube` {array}: cleaned data cube from amical.select_data().
    `pad` {int}: Padding added to each side or the image

    Returns:
    --------
    `ft_arr` {array}: complex array of the Fourier transform of the cube,\n
    `n_ps` {int}: Number of frames,\n
    `n_pix` {int}: Dimensions of one frames,\n
    """
    if cube.shape[1] % 2 == 1:
        cube = np.array([im[:-1, :-1] for im in cube])

    # Pad on each side of all images
    if pad != 0:
        pad_width = ((0, 0), (pad, pad), (pad, pad))
        cube = np.pad(cube, pad_width)

    n_pix = cube.shape[1]
    cube = np.roll(np.roll(cube, n_pix // 2, axis=1), n_pix // 2, axis=2)

    ft_arr = np.fft.fft2(cube)

    i_ps = ft_arr.shape
    n_ps = i_ps[0]

    return ft_arr, n_ps, n_pix


def get_unique_peaks(
    ft_arr: np.ndarray, n_baselines: int, mf: Munch, center: bool = False
) -> Tuple[List[float]]:
    """
    Get unique peak positions and powers in matched filter for each baseline.
    NOTE: There is a very similar function in amical (peak_info_2d or something like
    that). Maybe better to use this instead in the future

    :param ft_arr: Data cube with FT of each frame
    :type ft_arr: np.ndarray
    :param n_baselines: Number of baselines
    :type n_baselines: int
    :param mf: Matched filter created by AMICAL's make_mf function
    :type mf: Munch
    :param center: Center the FT with np.fft.fftshift before plotting, defaults to False
    :type center: bool, optional
    :return: Lists of x and y coordinates and of weights in the matched filter
    :rtype: Tuple[List[float]]
    """
    cchar = "c" if center else ""
    pvct = mf[f"{cchar}pvct"]
    gvct = mf[f"{cchar}gvct"]

    dim1, dim2 = ft_arr.shape[1], ft_arr.shape[2]
    x, y = np.arange(dim1), np.arange(dim2)
    X, Y = np.meshgrid(x, y)
    lX, lY, lC = [], [], []
    for j in range(n_baselines):
        l_x = X.ravel()[pvct[mf.ix[0, j] : mf.ix[1, j]]]
        l_y = Y.ravel()[pvct[mf.ix[0, j] : mf.ix[1, j]]]
        g = gvct[mf.ix[0, j] : mf.ix[1, j]]

        peak = [[l_y[k], l_x[k], g[k]] for k in range(len(l_x))]

        for x in peak:
            lX.append(x[1])
            lY.append(x[0])
            lC.append(x[2])

    return lX, lY, lC


def show_peak_position(
    ft_arr: np.ndarray,
    n_baselines: int,
    mf: Munch,
    i_fram: int = 0,
    aver: bool = False,
    center: bool = False,
    savepath: Optional[Union[str, Path]] = None,
) -> Figure:
    """
    Show the expected position of the peak in the Fourier space using the
    mask coordinates and the chosen method.

    Taken from AMICAL
    Edits:
        - rename i_frame to ft_frame in loop
        - Use function get_unique_peaks and center arg to get expected positions
        - Remove commented code

    :param ft_arr: Data cube with FT of each frame
    :type ft_arr: np.ndarray
    :param n_baselines: Number of baselines
    :type n_baselines: int
    :param mf: Matched filter created by AMICAL's make_mf function
    :type mf: Munch
    :param i_fram: Frame used in the plot, defaults to 0
    :type i_fram: int, optional
    :param aver: Average all frames before plotting if true, defaults to False
    :type aver: bool, optional
    :param center: Center the FT with np.fft.fftshift before plotting, defaults to False
    :type center: bool, optional
    """
    lX, lY, lC = get_unique_peaks(ft_arr, n_baselines, mf, center=center)

    ft_frame = ft_arr[i_fram]
    if center:
        ft_frame = np.fft.fftshift(ft_frame)
    ps = ft_frame.real

    if aver:
        ps = np.zeros(ps.shape)
        for ft_frame in ft_arr:
            ps += ft_frame.real
        ps /= ft_arr.shape[0]

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(ps, cmap="gist_stern", origin="lower")
    sc = ax.scatter(lX, lY, c=lC, s=20, cmap="viridis")
    divider = make_axes_locatable(ax)
    cax = divider.new_horizontal(size="3%", pad=0.5)
    fig.add_axes(cax)
    cb = fig.colorbar(im, cax=cax)

    cax2 = divider.new_horizontal(size="3%", pad=0.6, pack_start=True)
    fig.add_axes(cax2)
    cb2 = fig.colorbar(sc, cax=cax2)
    cb2.ax.yaxis.set_ticks_position("left")
    cb.set_label("Power Spectrum intensity")
    cb2.set_label("Relative weight [%]", fontsize=20)
    plt.subplots_adjust(
        top=0.965, bottom=0.035, left=0.025, right=0.965, hspace=0.2, wspace=0.2
    )

    if savepath is not None:
        plt.savefig(savepath)

    return fig
