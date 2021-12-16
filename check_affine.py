"""
Use AMICAL to predict splodge location and compare with FT of data.
"""
import glob
import inspect
import warnings
from argparse import ArgumentParser
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from amical.get_infos_obs import get_pixel_size, get_wavelength
from amical.mf_pipeline.ami_function import compute_index_mask, make_mf
from astropy.io import fits
from astropy.table import unique
from munch import Munch
from photutils import centroids
from photutils.detection import find_peaks

import utils as aut
from affine import (compute_affine2d, get_affine_map_sample,
                    reparametrize_affine_rot, reparametrize_affine_shear)
from lin_mini import linear_minimization

plt.style.use("tableau-colorblind10")


PEAK_FUNCTIONS = {
    "com": centroids.centroid_com,
    "quadratic": centroids.centroid_quadratic,
    "gauss1d": centroids.centroid_1dg,
    "gauss2d": centroids.centroid_2dg,
}


def plot_scatter(
    uv_diff: np.ndarray,
    display: bool = False,
    save_path: Optional[Union[Path, str]] = None,
):

    fig = plt.figure(figsize=(10, 10))
    for bl in range(uv_diff.shape[-1]):
        plt.scatter(*uv_diff[:, :, bl].T, label=str(bl))
        plt.text(*uv_diff[0, :, bl].T, str(bl), color="k", fontsize=12, zorder=1000)
        plt.axhline(0, linestyle="--", color="k", alpha=0.5)
        plt.axvline(0, linestyle="--", color="k", alpha=0.5)
        plt.xlabel("$\Delta u$ [pixel]")
        plt.ylabel("$\Delta v$ [pixel]")
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
    if display:
        plt.show()
    else:
        plt.close(fig)


def plot_vs_baseline(
    df: pd.DataFrame,
    xkey: str,
    ykey: str,
    nbase: int,
    xlab: Optional[str] = None,
    ylab: Optional[str] = None,
    display: bool = False,
    save_path: Optional[Union[Path, str]] = None,
):

    fig = plt.figure(figsize=(10, 10))
    for bl in range(nbase):
        df_bl = df.xs(bl, level="baseline")
        plt.plot(df_bl[xkey], df_bl[ykey], "o")
        plt.text(
            df_bl[xkey].values[0],
            df_bl[xkey].max(),
            str(bl),
            color="k",
            fontsize=12,
            zorder=1000,
        )
    if xlab is not None:
        plt.xlabel(xlab)
    if ylab is not None:
        plt.ylabel(ylab)
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
    if display:
        plt.show()
    else:
        plt.close(fig)


def print_param_summary(params: np.ndarray) -> None:

    # Print fitted parameters
    pmean = params.mean(axis=0)
    pstd = params.std(axis=0)
    print("Fitted parameters:")
    print(f"  dx: {pmean[0]} +/- {pstd[0]}")
    print(f"  dy: {pmean[1]} +/- {pstd[1]}")
    print(f"  a: {pmean[2]} +/- {pstd[2]}")
    print(f"  b: {pmean[3]} +/- {pstd[3]}")
    print(f"  c: {pmean[4]} +/- {pstd[4]}")
    print(f"  d: {pmean[5]} +/- {pstd[5]}")
    print()

    # Get parameters for "rotation" parametrization
    rotpar = np.array([reparametrize_affine_rot(*p[2:]) for p in params])
    rotpar_mean = rotpar.mean(axis=0)
    rotpar_std = rotpar.std(axis=0)
    print("'Rotation' parameters:")
    print(f"  scale: {rotpar_mean[0]} +/- {rotpar_std[0]}")
    print(f"  rot: {np.degrees(rotpar_mean[1])} +/- {rotpar_std[1]}")
    print(f"  xy ratio: {rotpar_mean[2]} +/- {rotpar_std[2]}")
    print(f"  diag ratio: {rotpar_mean[3]} +/- {rotpar_std[3]}")
    print()

    # Get "scale/shear" parametrization
    shpar = np.array([reparametrize_affine_shear(*p[2:]) for p in params])
    shpar_mean = shpar.mean(axis=0)
    shpar_std = shpar.std(axis=0)
    print("'Scale+Shear' parameters:")
    print(f"  x scale: {shpar_mean[0]} +/- {shpar_std[0]}")
    print(f"  y scale: {shpar_mean[1]} +/- {shpar_std[1]}")
    print(f"  x shear: {shpar_mean[2]} +/- {shpar_std[2]}")
    print(f"  y shear: {shpar_mean[3]} +/- {shpar_std[3]}")
    print("\n")


def find_max_pixels(
    ft_cube: np.ndarray,
    mf: Munch,
    nbase: int,
    pixelsize: float,
    filt_wl: np.ndarray,
    center: bool = True,
) -> np.ndarray:
    """
    Get X and Y positions for each peak in each frame of AMI FT

    :param ft_cube: FT cube
    :type ft_cube: np.ndarray
    :param mf: Matched filter created by AMICAL's make_mf function
    :type mf: Munch
    :param nbase: Number of baselines
    :type nbase: int
    :param pixelsize: Pixel size in rad
    :type pixelsize: float
    :param filt_wl: Length 2 array with filter wavelength in m
    :type filt_wl: np.ndarray
    :param center: Center the FT if True. Only set to false if ft_cube is already
                   centered, defaults to True
    :type center: bool, optional
    :return: Peak X and Y positions in an (nframes, 2, nbase) array
    :rtype: np.ndarray
    :raises RuntimeError: Raises error if a peak is found twice in the same frame
    :raises RuntimeError: Raises error if the number of peaks is not equal to the number
                          of baselines
    """
    # Use centered FFT because we don't want splodges to be split at the boundary
    if center:
        ft_cube = np.fft.fftshift(ft_cube)

    npix = ft_cube.shape[1]
    nframes = ft_cube.shape[0]

    ps_cube = np.abs(ft_cube)  # Don't square because values are already big

    splodge_size = aut.get_splodge_size(
        npix, pixelsize, filt_wl
    )  # Splodge size [pixels]

    # To find peaks only in one half of FT, use mask with AMICAL predicted locations
    xmod, ymod, _ = np.array(aut.get_unique_peaks(ft_cube, nbase, mf, center=True))

    # Mask central splodge, and also everything not roughly in expected splodge location
    central_mask = aut.mask_central_splodge(npix, pixelsize, filt_wl, center=True)
    splodge_mask = aut.mask_splodges(npix, xmod, ymod, splodge_size)
    peak_mask = central_mask | splodge_mask

    peaks = np.empty((nframes, 2, nbase))
    for i, ps in enumerate(ps_cube):

        # Find splodeg peaks in FT
        # mask central for threshold, it is much higher
        threshold = ps[central_mask].max() / 100
        pk = find_peaks(
            ps, threshold, box_size=splodge_size, mask=peak_mask, npeaks=nbase
        )
        if len(pk) != len(unique(pk)):
            raise RuntimeError(
                "The same splodge peak was found twice. This should not happen"
            )
        if len(unique(pk)) != nbase:
            word = "less" if len(unique(pk)) < nbase else "more"
            raise RuntimeError(
                f"Found {word} splodge peaks than the number of baselines in frame {i}."
                f"Only {len(unique(pk))} splodges were found, but there are {nbase} baselines"
            )

        pk_arr = np.array([pk["x_peak"].data, pk["y_peak"].data], dtype=int)
        pk_mod_dists = np.array(
            [
                (pk_arr[0] - xm) ** 2 + (pk_arr[1] - ym) ** 2
                for xm, ym in zip(xmod, ymod)
            ]
        )
        pk_inds = np.argmin(pk_mod_dists, axis=1)

        peaks[i] = pk_arr[:, pk_inds]

    return peaks


def find_peak_locations(
    ft_cube: np.ndarray,
    peaks: np.ndarray,
    index_mask: Munch,
    pixelsize: float,
    filt_wl: np.ndarray,
    box_size: Optional[int] = None,
    center: bool = True,
    peak_func: Callable = centroids.centroid_quadratic,
) -> np.ndarray:
    """
    Find exact position of each peak in AMI FT (i.e. below pixel level)

    :param ft_cube: FT cube
    :type ft_cube: np.ndarray
    :param peaks: Array of peak (nframes, 2, nbase), can be output of find_max_pixels
    :type peaks: np.ndarray
    :param index_mask: Object with max index info from AMICAL's compute_index_mask
    :type index_mask: Munch
    :param pixelsize: Pixel size in rad
    :type pixelsize: float
    :param box_size: Size (side length) of the box around the splodge center used in
                     the peak fitting
    :type box_size: int
    :param filt_wl: Length 2 array with filter wavelength in m
    :type filt_wl: np.ndarray
    :param center: Center the FT if True. Only set to false if ft_cube is already
                   centered, defaults to True
    :type center: bool, optional
    :return: Array of peak positions (similar to peaks, but sub-pixel)
    :rtype: np.ndarray
    """

    if center:
        ft_cube = np.fft.fftshift(ft_cube)

    npix = ft_cube.shape[1]

    ps_cube = np.abs(ft_cube)  # Don't square because values are already big

    splodge_size_float = aut.get_splodge_size(
        npix, pixelsize, filt_wl
    )  # Splodge size [pixels]
    splodge_size = int(splodge_size_float)

    # Meshgrid to compute square mask around each peak in the loop
    x, y = np.meshgrid(np.arange(npix), np.arange(npix))

    new_peaks = np.empty_like(peaks)
    for iframe, ps in enumerate(ps_cube):

        # NOTE: Photutil's centroid_sources could eliminate this loop,
        # but it does not accept xpeak and ypeak, which are essential here.
        # Related upstream issue: https://github.com/astropy/photutils/issues/1270
        for ibase in range(index_mask.n_baselines):
            pk = peaks[iframe, :, ibase]

            # We want to work with a square around the peak, not whole image
            xdiff = np.abs(x - pk[0])
            ydiff = np.abs(y - pk[1])

            # Define box around splodge
            # NOTE: Masking the corners makes the quad fits fail for some reason,
            # so simply using a box here
            mask = (xdiff < splodge_size) & (ydiff < splodge_size)
            side_len = int(np.sqrt(mask.nonzero()[0].size))

            splodge_ps = ps[mask].reshape(side_len, side_len).copy()

            # Get peak position in PS cropped around splodge pixel peak
            peak_func_kwargs = {}
            if peak_func == centroids.centroid_quadratic:
                peak_func_kwargs["xpeak"] = side_len // 2
                peak_func_kwargs["ypeak"] = side_len // 2

            # For NIRISS AMI splodges, a box size of three seems best to avoid
            # biasing the fit with neighbouring splodges or regions.
            if box_size is not None:
                peak_func_kwargs["fit_boxsize"] = box_size

            # Call the peak-finding function for the current splodge
            npk = peak_func(splodge_ps, **peak_func_kwargs)

            # Convert "splodge coordinates" to the full U-V coords and store peak
            npk_uv = pk + (npk - side_len // 2)
            new_peaks[iframe, :, ibase] = npk_uv

    return new_peaks


def fit_affine_model(df: pd.DataFrame, npix: int):
    fgroup = df.groupby("frame")
    nframes = len(fgroup)
    params = np.empty((nframes, 6))
    # uv_recon_all = np.empty_like(locs)
    for iframe, df_frame in fgroup:
        uv_vals = np.append(df_frame.u, df_frame.v) - npix // 2
        uv_mod = np.append(df_frame.u_mod, df_frame.v_mod) - npix // 2
        sample = get_affine_map_sample(uv_vals)
        pvect, _ = linear_minimization(uv_mod, sample)
        params[iframe] = pvect

    return params


def compute_uv_pos(
    in_files: List[Union[str, Path]],
    maskname,
    clean_params: Optional[dict] = None,
    extract_params: Optional[dict] = None,
    display: bool = False,
    pad_factor: int = 0,
    peakmethod: str = "unique",
    peak_func: Callable = centroids.centroid_quadratic,
    box_size: Union[int, Tuple[int]] = 3,
    out_dir: Optional[Union[str, Path]] = None,
    verbose: bool = False,
) -> pd.DataFrame:

    if out_dir is not None:
        out_dir = Path(out_dir)

    nskip_dither = 0
    n_ta = 0
    n_clearp = 0
    out_df_dict = dict()
    out_df_params_dict = dict()
    for i, fpath in enumerate(in_files):

        # Extract header info for current file
        hdr = fits.getheader(fpath)

        # Skip file if not AMI science file
        if hdr["SUBARRAY"] == "SUBTAAMI":
            # Skip TA files (they make AMICAL crash when one integration)
            print("Skipping TA file")
            n_ta += 1
            continue
        elif hdr["PUPIL"] == "CLEARP":
            # Skip CLEARP files
            print("Skipping CLEARP file")
            n_clearp += 1
            continue

        # Clean data before FT using AMICAL
        try:
            if clean_params is not None:
                cube = aut.amical_cleaning(fpath, clean_params, display=display)
            else:
                cube = fits.getdata(fpath, "SCI")
        except ValueError as e:
            # TODO: If padding doesn't become an option in AMICAL, will need to handle
            # these files in the same way as in run_amical.
            if hdr["PATT_NUM"] > 1:
                print(f"Skipping file with primary dithered (AMICAL gave error '{e}')")
                nskip_dither += 1
                continue
            else:
                raise e

        nframes = cube.shape[0]

        # Optionally pad the data to make peak fittting easier
        npix_og = cube.shape[1]
        if pad_factor > 1:
            pad = (npix_og // 2) * (pad_factor - 1)
        else:
            pad = 0

        # Get FT cube (one FT per frame)
        ft_cube, n_ps, npix = aut.construct_ft_arr(cube, pad=pad)
        if display:
            aut.show_complex_ps(ft_cube)

        # Get AMICAL matched filter for splodge position
        if extract_params is not None:
            make_mf_params = {
                k: v
                for k, v in inspect.signature(make_mf).parameters.items()
                if v.default != inspect.Parameter.empty
            }
            make_mf_kwargs = {
                k: extract_params[k] for k in extract_params if k in make_mf_params
            }
        else:
            make_mf_kwargs = {}

        instrument = hdr["INSTRUME"]
        filt_name = hdr["FILTER"]
        make_mf_kwargs["peakmethod"] = peakmethod

        # Generic info re-used multiple times for this file
        file_id = Path(fpath).stem
        index_mask = compute_index_mask(7)
        pixelsize = get_pixel_size(instrument)  # Pixel size of the detector [rad]
        filt_wl = get_wavelength(instrument, filt_name)

        # Get "model" peaks from AMICAL's matched filter
        mf = make_mf(
            maskname,
            instrument,
            filt_name,
            npix,
            display=display,
            diag_plot=display,
            **make_mf_kwargs,
        )

        # Get uv position below pixel level to compare later
        uv = np.array([mf.u, mf.v]) * pixelsize * npix / filt_wl[0] + npix // 2
        if display:
            aut.show_peak_position(ft_cube, index_mask.n_baselines, mf, center=True)
            plt.show()

        # Find pixel with peak of each splodge
        max_peaks = find_max_pixels(
            ft_cube, mf, index_mask.n_baselines, pixelsize, filt_wl
        )

        # Get U and V peak locations below pixel level
        locs = find_peak_locations(
            ft_cube,
            max_peaks,
            index_mask,
            pixelsize,
            filt_wl,
            peak_func=peak_func,
            box_size=box_size,
        )

        # Plot locations on top of FT
        if display:
            plt.show()  # AMICAL functions above might display stuff
            plt.imshow(
                np.fft.fftshift(np.abs(ft_cube))[0], cmap="gist_stern", origin="lower"
            )
            for j in range(index_mask.n_baselines):
                plt.text(*uv[:, j] + 1, str(j), color="white", fontsize=12)
            for i, loc in enumerate(locs):
                plt.scatter(*loc)
                for lo in loc.T:
                    plt.text(*lo, str(i), color="C0", fontsize=12)
            plt.show()

        # Re-scale results if we padded the image
        factor = npix_og / npix
        locs *= factor
        uv *= factor
        uv_diff = locs - uv

        # Plot how far away points are from 0
        opath = out_dir / "uv_diff" / f"uv_diff_{file_id}.pdf" if out_dir else None
        plot_scatter(uv_diff, display=display, save_path=opath)

        # Store locations and distances in a dataframe for this file
        # (index with frame and baseline)
        inds = [(f, b) for f in range(n_ps) for b in range(index_mask.n_baselines)]
        locs_t = np.swapaxes(locs, 1, 2)
        uv_diff_t = np.swapaxes(uv_diff, 1, 2)
        arr = np.append(locs_t, uv_diff_t, axis=2).reshape(-1, 4)
        arr = np.append(arr, np.tile(uv, n_ps).T, axis=1)
        inds = pd.MultiIndex.from_tuples(inds, names=["frame", "baseline"])
        df = pd.DataFrame(
            arr, index=inds, columns=["u", "v", "u_diff", "v_diff", "u_mod", "v_mod"]
        )
        out_df_dict[file_id] = df  # Keep df in a dict for later

        # Add distance of u and v difference to dataframe
        df["uv_off"] = np.sqrt(df.u_diff ** 2 + df.v_diff ** 2)  # abolute offsets
        # Record distance from center for each expected splodge location
        u_from_center = df.u_mod - npix // 2
        v_from_center = df.v_mod - npix // 2
        df["uv_dist"] = np.sqrt(u_from_center ** 2 + v_from_center ** 2)
        df["uv_ang"] = np.arctan2(v_from_center, u_from_center)

        opath = (
            out_dir / "off_vs_dist" / f"off_vs_dist_{file_id}.pdf" if out_dir else None
        )
        plot_vs_baseline(
            df,
            "uv_dist",
            "uv_off",
            index_mask.n_baselines,
            xlab="Distance $D$ from center [pixel]",
            ylab="$\sqrt{\Delta u^2 + \Delta v^2}$ [pixel]",
            display=display,
            save_path=opath,
        )
        opath = (
            out_dir / "off_vs_angle" / f"off_vs_angle_{file_id}.pdf"
            if out_dir
            else None
        )
        plot_vs_baseline(
            df,
            "uv_ang",
            "uv_off",
            xlab=r"Angle $\theta$ in U-V plane",
            ylab="$\sqrt{\Delta u^2 + \Delta v^2}$ [pixel]",
            nbase=index_mask.n_baselines,
            display=display,
            save_path=opath,
        )

        # Now that we have an overview of differences, fit Affine2d model
        params = fit_affine_model(df, npix)

        # Compute adjusted splodge locations from model parameters
        uv_all = np.append(df.u, df.v) - npix // 2  # Reshape U-V for affine2d functions
        pmean = params.mean(axis=0)  # Use mean to transform whole file the same way
        uv_recon_all = compute_affine2d(uv_all, *pmean)
        df[["u_corr_affine", "v_corr_affine"]] = uv_recon_all.reshape(
            len(df), 2, order="F"
        )
        uv_recon_vals = np.swapaxes(
            uv_recon_all.reshape(len(df), 2, order="F").reshape(nframes, 21, 2), 1, 2
        )  # Reshape to match uv model values
        uv_diff_recon = (uv_recon_vals - uv) + npix // 2
        if verbose:
            print_param_summary(params)

        df_params = pd.DataFrame(params, columns=["dx", "dy", "a", "b", "c", "d"])
        out_df_params_dict[file_id] = df_params

        # Plot how far away points are from 0
        opath = (
            out_dir / "uv_diff_recon" / f"uv_diff_recon_{file_id}.pdf"
            if out_dir
            else None
        )
        plot_scatter(uv_diff_recon, display=display, save_path=opath)

        if display:
            plt.show()
            display = False

    # Save everything in one big csv file, add index level for file
    all_df = pd.concat(out_df_dict, names=["file", *df.index.names])
    all_df_params = pd.concat(out_df_params_dict, names=["file", "frame"])
    if out_dir is not None:
        all_df.to_csv(out_dir / "uv_output.csv")
        all_df_params.to_csv(out_dir / "params.csv")

    print("FINAL SUMMARY")
    print(
        f"Average absolute u difference: {all_df.u_diff.abs().mean()} +/- {all_df.u_diff.abs().std()}"
    )
    print(
        f"Average absolute v difference: {all_df.v_diff.abs().mean()} +/- {all_df.v_diff.abs().std()}"
    )
    print(
        f"Average absolute u-v offset radius: {all_df.uv_off.mean()} +/- {all_df.uv_off.std()}"
    )

    all_params = all_df_params.values
    print_param_summary(all_params)

    print(f"Skipped {n_ta} TA files")
    print(f"Skipped {n_clearp} CLEARP files")
    print(f"Skipped {nskip_dither} files with primary dither")
    print(f"Total files: {len(in_files)}")
    print(f"Files used: {len(in_files) - (n_ta + n_clearp + nskip_dither)}")

    return all_df


def main():
    psr = ArgumentParser(
        description="Check splodge (aperture) locations and fit with 2d model",
    )
    psr.add_argument(
        "datadir",
        type=str,
        help="Directory of the input (calints) files.",
    )
    psr.add_argument(
        "amical_params",
        type=str,
        default=None,
        help="Path to yaml file with AMICAL parameters",
    )
    psr.add_argument(
        "-p",
        "--pattern",
        type=str,
        default="jw*calints.fits",
        help="Pattern to match files in data directory.",
    )
    psr.add_argument(
        "-s",
        "--save-path",
        dest="out_dir",
        type=str,
        default="affine_out",
        help="Directory where outputs will be saved",
    )
    psr.add_argument(
        "-d",
        "--dry",
        action="store_true",
        help="Perform 'dry' run (do not save results).",
    )
    psr.add_argument(
        "-b",
        "--box-size",
        dest="box_size",
        type=int,
        default=3,
        help="Box size to fit splodge max position",
    )
    psr.add_argument(
        "-m",
        "--mask",
        type=str,
        default="g7",
        help="Mask name to give to AMICAL.",
    )
    psr.add_argument(
        "--peak-method",
        dest="peakmethod",
        type=str,
        choices=["unique", "fft", "square", "gauss"],
        default="unique",
        help="Method used to predict splodge location with AMICAL",
    )
    psr.add_argument(
        "--peak-function",
        dest="peak_function",
        type=str,
        choices=list(PEAK_FUNCTIONS.keys()),
        default="quadratic",
        help="Method used to fit peaks with photutils",
    )
    psr.add_argument(
        "-f",
        "--pad-factor",
        dest="pad_factor",
        type=int,
        default=0,
        help="Pad the array to have factor*orignal size array before FT. Use 0 or 1 to turn off.",
    )
    psr.add_argument(
        "--skip-clean",
        action="store_false",
        dest="clean",
        help="Skip data cleaning.",
    )
    psr.add_argument(
        "-c",
        "--clip",
        action="store_true",
        help="Clip bad frames with AMICAL.",
    )
    psr.add_argument(
        "--show-first",
        dest="show_first",
        action="store_true",
        help="Show plots for first file",
    )
    psr.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=None,  # Override False as default even if store true
        help="Verbose mode (off by default)",
    )
    args = psr.parse_args()

    datadir = Path(args.datadir)

    in_files = sorted(glob.glob(str(datadir / args.pattern)))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.amical_params is not None:
        with open(args.amical_params, "r") as amical_file:
            amical_params = yaml.safe_load(amical_file)
    else:
        amical_params = {}

    if "clean" in amical_params and args.clean:
        clean_params = amical_params["clean"]
        clean_params["clip"] = args.clip
    elif args.clean:
        raise KeyError("amical params should have an 'clean' key.")
    else:
        clean_params = None

    if "extraction" in amical_params:
        extract_params = amical_params["extraction"]
    else:
        extract_params = None

    if extract_params is not None:
        try:
            maskname = extract_params["maskname"]
        except KeyError:
            maskname = args.mask
    elif "maskname" in amical_params:
        maskname = amical_params["maskname"]
    else:
        maskname = args.mask

    display = args.show_first

    df = compute_uv_pos(
        in_files,
        maskname,
        box_size=args.box_size,
        clean_params=clean_params,
        peakmethod=args.peakmethod,
        peak_func=PEAK_FUNCTIONS[args.peak_function],
        extract_params=extract_params,
        display=display,
        pad_factor=args.pad_factor,
        out_dir=out_dir if not args.dry else None,
        verbose=args.verbose,
    )

    return df


if __name__ == "__main__":

    all_df = main()
