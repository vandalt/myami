# %% [markdown]
# # Extraction and calibration with AMICAL
# **NOTE (2021-12-11): You will need the master branch AMICAL to run this notebook**
#
# At the time of writing this notebook, the PyPI version of AMICAL cannot save raw data,
# but the master branch can. The master branch can be installed with:
# ```
# python -m pip install -U git+https://github.com/SydneyAstrophotonicInstrumentationLab/AMICAL.git
# ```

# %%
from pathlib import Path

import amical
import matplotlib.pyplot as plt
from astropy.io import fits

import utils as ut

# %%
# This assumes a directory where all the fits files can be used.
# You can edit the path and glob pattern to use another setup
dirp = Path("/home/vandal/Documents/data/lower_contrast/")
show_figs = "first"  # "first" or "all" or "none"
clean = True
save_id = "clean"
save_id_for_fname = save_id + "_"
# save_id_for_fname = ""

# %%
# Extract and save raw observables for all fits files
bs_dict = dict()
for i, fp in enumerate(dirp.glob("*.fits")):
    print(fp)
    display = show_figs == "all" or (show_figs == "first" and i == 0)

    # Do cleaning with AMICAL built-in tools
    if clean:
        cube = amical.select_clean_data(
            fp, 79, 31, 2, apod=True, window=32, ihdu="SCI", clip=True, display=display
        )
        # If clipping removed everything, all same level -> don't clip
        if cube.shape[0] == 0:
            cube = amical.select_clean_data(
                fp, 79, 31, 2, apod=True, window=32, ihdu="SCI", clip=False, display=display
            )
        if display:
            amical.show_clean_params(fp, 79, 31, 2, apod=True, window=32, ihdu="SCI")
    else:
        cube = fits.getdata(fp, "SCI")

    # Extract bispectrum for each fits file, calib or target
    bs = amical.extract_bs(
        cube,
        fp,
        peakmethod="fft",
        maskname="g7",
        bs_multi_tri=False,
        fw_splodge=0.7,
        targetname="simbinary",
        display=display,
    )
    ut.amical_save_raw_bs(bs, fp.stem, dirp / f"{save_id_for_fname}amical_raw")

    # Store extraction results for next step (calibration)
    bs_dict[fp.stem] = bs

    if display:
        plt.show(block=True)

# %%
cal_dict = dict()
# Loop over target files
for i, tfile in enumerate(dirp.glob("t*.fits")):
    # Get corresponding calib file
    cfile = dirp / ("c" + tfile.name[1:])

    # If calib file is missing, skip calibration
    if not cfile.is_file():
        print(f"File {cfile} is missing, skipping target file {tfile}")
        continue

    bs_t = bs_dict[tfile.stem]
    bs_c = bs_dict[cfile.stem]
    cal = amical.calibrate(bs_t, bs_c)

    display = show_figs == "all" or (show_figs == "first" and i == 0)
    if display:
        amical.show(cal)
        plt.show()

    amical.save(
        cal,
        oifits_file=tfile.stem + ".oifits",
        datadir=dirp / f"{save_id_for_fname}amical_calib",
    )
    cal_dict[tfile.stem] = cal
