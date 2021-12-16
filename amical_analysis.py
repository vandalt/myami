# %% [markdown]
# Short AMICAL analysis (binary fit) script, inspired from AMICAL tutorial

# Disclaimer: I have very little experience with CANDID, feel free to edit this
# and send a PR if it can be improved

# %%
from pathlib import Path
from typing import Union

import amical
import matplotlib.pyplot as plt
from astropy import units as u

# %%
dirp = Path("/home/vandal/Documents/data/lower_contrast/")
show_figs = "first"  # "first" or "all" or "none"
extract_id = "clean"
oifits_subdir = f"{extract_id}_amical_calib"

# %%
param_candid = {
    "rmin": 10,  # inner radius of the grid
    "rmax": 400,  # outer radius of the grid
    "step": 50,  # grid sampling
    "ncore": 1,  # core for multiprocessing
}
cen_wavels = {
    "F480M": 4.8e-6,
    "F430M": 4.3e-6,
    "F380M": 3.8e-6,
}
bmax = 5.28


# %%
def analyze_one_fle(inputdata: Union[Path, str], display: bool = False) -> dict:

    inputdata = Path(inputdata)

    filt = inputdata.stem.split("_")[4]
    step = cen_wavels[filt] / (4 * bmax) * u.rad
    intstep = int(step.to("mas").value)  # make integer mas
    param_candid["step"] = intstep

    # %%
    fit1 = amical.candid_grid(
        inputdata, **param_candid, diam=0, doNotFit=["diam*"], save=False
    )

    fit_dict[inputdata.stem] = fit1

    # Plot and save the fitted model
    if display:
        amical.plot_model(inputdata, fit1["best"], save=False)

        amical.candid_cr_limit(
            inputdata, **param_candid, fitComp=fit1["comp"], save=False
        )

    if display:
        plt.show()
    else:
        plt.close("all")

    return fit1


# %%
# Either run this loop or use function above with path to a file
# TODO: Save fits in pandas CSV
fit_dict = dict()
for i, inputdata in enumerate((dirp / oifits_subdir).glob("t_*.oifits")):
    display = show_figs == "all" or (show_figs == "first" and i == 0)
    fit_dict[inputdata.stem] = analyze_one_fle(inputdata, display=display)
    if i == 2:
        break
