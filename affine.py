from typing import Tuple

import numpy as np


def get_affine_map_sample(xy: np.ndarray) -> np.ndarray:
    """
    Get 'sample' matrix to represent affine 2d mapping for linear minizmation.

    :param xy: 1-D xy vector with the x values in the first half followed by the y values
    :type xy: np.ndarray
    :return: 6 by len(xy) matrix with one row per affine parameter and one column per x or y point
    :rtype: np.ndarray
    """

    xy = np.atleast_1d(xy)
    ntot = xy.size
    npts = ntot // 2

    x, y = np.split(xy, 2)
    sample = np.zeros([6, ntot])
    sample[0, :npts] = 1  # dx, set to 1 in 1st half because additive in u
    sample[1, npts:] = 1  # dy, set to 1 in 2nd half because additive in v
    sample[2, :] = np.append(x, np.zeros_like(y))  # A, maps from u to u'
    sample[3, :] = np.append(y, np.zeros_like(y))  # B, maps from v to u'
    sample[4, :] = np.append(np.zeros_like(x), x)  # C, maps from u to v'
    sample[5, :] = np.append(np.zeros_like(x), y)  # D, maps from v to v'

    return sample


def compute_affine2d(
    xy: np.ndarray, x0: float, y0: float, a: float, b: float, c: float, d: float
) -> np.ndarray:
    """
    Compute affine 2d transform using the format of `linear_minimzation` outputs.

    :param xy: 1-D xy vector with the x values in the first half followed by the y values
    :type xy: np.ndarray
    :param x0: Offset along the x axis
    :type x0: float
    :param y0: Offset along the y axis
    :type y0: float
    :param a: Parameter mapping u to u'
    :type a: float
    :param b: Parameter mapping v to u'
    :type b: float
    :param c: Parameter mapping u to v'
    :type c: float
    :param d: Parameter mapping v to v'
    :type d: float
    :return: xy' vector of transformed coordinates
    :rtype: np.ndarray
    """

    # 1-D Parameter vector representing affine transform
    pvect = np.append([x0, y0], [a, b, c, d])

    # Sample is a 2d matrix with one row per parameter.
    # Each column maps to a u' or v' point (first half u' second half v')
    sample = get_affine_map_sample(xy)

    recon = np.dot(pvect, sample)

    return recon


def reparametrize_affine_rot(a: float, b: float, c: float, d: float) -> Tuple[float]:
    """
    Go from [[a, b], [c, d]] matrix to uniform scale, rotation, xy scale and diagonal scale
    This is a representation of the following matrix addition for the transform:
        M = scale * (  [cos(t) -sin(t)] + [ xy_scale_ratio  diag_scale_ratio] )
                       [sin(t)  cos(t)]   [ diag_scale_ratio -xy_scale_ratio]

    :param a: First affine matrix parameter ([0, 0], u->u')
    :type a: float
    :param b: Second affine matrix parameter ([0, 1], v->u')
    :type b: float
    :param c: Third affine matrix parameter ([1, 0], u->v')
    :type c: float
    :param d: Fourth affine matrix parameter ([1, 1], v->v')
    :type d: float
    :return: Transofmed parameters (scale, angle, and xy and diagonal scale ratios).
             Angle is in radians.
    :rtype: Tuple[float]
    """

    mat = np.array([[a, b], [c, d]])

    scale = np.sqrt(np.linalg.det(mat))

    mat_u = mat / scale

    a = mat_u[0, 0]
    b = mat_u[0, 1]
    c = mat_u[1, 0]
    d = mat_u[1, 1]

    theta = np.arcsin((c - b) / 2)

    xy_ratio = (a - d) / 2
    diag_ratio = (c + b) / 2

    return scale, theta, xy_ratio, diag_ratio


def reparametrize_affine_shear(a: float, b: float, c: float, d: float) -> Tuple[float]:
    """
    Go from [[A, B], [C, D]] matrix to x scale, y scale, x shear, y shear
    This is a representation of the following matrix addition for the transform:
        M = [[xscale, 0]] . [[1, xshear]] = [[xscale, xscale*xhsear]]
            [[0, yscale]]   [[yshear, 1]]   [[yscale*yshear, yscale]]

    :param a: First affine matrix parameter ([0, 0], u->u')
    :type a: float
    :param b: Second affine matrix parameter ([0, 1], v->u')
    :type b: float
    :param c: Third affine matrix parameter ([1, 0], u->v')
    :type c: float
    :param d: Fourth affine matrix parameter ([1, 1], v->v')
    :type d: float
    :return: Transofmed parameters (x scale, y scale, x shear, y shear).
    :rtype: Tuple[float]
    """

    xscale = a
    yscale = d
    xshear = b / a
    yshear = c / d

    return xscale, yscale, xshear, yshear
