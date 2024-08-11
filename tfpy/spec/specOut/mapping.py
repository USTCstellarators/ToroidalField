import numpy as np
from coilpy import FourSurf
from typing import Tuple


def cylinder2spec(self, lvol: int, r: np.ndarray, phi: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray]:
    """
    Returns:
        s, theta, zeta
    """

    assert r.shape == phi.shape == z.shape
    shape = r.shape
    r, phi, z = r.flatten(), phi.flatten(), z.flatten()

    innerSurf = FourSurf.read_spec_output(self, lvol)
    outSurf = FourSurf.read_spec_output(self, lvol+1)
    axis = FourSurf.read_spec_output(self, 0)

    zeta = phi
    
    axisR, axisZ = axis.rz(0, 0)
    axisR, axisZ = float(axisR), float(axisZ)
    # theta = np.arctan2(z-axisZ, r-axisR)
    theta = (-np.arctan2(z-axisZ, r-axisR)+2*np.pi) % (2*np.pi)

    innerR, innerZ = innerSurf.rz(theta, zeta)
    outR, outZ = outSurf.rz(theta, zeta)
    s = (np.power(r-innerR,2) + np.power(z-innerZ,2)) / (np.power(outR-innerR,2) + np.power(outZ-innerZ,2))
    s = 2*np.sqrt(s) - 1
    return s.reshape(shape), theta.reshape(shape), zeta.reshape(shape)

