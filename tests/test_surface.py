import numpy as np
import pytest

from tfpy.geometry import Surface_cylindricalAngle
from tfpy.toroidalField import ToroidalField


def test_surface_coordinates_and_symmetry():
    r = ToroidalField.constantField(2.0, nfp=2, mpol=0, ntor=0)
    z = ToroidalField.constantField(0.5, nfp=2, mpol=0, ntor=0)
    surface = Surface_cylindricalAngle(r, z, reverseToroidalAngle=False)
    theta = np.array([[0.0, np.pi]])
    zeta = np.array([[0.0, np.pi / 2]])
    rr, zz = surface.getRZ(theta, zeta)
    np.testing.assert_allclose(rr, 2.0)
    np.testing.assert_allclose(zz, 0.5)
    x, y, _ = surface.getXYZ(theta, zeta)
    np.testing.assert_allclose(x, 2 * np.cos(zeta))
    np.testing.assert_allclose(y, 2 * np.sin(zeta))


def test_surface_rejects_mismatched_fields():
    r = ToroidalField.constantField(1.0, nfp=1, mpol=1, ntor=0)
    z = ToroidalField.constantField(1.0, nfp=2, mpol=1, ntor=0)
    with pytest.raises(ValueError):
        Surface_cylindricalAngle(r, z)
