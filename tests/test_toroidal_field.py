import numpy as np
import pytest

from tfpy.toroidalField import ToroidalField, derivatePol, derivateTor


def test_constant_field_evaluation_and_broadcasting():
    field = ToroidalField.constantField(3.5, nfp=2, mpol=2, ntor=1)
    theta = np.array([[0.0], [np.pi / 2]])
    zeta = np.array([0.0, 0.2, 0.4])
    values = field.getValue(theta, zeta)
    np.testing.assert_allclose(values, 3.5)
    assert values.shape == (2, 3)


def test_fourier_coefficients_and_derivatives():
    size = (2 * 1 + 1) * 2 + 1 + 1
    re = np.zeros(size)
    im = np.zeros(size)
    field = ToroidalField(2, 2, 1, re, im)
    field.setRe(1, 0, 2.0)
    field.setIm(1, 1, 3.0)
    assert field.getRe(1, 0) == 2.0
    assert field.getIm(-1, -1) == -3.0
    theta, zeta = np.array([0.3]), np.array([0.1])
    eps = 1e-6
    numerical_pol = (field.getValue(theta + eps, zeta) - field.getValue(theta - eps, zeta)) / (2 * eps)
    numerical_tor = (field.getValue(theta, zeta + eps) - field.getValue(theta, zeta - eps)) / (2 * eps)
    np.testing.assert_allclose(derivatePol(field).getValue(theta, zeta), numerical_pol, rtol=1e-5)
    np.testing.assert_allclose(derivateTor(field).getValue(theta, zeta), numerical_tor, rtol=1e-5)


def test_invalid_shapes_and_resolution_are_explicit():
    with pytest.raises(ValueError):
        ToroidalField(1, 1, 1, np.zeros(4), np.zeros(4))
    with pytest.raises(IndexError):
        ToroidalField(1, 1, 1, np.zeros(5), np.zeros(5)).indexMap(2, 0)
