#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# potentialBoozerAngle.py


import numpy as np
from .surfaceBoozerAngle import Surface_BoozerAngle
from ..toroidalField import ToroidalField


def singlePotential_BoozerSurf(x: np.ndarray, y: np.ndarray, z: np.ndarray, 
                               surf: Surface_BoozerAngle, 
                               potential: ToroidalField,
                               npol: int=256, ntor: int=256) -> np.ndarray:
    r"""
    Calculate the single potential on a surface using Boozer coordinates.   
    Args:
        x (np.ndarray): x-coordinates of the points.
        y (np.ndarray): y-coordinates of the points.
        z (np.ndarray): z-coordinates of the points.
        surf (Surface_BoozerAngle): Surface \Omgea using Boozer coordinates.
        potential (ToroidalField): potential \sigma on the Boozer surface.
        npol (int): resolution of numerical integration in the poloidal direction.
        ntor (int): resolution of numerical integration in the toroidal direction.
    Returns:       
        np.ndarray: the single potential at the points \mathbf{x}(x, y, z).
            $$
                \S(\mathbf{x}) = \int_\partial\Omega \frac{\sigma(\mathbf{y})}{|\mathbf{x}-\mathbf{y}|} d\omega(\mathbf{y})
            $$
    """
    assert x.shape == y.shape == z.shape
    dtheta, dzeta = 2*np.pi/npol, 2*np.pi/surf.nfp/ntor
    _thetaarr = np.linspace(0, 2*np.pi, npol, endpoint=False) + dtheta/10
    _zetaarr = np.linspace(0, 2*np.pi, surf.nfp*ntor, endpoint=False) + dzeta/10
    zetaGrid, thetaGrid = np.meshgrid(_zetaarr, _thetaarr)
    sigmaGrid = potential.getValue(thetaGrid, zetaGrid)
    rGrid, zGrid = surf.getRZ(thetaGrid, zetaGrid)
    phiGrid = surf.getPhi(thetaGrid, zetaGrid)
    xGrid, yGrid = np.cos(phiGrid) * rGrid, np.sin(phiGrid) * rGrid
    try:
        position_theta = np.transpose(np.array([surf.position_theta[_i].getValue(thetaGrid, zetaGrid) for _i in range(3)]))
        position_zeta = np.transpose(np.array([surf.position_zeta[_i].getValue(thetaGrid, zetaGrid) for _i in range(3)]))
    except AttributeError:
        surf.updateBasis()
        position_theta = np.transpose(np.array([surf.position_theta[_i].getValue(thetaGrid, zetaGrid) for _i in range(3)]))
        position_zeta = np.transpose(np.array([surf.position_zeta[_i].getValue(thetaGrid, zetaGrid) for _i in range(3)]))
    normalGrid = np.cross(position_theta, position_zeta)
    dareaGrid = np.transpose(np.linalg.norm(normalGrid, axis=-1))
    deltaPosition = np.array([
        x[..., np.newaxis, np.newaxis] - xGrid,
        y[..., np.newaxis, np.newaxis] - yGrid,
        z[..., np.newaxis, np.newaxis] - zGrid
    ])
    lengthDeltaPosition = np.linalg.norm(deltaPosition, axis=0)
    return dtheta * dzeta * np.einsum(
        'ij,...ij->...', sigmaGrid*dareaGrid, 1/lengthDeltaPosition
    )
    
    
def doublePotential_BoozerSurf(x: np.ndarray, y: np.ndarray, z: np.ndarray, 
                               surf: Surface_BoozerAngle, 
                               potential: ToroidalField, 
                               npol: int=256, ntor: int=256) -> np.ndarray:
    r"""
    Calculate the double potential on a surface using Boozer coordinates.
    Args:
        x (np.ndarray): x-coordinates of the points.
        y (np.ndarray): y-coordinates of the points.
        z (np.ndarray): z-coordinates of the points.
        surf (Surface_BoozerAngle): Surface \Omgea using Boozer coordinates.
        potential (ToroidalField): potential \mu on the Boozer surface.
        npol (int): resolution of numerical integration in the poloidal direction.
        ntor (int): resolution of numerical integration in the toroidal direction.
    Returns:
        np.ndarray: the double potential at the points \mathbf{x}(x, y, z).
            $$
                \D(\mathbf{x}) = \int_\partial\Omega \frac{\mu(\mathbf{y})(\mathbf{x}-\mathbf{y})}{|\mathbf{x}-\mathbf{y}|^3}\cdot(\partial_\theta\times\partial_zeta)d\omega(\mathbf{y})
            $$
    """
    assert x.shape == y.shape == z.shape
    dtheta, dzeta = 2*np.pi/npol, 2*np.pi/surf.nfp/ntor
    _thetaarr = np.linspace(0, 2*np.pi, npol, endpoint=False) + dtheta/10
    _zetaarr = np.linspace(0, 2*np.pi, surf.nfp*ntor, endpoint=False) + dzeta/10
    zetaGrid, thetaGrid = np.meshgrid(_zetaarr, _thetaarr)
    muGrid = potential.getValue(thetaGrid, zetaGrid)
    rGrid, zGrid = surf.getRZ(thetaGrid, zetaGrid)
    phiGrid = surf.getPhi(thetaGrid, zetaGrid)
    xGrid, yGrid = np.cos(phiGrid) * rGrid, np.sin(phiGrid) * rGrid
    try:
        _position_theta = np.array([surf.position_theta[_i].getValue(thetaGrid, zetaGrid) for _i in range(3)])
        _position_zeta = np.array([surf.position_zeta[_i].getValue(thetaGrid, zetaGrid) for _i in range(3)])
    except AttributeError:
        surf.updateBasis()
        _position_theta = np.array([surf.position_theta[_i].getValue(thetaGrid, zetaGrid) for _i in range(3)])
        _position_zeta = np.array([surf.position_zeta[_i].getValue(thetaGrid, zetaGrid) for _i in range(3)])
    position_theta = np.transpose(np.array([
        _position_theta[0]*np.cos(phiGrid) - _position_theta[1]*np.sin(phiGrid),
        _position_theta[0]*np.sin(phiGrid) + _position_theta[1]*np.cos(phiGrid),
        _position_theta[2]
    ]))
    position_zeta = np.transpose(np.array([
        _position_zeta[0]*np.cos(phiGrid) - _position_zeta[1]*np.sin(phiGrid),
        _position_zeta[0]*np.sin(phiGrid) + _position_zeta[1]*np.cos(phiGrid),
        _position_zeta[2]
    ]))
    normalGrid = np.transpose(np.cross(position_theta, position_zeta))
    deltaPosition = np.array([
        x[..., np.newaxis, np.newaxis] - xGrid,
        y[..., np.newaxis, np.newaxis] - yGrid,
        z[..., np.newaxis, np.newaxis] - zGrid
    ])
    lengthDeltaPosition = np.linalg.norm(deltaPosition, axis=0)
    return dtheta * dzeta * np.einsum(
        'ij,...ij->...', muGrid, normalGrid[0]*deltaPosition[0]/lengthDeltaPosition**3
    )
    

def potential_BoozerSurf(x: np.ndarray, y: np.ndarray, z: np.ndarray, 
                         surf: Surface_BoozerAngle, 
                         singlepotential: ToroidalField, 
                         doublepotential: ToroidalField,
                         npol: int=256, ntor: int=256) -> np.ndarray:
    r"""
    Calculate the single potential and double potential at the same time on a surface using Boozer coordinates.
    Args:
        x (np.ndarray): x-coordinates of the points.
        y (np.ndarray): y-coordinates of the points.
        z (np.ndarray): z-coordinates of the points.
        surf (Surface_BoozerAngle): Surface \Omgea using Boozer coordinates.
        singlepotential (ToroidalField): singlepotential \sigma on the Boozer surface.
        doublepotential (ToroidalField): doublepotential \mu on the Boozer surface.
        npol (int): resolution of numerical integration in the poloidal direction.
        ntor (int): resolution of numerical integration in the toroidal direction.
    Returns:
        single potential and double potential at the points \mathbf{x}(x, y, z).
            $$
                \S(\mathbf{x}) = \int_\partial\Omega \frac{\sigma(\mathbf{y})}{|\mathbf{x}-\mathbf{y}|} d\omega(\mathbf{y})
            $$
            $$
                \D(\mathbf{x}) = \int_\partial\Omega \frac{\mu(\mathbf{y})(\mathbf{x}-\mathbf{y})}{|\mathbf{x}-\mathbf{y}|^3}\cdot(\partial_\theta\times\partial_zeta)d\omega(\mathbf{y})
            $$
    """
    assert x.shape == y.shape == z.shape
    dtheta, dzeta = 2*np.pi/npol, 2*np.pi/surf.nfp/ntor
    _thetaarr = np.linspace(0, 2*np.pi, npol, endpoint=False) + dtheta/10
    _zetaarr = np.linspace(0, 2*np.pi, surf.nfp*ntor, endpoint=False) + dzeta/10
    zetaGrid, thetaGrid = np.meshgrid(_zetaarr, _thetaarr)
    sigmaGrid = singlepotential.getValue(thetaGrid, zetaGrid)
    muGrid = doublepotential.getValue(thetaGrid, zetaGrid)
    rGrid, zGrid = surf.getRZ(thetaGrid, zetaGrid)
    phiGrid = surf.getPhi(thetaGrid, zetaGrid)
    xGrid, yGrid = np.cos(phiGrid) * rGrid, np.sin(phiGrid) * rGrid
    try:
        position_theta = np.transpose(np.array([surf.position_theta[_i].getValue(thetaGrid, zetaGrid) for _i in range(3)]))
        position_zeta = np.transpose(np.array([surf.position_zeta[_i].getValue(thetaGrid, zetaGrid) for _i in range(3)]))
    except AttributeError:
        surf.updateBasis()
        position_theta = np.transpose(np.array([surf.position_theta[_i].getValue(thetaGrid, zetaGrid) for _i in range(3)]))
        position_zeta = np.transpose(np.array([surf.position_zeta[_i].getValue(thetaGrid, zetaGrid) for _i in range(3)]))
    normalGrid = np.transpose(np.cross(position_theta, position_zeta))
    dareaGrid = np.linalg.norm(normalGrid, axis=0)
    _deltaPosition = np.array([
        x[..., np.newaxis, np.newaxis] - xGrid,
        y[..., np.newaxis, np.newaxis] - yGrid,
        z[..., np.newaxis, np.newaxis] - zGrid
    ])
    deltaPosition = np.array([
        _deltaPosition[0]*np.cos(phiGrid) - _deltaPosition[1]*np.sin(phiGrid),
        _deltaPosition[0]*np.sin(phiGrid) + _deltaPosition[1]*np.cos(phiGrid),
        _deltaPosition[2]
    ])
    lengthDeltaPosition = np.linalg.norm(deltaPosition, axis=0)
    return (
        dtheta * dzeta * np.einsum('ij,...ij->...', sigmaGrid*dareaGrid, 1/lengthDeltaPosition), 
        dtheta * dzeta * np.einsum('ij,...ij->...', muGrid, normalGrid[0]*deltaPosition[0]/lengthDeltaPosition**3)
    )
   

if __name__ == '__main__':
    pass
