#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vmecTransform.py 


from ..toroidalField import ToroidalField
from ..toroidalField import fftToroidalField
from typing import Tuple


def transBoozer(self, surfIndex: int, valueField: ToroidalField , mpol: int=None, ntor: int=None, **kwargs) -> ToroidalField:
    
    import numpy as np
    from scipy.optimize import fixed_point
    try:
        from collections.abc import Iterable
    except ImportError:
        from collections import Iterable

    _surf, lam = self.getSurface(surfIndex)
    iota = self.iota(surfIndex)
    omega = self.getOmega(surfIndex)
    
    if mpol is None and ntor is None: 
        if isinstance(valueField, ToroidalField):
            mpol = valueField.mpol + max(omega.mpol, lam.mpol)
            ntor = valueField.ntor + max(omega.ntor, lam.ntor)
        elif isinstance(valueField, Iterable) and isinstance(valueField[0], ToroidalField):
            mpol = valueField[0].mpol + max(omega.mpol, lam.mpol) 
            ntor = valueField[0].ntor + max(omega.ntor, lam.ntor)
        else:
            print("Wrong type of the valuefield... ")
    sampleTheta = np.linspace(0, 2*np.pi, 2*mpol+1, endpoint=False) 
    sampleZeta = -np.linspace(0, 2*np.pi/self.nfp, 2*ntor+1, endpoint=False) 
    gridSampleZeta, gridSampleTheta = np.meshgrid(sampleZeta, sampleTheta) 

    # find the fixed point of vartheta and varphi 
    def varthetaphiValue(inits, theta, zeta):
        vartheta, varphi = inits[0], inits[1]
        lamValue = lam.getValue(vartheta, varphi) 
        omegaValue = omega.getValue(vartheta, varphi)
        return np.array([
            theta - lamValue - iota*omegaValue, 
            zeta - omegaValue
        ])

    gridVartheta, gridVarphi = np.zeros_like(gridSampleTheta), np.zeros_like(gridSampleZeta) 
    for i in range(len(gridVartheta)): 
        for j in range(len(gridVartheta[0])): 
            try:
                varthetaphi = fixed_point(
                    varthetaphiValue, [gridSampleTheta[i,j],gridSampleZeta[i,j]], args=(gridSampleTheta[i,j],gridSampleZeta[i,j]), **kwargs
                )
            except:
                varthetaphi = fixed_point(
                    varthetaphiValue, [gridSampleTheta[i,j],gridSampleZeta[i,j]], args=(gridSampleTheta[i,j],gridSampleZeta[i,j]), method="iteration",**kwargs
                )
            gridVartheta[i,j] = float(varthetaphi[0,0])
            gridVarphi[i,j] = float(varthetaphi[1,0])
    
    if isinstance(valueField, ToroidalField):
        sampleValue = valueField.getValue(gridVartheta, gridVarphi)
        return fftToroidalField(sampleValue, nfp=self.nfp)
    elif isinstance(valueField, Iterable) and isinstance(valueField[0], ToroidalField):
        ans = list()
        for _field in valueField:
            _sampleValue = _field.getValue(gridVartheta, gridVarphi)
            ans.append(fftToroidalField(_sampleValue, nfp=self.nfp))
        return ans
    else:
        print("Wrong type of the valuefield... ")


def surf2Boozer(self, surfIndex: int=-1, mpol: int=None, ntor: int=None, reverseToroidalAngle: bool=False, reverseOmegaAngle:bool=True) :
    ''' Transform the surface to Boozer coordinates.
    Args:
        surfIndex (int): The index of the surface to be transformed.
        mpol (int): The poloidal mode number.
        ntor (int): The toroidal mode number.
    Returns:
        surfBoozer (tfpy.geometry.Surf_BoozerAngle): The surface in Boozer coordinates.
        B (tfpy.toroidalField.ToroidalField): The magnetic field in Boozer coordinates.
    '''
    from ..geometry import Surface_BoozerAngle
    _surf, _lam = self.getSurface(surfIndex)
    _omgea = self.getOmega(surfIndex)
    _B = self.getB(surfIndex)
    rzomegaB_v2boozer = self.transBoozer(surfIndex, [_surf.r, _surf.z, _omgea, _B], mpol=mpol, ntor=ntor, )
    surfBoozer = Surface_BoozerAngle(
        rzomegaB_v2boozer[0], 
        rzomegaB_v2boozer[1], 
        rzomegaB_v2boozer[2], 
        reverseToroidalAngle=reverseToroidalAngle, reverseOmegaAngle=reverseOmegaAngle
    )
    return surfBoozer, rzomegaB_v2boozer[3]



if __name__ == "__main__":
    pass
