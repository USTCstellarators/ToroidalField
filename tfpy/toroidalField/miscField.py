#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# miscField.py


import numpy as np 
from .field import ToroidalField
from .misc import resize_center_pad_zeros


def changeResolution(originalField: ToroidalField, mpol: int, ntor: int) -> ToroidalField:
    
    newReMatrix = resize_center_pad_zeros(originalField.reMatrix, 2*mpol+1, 2*ntor+1)
    newImMatrix = resize_center_pad_zeros(originalField.imMatrix, 2*mpol+1, 2*ntor+1)
    _field = ToroidalField(
        nfp = originalField.nfp, 
        mpol = mpol, 
        ntor = ntor, 
        reArr = newReMatrix.flatten()[(2*ntor+1)*mpol+ntor :], 
        imArr = newImMatrix.flatten()[(2*ntor+1)*mpol+ntor :], 
        reIndex = originalField.reIndex, 
        imIndex = originalField.imIndex
    )
    return _field 

def normalize(originalField: ToroidalField) -> ToroidalField:
    assert originalField.reIndex
    return ToroidalField(
        nfp = originalField.nfp, 
        mpol = originalField.mpol, 
        ntor = originalField.ntor, 
        reArr = originalField.reArr / originalField.reArr[0], 
        imArr = originalField.imArr / originalField.reArr[0]
    )

def power(field: ToroidalField, index: float) -> ToroidalField:
    mpol, ntor = field.mpol, field.ntor
    nfp = field.nfp
    deltaTheta = 2*np.pi / (2*mpol+1)
    deltaZeta = 2*np.pi / nfp / (2*ntor+1) 
    sampleTheta, sampleZeta = np.arange(2*mpol+1)*deltaTheta, np.arange(2*ntor+1)*deltaZeta
    gridSampleZeta, gridSampleTheta = np.meshgrid(sampleZeta, sampleTheta)
    sampleValue = field.getValue(gridSampleTheta, -gridSampleZeta)
    from .sample import fftToroidalField
    _field = fftToroidalField(np.power(sampleValue, index), nfp=nfp)
    return _field


if __name__ == "__main__": 
    pass
