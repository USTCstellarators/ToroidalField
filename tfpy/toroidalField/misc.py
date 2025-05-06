#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# misc.py


import numpy as np 
from .field import ToroidalField


def changeResolution(originalField: ToroidalField, mpol: int, ntor: int) -> ToroidalField:
    nums = (2*ntor+1)*mpol+ntor+1
    _field = ToroidalField(
        nfp = originalField.nfp, 
        mpol = mpol, 
        ntor = ntor, 
        reArr = np.zeros(nums), 
        imArr = np.zeros(nums), 
        reIndex = originalField.reIndex, 
        imIndex = originalField.imIndex
    )
    for i in range(nums):
        m, n = _field.indexReverseMap(i)
        if _field.reIndex:
            _field.setRe(m, n, originalField.getRe(m, n))
        if _field.imIndex:
            _field.setIm(m, n, originalField.getIm(m, n))
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
