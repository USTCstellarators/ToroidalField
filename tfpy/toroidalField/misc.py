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


if __name__ == "__main__": 
    pass
