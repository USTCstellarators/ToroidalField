#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# reverseAngle.py


import numpy as np
from .field import ToroidalField


def reversePoloidalAngle(field: ToroidalField) -> ToroidalField:
    r"""
    $$ \vartheta = \pi - \theta $$
    """
    reArr = np.zeros_like(field.reArr)
    imArr = np.zeros_like(field.imArr)
    ans = ToroidalField(
        nfp = field.nfp,
        mpol = field.mpol,
        ntor = field.ntor,
        reArr = reArr,
        imArr = imArr,
        reIndex = field.reIndex,
        imIndex = field.imIndex
    )
    for i in range(reArr.size):
        m, n = ans.indexReverseMap(i)
        if m % 2 == 0:
            _label = 1
        else:
            _label = -1
        if ans.reIndex:
            ans.setRe(m,-n, _label*field.getRe(m,n))
        if ans.imIndex:
            ans.setIm(m,-n,-1*_label*field.getIm(m,n))
    return ans
