#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# derivative.py


from .field import ToroidalField


def derivatePol(field: ToroidalField) -> ToroidalField:
    r"""
    Get the field $\frac{\partial f}{\partial\theta}$
    """
    return ToroidalField(
        nfp = field.nfp, 
        mpol = field.mpol, 
        ntor = field.ntor, 
        reArr = - field.xm*field.imArr, 
        imArr =  field.xm*field.reArr, 
        reIndex = field.imIndex, 
        imIndex = field.reIndex
    )


def derivateTor(field: ToroidalField) -> ToroidalField:
    r"""
    Get the field $\frac{\partial f}{\partial\varphi}$
    """
    return ToroidalField(
        nfp = field.nfp, 
        mpol = field.mpol, 
        ntor = field.ntor, 
        reArr = field.nfp*field.xn*field.imArr, 
        imArr = - field.nfp*field.xn*field.reArr, 
        reIndex = field.imIndex, 
        imIndex = field.reIndex
    )


if __name__ == "__main__":
    pass
