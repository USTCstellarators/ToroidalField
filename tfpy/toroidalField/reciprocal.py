#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# reciprocal.py


import numpy as np 
from scipy.linalg import solve
from .field import ToroidalField
from .sample import fftToroidalField
from typing import Tuple


def multiplicativeInverse(originalField: ToroidalField, method: str="dft") -> ToroidalField:

    mpol, ntor = originalField.mpol, originalField.ntor
    nfp = originalField.nfp

    if method == "dft":
        deltaTheta = 2*np.pi / (2*mpol+1)
        deltaZeta = 2*np.pi / nfp / (2*ntor+1) 
        sampleTheta, sampleZeta = np.arange(2*mpol+1)*deltaTheta, np.arange(2*ntor+1)*deltaZeta
        gridSampleZeta, gridSampleTheta = np.meshgrid(sampleZeta, sampleTheta)
        sampleValue = originalField.getValue(gridSampleTheta, -gridSampleZeta)
        _field = fftToroidalField(1/sampleValue, nfp=nfp)

    # TODO
    elif method == "convolution":
        """
        This method can not be trusted. 
        """

        def indexMap(index: int) -> Tuple:
            assert 1 <= index <= 2*ntor+2*mpol*(2*ntor+1)
            if index <= ntor+mpol*(2*ntor+1):
                if 1 <= index <= ntor:
                    m = 0
                    n = index
                elif ntor+1 <= index <= ntor+mpol*(2*ntor+1):
                    m = (index-ntor-1) // (2*ntor+1) + 1
                    n = (index-ntor-1) % (2*ntor+1) - ntor 
                label = "re"
            elif ntor+mpol*(2*ntor+1)+1 <= index <= 2*ntor+2*mpol*(2*ntor+1):
                if ntor+mpol*(2*ntor+1)+1 <= index <= 2*ntor+mpol*(2*ntor+1):
                    m = 0
                    n = index - ntor - mpol*(2*ntor+1)
                elif 2*ntor+mpol*(2*ntor+1)+1 <= index <= 2*ntor+2*mpol*(2*ntor+1):
                    m = (index-2*ntor-mpol*(2*ntor+1)-1) // (2*ntor+1) + 1
                    n = (index-2*ntor-mpol*(2*ntor+1)-1) % (2*ntor+1) - ntor 
                label = "im"
            return m, n, label

        def getRe_CoefMN(m: int, n : int, _m : int, _n: int) -> float:
            return (
                originalField.getRe(m-_m, n-_n)
            )

        def getIm_CoefMN(m: int, n : int, _m : int, _n: int) -> float:
            return (
                originalField.getIm(m-_m, n-_n)
            )

        def getVectorB() -> np.ndarray:
            vectorB = np.zeros(1+2*ntor+2*mpol*(2*ntor+1))
            vectorB[0] = 1 
            return vectorB

        def getMatrixCoef() -> np.ndarray:
            matrixCoef = np.zeros([1+2*ntor+2*mpol*(2*ntor+1), 1+2*ntor+2*mpol*(2*ntor+1)]) 
            matrixCoef[0,0] = getRe_CoefMN(0,0,0,0)
            for i in range(2*ntor+2*mpol*(2*ntor+1)):
                m, n, equationLabel = indexMap(i+1)
                if equationLabel == "re":
                    matrixCoef[i+1,0] = getRe_CoefMN(m,n,0,0)
                elif equationLabel == "im":
                    matrixCoef[i+1,0] = getIm_CoefMN(m,n,0,0)
            for j in range(2*ntor+2*mpol*(2*ntor+1)):
                _m, _n, variableLabel = indexMap(j+1) 
                if variableLabel == "re":
                    matrixCoef[0,j+1] = getRe_CoefMN(0,0,_m,_n) + getRe_CoefMN(0,0,-_m,-_n)
                elif variableLabel == "im":
                    matrixCoef[0,j+1] = - getIm_CoefMN(0,0,_m,_n) + getIm_CoefMN(0,0,-_m,-_n)
            for i in range(2*ntor+2*mpol*(2*ntor+1)):
                m, n, equationLabel = indexMap(i+1) 
                for j in range(2*ntor+2*mpol*(2*ntor+1)):
                    _m, _n, variableLabel = indexMap(j+1) 
                    if equationLabel == "re":
                        if variableLabel == "re":
                            matrixCoef[i+1,j+1] = getRe_CoefMN(m,n,_m,_n) + getRe_CoefMN(m,n,-_m,-_n)
                        elif variableLabel == "im":
                            matrixCoef[i+1,j+1] = - getIm_CoefMN(m,n,_m,_n) + getIm_CoefMN(m,n,-_m,-_n)
                    elif equationLabel == "im":
                        if variableLabel == "re":
                            matrixCoef[i+1,j+1] = getIm_CoefMN(m,n,_m,_n) + getIm_CoefMN(m,n,-_m,-_n)
                        elif variableLabel == "im":
                            matrixCoef[i+1,j+1] = getRe_CoefMN(m,n,_m,_n) - getRe_CoefMN(m,n,-_m,-_n)
            return matrixCoef

        matrixCoef = getMatrixCoef() 
        vectorB = getVectorB()
        vectorComp = solve(matrixCoef, vectorB)
        
        reArr = np.zeros(ntor+1+mpol*(2*ntor+1))
        imArr = np.zeros(ntor+1+mpol*(2*ntor+1))
        _field = ToroidalField(
            nfp = originalField.nfp, 
            mpol = mpol, 
            ntor = ntor, 
            reArr = reArr,
            imArr = imArr
        )
        _field.setRe(0, 0, vectorComp[0])
        for i in range(ntor+mpol*(2*ntor+1)):
            m, n, label = indexMap(i+1)
            if label == "re":
                _field.setRe(m, n, vectorComp[i+1])
            elif label == "im":
                _field.setIm(m, n, vectorComp[i+1])
        
    return _field


if __name__ == "__main__": 
    pass
