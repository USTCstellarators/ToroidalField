#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# field.py


import numpy as np
from scipy.signal import convolve2d 
from typing import Tuple
from .misc import resize_center_pad_zeros
from ..config import tfParams


class ToroidalField:
    r"""
    The Fourier representation of the field f defined on the toroidal surface. 
    $$ f(\theta, \zeta) = \sum_{m,n} F_{m,n}\exp(i(m\theta-nN_{fp}\zeta)) $$
    """

    def __init__(self, nfp: int, mpol: int, ntor: int, reArr: np.ndarray, imArr: np.ndarray, 
    reIndex: bool=True, imIndex: bool=True) -> None:
        """
        ### Initialization with Fourier harmonics. 
        Args:
            nfp: the number of field periods. 
            mpol, ntor: the resolution in the poloidal/toroidal direction. 
            reArr, imArr: the real/imaginary part of the Fourier coefficients. 
        """
        assert reArr.shape == imArr.shape
        assert (2*ntor+1)*mpol+ntor+1 == reArr.size
        self.nfp = nfp
        self.mpol = mpol
        self.ntor = ntor
        self._reArr = reArr
        self._imArr = imArr
        assert reIndex or imIndex
        self.reIndex = reIndex
        self.imIndex = imIndex
        # Precompute index grids for speed and reuse.
        zero_block = np.zeros(self.ntor + 1, dtype=int)
        if self.mpol:
            m_block = np.repeat(np.arange(1, self.mpol + 1, dtype=int), 2 * self.ntor + 1)
            n_block = np.tile(np.arange(-self.ntor, self.ntor + 1, dtype=int), self.mpol)
            self._xm = np.concatenate((zero_block, m_block))
            self._xn = np.concatenate((np.arange(0, self.ntor + 1, dtype=int), n_block))
        else:
            self._xm = zero_block
            self._xn = np.arange(0, self.ntor + 1, dtype=int)

    @property
    def arrlen(self) -> int: 
        return (2*self.ntor+1)*self.mpol+self.ntor+1 

    @property
    def reArr(self) -> np.ndarray:
        if not self.reIndex:
            return np.zeros(self.arrlen)
        else:
            return self._reArr

    @property
    def imArr(self) -> np.ndarray:
        if not self.imIndex:
            return np.zeros(self.arrlen)
        else:
            return self._imArr

    @property
    def reMatrix(self) -> np.ndarray: 
        if not self.reIndex:
            return np.zeros((2*self.mpol+1, 2*self.ntor+1))
        else:
            return (
                np.concatenate((np.flip(self.reArr), self.reArr[1:]))
            ).reshape((2*self.mpol+1, 2*self.ntor+1))

    @property
    def imMatrix(self) -> np.ndarray: 
        if not self.imIndex:
            return np.zeros((2*self.mpol+1, 2*self.ntor+1))
        else:
            return (
                np.concatenate((-np.flip(self.imArr), self.imArr[1:]))
            ).reshape((2*self.mpol+1, 2*self.ntor+1))

    @property
    def xm(self) -> np.ndarray:
        return self._xm

    @property
    def xn(self) -> np.ndarray:
        return self._xn

    def indexMap(self, m: int, n: int) -> int:
        assert abs(m) <= self.mpol and abs(n) <= self.ntor
        return self.ntor + (2*self.ntor+1)*(m-1) + (n+self.ntor+1)

    def indexReverseMap(self, index: int) -> Tuple[int]: 
        assert index < (self.mpol*(2*self.ntor+1)+self.ntor+1)
        if index <= self.ntor:
            return 0, index
        else:
            return (index-self.ntor-1)//(2*self.ntor+1)+1, (index-self.ntor-1)%(2*self.ntor+1)-self.ntor

    def getValue(self, thetaArr: np.ndarray, zetaArr: np.ndarray) -> np.ndarray:
        assert type(thetaArr) == type(zetaArr)
        if not isinstance(thetaArr, np.ndarray):
            try:
                thetaArr, zetaArr = np.array(thetaArr), np.array(zetaArr)
            except:
                thetaArr, zetaArr = np.array([thetaArr]), np.array([zetaArr])
        angleMat = (
            np.dot(self.xm.reshape(-1,1), thetaArr.reshape(1,-1)) -  
            self.nfp * np.dot(self.xn.reshape(-1,1), zetaArr.reshape(1,-1))
        )
        valueArr = 2 * (
            np.dot(self.reArr.reshape(1,-1), np.cos(angleMat)) - 
            np.dot(self.imArr.reshape(1,-1), np.sin(angleMat))
        )
        valueArr -= self.reArr[0]
        try:
            m, n = thetaArr.shape
            return valueArr.reshape(m, n)
        except:
            if isinstance(valueArr, np.ndarray) and valueArr.shape[0] == 1: 
                return valueArr.flatten()
            elif isinstance(valueArr, np.ndarray) and valueArr.shape[1] == 1: 
                return valueArr.flatten()
            else:
                return valueArr

    def getRe(self, m: int=0, n: int=0) -> float: 
        if not self.reIndex:
            return 0
        if abs(m) > self.mpol or abs(n) > self.ntor:
            return 0
        elif m == 0 and n < 0:
            return self.reArr[self.indexMap(0, -n)] 
        elif m < 0:
            return self.reArr[self.indexMap(-m, -n)] 
        else:
            return self.reArr[self.indexMap(m, n)] 

    def getIm(self, m: int, n: int) -> float:
        if not self.imIndex:
            return 0
        if abs(m) > self.mpol or abs(n) > self.ntor:
            return 0
        elif m == 0 and n < 0:
            return -self.imArr[self.indexMap(0, -n)] 
        elif m < 0:
            return -self.imArr[self.indexMap(-m, -n)] 
        else:
            return self.imArr[self.indexMap(m, n)] 

    def setRe(self, m: int=0, n: int=0, value: float=0): 
        assert self.reIndex
        assert 0 <= m <= self.mpol and -self.ntor <= n <= self.ntor
        self._reArr[self.indexMap(m, n)] = value 

    def setIm(self, m: int=0, n: int=0, value: float=0): 
        assert self.imIndex 
        assert 0 <= m <= self.mpol and -self.ntor <= n <= self.ntor
        self._imArr[self.indexMap(m, n)] = value
        
    # plotting ###############################################################
    def plot_plt(self, ntheta: int=128, nzeta: int=128, fig=None, ax=None, onePeriod: bool=True, fill: bool=True, **kwargs):
        if kwargs.get('cmap') is None:
            try:
                from cmap import Colormap
                kwargs.update({'cmap': Colormap('tol:sunset').to_matplotlib()})
            except:
                pass
        import matplotlib.pyplot as plt 
        thetaArr = np.linspace(0, 2*np.pi, ntheta)
        thetaValue =  np.linspace(0, 2*np.pi, 3)
        if onePeriod:
            zetaArr = np.linspace(0, 2*np.pi/self.nfp, nzeta)
            zetaValue =  np.linspace(0, 2*np.pi/self.nfp, 3)
        else:
            zetaArr = np.linspace(0, 2*np.pi, nzeta) 
            zetaValue =  np.linspace(0, 2*np.pi, 3)
        if ax is None: 
            fig, ax = plt.subplots() 
        plt.sca(ax) 
        thetaGrid, zetaGrid = np.meshgrid(thetaArr, zetaArr) 
        valueGrid = self.getValue(thetaGrid, zetaGrid) 
        if fill: 
            ctrig = ax.contourf(zetaGrid, thetaGrid, valueGrid, **kwargs)
            colorbar = fig.colorbar(ctrig)
            colorbar.ax.tick_params(labelsize=18) 
        else: 
            ctrig = ax.contour(zetaGrid, thetaGrid, valueGrid, **kwargs)
            colorbar = fig.colorbar(ctrig)
            colorbar.ax.tick_params(labelsize=18) 
        try:
            colorbar.ax.yaxis.get_offset_text().set_fontsize(15)
        except:
            pass
        if kwargs.get("toroidalLabel") == None:
            if onePeriod:
                kwargs.update({"toroidalLabel": r"$N_\mathrm{fp}\zeta$"})
            else:
                kwargs.update({"toroidalLabel": r"$\zeta$"})
        if kwargs.get("poloidalLabel") == None:
            kwargs.update({"poloidalLabel": r"$\theta$"})
        ax.set_xlabel(kwargs.get("toroidalLabel"), fontsize=18)
        ax.set_ylabel(kwargs.get("poloidalLabel"), fontsize=18)
        ax.set_xticks(zetaValue)
        ax.set_xticklabels(["$0$", r"$\pi$", r"$2\pi$"], fontsize=18) 
        ax.set_yticks(thetaValue)
        ax.set_yticklabels(["$0$", r"$\pi$", r"$2\pi$"], fontsize=18)
        return

    # operator overloading ####################################################
    def __add__(self, other):
        if isinstance(other, ToroidalField):
            assert self.nfp == other.nfp
            if self.mpol==other.mpol and self.ntor==other.ntor: 
                return ToroidalField(
                    nfp = self.nfp, 
                    mpol = self.mpol, 
                    ntor = self.ntor,
                    reArr = self.reArr + other.reArr, 
                    imArr = self.imArr + other.imArr, 
                    reIndex = self.reIndex or other.reIndex, 
                    imIndex = self.imIndex or other.imIndex
                )
            else:
                _mpol = max(self.mpol, other.mpol) 
                _ntor = max(self.ntor, other.ntor) 
                reMatrix = resize_center_pad_zeros(self.reMatrix, 2*_mpol+1, 2*_ntor+1) + resize_center_pad_zeros(other.reMatrix, 2*_mpol+1, 2*_ntor+1)
                imMatrix = resize_center_pad_zeros(self.imMatrix, 2*_mpol+1, 2*_ntor+1) + resize_center_pad_zeros(other.imMatrix, 2*_mpol+1, 2*_ntor+1)
                _field = ToroidalField(
                    nfp = self.nfp,
                    mpol = _mpol, 
                    ntor = _ntor, 
                    reArr = reMatrix.flatten()[(2*_ntor+1)*_mpol+_ntor :], 
                    imArr = imMatrix.flatten()[(2*_ntor+1)*_mpol+_ntor :],
                    reIndex = self.reIndex or other.reIndex, 
                    imIndex = self.imIndex or other.imIndex
                )
                return _field
        else:
            _reArr = np.zeros_like(self.reArr) + self.reArr
            _reArr[0] = _reArr[0] + other
            return ToroidalField(
                nfp = self.nfp, 
                mpol = self.mpol, 
                ntor = self.ntor, 
                reArr = _reArr, 
                imArr = self.imArr, 
                reIndex = self.reIndex, 
                imIndex = self.imIndex
            )

    def __radd__(self, other): 
        _reArr = np.zeros_like(self.reArr) + self.reArr 
        _imArr = np.zeros_like(self.imArr) + self.imArr
        _reArr[0] = _reArr[0] + other
        return ToroidalField(
            nfp = self.nfp, 
            mpol = self.mpol, 
            ntor = self.ntor,
            reArr = _reArr, 
            imArr = _imArr, 
            reIndex = self.reIndex, 
            imIndex = self.imIndex
        )

    def __sub__(self, other):
        if isinstance(other, ToroidalField):
            assert self.nfp == other.nfp
            if self.mpol==other.mpol and self.ntor==other.ntor: 
                return ToroidalField(
                    nfp = self.nfp, 
                    mpol = self.mpol, 
                    ntor = self.ntor,
                    reArr = self.reArr - other.reArr, 
                    imArr = self.imArr - other.imArr, 
                    reIndex = self.reIndex or other.reIndex, 
                    imIndex = self.imIndex or other.imIndex
                )
            else:
                _mpol = max(self.mpol, other.mpol) 
                _ntor = max(self.ntor, other.ntor) 
                reMatrix = resize_center_pad_zeros(self.reMatrix, 2*_mpol+1, 2*_ntor+1) - resize_center_pad_zeros(other.reMatrix, 2*_mpol+1, 2*_ntor+1)
                imMatrix = resize_center_pad_zeros(self.imMatrix, 2*_mpol+1, 2*_ntor+1) - resize_center_pad_zeros(other.imMatrix, 2*_mpol+1, 2*_ntor+1)
                _field = ToroidalField(
                    nfp = self.nfp,
                    mpol = _mpol, 
                    ntor = _ntor, 
                    reArr = reMatrix.flatten()[(2*_ntor+1)*_mpol+_ntor :], 
                    imArr = imMatrix.flatten()[(2*_ntor+1)*_mpol+_ntor :],
                    reIndex = self.reIndex or other.reIndex, 
                    imIndex = self.imIndex or other.imIndex
                )
                return _field
        else:
            _reArr = np.zeros_like(self.reArr) + self.reArr
            _reArr[0] = _reArr[0] - other
            return ToroidalField(
                nfp = self.nfp, 
                mpol = self.mpol, 
                ntor = self.ntor,
                reArr = _reArr, 
                imArr = self.imArr, 
                reIndex = self.reIndex, 
                imIndex = self.imIndex
            )

    def __rsub__(self, other): 
        _reArr = np.zeros_like(self.reArr) - self.reArr
        _imArr = np.zeros_like(self.imArr) - self.imArr
        _reArr[0] = _reArr[0] + other
        return ToroidalField(
            nfp = self.nfp, 
            mpol = self.mpol, 
            ntor = self.ntor,
            reArr = _reArr, 
            imArr = _imArr, 
            reIndex = self.reIndex, 
            imIndex = self.imIndex
        )

    def __mul__(self, other):
        if isinstance(other, ToroidalField):
            assert self.nfp == other.nfp
            mpol, ntor = self.mpol+other.mpol, self.ntor+other.ntor 
            reMat = np.zeros((2*mpol+1, 2*ntor+1))
            imMat = np.zeros((2*mpol+1, 2*ntor+1))
            if tfParams.jit:
                from .misc import numba_convolve2d
                if self.reIndex and other.reIndex:
                    reMat += numba_convolve2d(self.reMatrix, other.reMatrix)
                if self.imIndex and other.imIndex:
                    reMat -= numba_convolve2d(self.imMatrix, other.imMatrix)
                if self.reIndex and other.imIndex:
                    imMat += numba_convolve2d(self.reMatrix, other.imMatrix)
                if self.imIndex and other.reIndex:
                    imMat += numba_convolve2d(self.imMatrix, other.reMatrix)
            else:
                if self.reIndex and other.reIndex:
                    reMat += convolve2d(self.reMatrix, other.reMatrix, mode='full')
                if self.imIndex and other.imIndex:
                    reMat -= convolve2d(self.imMatrix, other.imMatrix, mode='full')
                if self.reIndex and other.imIndex:
                    imMat += convolve2d(self.reMatrix, other.imMatrix, mode='full')
                if self.imIndex and other.reIndex:
                    imMat += convolve2d(self.imMatrix, other.reMatrix, mode='full')
            reIndex = (self.reIndex and other.reIndex) or (self.imIndex and other.imIndex)
            imIndex = (self.reIndex and other.imIndex) or (self.imIndex and other.reIndex)
            if mpol > tfParams.max_mpol or ntor > tfParams.max_ntor:
                _mpol = tfParams.max_mpol if mpol>tfParams.max_mpol else mpol
                _ntor = tfParams.max_ntor if ntor>tfParams.max_ntor else ntor
                reMat= resize_center_pad_zeros(reMat, 2*_mpol+1, 2*_ntor+1)
                imMat= resize_center_pad_zeros(imMat, 2*_mpol+1, 2*_ntor+1)
                return  ToroidalField(
                    nfp = self.nfp, 
                    mpol = _mpol, 
                    ntor = _ntor,
                    reArr = reMat.flatten()[(2*_ntor+1)*_mpol+_ntor :],
                    imArr = imMat.flatten()[(2*_ntor+1)*_mpol+_ntor :], 
                    reIndex = reIndex, 
                    imIndex = imIndex
                )
            return  ToroidalField(
                nfp = self.nfp, 
                mpol = mpol, 
                ntor = ntor,
                reArr = reMat.flatten()[(2*ntor+1)*mpol+ntor :],
                imArr = imMat.flatten()[(2*ntor+1)*mpol+ntor :], 
                reIndex = reIndex, 
                imIndex = imIndex
            )
        else:
            return ToroidalField(
                nfp = self.nfp, 
                mpol = self.mpol, 
                ntor = self.ntor, 
                reArr = other * self.reArr,
                imArr = other * self.imArr, 
                reIndex = self.reIndex, 
                imIndex = self.imIndex
            )

    def __rmul__(self, other): 
        return ToroidalField(
            nfp = self.nfp, 
            mpol = self.mpol, 
            ntor = self.ntor, 
            reArr = other * self.reArr,
            imArr = other * self.imArr, 
            reIndex = self.reIndex, 
            imIndex = self.imIndex
        )

    def __eq__(self, other) -> bool:
        if not isinstance(other, ToroidalField):
            return False
        return (
            self.nfp == other.nfp
            and self.mpol == other.mpol
            and self.ntor == other.ntor
            and np.array_equal(self.reArr, other.reArr)
            and np.array_equal(self.imArr, other.imArr)
            and self.reIndex == other.reIndex
            and self.imIndex == other.imIndex
        )

    @classmethod
    def constantField(cls, constant: int, nfp: int, mpol: int, ntor: int):
        reArr = np.zeros((2*ntor+1)*mpol+ntor+1)
        imArr = np.zeros((2*ntor+1)*mpol+ntor+1)
        reArr[0] = constant
        reIndex, imIndex = True, False 
        field =  cls(
            nfp, mpol, ntor, reArr, imArr, reIndex, imIndex
        )
        return field



if __name__ == "__main__": 
    pass
