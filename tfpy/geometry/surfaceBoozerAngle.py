#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# surfaceBoozerAngle.py


import h5py
import numpy as np 
import matplotlib.pyplot as plt 
from .surface import Surface
from .surfaceCylindricalAngle import Surface_cylindricalAngle
from ..toroidalField import ToroidalField
from ..toroidalField import derivatePol, derivateTor
from ..toroidalField import changeResolution, fftToroidalField 
from ..misc import print_progress
from scipy.optimize import fixed_point
from scipy.integrate import dblquad
from typing import Tuple


class Surface_BoozerAngle(Surface):
    r"""
    The magnetic surface in Boozer coordinates $(\theta, \zeta)$.  
        $$ \phi = -\zeta + \omega(\theta,\zete) $$ 
    or
        $$ \phi = \mp\zeta \pm \omega(\theta,\zeta) $$
    When `reverseToroidalAngle` is `False`, the sign of zeta will become positive. 
        $$ \phi = \zeta + \omega(\theta,\zete) $$
    When `reverseOmegaAngle` is `True`, the sign of omega will become negative. 
        $$ \phi = -\zeta - \omega(\theta,\zete) $$
    There is a mapping with coordinates corrdinates $(R, \phi, Z)$
        $$ R = R(\theta, \zeta) $$ 
        $$ \phi = \phi(\theta, \zeta) $$ 
        $$ Z = Z(\theta, \zeta) $$ 
    """

    def __init__(self, r: ToroidalField, z: ToroidalField, omega: ToroidalField, reverseToroidalAngle: bool=True, reverseOmegaAngle: bool=False) -> None:
        super().__init__(r, z)
        self.omega = omega
        self.reverseToroidalAngle = reverseToroidalAngle
        self.reverseOmegaAngle = reverseOmegaAngle

    @property
    def stellSym(self) -> bool:
        return (not self.r.imIndex) and (not self.z.reIndex) and (not self.omega.reIndex)

    def changeStellSym(self, stellSym: bool) -> None: 
        self.r.reIndex, self.z.imIndex, self.omega.imIndex = True, True, True 
        if stellSym: 
            self.r.imIndex, self.z.reIndex, self.omega.reIndex = False, False, False
        else: 
            self.r.imIndex, self.z.reIndex, self.omega.reIndex = True, True, True

    def changeResolution(self, mpol: int, ntor: int): 
        self.r = changeResolution(self.r, mpol, ntor)
        self.z = changeResolution(self.z, mpol, ntor)
        self.omega = changeResolution(self.omega, mpol, ntor)

    @property
    def metric(self): 
        dPhidTheta = derivatePol(self.omega)
        if self.reverseOmegaAngle: 
            dPhidTheta = dPhidTheta*(-1)
        if not self.reverseOmegaAngle and self.reverseToroidalAngle:
            dPhidZeta = derivateTor(self.omega) - 1 
        elif not self.reverseOmegaAngle and not self.reverseToroidalAngle: 
            dPhidZeta = derivateTor(self.omega) + 1 
        if self.reverseOmegaAngle and self.reverseToroidalAngle:
            dPhidZeta = derivateTor(self.omega)*(-1) - 1 
        else:
            dPhidZeta = derivateTor(self.omega)*(-1) + 1 
        g_thetatheta = (
            self.dRdTheta*self.dRdTheta + 
            self.r*self.r*dPhidTheta*dPhidTheta +
            self.dZdTheta*self.dZdTheta
        )
        g_thetazeta = (
            self.dRdTheta*self.dRdZeta + 
            self.r*self.r*dPhidTheta*dPhidZeta +
            self.dZdTheta*self.dZdZeta
        )
        g_zetazeta = (
            self.dRdZeta*self.dRdZeta + 
            self.r*self.r*dPhidZeta*dPhidZeta +
            self.dZdZeta*self.dZdZeta
        )
        return g_thetatheta, g_thetazeta, g_zetazeta

    def getRZ(self, thetaGrid: np.ndarray, zetaGrid: np.ndarray) -> Tuple[np.ndarray]: 
        # if self.reverseToroidalAngle: 
        #     zetaGrid = -zetaGrid
        rArr = self.r.getValue(thetaGrid, zetaGrid)
        zArr = self.z.getValue(thetaGrid, zetaGrid)
        return rArr, zArr

    def getZeta(self, theta: np.ndarray, phi: np.ndarray, xtol: float=1e-15) -> np.ndarray:
        def zetaValue(zeta, theta, phi):
            if self.reverseToroidalAngle and not self.reverseOmegaAngle:
                return (
                    self.omega.getValue(theta, zeta) - phi
                )
            elif self.reverseToroidalAngle and self.reverseOmegaAngle:
                return (
                    - self.omega.getValue(theta, zeta) - phi
                )
            elif not self.reverseToroidalAngle and self.reverseOmegaAngle:
                return (
                    self.omega.getValue(theta, zeta) + phi
                )
            else: 
                return (
                    - self.omega.getValue(theta, zeta) + phi
                )
        return (
            fixed_point(zetaValue, phi, args=(theta, phi), xtol=xtol)
        )
        
    def getPhi(self, thetaArr: np.ndarray, zetaArr: np.ndarray) -> np.ndarray:
        omegaArr = self.omega.getValue(thetaArr, zetaArr)
        if self.reverseToroidalAngle and not self.reverseOmegaAngle:
            return (-zetaArr + omegaArr)
        elif self.reverseToroidalAngle and self.reverseOmegaAngle:
            return (-zetaArr - omegaArr)
        elif not self.reverseToroidalAngle and self.reverseOmegaAngle:
            return (zetaArr - omegaArr)
        else:
            return (zetaArr + omegaArr)

    def toCylinder(self, method: str="DFT", **kwargs) -> Surface_cylindricalAngle:
        if method == "DFT":
            return self.toCylinder_dft(**kwargs)
        else:
            return self.toCylinder_integrate()

    def toCylinder_dft(self, mpol: int=None, ntor: int=None, xtol: float=1e-13) -> Surface_cylindricalAngle:
        
        if mpol is None:
            mpol = 2*self.mpol+1
        if ntor is None:
            ntor = 2*self.ntor+1
        deltaTheta = 2*np.pi / (2*mpol+1) 
        deltaPhi = 2*np.pi / self.nfp / (2*ntor+1) 
        sampleTheta, samplePhi = np.arange(2*mpol+1)*deltaTheta, -np.arange(2*ntor+1)*deltaPhi 
        gridPhi, gridTheta = np.meshgrid(samplePhi, sampleTheta) 
        
        # Find fixed point of zeta. 
        def zetaValue(zeta, theta, phi):
            if self.reverseToroidalAngle and not self.reverseOmegaAngle:
                return (
                    self.omega.getValue(float(theta), float(zeta)) - phi
                )
            elif self.reverseToroidalAngle and self.reverseOmegaAngle:
                return (
                    - self.omega.getValue(float(theta), float(zeta)) - phi
                )
            elif not self.reverseToroidalAngle and self.reverseOmegaAngle:
                return (
                    self.omega.getValue(float(theta), float(zeta)) + phi
                )
            else: 
                return (
                    - self.omega.getValue(float(theta), float(zeta)) + phi
                )
        
        from ..misc import print_progress
        gridZeta = np.zeros_like(gridPhi)
        print("Convert a toroidal surface from Boozer coordinates to cylindrical coordinates... ")
        for i in range(len(gridZeta)): 
            for j in range(len(gridZeta[0])): 
                gridZeta[i,j] = float(
                    fixed_point(zetaValue, gridZeta[i,j], args=(gridTheta[i,j], gridPhi[i,j]), xtol=xtol)
                )
                print_progress(i*len(gridZeta[0])+j+1, len(gridZeta)*len(gridZeta[0]))
        
        sampleR = self.r.getValue(gridTheta, gridZeta)
        sampleZ = self.z.getValue(gridTheta, gridZeta)
        _fieldR = fftToroidalField(sampleR, nfp=self.nfp) 
        _fieldZ = fftToroidalField(sampleZ, nfp=self.nfp)
        return Surface_cylindricalAngle(
            _fieldR, 
            _fieldZ, 
            reverseToroidalAngle = True
        )

    # TODO: An untested function! 
    def toCylinder_integrate(self) -> Surface_cylindricalAngle:

        def substitutionFactor(theta, zeta) -> float:
            return float(
                derivateTor(self.omega).getValue(theta, zeta) - 1 
            )
        
        def helicalAngle(theta, zeta, m, n, _m, _n):
            return (
                m*theta - n*self.nfp*zeta - _m*theta + _n*self.nfp*(-zeta+self.omega.getValue(theta, zeta))
            )

        def rRe(zeta, theta, m, n, _m, _n): 
            factor = substitutionFactor(theta, zeta)
            _angle = helicalAngle(theta, zeta, m, n, _m, _n)
            return ((
                self.r.getRe(m,n) * np.cos(_angle) - 
                self.r.getIm(m,n) * np.sin(_angle)
            )*factor).flatten() 

        def rIm(zeta, theta, m, n, _m, _n): 
            factor = substitutionFactor(theta, zeta)
            _angle = helicalAngle(theta, zeta, m, n, _m, _n)
            return ((
                self.r.getRe(m,n) * np.sin(_angle) + 
                self.r.getIm(m,n) * np.cos(_angle)
            )*factor).flatten() 

        def zRe(zeta, theta, m, n, _m, _n): 
            factor = substitutionFactor(theta, zeta)
            _angle = helicalAngle(theta, zeta, m, n, _m, _n)
            return ((
                self.z.getRe(m,n) * np.cos(_angle) - 
                self.z.getIm(m,n) * np.sin(_angle)
            )*factor).flatten() 

        def zIm(zeta, theta, m, n, _m, _n): 
            factor = substitutionFactor(theta, zeta)
            _angle = helicalAngle(theta, zeta, m, n, _m, _n)
            return ((
                self.z.getRe(m,n) * np.sin(_angle) + 
                self.z.getIm(m,n) * np.cos(_angle)
            )*factor).flatten() 

        print("Transformation from Boozer to Cylindrical Coordinates... ")

        print("Compute the R component: ")
        rReArr = np.zeros((2*self.ntor+1)*self.mpol+self.ntor+1)
        rImArr = np.zeros((2*self.ntor+1)*self.mpol+self.ntor+1)
        for index in range((2*self.ntor+1)*self.mpol+self.ntor+1): 
            _m, _n = self.r.indexReverseMap(index)
            for _i ,m in enumerate(range(-self.mpol, self.mpol+1)):
                for _j, n in enumerate(range(-self.ntor, self.ntor+1)): 
                    rReArr[index] += dblquad(
                        rRe, 
                        0, 2*np.pi, 
                        0, 2*np.pi, 
                        args = (m, n, _m, _n)
                    )[0] * (1/4/np.pi/np.pi)
                    rImArr[index] += dblquad(
                        rIm, 
                        0, 2*np.pi, 
                        0, 2*np.pi, 
                        args = (m, n, _m, _n)
                    )[0] * (1/4/np.pi/np.pi)
                    print_progress(
                        index*((2*self.mpol+1)*(2*self.ntor+1)) + 
                        _i * (2*self.ntor+1) + _j+1, 
                        ((2*self.ntor+1)*self.mpol+self.ntor+1) * ((2*self.mpol+1)*(2*self.ntor+1))
                    ) 

        print("Compute the Z component: ")
        zReArr = np.zeros((2*self.ntor+1)*self.mpol+self.ntor+1)
        zImArr = np.zeros((2*self.ntor+1)*self.mpol+self.ntor+1)
        for index in range((2*self.ntor+1)*self.mpol+self.ntor+1): 
            _m, _n = self.r.indexReverseMap(index)
            for _i ,m in enumerate(range(-self.mpol, self.mpol+1)):
                for _j, n in enumerate(range(-self.ntor, self.ntor+1)): 
                    zReArr[index] += dblquad(
                        zRe, 
                        0, 2*np.pi, 
                        0, 2*np.pi, 
                        args = (m, n, _m, _n)
                    )[0] * (1/4/np.pi/np.pi)
                    zImArr[index] += dblquad(
                        zIm, 
                        0, 2*np.pi, 
                        0, 2*np.pi, 
                        args = (m, n, _m, _n)
                    )[0] * (1/4/np.pi/np.pi)
                    print_progress(
                        index*((2*self.mpol+1)*(2*self.ntor+1)) + 
                        _i * (2*self.ntor+1) + _j+1, 
                        ((2*self.ntor+1)*self.mpol+self.ntor+1) * ((2*self.mpol+1)*(2*self.ntor+1))
                    )

        _rField = ToroidalField(
            nfp = self.nfp, 
            mpol = self.mpol, 
            ntor = self.ntor,
            reArr = rReArr,
            imArr = rImArr
        )
        _zField = ToroidalField(
            nfp = self.nfp, 
            mpol = self.mpol, 
            ntor = self.ntor,
            reArr = zReArr,
            imArr = zImArr
        )
        return Surface_cylindricalAngle(
            _rField, 
            _zField, 
            reverseToroidalAngle = False
        )

    def plot_plt(self, ntheta: int=360, nzeta: int=360, fig=None, ax=None, **kwargs):
        if ax is None: 
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
        plt.sca(ax) 
        thetaArr = np.linspace(0, 2*np.pi, ntheta) 
        zetaArr = np.linspace(0, 2*np.pi, nzeta) 
        thetaGrid, zetaGrid = np.meshgrid(thetaArr, zetaArr) 
        rArr = self.r.getValue(thetaGrid, zetaGrid)
        zArr = self.z.getValue(thetaGrid, zetaGrid)
        # omegaArr = self.omega.getValue(thetaGrid, zetaGrid)
        # phiArr = zetaGrid - omegaArr
        phiArr = self.getPhi(thetaGrid, zetaGrid)
        # if self.reverseToroidalAngle:
        #     phiArr = -phiArr
        xArr = rArr * np.cos(phiArr)
        yArr = rArr * np.sin(phiArr)
        ax.plot_surface(xArr, yArr, zArr, color="coral") 
        plt.axis("equal")
        return fig

    def writeH5(self, filename="surf"):
        stellsym = (not self.r.imIndex) and (not self.z.reIndex) and (not self.omega.reIndex)
        with h5py.File(filename+".h5", 'w') as f:
            f.create_dataset(
                "resolution", 
                data = (self.nfp, self.mpol, self.ntor, int(stellsym), int(self.reverseToroidalAngle), int(self.reverseOmegaAngle)), 
                dtype = "int32"
            )
            f.create_group("r") 
            f.create_group("z") 
            f.create_group("omega")
            f["r"].create_dataset("re", data=self.r.reArr)
            f["z"].create_dataset("im", data=self.z.imArr)
            f["omega"].create_dataset("im", data=self.omega.imArr)
            if not stellsym:
                f["r"].create_dataset("im", data=self.r.imArr)
                f["z"].create_dataset("re", data=self.z.reArr)
                f["omega"].create_dataset("re", data=self.omega.reArr)

    @classmethod
    def readH5(cls, filename):
        with h5py.File(filename, 'r') as f:
            nfp = int(f["resolution"][0])
            mpol = int(f["resolution"][1])
            ntor = int(f["resolution"][2])
            stellsym = bool(f["resolution"][3])
            reverseToroidalAngle = bool(f["resolution"][4])
            reverseOmegaAngle = bool(f["resolution"][5])
            if stellsym:
                _r = ToroidalField(
                    nfp=nfp, mpol=mpol, ntor=ntor, 
                    reArr=f["r"]["re"][:], 
                    imArr=np.zeros_like(f["r"]["re"][:]), 
                    imIndex=False
                )
                _z = ToroidalField(
                    nfp=nfp, mpol=mpol, ntor=ntor, 
                    reArr=np.zeros_like(f["z"]["im"][:]),
                    imArr=f["z"]["im"][:],  
                    reIndex=False
                )
                _omega = ToroidalField(
                    nfp=nfp, mpol=mpol, ntor=ntor, 
                    reArr=np.zeros_like(f["omega"]["im"][:]),
                    imArr=f["omega"]["im"][:],  
                    reIndex=False
                )
            else:
                _r = ToroidalField(
                    nfp=nfp, mpol=mpol, ntor=ntor, 
                    reArr=f["r"]["re"][:], 
                    imArr=f["r"]["im"][:]
                )
                _z = ToroidalField(
                    nfp=nfp, mpol=mpol, ntor=ntor, 
                    reArr=f["z"]["re"][:],
                    imArr=f["z"]["im"][:] 
                )
                _omega = ToroidalField(
                    nfp=nfp, mpol=mpol, ntor=ntor, 
                    reArr=f["omega"]["re"][:],
                    imArr=f["omega"]["im"][:]
                )
            return cls(
                _r, _z, _omega, 
                reverseToroidalAngle=reverseToroidalAngle, 
                reverseOmegaAngle=reverseOmegaAngle
            )

if __name__ == "__main__":
    pass
