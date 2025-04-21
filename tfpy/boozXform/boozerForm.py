#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# boozerForm.py


import numpy as np 
from booz_xform import Booz_xform
from ..toroidalField import ToroidalField
from ..geometry import Surface_BoozerAngle


class BoozerForm(Booz_xform): 
    
    def __init__(self, fileName: str) -> None:
        super().__init__()
        self.verbose = 0
        self.read_boozmn(fileName)

    def getSurface(self, surfaceIndex: int=-1, asym: bool=True, reverseToroidal: bool=False, reverseOmegaAngle:bool=True) -> Surface_BoozerAngle: 
        nfp = int(self.nfp)
        mpol = int(self.mboz) - 1
        ntor = int(self.nboz) 
        rbc = self.rmnc_b[:, surfaceIndex].copy()
        zbs = self.zmns_b[:, surfaceIndex].copy()
        if not asym: 
            rbs = self.rmns_b[:, surfaceIndex].copy()
            zbc = self.zmnc_b[:, surfaceIndex].copy()
        else: 
            rbs = np.zeros_like(rbc)
            zbc = np.zeros_like(zbs)
        rbc[1:-1] = rbc[1:-1] / 2
        rbs[1:-1] = rbs[1:-1] / 2
        zbc[1:-1] = zbc[1:-1] / 2
        zbs[1:-1] = zbs[1:-1] / 2
        _rfield = ToroidalField(
            nfp = nfp, 
            mpol = mpol, 
            ntor = ntor,
            reArr = rbc, 
            imArr = -rbs, 
            reIndex = True, 
            imIndex = not asym
        ) 
        _zfield = ToroidalField(
            nfp = nfp, 
            mpol = mpol, 
            ntor = ntor,
            reArr = zbc, 
            imArr = -zbs, 
            reIndex = not asym, 
            imIndex = True
        ) 
        nus = self.numns_b[:, surfaceIndex].copy()
        if not asym:
            nuc = self.numnc_b[:, surfaceIndex].copy()
        else: 
            nuc = np.zeros_like(nus)
        nuc[1:-1] = nuc[1:-1] / 2
        nus[1:-1] = nus[1:-1] / 2
        _omegafield = ToroidalField(
            nfp = nfp,
            mpol = mpol, 
            ntor = ntor,
            reArr = -nuc, 
            imArr = -nus, 
            reIndex = not asym, 
            imIndex = True
        )
        return Surface_BoozerAngle(_rfield, _zfield, _omegafield, reverseToroidalAngle=reverseToroidal, reverseOmegaAngle=reverseOmegaAngle) 

    def getB(self, surfaceIndex: int=-1, asym: bool=True) -> ToroidalField:
        """
        returns: 
            B
        """
        nfp = int(self.nfp)
        mpol = int(self.mboz) - 1
        ntor = int(self.nboz) 
        bmnc = self.bmnc_b[:, surfaceIndex].copy()
        if not asym: 
            bmns = self.bmns_b[:, surfaceIndex].copy()
        else: 
            bmns = np.zeros_like(bmnc)
        bmnc[1:-1] = bmnc[1:-1] / 2
        bmns[1:-1] = bmns[1:-1] / 2
        _bfield = ToroidalField(
            nfp = nfp, 
            mpol = mpol, 
            ntor = ntor,
            reArr = bmnc, 
            imArr = -bmns, 
            reIndex = True, 
            imIndex = not asym
        ) 
        return _bfield 
    
    def getJacobian(self, surfaceIndex: int=-1, asym: bool=True) -> ToroidalField:
        """
        returns: 
            Jacobian
        """
        nfp = int(self.nfp)
        mpol = int(self.mboz) - 1
        ntor = int(self.nboz)
        jmnc = self.gmnc_b[:, surfaceIndex].copy()
        if not asym: 
            jmns = self.gmns_b[:, surfaceIndex].copy()
        else:
            jmns = np.zeros_like(jmnc)
        jmnc[1:-1] = jmnc[1:-1] / 2
        jmns[1:-1] = jmns[1:-1] / 2
        _jfield = ToroidalField(
            nfp = nfp, 
            mpol = mpol, 
            ntor = ntor,
            reArr = jmnc, 
            imArr = -jmns, 
            reIndex = True, 
            imIndex = not asym
        )
        return _jfield

    def getIota(self, surfaceIndex: int=-1):
        return self.iota[surfaceIndex]


if __name__ == "__main__": 
    pass
