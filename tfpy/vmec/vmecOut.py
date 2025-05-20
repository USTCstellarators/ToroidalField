#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vmecOut.py


import xarray
import numpy as np
from ..toroidalField import ToroidalField
from ..geometry import Surface_cylindricalAngle
from typing import Tuple


class VMECOut():
    
    def __init__(self, fileName: str, ssym: bool=True) -> None:
        _vmeclib = xarray.open_dataset(fileName)
        self.keys = [key for key in _vmeclib.data_vars] 
        for key in self.keys:
            setattr(self, key, _vmeclib[key].values)
        self.ns = int(self.ns)
        self.nfp = int(self.nfp) 
        self.ssym = ssym

    # plot functions
    from ._plotCrossSection import plotCrossSection
    from ._plotProfile import plotProfile

    # transform functions
    from ._vmecTransform import transBoozer

    def getSurface(self, surfaceIndex: int=-1, reverseToroidal: bool=True) -> Tuple:
        """
        returns:
            surface, lambda
        """
        mpol = int(self.mpol) - 1 
        ntor = int(self.ntor) 
        rbc = self.rmnc[surfaceIndex, :].copy()
        zbs = self.zmns[surfaceIndex, :].copy()
        lams = self.lmns[surfaceIndex, :].copy()
        if not self.ssym: 
            rbs = self.rmns[surfaceIndex, :].copy()
            zbc = self.zmnc[surfaceIndex, :].copy()
            lamc = self.lmnc[surfaceIndex, :].copy()
        else:
            rbs = np.zeros_like(rbc) 
            zbc = np.zeros_like(zbs) 
            lamc = np.zeros_like(lams) 
        rbc[1: ] = rbc[1: ] / 2 
        zbs[1: ] = zbs[1: ] / 2 
        rbs[1: ] = rbs[1: ] / 2 
        zbc[1: ] = zbc[1: ] / 2 
        lams[1: ] = lams[1: ] / 2
        lamc[1: ] = lamc[1: ] / 2
        _rField = ToroidalField(
            nfp = self.nfp, 
            mpol = mpol, 
            ntor = ntor, 
            reArr = rbc, 
            imArr = -rbs, 
            reIndex = True, 
            imIndex = not self.ssym
        )
        _zField = ToroidalField(
            nfp = self.nfp, 
            mpol = mpol, 
            ntor = ntor, 
            reArr = zbc, 
            imArr = -zbs, 
            reIndex = not self.ssym, 
            imIndex = True
        )
        lam = ToroidalField(
            nfp = self.nfp, 
            mpol = mpol, 
            ntor = ntor, 
            reArr = lamc, 
            imArr = -lams, 
            reIndex = not self.ssym, 
            imIndex = True
        )
        surf = Surface_cylindricalAngle(_rField, _zField, reverseToroidalAngle=reverseToroidal)
        return surf, lam

    def getJacobian(self, surfaceIndex: int=-1) -> ToroidalField:
        mpol_nyq = int(np.max(self.xm_nyq))
        ntor_nyq = int(np.max(self.xn_nyq/self.nfp))
        gc = self.gmnc[surfaceIndex, :].copy()
        if not self.ssym:
            gs = self.gmns[surfaceIndex, :].copy()
        else:
            gs = np.zeros_like(gc)
        gc[1: ] = gc[1: ] / 2
        gs[1: ] = gs[1: ] / 2
        _Jacobian = ToroidalField(
            nfp = self.nfp,
            mpol = mpol_nyq, 
            ntor = ntor_nyq,
            reArr = gc,
            imArr = -gs, 
            reIndex = True, 
            imIndex = not self.ssym
        )
        return _Jacobian
    
    def getBsup(self, surfaceIndex: int=-1) -> Tuple:
        """
        returns:
            bsupu, bsupv
        """
        mpol_nyq = int(np.max(self.xm_nyq))
        ntor_nyq = int(np.max(self.xn_nyq/self.nfp))
        bSupUc = self.bsupumnc[surfaceIndex, :].copy() 
        bSupVc = self.bsupvmnc[surfaceIndex, :].copy()
        if not self.ssym:
            bSupUs = self.bsupumns[surfaceIndex, :].copy()
            bSupVs = self.bsupvmns[surfaceIndex, :].copy()
        else:
            bSupUs = np.zeros_like(bSupUc)
            bSupVs = np.zeros_like(bSupVc)
        bSupUc[1: ] = bSupUc[1: ] / 2
        bSupVc[1: ] = bSupVc[1: ] / 2
        bSupUs[1: ] = bSupUs[1: ] / 2
        bSupVs[1: ] = bSupVs[1: ] / 2
        _bSupU = ToroidalField(
            nfp = self.nfp,
            mpol = mpol_nyq, 
            ntor = ntor_nyq,
            reArr = bSupUc,
            imArr = -bSupUs, 
            reIndex = True, 
            imIndex = not self.ssym
        )
        _bSupV = ToroidalField(
            nfp = self.nfp,
            mpol = mpol_nyq, 
            ntor = ntor_nyq,
            reArr = bSupVc,
            imArr = -bSupVs, 
            reIndex = True, 
            imIndex = not self.ssym
        )
        return _bSupU, _bSupV

    def getBsub(self, surfaceIndex: int=-1) -> Tuple:
        """
        returns: 
            bsubu, bsubv, bsubs 
        """
        mpol_nyq = int(np.max(self.xm_nyq))
        ntor_nyq = int(np.max(self.xn_nyq/self.nfp))
        bSubUc = self.bsubumnc[surfaceIndex, :].copy() 
        bSubVc = self.bsubvmnc[surfaceIndex, :].copy()
        bSubSs = self.bsubsmns[surfaceIndex, :].copy()
        if not self.ssym:
            bSubUs = self.bsupumns[surfaceIndex, :].copy()
            bSubVs = self.bsupvmns[surfaceIndex, :].copy()
            bSubSc = self.bsupsmnc[surfaceIndex, :].copy()
        else:
            bSubUs = np.zeros_like(bSubUc)
            bSubVs = np.zeros_like(bSubVc)
            bSubSc = np.zeros_like(bSubSs)
        bSubUc[1: ] = bSubUc[1: ] / 2 
        bSubUs[1: ] = bSubUs[1: ] / 2  
        bSubVc[1: ] = bSubVc[1: ] / 2 
        bSubVs[1: ] = bSubVs[1: ] / 2 
        bSubSc[1: ] = bSubSc[1: ] / 2 
        bSubSs[1: ] = bSubSs[1: ] / 2 
        _bSubU = ToroidalField(
            nfp = self.nfp,
            mpol = mpol_nyq, 
            ntor = ntor_nyq,
            reArr = bSubUc,
            imArr = -bSubUs, 
            reIndex = True, 
            imIndex = not self.ssym
        ) 
        _bSubV = ToroidalField(
            nfp = self.nfp,
            mpol = mpol_nyq, 
            ntor = ntor_nyq,
            reArr = bSubVc,
            imArr = -bSubVs, 
            reIndex = True, 
            imIndex = not self.ssym
        )
        _bSubS = ToroidalField(
            nfp = self.nfp,
            mpol = mpol_nyq, 
            ntor = ntor_nyq,
            reArr = bSubSc,
            imArr = -bSubSs, 
            reIndex = not self.ssym, 
            imIndex = True
        )
        return _bSubU, _bSubV, _bSubS

    def getB(self, surfaceIndex: int=-1) -> ToroidalField:
        """
        returns:
            B
        """
        mpol_nyq = int(np.max(self.xm_nyq))
        ntor_nyq = int(np.max(self.xn_nyq/self.nfp))
        bc = self.bmnc[surfaceIndex, :].copy()
        if not self.ssym:
            bs = self.bmns[surfaceIndex, :].copy()
        else:
            bs = np.zeros_like(bc)
        bc[1: ] = bc[1: ] / 2
        bs[1: ] = bs[1: ] / 2
        _bField = ToroidalField(
            nfp = self.nfp,
            mpol = mpol_nyq, 
            ntor = ntor_nyq,
            reArr = bc,
            imArr = -bs, 
            reIndex = True, 
            imIndex = not self.ssym
        )
        return _bField

    def iota(self, surfaceIndex: int=-1): 
        return float(self.iotas[surfaceIndex])

    def getIG(self, bsubu: ToroidalField=None, bsubv: ToroidalField=None, surfaceIndex: int=-1): 
        if (bsubu is None) or (bsubv is None): 
            bsubu, bsubv, _ = self.getBsub(surfaceIndex)
        return float(bsubu.getRe(0,0)),  float(bsubv.getRe(0,0))

    def getOmega(self, bsubu: ToroidalField=None, bsubv: ToroidalField=None, lam: ToroidalField=None, iota: float=None, surfaceIndex: int=-1): 
        if (bsubu is None) or (bsubv is None) or (lam is None) or (iota is None): 
            bsubu, bsubv, _ = self.getBsub(surfaceIndex)
            _, lam = self.getSurface(surfaceIndex) 
            iota = self.iota(surfaceIndex)
        curI, curG = self.getIG(bsubu, bsubv) 
        from ..toroidalField import derivatePol, derivateTor
        overI = bsubu - curI - curI*derivatePol(lam)
        overG = bsubv - curG - curI*derivateTor(lam)
        _omega = ToroidalField(
            nfp = self.nfp, 
            mpol = overI.mpol, 
            ntor = overI.ntor, 
            reArr = np.zeros_like(overI.reArr), 
            imArr = np.zeros_like(overI.reArr)
        )
        for index in range(len(overI.reArr)):
            m, n = overI.indexReverseMap(index)
            if m!=0: 
                _omega.setRe(m, n, overI.getIm(m,n)/m) 
                _omega.setIm(m, n, overI.getRe(m,n)/m) 
            elif n!=0: 
                _omega.setRe(m, n, -overG.getIm(m,n)/n/self.nfp)  
                _omega.setIm(m, n, -overG.getRe(m,n)/n/self.nfp)
        return _omega * (-1/(iota*curI+curG))


if __name__ == "__main__":
    pass
