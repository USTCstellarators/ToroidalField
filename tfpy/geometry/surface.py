#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# surface.py


from ..toroidalField import ToroidalField 
from ..toroidalField import derivatePol, derivateTor 
from ..toroidalField import changeResolution 


class Surface:

    def __init__(self, r: ToroidalField, z: ToroidalField) -> None:
        assert r.nfp == z.nfp
        assert r.mpol == z.mpol
        assert r.ntor == z.ntor
        # self._mpol, self._ntor = r.mpol, r.ntor
        self.r = r
        self.z = z

    @property
    def nfp(self) -> int:
        return self.r.nfp 

    @property
    def mpol(self) -> int:
        return self.r.mpol 

    @property
    def ntor(self) -> int: 
        return self.r.ntor 

    @property
    def dRdTheta(self) -> ToroidalField:
        return derivatePol(self.r)

    @property
    def dRdZeta(self) -> ToroidalField:
        return derivateTor(self.r)

    @property
    def dZdTheta(self) -> ToroidalField:
        return derivatePol(self.z)

    @property
    def dZdZeta(self) -> ToroidalField:
        return derivateTor(self.z) 

    def changeResolution(self, mpol: int, ntor: int): 
        self._mpol, self._ntor = mpol, ntor
        self.r = changeResolution(self.r, mpol, ntor)
        self.z = changeResolution(self.z, mpol, ntor)


if __name__ == "__main__":
    pass
