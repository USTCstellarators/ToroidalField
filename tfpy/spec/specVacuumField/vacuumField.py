#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vacuumField.py


from typing import List
from tfpy.geometry import Surface_BoozerAngle
from ..specOut import SPECOut


class SPECVacuumField: 
    
    def __init__(self, file: str) -> None:
        self.data = SPECOut(file)
        if self.data.input.physics.nvol != 1:
            raise ValueError("SPECVacuumField requires a single-volume SPEC result")
        self.nfp = self.data.input.physics.nfp
    
    def getBoozerSurfaces(self, tflux: List[float]) -> List[Surface_BoozerAngle]:
        raise NotImplementedError("Boozer surface extraction from SPEC vacuum data is not implemented")
