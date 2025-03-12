#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vacuumField.py


from typing import List
from tfpy.geometry import Surface_BoozerAngle
from ..specOut import SPECOut


class SPECVacuumField: 
    
    def __init__(self, file: str) -> None:
        self.data = SPECOut(file)
        assert self.data.input.physics.nvol == 1 
        self.nfp = self.data.input.physics.nvol
    
    # TODO
    def getBoozerSurfaces(self, tflux: List[float]) -> List[Surface_BoozerAngle]:
        pass
        


if __name__ == "__main__": 
    pass
