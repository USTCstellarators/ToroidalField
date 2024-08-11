#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# _getInterface.py


from coilpy import FourSurf
from typing import List


def getInterface(self) -> List[FourSurf]:
    surface = list()
    xm = list()
    xn = list()
    rbcList = [list() for i in range(self._Nvol)]
    zbsList = [list() for i in range(self._Nvol)]
    rbsList = [list() for i in range(self._Nvol)]
    zbcList = [list() for i in range(self._Nvol)]
    for (m, n), data in self.interface_guess.items():
        xm.append(m)
        xn.append(n)
        for i in range(self._Nvol):
            rbcList[i].append(data["Rbc"][i])
            zbsList[i].append(data["Zbs"][i])
            rbsList[i].append(data["Rbs"][i])
            zbcList[i].append(data["Zbc"][i])
    surface = [FourSurf(xm=xm, xn=xn, rbc=rbcList[i], zbs=zbsList[i], rbs=rbsList[i], zbc=zbcList[i]) for i in range(self._Nvol)]
    return surface
    