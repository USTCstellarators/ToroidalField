#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# _plotCrossSection.py


def plotCrossSection(self, zeta: float=0.0, ntheta: int=256, nSurf: int=2, ax=None, **kwargs):
    import numpy as np
    import matplotlib.pyplot as plt
    thetaArr = np.linspace(0, 2*np.pi, ntheta)
    zetaArr = np.ones(ntheta)*zeta
    if ax is None:
        fig, ax = plt.subplots()
    if kwargs.get("c") == None:
        kwargs.update({"c": "coral"})
    for i in range(self.ns):
        index = self.ns-1-i
        if index % nSurf == 0:
            _surf, _lam = self.getSurface(index)
            _rArr, _zArr = _surf.getRZ(thetaArr, zetaArr)
            ax.plot(_rArr, _zArr, **kwargs)
    ax.set_xlabel(r"$R$", fontsize=18)
    ax.set_ylabel(r"$Z$", fontsize=18)
    ax.axis('equal')
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    return