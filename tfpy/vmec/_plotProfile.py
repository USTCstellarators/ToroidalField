#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# _plotProfile.py


def plotProfile(self, name: str, xAxis: str='sqrtflux', ax=None, **kwargs):
    r'''
    Plot the profile of rotational transform. 
    Args: 
        `name`: shoule be one of the `'iota'` or `Iterable`
        `xAixs`: the x label, should be one of `'flux'`, `'sqrtflux'`(default) or `'r'`
    '''
    import numpy as np
    import matplotlib.pyplot as plt 
    if ax is None:
        fig, ax = plt.subplot()
    if xAxis == 'flux':
        xArr = np.linspace(0, 1, self.ns)
        kwargs.update({"xlabel": r"$\psi$"})
    elif xAxis == 'sqrtflux':
        xArr = np.power(np.linspace(0, 1, self.ns), 0.5)
        kwargs.update({"xlabel": r"$\sqrt{\psi}$"})
    elif xAxis == 'r':
        xArr = np.zeros(2*self.ns-1)
        kwargs.update({"xlabel": r"$R$"})
        for i in range(self.ns):
            rc = self.rmnc[i, :].copy()
            if i == 0:
                xArr[self.ns-1] = np.sum(rc)
            else:
                xArr[self.ns-1+i] = np.sum(rc)
                xArr[self.ns-1-i] = np.sum(rc * np.cos(self.xm*np.pi))
    else:
        print('There is no figure because of the wrong axis... ')
        return
    try:
        from collections.abc import Iterable
    except ImportError:
        from collections import Iterable
    if name == "iota":
        yArr = np.zeros(2*self.ns-1)
        kwargs.update({"ylabel": r"$\iota$"})
        if kwargs.get('label') == None:
            kwargs.update({'label': r"$\iota$"})
        for i in range(self.ns):
            if i == 0:
                yArr[self.ns-1] = self.iota(i)
            else:
                yArr[self.ns-1+i] = self.iota(i)
                yArr[self.ns-1-i] = self.iota(i)
    elif isinstance(name, Iterable):
        # TODO
        print("To Do... ")
    else:
        print('There is no figure because of the wrong name... ')
        return
    if kwargs.get('c') == None:
        kwargs.update({'c': 'coral'})
    if kwargs.get('ls') == None:
        kwargs.update({'ls': '--'})
    ax.plot(xArr, yArr, c=kwargs.get('c'), ls=kwargs.get('ls'), label=kwargs.get('label'))
    ax.set_xlabel(kwargs.get("xlabel"), fontsize=18)
    ax.set_ylabel(kwargs.get("ylabel"), fontsize=18)
    ax.legend(fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    return