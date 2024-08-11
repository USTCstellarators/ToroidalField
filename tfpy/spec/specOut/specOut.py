#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# specOut.py


import py_spec


class SPECOut(py_spec.SPECout):
    """
    Class that contains the output of a SPEC calculation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # from ._plot_kam_surface import plot_kam_surface

    from ._plot_pressure import plot_pressure

    from ._plot_poincare import plot_poincare

    from .mapping import cylinder2spec


if __name__ == "__main__":
    pass
