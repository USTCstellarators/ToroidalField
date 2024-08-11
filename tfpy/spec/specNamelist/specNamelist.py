#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# specNamelist.py


import py_spec


class SPECNamelist(py_spec.SPECNamelist):
    """
    The SPEC namelist class. 
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    from ._getInterface import getInterface


if __name__ == "__main__":
    pass
