#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# laplaceField.py


from ..specOut import SPECOut


class LaplaceFiled: 
    
    def __init__(self, file: str) -> None:
        self.data = SPECOut(file)
        assert self.data.input.physics.nvol == 1 
        self.nfp = self.data.input.physics.nvol
        


if __name__ == "__main__": 
    pass
