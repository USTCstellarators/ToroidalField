#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# misc.py


import math
import numpy as np
from typing import List, Dict


# permeability of vacuum
mu0 = 4 * math.pi * 1e-7


def integrate(baseX: List[float] or np.ndarray, baseY: List[float] or np.ndarray, xLeft: float, xRight: float) -> float:
    if xLeft < min(baseX):
        raise ValueError(
            "The value of xLeft is out of range. "
        )
    if xRight > max(baseX):
        raise ValueError(
            "The value of xRight is out of range. "
        )
    if len(baseX) != len(baseY):
        raise ValueError(
            "The length of baseX and baseY should be equal. "
        )
    nums = len(baseX)
    ans = 0
    base = sorted([i for i in zip(baseX, baseY)], key=lambda k: [k[0], k[1]])
    index = 0
    while index < nums:
        if base[index][0] > xLeft:
            tempXLeft = xLeft
            tempYLeft = np.interp(xLeft, [base[i][0] for i in range(nums)], [
                                  base[i][1] for i in range(nums)])
            while base[index][0] < xRight:
                tempXRight = base[index][0]
                tempYRight = base[index][1]
                ans += (tempXRight - tempXLeft) * (tempYRight + tempYLeft) / 2
                tempXLeft = base[index][0]
                tempYLeft = base[index][1]
                index += 1
            tempXRight = xRight
            tempYRight = np.interp(xRight, [base[i][0] for i in range(nums)], [
                                   base[i][1] for i in range(nums)])
            ans += (tempXRight - tempXLeft) * (tempYRight + tempYLeft) / 2
            break
        else:
            index += 1
    return ans


def writeSPECInput(SPEC_input: str, datas: Dict, interfaceShape: bool) -> None:
    """
    Write SPEC input file. 
    """

    nvol = datas["nvol"]
    ninterface = nvol +1

    file = open(SPEC_input, "w")

    # physicslist
    file.write("&physicslist\n")

    file.write("    " + "{:12} = {:d}".format("igeometry",   3) + "\n")
    file.write("    " + "{:12} = {:d}".format("istellsym",   1) + "\n")
    file.write("    " + "{:12} = {:d}".format("lfreebound",  0) + "\n")

    file.write("    " + "{:12} = {:.5e}".format("phiedge", datas["phiedge"]) + "\n")
    file.write("    " + "{:12} = {:.5e}".format("curtor", datas["curtor"]) + "\n")
    file.write("    " + "{:12} = {:.5e}".format("curpol", datas["curpol"]) + "\n")
    
    file.write("    " + "{:12} = {:.5e}".format("gamma", datas["gamma"]) + "\n")
    
    file.write("    " + "{:12} = {:d}".format("nfp",  datas["nfp"]) + "\n")
    file.write("    " + "{:12} = {:d}".format("nvol", datas["nvol"]) + "\n")
    file.write("    " + "{:12} = {:d}".format("mpol", datas["mpol"]) + "\n")
    file.write("    " + "{:12} = {:d}".format("ntor", datas["ntor"]) + "\n")
    
    file.write("    {:12} = ".format("lrad"))
    for i in range(nvol):
        file.write("{:<11d}".format(datas["lrad"][i]))
        file.write("  ")
    file.write("\n")
    
    file.write("    " + "{:12} = {:d}".format("lconstraint", int(datas["lconstraint"])) + "\n")
    
    file.write("    {:12} = ".format("tflux"))
    for i in range(nvol):
        file.write("{:.5e}".format(datas["tflux"][i]))
        file.write("  ")
    file.write("\n")
    file.write("    {:12} = ".format("pflux"))
    for i in range(nvol):
        file.write("{:.5e}".format(datas["pflux"][i]))
        file.write("  ")
    file.write("\n")
    
    file.write("    {:12} = ".format("helicity"))
    for i in range(nvol):
        file.write("{:.5e}".format(datas["helicity"][i]))
        file.write("  ")
    file.write("\n")
    
    file.write("    " + "{:12} = {:.5e}".format("pscale", mu0) + "\n")
    file.write("    {:12} = ".format("pressure"))
    for i in range(nvol):
        file.write("{:.5e}".format(datas["pressure"][i]))
        file.write("  ")
    file.write("\n")
    file.write("    " + "{:12} = {:d}".format("ladiabatic", 1) + "\n")
    file.write("    {:12} = ".format("adiabatic"))
    for i in range(nvol):
        file.write("{:.5e}".format(datas["adiabatic"][i]))
        file.write("  ")
    file.write("\n")

    file.write("    {:12} = ".format("mu"))
    for i in range(nvol):
        file.write("{:.5e}".format(datas["mu"][i]))
        file.write("  ")
    file.write("\n")

    file.write("    {:12} = ".format("ivolume"))
    for i in range(nvol):
        file.write("{:.5e}".format(datas["ivolume"][i]))
        file.write("  ")
    file.write("\n")
    file.write("    {:12} = ".format("isurf"))
    for i in range(nvol):
        file.write("{:.5e}".format(datas["isurf"][i]))
        file.write("  ")
    file.write("\n")
    
    file.write("    {:12} = ".format("pl"))
    for i in range(ninterface):
        file.write("{:<11d}".format(0))
        file.write("  ")
    file.write("\n")
    file.write("    {:12} = ".format("ql"))
    for i in range(ninterface):
        file.write("{:<11d}".format(0))
        file.write("  ")
    file.write("\n")
    file.write("    {:12} = ".format("pr"))
    for i in range(ninterface):
        file.write("{:<11d}".format(0))
        file.write("  ")
    file.write("\n")
    file.write("    {:12} = ".format("qr"))
    for i in range(ninterface):
        file.write("{:<11d}".format(0))
        file.write("  ")
    file.write("\n")
    file.write("    {:12} = ".format("iota"))
    for i in range(ninterface):
        file.write("{:.12e}".format(datas["iota"][i]))
        file.write("  ")
    file.write("\n")
    file.write("    {:12} = ".format("lp"))
    for i in range(ninterface):
        file.write("{:<11d}".format(0))
        file.write("  ")
    file.write("\n")
    file.write("    {:12} = ".format("lq"))
    for i in range(ninterface):
        file.write("{:<11d}".format(0))
        file.write("  ")
    file.write("\n")
    file.write("    {:12} = ".format("rp"))
    for i in range(ninterface):
        file.write("{:<11d}".format(0))
        file.write("  ")
    file.write("\n")
    file.write("    {:12} = ".format("rq"))
    for i in range(ninterface):
        file.write("{:<11d}".format(0))
        file.write("  ")
    file.write("\n")
    file.write("    {:12} = ".format("oita"))
    for i in range(ninterface):
        file.write("{:.12e}".format(datas["iota"][i]))
        file.write("  ")
    file.write("\n")

    file.write("    {:12} = ".format("rac"))
    for i in range(datas["ntor"] + 1):
        file.write("{:.5e}".format(datas["rac"][i]))
        file.write("  ")
    file.write("\n")
    file.write("    {:12} = ".format("zas"))
    for i in range(datas["ntor"] + 1):
        file.write("{:.5e}".format(datas["zas"][i]))
        file.write("  ")
    file.write("\n")
    for i in range(len(datas["rbc"])):
        file.write("    {:12} = ".format("rbc(" + str(int(datas["in"][i])) + ", " + str(int(datas["im"][i])) + ")"))
        file.write("{:.5e}".format(datas["rbc"][i]))
        file.write("    {:12} = ".format("zbs(" + str(int(datas["in"][i])) + ", " + str(int(datas["im"][i])) + ")"))
        file.write("{:.5e}".format(datas["zbs"][i]))
        file.write("    {:12} = ".format("rbs(" + str(int(datas["in"][i])) + ", " + str(int(datas["im"][i])) + ")"))
        file.write("{:.5e}".format(0))
        file.write("    {:12} = ".format("zbc(" + str(int(datas["in"][i])) + ", " + str(int(datas["im"][i])) + ")"))
        file.write("{:.5e}".format(0))
        file.write("\n")

    file.write("    " + "{:12} = {:.5e}".format("mupftol", 1e-14) + "\n")

    file.write("    " + "{:12} = {:d}".format("mupfits",   8) + "\n")

    file.write("/\n")
    file.write("\n")

    # numericlist
    file.write("&numericlist\n")
    if nvol == 1 or interfaceShape == False:
        file.write("    " + "{:12} = {:d}".format("linitialize", 1) + "\n")
    else:
        file.write("    " + "{:12} = {:d}".format("linitialize", 0) + "\n")
    file.write("    " + "{:12} = {:d}".format("ndiscrete",   2) + "\n")
    file.write("    " + "{:12} = {:d}".format("nquad",      -1) + "\n")
    file.write("    " + "{:12} = {:d}".format("impol",      -4) + "\n")
    file.write("    " + "{:12} = {:d}".format("intor",      -4) + "\n")
    file.write("    " + "{:12} = {:d}".format("lsparse",     0) + "\n")
    file.write("    " + "{:12} = {:d}".format("lsvdiota",    0) + "\n")
    file.write("    " + "{:12} = {:d}".format("imethod",     3) + "\n")
    file.write("    " + "{:12} = {:d}".format("iorder",      2) + "\n")
    file.write("    " + "{:12} = {:d}".format("iprecon",     0) + "\n")
    file.write("    " + "{:12} = {:.5e}".format("iotatol", -1.0) + "\n")
    file.write("/\n")
    file.write("\n")

    # locallist
    file.write("&locallist\n")
    file.write("    " + "{:12} = {:d}".format("lbeltrami", 4) + "\n")
    file.write("    " + "{:12} = {:d}".format("linitgues", 1) + "\n")
    file.write("/\n")
    file.write("\n")

    # globallist
    file.write("&globallist\n")
    file.write("    " + "{:12} = {:d}".format("lfindzero", 2) + "\n")
    file.write("    " + "{:12} = {:.5e}".format("escale", 0.0) + "\n")
    file.write("    " + "{:12} = {:.5e}".format("pcondense", 4.0) + "\n")
    file.write("    " + "{:12} = {:.5e}".format("forcetol", 1e-10) + "\n")
    file.write("    " + "{:12} = {:.5e}".format("c05xtol", 1e-12) + "\n")
    file.write("    " + "{:12} = {:.5e}".format("c05factor", 1e-2) + "\n")
    file.write("    " + "{:12} = .true.".format("lreadgf") + "\n")
    file.write("    " + "{:12} = {:.5e}".format("opsilon", 1.0) + "\n")
    file.write("    " + "{:12} = {:.5e}".format("epsilon", 0.0) + "\n")
    file.write("    " + "{:12} = {:.5e}".format("upsilon", 1.0) + "\n")
    file.write("/\n")
    file.write("\n")

    # diagnosticslist
    file.write("&diagnosticslist\n")
    file.write("    " + "{:12} = {:.5e}".format("odetol", 1e-7) + "\n")
    file.write("    " + "{:12} = {:.5e}".format("absreq", 1e-8) + "\n")
    file.write("    " + "{:12} = {:.5e}".format("relreq", 1e-8) + "\n")
    file.write("    " + "{:12} = {:.5e}".format("absacc", 1e-4) + "\n")
    file.write("    " + "{:12} = {:.5e}".format("epsr",   1e-8) + "\n")
    file.write("    " + "{:12} = {:d}".format("nppts", 400) + "\n")
    file.write("    " + "{:12} = {:d}".format("nptrj", -1) + "\n")
    file.write("    " + "{:12} = .false.".format("lhevalues") + "\n")
    file.write("    " + "{:12} = .false.".format("lhevectors") + "\n")
    file.write("/\n")
    file.write("\n")

    # screenlist
    file.write("&screenlist\n")
    file.write("    " + "{:12} = .true.".format("wpp00aa") + "\n")
    file.write("/\n")
    file.write("\n")

    if nvol != 1:
        for j in range(len(datas["im"])):
            file.write("    " + "{:<5d}".format(int(datas["im"][j])) + "{:<5d}".format(int(datas["in"][j])))
            for i in range(1, ninterface):
                file.write("{:.5e}".format(float(datas["interface_rc"][i, j])) + "{:5}".format("") +
                "{:.5e}".format(float(datas["interface_zs"][i, j])) + "{:5}".format("") +
                "{:.5e}".format(0) + "{:5}".format("") +"{:.5e}".format(0) + "{:5}".format(""))
            file.write("\n")

    file.close()

    return


if __name__ == "__main__":
    pass
