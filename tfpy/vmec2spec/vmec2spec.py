#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vmec2spec.py


import math
import xarray
import numpy as np
from typing import List
from .misc import integrate, writeSPECInput
from .misc import mu0


def vmecOut2spec(VMEC_output: str, SPEC_input: str, interfaceLabel: List[float], 
                fluxLabel: str="toroidal", 
                lconstraint: int=0, 
                interfaceShape: bool = True, 
                changePoloidalAngle: bool=True, 
                changePflux: bool=False, 
                changeIota: bool=False, 
                **kwargs) -> None:
    """
    This function creates a SPEC input namelist from a VMEC output file. 
    Args:
        VMEC_output: The VMEC outputput file. 
        SPEC_input: The SPEC input file. 
        interfaceLabel: The normalized magnetic flux in subvolumes. 
        fluxLabel : The surface label shoulde be "toroidal" or "poloidal". 
        lconstraint: selects constraints.
    """

    try:
        VMECout = xarray.open_dataset(VMEC_output)
    except:
        raise FileExistsError(
            "Please cheak your argument. The first argument should be a VMEC output. "
        )
    if abs(min(interfaceLabel)) > 1e-10 or abs(max(interfaceLabel) - 1) > 1e-10:
        raise ValueError(
            "The maximum and minimum value of interface label should be 1 and 0. "
        )
    if not all([interfaceLabel[i] < interfaceLabel[i+1] for i in range(len(interfaceLabel)-1)]):
        raise ValueError(
            "List interfaceLabel should be strictly monotonically increasing. "
        )

    VMEC_ns = int(VMECout["ns"].values)
    VMEC_tflux = VMECout["phi"].values
    VMEC_pflux = np.abs(VMECout["chi"].values)
    if changePflux == True:
        VMEC_pflux *= -1
    VMEC_tpflux2 = VMEC_tflux * VMEC_pflux
    VMEC_psi = VMEC_tflux / np.pi / 2
    VMEC_chi = VMEC_pflux / np.pi / 2
    flux_label = np.linspace(0, 1, VMEC_ns)
    if not (fluxLabel == "toroidal" or fluxLabel == "poloidal"):
        raise ValueError(
            "The flux label should be toroidal or poloidal, cheak the flux label. "
        )
    VMEC_iota = np.abs(VMECout["iotaf"].values)
    if changeIota == True:
        VMEC_iota *= -1
    VMEC_g = [abs(VMECout["gmnc"].values[i][0]) for i in range(VMEC_ns)]
    VMEC_jpol = np.abs(VMECout["jcuru"].values)
    VMEC_jtor = np.abs(VMECout["jcurv"].values)
    VMEC_gamma = VMECout["gamma"].values
    VMEC_nfp = int(VMECout["nfp"].values)
    VMEC_mpol = int(VMECout["mpol"].values)
    VMEC_ntor = int(VMECout["ntor"].values)
    VMEC_pressure = VMECout["presf"].values
    VMEC_im = VMECout["xm"].values
    VMEC_in = VMECout["xn"].values / VMEC_nfp
    VMEC_rmnc = VMECout["rmnc"].values
    VMEC_zmns = VMECout["zmns"].values
    if changePoloidalAngle == True:
        VMEC_in *= -1
        VMEC_zmns *= -1
        for i in range(len(VMEC_im)):
            if VMEC_im[i] == 0:
                VMEC_in[i] *= -1
                VMEC_zmns[:, i] *= -1


    nvol = len(interfaceLabel) - 1
    # interfacePsi = np.interp(interfaceLabel, flux_label, VMEC_psi)
    datas = {}
    datas["phiedge"] = VMEC_tflux[-1]
    datas["curtor"] = mu0 * 2 * np.pi * integrate(
        flux_label, VMEC_jtor * VMEC_g, flux_label[0], flux_label[-1]
    )
    datas["curpol"] = mu0 * 2 * np.pi * integrate(
        flux_label, VMEC_jpol * VMEC_g, flux_label[0], flux_label[-1]
    )
    datas["gamma"] = VMEC_gamma
    datas["nfp"] = VMEC_nfp
    datas["nvol"] = nvol
    datas["mpol"] = VMEC_mpol
    datas["ntor"] = VMEC_ntor
    datas["lrad"] = [8 for i in range(nvol)]
    datas["lconstraint"] = int(lconstraint)
    datas["tflux"] = np.interp(interfaceLabel[1:], flux_label,
                               VMEC_tflux) / VMEC_tflux[-1]
    datas["pflux"] = np.interp(interfaceLabel[1:], flux_label,
                               VMEC_pflux) / VMEC_tflux[-1]
    # datas["helicity"] = [(4 * math.pi * math.pi * integrate(
    #     VMEC_psi, VMEC_iota * VMEC_psi -
    #     VMEC_chi, interfacePsi[i], interfacePsi[i + 1]) 
    # + np.interp(interfaceLabel[i+1], flux_label, VMEC_tpflux2)
    # - np.interp(interfaceLabel[i], flux_label, VMEC_tpflux2)
    # ) for i in range(nvol)]
    if fluxLabel == "toroidal":
        datas["helicity"] = [(
            4 * math.pi * math.pi * integrate(
                flux_label, VMEC_iota * VMEC_psi - VMEC_chi, interfaceLabel[i], interfaceLabel[i+1]
            ) + np.interp(interfaceLabel[i+1], flux_label, VMEC_tpflux2)
            - np.interp(interfaceLabel[i], flux_label, VMEC_tpflux2)
        ) for i in range(nvol)]
    elif fluxLabel == "poloidal":
        datas["helicity"] = [(
            4 * math.pi * math.pi * integrate(
                flux_label, VMEC_psi - np.divide(VMEC_chi, VMEC_iota), interfaceLabel[i], interfaceLabel[i+1]
            ) + np.interp(interfaceLabel[i+1], flux_label, VMEC_tpflux2)
            - np.interp(interfaceLabel[i], flux_label, VMEC_tpflux2)
        ) for i in range(nvol)]
    datas["pressure"] = [((integrate(
        flux_label, VMEC_pressure * VMEC_g, interfaceLabel[i], interfaceLabel[i + 1])
        / integrate(flux_label, VMEC_g, interfaceLabel[i], interfaceLabel[i + 1])
    )) for i in range(nvol)]
    datas["adiabatic"] = [(
        datas["pressure"][i] * math.pow(
        integrate(flux_label, VMEC_g, interfaceLabel[i], interfaceLabel[i+1]), VMEC_gamma)
    ) for i in range(nvol)]
    datas["ivolume"] = [(mu0 * 2 * math.pi * integrate(
        flux_label, VMEC_g * VMEC_jtor, interfaceLabel[i], interfaceLabel[i + 1]))for i in range(nvol)]
    # if fluxLabel == "toroidal":
    #     datas["mu"] = [(datas["ivolume"][i] / datas["tflux"][i])
    #                for i in range(nvol)]
    # elif fluxLabel == "poloidal":
    #     datas["mu"] = [([(mu0 * 2 * math.pi * integrate(
    #                 flux_label, VMEC_g * VMEC_jpol, interfaceLabel[i], interfaceLabel[i + 1]))for i in range(nvol)][i] 
    #                 / datas["pflux"][i]) for i in range(nvol)]
    mu = VMECout["jdotb"].values*mu0/VMECout["bdotb"].values
    datas["mu"] = [(
        integrate(flux_label, mu*VMEC_g, interfaceLabel[i], interfaceLabel[i+1]) / 
        integrate(flux_label, VMEC_g, interfaceLabel[i], interfaceLabel[i+1])
    ) for i in range(nvol)]
    datas["isurf"] = [0 for i in range(nvol)]
    datas["iota"] = np.interp(interfaceLabel, flux_label, VMEC_iota)
    datas["rac"] = VMEC_rmnc[0, :]
    datas["zas"] = VMEC_zmns[0, :]
    datas["im"] = VMEC_im
    datas["in"] = VMEC_in
    datas["rbc"] = VMEC_rmnc[-1, :]
    datas["zbs"] = VMEC_zmns[-1, :]
    datas["interface_rc"] = np.zeros([len(interfaceLabel), len(VMEC_im)])
    datas["interface_zs"] = np.zeros([len(interfaceLabel), len(VMEC_im)])
    for j in range(len(VMEC_im)):
        datas["interface_rc"][:, j] = np.interp(
            interfaceLabel, flux_label, VMEC_rmnc[:, j]
        )
        datas["interface_zs"][:, j] = np.interp(
            interfaceLabel, flux_label, VMEC_zmns[:, j]
        )

    for key in kwargs.keys():
        datas[key] = kwargs[key]
        
    writeSPECInput(SPEC_input, datas, interfaceShape)

    return


if __name__ == "__main__":
    pass
