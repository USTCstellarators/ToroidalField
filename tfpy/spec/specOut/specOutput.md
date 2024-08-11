specOutput的结构
===========================================================================
```
specOutput.h5: 
+-- /
|   +-- group: grid
|       +-- dataset: BR
|       +-- dataset: BZ
|       +-- dataset: Bp
|       +-- dataset: Nt
|       +-- dataset: Ntz
|       +-- dataset: Rij
|       +-- dataset: Zij
|       +-- dataset: pi2nfp
|       +-- dataset: sg
|   +-- group: input
|       +-- group: physics
|       +-- group: numerics
|       +-- group: global
|       +-- group: local
|       +-- group: diagnostics
|   +-- dataset: iterations
|   +-- group: output
|       +-- dataset: Bnc
|       +-- dataset: Bns
|       +-- dataset: Btemn
|       +-- dataset: Btomn
|       +-- dataset: Bzemn
|       +-- dataset: Bzomn
|       +-- dataset: ForceErr
|       +-- dataset: IPDt
|       +-- dataset: Ivolume
|       +-- dataset: Mrad
|       +-- dataset: Mvol
|       +-- dataset: Rbc
|       +-- dataset: Rbs
|       +-- dataset: TT
|       +-- dataset: Vnc
|       +-- dataset: Vns
|       +-- dataset: Zbc
|       +-- dataset: Zbs
|       +-- dataset: adiabatic
|       +-- dataset: beltramierror
|       +-- dataset: helicity
|       +-- dataset: im
|       +-- dataset: ims
|       +-- dataset: in
|       +-- dataset: ins
|       +-- dataset: lambdamn
|       +-- dataset: lmns
|       +-- dataset: mn
|       +-- dataset: mns
|       +-- dataset: mu
|       +-- dataset: pflux
|       +-- dataset: tflux
|       +-- dataset: volume
|   +-- group poincare
|       +-- dataset: R
|       +-- dataset: Z
|       +-- dataset: s
|       +-- dataset: success
|       +-- dataset: t
|   +-- group: transform
|       +-- dataset: diotadxup
|       +-- dataset: fiota
|   +-- group: vector_potential
|       +-- dataset: Ate
|       +-- dataset: Ato
|       +-- dataset: Aze
|       +-- dataset: Azo
|   +-- dataset: version
```



grid
----------------------------------------------------------------------------------------
`BR.shape = BZ.shape = Bp.shape = Rij.shape = Zij.shape`


input
----------------------------------------------------------------------------------------


iterations
----------------------------------------------------------------------------------------


output
----------------------------------------------------------------------------------------


poincare
----------------------------------------------------------------------------------------


transform
----------------------------------------------------------------------------------------


vector_potential
----------------------------------------------------------------------------------------


version
----------------------------------------------------------------------------------------