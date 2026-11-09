from pathlib import Path
import numpy as np

from tfpy.vmec import VMECOut
from tfpy.spec import SPECOut

ROOT = Path(__file__).parents[1]
VMEC_FILE = ROOT / 'example/datas/VMECDatas/wout_QAS.nc'
SPEC_FILE = ROOT / 'example/datas/SPECDatas/LandremanPaul2021_QA.sp.h5'


def test_vmec_example_loads_and_builds_surface():
    output = VMECOut(str(VMEC_FILE))
    assert output.nfp == 3
    assert output.ns == 128
    surface, lam = output.getSurface(surfaceIndex=64)
    assert (surface.mpol, surface.ntor) == (10, 6)
    assert lam.arrlen == surface.r.arrlen
    values = surface.r.getValue(np.array([0.0, 0.4]), np.array([0.0, 0.2]))
    assert np.all(np.isfinite(values))
    jacobian = output.getJacobian(surfaceIndex=64)
    assert np.all(np.isfinite(jacobian.getValue(np.array([0.1]), np.array([0.2]))))


def test_spec_example_loads():
    output = SPECOut(str(SPEC_FILE))
    assert output is not None
    assert len(vars(output)) > 0
