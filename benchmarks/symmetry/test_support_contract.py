"""Adversarial controls for the independent stored-field benchmark auditor."""
from copy import deepcopy
import unittest
import numpy as np
from support_contract import quality, pair_quality, fields, common_guarded_prefix


def encode(a):
    return np.asarray(a,dtype=np.float64).view(np.uint64).tolist()


def encode_fields(u):
    return encode(np.stack((u.real,u.imag),axis=-1))


def uniform():
    n=5; q=6; tau=2*np.pi
    modes=[[0,0],[1,0],[-1,0],[0,1],[0,-1],[1,1]]
    y,x=np.mgrid[:n,:n]
    u=np.asarray([np.exp(1j*tau*(gx*x+gy*y)/n).ravel()/(n*2) for gx,gy in modes])
    vals=[tau*tau*(gx*gx+gy*gy)/4 for gx,gy in modes]
    raw=encode([1,1])+[0]+encode([0,0,tau,0,0,tau]+[4.]*(n*n)+[.25]*(n*n))+[0]
    identity=dict(nx=n,ny=n,storage_bits=64,centered=False,operator_bits=raw)
    space=dict(grid=[n,n],k_bits=encode([0,0]),modes=modes)
    report=dict(accepted=True,solver_converged=True,rejection=None,method="lobpcg",iterations=1,
                coefficient_count=len(modes),analytic_kernel=True,
                projected=dict(residuals=[0.]*q,absolute_residuals=[0.]*q,b_orthogonality_defect=0.))
    result=dict(spectrum=vals,field_bits=encode_fields(u),field_count=q,field_payload_bytes=q*n*n*16,
                support_identity=space,report=report)
    return dict(case="synthetic",grid=[n,n],bands=q,polarization="TM",radius=2,precision="f64",eigensolver={},
                mode="unreduced",operator_identity=identity,support_identity=space,result=result,sectors=None,error=None)


class StoredFieldContracts(unittest.TestCase):
    def test_plane_waves_pass_independent_certificate(self):
        self.assertEqual(quality(uniform())["issues"],[])

    def test_forged_success_and_certificate_cannot_hide_wrong_eigenvalue(self):
        record=uniform();record["result"]["spectrum"][1]*=.9
        self.assertTrue(any("independent FFT fails projected residual" in s for s in quality(record)["issues"]))

    def test_missing_field_is_rejected(self):
        record=uniform();record["result"]["field_bits"].pop()
        self.assertTrue(quality(record)["issues"])

    def test_outside_support_vector_is_rejected(self):
        record=uniform();u=fields(record["result"])
        y,x=np.mgrid[:5,:5];u[1]=np.exp(1j*4*np.pi*x/5).ravel()/10
        record["result"]["field_bits"]=encode_fields(u)
        self.assertTrue(quality(record)["issues"])

    def test_changed_material_identity_cannot_be_paired(self):
        a=uniform();b=deepcopy(a);b["operator_identity"]["operator_bits"][9]=encode([4.1])[0]
        self.assertIn("paired physical pencil/configuration differs",pair_quality(a,b)["issues"])

    def test_complete_degenerate_rotation_preserves_the_subspace(self):
        a=uniform();b=deepcopy(a);u=fields(b["result"])
        first,second=u[1].copy(),u[2].copy()
        u[1]=(first+second)/np.sqrt(2);u[2]=(first-second)/np.sqrt(2)
        b["result"]["field_bits"]=encode_fields(u)
        self.assertEqual(quality(b)["issues"],[])
        self.assertEqual(pair_quality(a,b)["issues"],[])
        self.assertEqual(pair_quality(a,b)["comparison"]["guarded_prefix"],5)

    def test_guard_checks_both_spectra(self):
        self.assertEqual(common_guarded_prefix([0,1,2,2.1],[0,1,2,2],3),2)
        self.assertEqual(common_guarded_prefix([0,1,2,2],[0,1,2,2.1],3),2)

    def test_normalization_cannot_be_forged_in_report(self):
        record=uniform();record["result"]["field_bits"]=encode_fields(fields(record["result"])*2)
        self.assertTrue(any("independent FFT fails B orthogonality" in s for s in quality(record)["issues"]))


if __name__=="__main__":
    unittest.main()
