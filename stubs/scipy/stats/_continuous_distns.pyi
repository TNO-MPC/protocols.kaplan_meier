from ._distn_infrastructure import rv_continuous

class chi2_gen(rv_continuous): ...

chi2 = chi2_gen(a=0.0, name="chi2")
