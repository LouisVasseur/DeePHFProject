# AI Use: none, hyperparameters chosen by hand and just calls APIs 

"""SOAP descriptor via DScribe."""

import numpy as np

# helper to build a SOAP object with our chosen hyperparameters and species set.
def build_soap(species):
    from dscribe.descriptors import SOAP
    return SOAP(species=sorted(set(species)),
                r_cut=6.0, n_max=8, l_max=6, sigma=0.3,
                periodic=False, compression={"mode": "crossover"})

# helper to compute SOAP descriptors for a molecule given (Z, R) using a SOAP object.
def compute_for_mol(soap, Z, R):
    from ase import Atoms
    return soap.create(Atoms(numbers=Z, positions=R)).astype(np.float32)
