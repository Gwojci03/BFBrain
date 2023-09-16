"""This module contains a number of SymPy functions for the quartic parts of 
scalar potentials which the program can parse for analysis. Any other potential 
functions you may want to use can be implemented by writing them in the same 
format as these.
"""
import sympy as sym
import numpy as np
from sympy import I
from sympy.physics.quantum.dagger import Dagger
from sympy.matrices.expressions import Trace

def _sim_Trace(M):
    """A function which simplifies the trace of some matrix."""
    return sym.simplify(Trace(M))

def _sim_expand(x):
    """A function which expands and simplifies an expression. Useful for eliminating 
    complex numbers from the potential, so NumPy can work with only real numbers.
    """
    return sym.simplify(sym.expand(x))

#The square root of 2-- it comes up a lot.
_sqrt2 = sym.sqrt(2)



def V_Precustodial(phi, lam):
    """The "precustodial" generalization of the Georgi-Machacheck model given in 
    arXiv:hep-ph/2012.13947. The quartic coefficients are parameterized in the 
    same manner as Eq.(3.4) of that work."""
    H = sym.Matrix([0, phi[0]])
    A = sym.Matrix([[(phi[1] + I*phi[4])/_sqrt2, -phi[2] - I*phi[5]],[phi[3] + I*phi[6], -(phi[1] + I*phi[4])/_sqrt2]])
    B = sym.Matrix([[phi[7], -phi[8] - I*phi[9]], [-phi[8] + I*phi[9], -phi[7]]])/_sqrt2
    is2 = sym.Matrix([[0,1],[-1,0]])

    TrHsq = (Dagger(H).dot(H)).simplify()
    TrAsq = Trace(A*Dagger(A))
    TrBsq = Trace(B*Dagger(B))
    QVec = sym.Matrix([(TrHsq**2)/4, (TrAsq**2)/4, _sim_Trace(A*Dagger(A)*A*Dagger(A))/4,
                                    (TrBsq**2)/24, TrHsq*TrAsq, Dagger(H)*A*Dagger(A)*H,
                                    TrHsq*TrBsq/2, TrAsq*TrBsq/2, _sim_expand(_sim_Trace(A*B)*_sim_Trace(Dagger(A)*B))/2,
                                    ((H.T)*is2*Dagger(A)*B*H-Dagger(H)*B*A*is2*(sym.conjugate(H)))/2
                                    ]).applyfunc(sym.simplify)
    return QVec.dot(lam)

def V_GM(phi, lam):
    """The Georgi-Machacheck potential, following the conventions of Eq.(5) of arXiv:hep-ph/1404.2640
    """
    Phi = sym.Matrix([[phi[0], 0],[0, phi[0]]])/_sqrt2
    X = sym.Matrix([[phi[1] - I*phi[2], phi[3] + I*phi[4], phi[5] + I*phi[6]],[-phi[7] + I*phi[8], _sqrt2*phi[9], phi[7] + I*phi[8]],[phi[5] - I*phi[6], -phi[3] + I*phi[4], phi[1] + I*phi[2]]])/_sqrt2
    TrPhisq = _sim_Trace(Dagger(Phi).multiply(Phi))
    TrXsq = _sim_Trace(Dagger(X).multiply(X))
    tau = [sym.Matrix([[0,1/2],[1/2,0]]), sym.Matrix([[0,-I/2],[I/2,0]]), sym.Matrix([[1/2,0],[0,-1/2]])]
    # tau = sym.Array([[[0,1],[1,0]], [[0,-I],[I,0]], [[1,0],[0,-1]]])
    t = [sym.Matrix([[0,1,0], [1,0,1], [0,1,0]])/_sqrt2, sym.Matrix([[0,-I,0],[I,0,-I],[0,I,0]])/_sqrt2, sym.Matrix([[1,0,0],[0,0,0],[0,0,-1]])]
    lam5_term = 0
    for i in range(3):
        for j in range(3):
            lam5_term += _sim_Trace(Dagger(Phi)*tau[i]*Phi*tau[j])*_sim_Trace(Dagger(X)*t[i]*X*t[j])
    QVec = sym.Matrix([TrPhisq**2, TrPhisq*TrXsq, _sim_expand(_sim_Trace(Dagger(X)*X*Dagger(X)*X)),
                       TrXsq**2, - _sim_expand(lam5_term)]).applyfunc(sym.simplify)
    
    return QVec.dot(lam)

def V_2HDM(phi, lam):
    """The most general 2-Higgs doublet model, following the conventions of Eq.(1) of hep-ph/0609018
    """
    Phi1 = sym.Matrix([0, phi[0]])
    Phi2 = sym.Matrix([phi[1] + I*phi[3], phi[2] + I*phi[4]])
    phi1sq = Dagger(Phi1).dot(Phi1)
    phi2sq = sym.simplify(Dagger(Phi2).dot(Phi2))
    phi12 = sym.simplify(Dagger(Phi1).dot(Phi2))
    phi21 = sym.simplify(Dagger(Phi2).dot(Phi1))

    QVec = (sym.Matrix([(phi1sq**2)/2, (phi2sq**2)/2, phi1sq*phi2sq, phi12*phi21, 
                       (phi12**2 + phi21**2)/2, I*(phi12**2 - phi21**2)/2,
                       phi1sq*(phi12 + phi21), I*phi1sq*(phi12 - phi21), phi2sq*(phi12 + phi21), I*phi2sq*(phi12-phi21)])).applyfunc(sym.simplify)
    return QVec.dot(lam)

def V_3HDM(phi, lam):
    """A 3HDM with Z2xZ2 symmetry, for which no BFB conditions are known.
    """
    phi1 = sym.Matrix([0, phi[0]])
    phi2 = sym.Matrix([phi[1] + I*phi[2], phi[3] + I*phi[4]])
    phi3 = sym.Matrix([phi[5] + I*phi[6], phi[7] + I*phi[8]])
    phi1sq = _sim_Trace(Dagger(phi1).multiply(phi1))
    phi2sq = _sim_Trace(Dagger(phi2).multiply(phi2))
    phi3sq = _sim_Trace(Dagger(phi3).multiply(phi3))
    phi12 = _sim_Trace(Dagger(phi1).multiply(phi2))
    phi21 = _sim_Trace(Dagger(phi2).multiply(phi1))
    phi13 = _sim_Trace(Dagger(phi1).multiply(phi3))
    phi31 = _sim_Trace(Dagger(phi3).multiply(phi1))
    phi23 = _sim_Trace(Dagger(phi2).multiply(phi3))
    phi32 = _sim_Trace(Dagger(phi3).multiply(phi2))

    QVec = sym.Matrix([phi1sq**2, phi2sq**2, phi3sq**2, phi1sq*phi2sq, phi1sq*phi3sq, phi2sq*phi3sq, phi12*phi21, phi13*phi31, phi23*phi32, (phi12**2 + phi21**2)/2, I*(phi12**2 - phi21**2)/2, 
                       (phi13**2 + phi31**2)/2, I*(phi13**2 - phi31**2)/2, (phi23**2 + phi32**2)/2, I*(phi23**2-phi32**2)/2]).applyfunc(_sim_expand).applyfunc(sym.simplify)
    return QVec.dot(lam)

def V_3HDM_U1(phi, lam):
    """A 3HDM with U1xU1 symmetry.
    """
    phi1 = sym.Matrix([0, phi[0]])
    phi2 = sym.Matrix([phi[1] + I*phi[2], phi[3] + I*phi[4]])
    phi3 = sym.Matrix([phi[5] + I*phi[6], phi[7] + I*phi[8]])
    phi1sq = _sim_Trace(Dagger(phi1).multiply(phi1))
    phi2sq = _sim_Trace(Dagger(phi2).multiply(phi2))
    phi3sq = _sim_Trace(Dagger(phi3).multiply(phi3))
    phi12 = _sim_Trace(Dagger(phi1).multiply(phi2))
    phi21 = _sim_Trace(Dagger(phi2).multiply(phi1))
    phi13 = _sim_Trace(Dagger(phi1).multiply(phi3))
    phi31 = _sim_Trace(Dagger(phi3).multiply(phi1))
    phi23 = _sim_Trace(Dagger(phi2).multiply(phi3))
    phi32 = _sim_Trace(Dagger(phi3).multiply(phi2))

    QVec = sym.Matrix([phi1sq**2, phi2sq**2, phi3sq**2, phi1sq*phi2sq, phi1sq*phi3sq, phi2sq*phi3sq, phi12*phi21, phi13*phi31, phi23*phi32]).applyfunc(_sim_expand).applyfunc(sym.simplify)
    return QVec.dot(lam)