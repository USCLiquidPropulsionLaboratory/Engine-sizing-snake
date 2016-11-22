##1D Fluid Dynamics originally written for Laughing Candle, and expanded
# for use with ASTE572.
#@author Alexander Adams

import scipy.optimize as opt
from numpy import sqrt, log

#when solving numerically, it is assumed that all mach numbers are less than this
MAX_M = 100
MIN_M = 1e-6

#thermo properties
def getT0(T, gamma, M):
    return T*(1 + (gamma-1)*M**2/2)

def getP0(P, gamma, M):
    return P*(1 + (gamma-1)*M**2/2)**(gamma/(gamma-1))

def getcstar(gamma, Rgas, T0):
    return sqrt(((gamma+1)/2)**((gamma+1)/(gamma-1))*Rgas*T0/gamma)

#ideal nozzle
def getIdealCf(gamma, Pe, P0, Pa, Ae, Astar):
    return sqrt(2*gamma**2/(gamma-1)* \
        (2/(gamma+1))**((gamma+1)/(gamma-1))* \
        (1-(Pe/P0)**((gamma-1)/gamma))) + (Pe-Pa)/P0*Ae/Astar

#isentropic flow functions
def isentropicT(T0, gamma, M):
    return T0/(1 + (gamma-1)*M**2/2)

def isentropicP(P0, gamma, M):
    return P0/(1 + (gamma-1)*M**2/2)**(gamma/(gamma-1))

def isentropicRho(rho0, gamma, M):
    return rho0/(1 + (gamma-1)*M**2/2)**(1/(gamma-1))

def isentropicmdot(A, P0, T0, Rgas, gamma, M):
    return M*P0*A*sqrt(gamma/Rgas/T0)/ \
        (1 + (gamma-1)*M**2/2)**((gamma+1)/2/(gamma-1))

def isentropicA(Astar, gamma, M):
    return (Astar/M)*(2/(gamma+1)* \
        (1 + (gamma-1)*M**2/2))**((gamma+1)/2/(gamma-1))
        
def getIsentropicMs(Astar, A, gamma):
    f = lambda M: isentropicA(Astar, gamma, M) - A
    subsonicM = opt.brentq(f, MIN_M, 1)
    supersonicM = opt.brentq(f, 1, MAX_M)
    return subsonicM, supersonicM
    
def getIsentropicMFromP(P0, P, gamma):
    return sqrt(2*((P0/P)**((gamma-1)/gamma) - 1)/(gamma-1))

#Fanno flow functions
def fannoLstar(Cf, D, M, gamma):
    return (D/4/Cf)*((1-M**2)/gamma/M**2 + \
        (gamma+1)*log((gamma+1)*M**2/2/(1+(gamma-1)*M**2/2))/2/gamma)
#returns subsonic M
def fannoMafter(Cf, D, Min, gamma, L):
    Lstar1 = fannoLstar(Cf, D, Min, gamma)
    Lstar2 = Lstar1 - L
    f = lambda M: Lstar2 - fannoLstar(Cf, D, M, gamma)
    return opt.brentq(f, Min, 1)

def fannoPstar(P, M, gamma):
    return P*M/sqrt((gamma+1)/(2+(gamma-1)*M**2))
    
def fannoP0star(P0, M, gamma):
    return P0*M/((2/(gamma+1))*(1+(gamma-1)*M**2/2)) \
        **((gamma+1)/2/(gamma-1))

def fannoP0(P0star, M, gamma):
    return P0star* \
        ((2/(gamma+1))*(1+(gamma-1)*M**2/2))**((gamma+1)/2/(gamma-1))/M

#Rayleigh Flow
def rayleighP0toP01(M, M1, gamma):
    return ((1+gamma*M1**2)/(1+gamma*M**2))* \
        ((1+(gamma-1)*M**2/2)/(1+(gamma-1)*M1**2/2))

def rayleighT0toT01(M, M1, gamma):
    return ((1+gamma*M1**2)/(1+gamma*M**2))**2*(M/M1)**2 \
        *((1+(gamma-1)*M**2/2)/(1+(gamma-1)*M1**2/2))

## Returns the subsonic machnumber for a given ratio of T0s
# @return subsonic M
def rayleighM(T0, T01, M, gamma):
    f = lambda M1: T0/T01 - rayleighT0toT01(M, M1, gamma)
    return opt.brentq(f, MIN_M, 1)

##Calculates conditions after a normal shock given conditions before the shock
#@return P2, M2, T2
def normalShockP(P1, M1, T1, gamma):
    M2 = sqrt((M1**2 + 2/(gamma-1))/(2*M1**2*gamma/(gamma-1) - 1))
    P2 = P1*(1+gamma*M1**2)/(1+gamma*M2**2)
    T2 = T1*(M2*P2/M1/P1)**2
    return P2, M2, T2
    