## Liquid rocket engine component library
#@ Authors Juha Nieminen & Alexander Adams

import Flows1D
from physical_constants import Runiv, psi, lbm, atm
from numpy import sqrt, pi, exp, log, log10
from scipy import interpolate
import scipy.optimize as opt
import numpy as np
import Moody_diagram
from scipy.integrate import odeint
import CoolProp.CoolProp as CP


class MagnumOxPintle:

    # References to Fluid Mechanics by Frank M. White, 7th edition
    
    def __init__(self, d_in, d_mani, d_ori, OD_ann, ID_ann, L_shaft, Nori, cd_ori, rou):
        
        self.d_in   = d_in                      # injector inlet diameter
        self.d_mani = d_mani                    # circulation chamber diameter upstream from orifice plate
        self.OD_ann = OD_ann                    # converging section end/annulus outer diameter, m
        self.ID_ann = ID_ann                    # central shaft outer diameter, m
        self.L_shaft= L_shaft                   # Length of annular flow path, m
        self.Nori   = Nori                      # number of orifices in orifice plate
        self.d_ori  = d_ori                     # single orifice diameter, m
        self.cd_ori = cd_ori                    # orifice discharge coefficient in orifice plate, dimensionless
        self.Dh     = self.OD_ann - self.ID_ann # annulus hydraulic diameter, m
        self.rou    = rou                       # annulus _DIMENSIONLESS_ roughness
    
    def getVelocities(self, mdot, rho, mu):
    
        v1      = 4*mdot/(rho*pi*self.d_in**2)                      # velocity in feed lines, m/s
        v2      = 4*mdot/(rho*pi*(self.d_mani**2-self.ID_ann**2))   # velocity in circulation chamber upstream of orifices,m/s
        v_ori   = 4*mdot/(rho*pi*self.d_ori**2*self.Nori)           # velocity in orifices, m/s
        v3      = v2                                                # velocity downstream of orifices, m/s
        v4      = 4*mdot/(rho*pi*(self.OD_ann**2-self.ID_ann**2))   # velocity in annulus around the shaft, m/s
       
        return (v1, v2, v_ori, v3, v4)
        
    def getPressureDrops(self, mdot, rho, mu, Pin): #Pin needed only for diagnostic purposes
    
        (v1, v2, v_ori, v3, v4) = self.getVelocities(mdot, rho, mu)
        
        A_ori   = pi*self.d_ori**2/4                            # single orifice cross sectional area, m
        
        Re      = rho*v4*self.Dh/mu                             # Reynold's number in annulus, dimensionless
        f       = Moody_diagram.getAnnularF(Re, self.OD_ann, self.ID_ann, mu, v3, self.rou)
        alfa_t  = 1                                             # turbulent correction factor, see White p.191
        
        dp1     = 0.5*rho*alfa_t*(v2**2 - v1**2)                # velocity loss going from feedlines to circ. chamber, Pa
        dp2     = (mdot/self.Nori)**2/(2*rho*A_ori**2*self.cd_ori**2) \
                    - 0.5*rho*alfa_t*(v_ori**2 - v3**2)         # unrecoverable pressure drop in orifices, Pa
        dp3     = 0.5*rho*alfa_t*(v4**2 - v3**2)                # velocity loss in converging section, Pa
        dp4     = 0.5*rho*v4**2*f*self.L_shaft/self.Dh          # friction losses in the annulus, Pa
        
        '''
        print("")
        print("P_inlet is", '%.1f'%(Pin/psi), "psi")
        print("dp1 (feedline to manifold) is", '%.1f'%(dp1/psi), "psi")
        print("P_ox_manifold is", '%.1f'%((Pin-dp1)/psi), "psi")
        print("dp2 (orifice loss) is", '%.1f'%(dp2/psi), "psi")
        print("P_converging_in is", '%.1f'%((Pin-dp1-dp2)/psi), "psi")
        print("dp3 (velocity losses in converging section) is", '%.1f'%(dp3/psi), "psi")
        print("P_annulus_in is", '%.1f'%((Pin-dp1-dp2-dp3)/psi), "psi")
        print("dp4 (friction losses in annulus) is", '%.1f'%(dp4/psi), "psi")
        print("P_chamber_ox is", '%.1f'%((Pin-dp1-dp2-dp3-dp4)/psi), "psi")
        print("")
        print("v_feedline_ox is", '%.3f'%(v1), "m/s")
        print("v2_ox is", '%.3f'%(v2), "m/s")
        print("v_ori is", '%.3f'%(v_ori), "m/s")
        print("v3_ox is", '%.3f'%(v3), "m/s")
        print("v4_annulus_ox is", '%.3f'%(v4), "m/s")
        print("")
        '''
        return (dp1, dp2, dp3, dp4, dp1 + dp2 + dp3 + dp4)
        
    def getPressures(self, mdot, rho, mu, Pin):
        
        (dp1, dp2, dp3, dp4, dpTot)   = self.getPressureDrops(mdot, rho, mu, Pin)
        P1          = Pin - dp1                         # Ox manifold
        P2          = P1 - dp2                          # Converging section in (after orifices)
        P3          = P2 - dp3                          # Annulus inlet
        P4          = P3 - dp4                          # Annulus outlet
        
        return (Pin, P1, P2, P3, P4)

        
class MagnumFuelPintle:
    
    # References to Fluid Mechanics by Frank M. White, 7th edition
    
    def __init__(self, d_in, d2, ID_shaft, OD_shaft, L_shaft, r_tip, h_exit, rou):
        
        self.d_in           = d_in                                      # injector inlet diameter (two of them!)
        self.d2             = d2                                        # manifold/converging section begin diameter, m
        self.ID_shaft       = ID_shaft                                  # fuel annulus ID, m
        self.OD_shaft       = OD_shaft                                  # fuel annulus OD, m
        self.L_shaft        = L_shaft                                   # Length of annular flow path, m
        self.r_tip          = r_tip                                     # pintle tip radius, m
        self.h_exit         = h_exit                                    # pintle exit slot height, m
        self.Dh             = self.OD_shaft - self.ID_shaft             # annulus hydraulic diameter, m
        self.rou            = rou                                       # annulus _DIMENSIONLESS_ roughness]
        
        # Interpolation of k_bend from COMSOL data (x-coord= OD, y-coord= mdot). r_inside = 1.5mm
        # (extrapolation = assumes closest valid data point)
        
        ODmin,ODmax         = 14e-3, 17e-3                              # Fuel annulus OD,m
        ODstep              = 1e-3                                      # m
        mdot_min, mdot_max  = 0.5, 2.5                                  # kg/s 
        mdot_step           = 0.5                                       # kg/s

        OD_vector           = np.linspace(ODmin,ODmax,(ODmax-ODmin)/ODstep+1)
        mdot_vector         = np.linspace(mdot_min,mdot_max,(mdot_max-mdot_min)/mdot_step+1)
        
        kbend_grid = np.array([[1.2838, 1.3038, 1.2904, 1.28],
        [ 1.0860, 1.1604, 1.1840, 1.27],
        [ 0.9514, 1.0375, 1.0902, 1.15],
        [ 0.8860, 0.9648, 1.0150, 1.10],
        [ 0.7860, 0.9204, 0.9567, 1.01]])
        
        
        self.kbend_int      = interpolate.RectBivariateSpline(mdot_vector, OD_vector, kbend_grid)     # Note the order of X&Y!
     
    def get_kbend(self, mdot, OD):
        
        return self.kbend_int.ev(OD, mdot)
       
    def getVelocities(self, mdot, rho, mu):
    
        v1      = 4*mdot/(rho*pi*self.d_in**2)/2                        # velocity in feed lines (two of them!), m/s
        v2      = 4*mdot/(rho*pi*(self.d2**2-self.ID_shaft**2))         # velocity in manifold, m/s
        v3      = 4*mdot/(rho*pi*(self.OD_shaft**2-self.ID_shaft**2))   # velocity in shaft, m/s
        v4      = mdot/(rho*2*pi*self.r_tip*self.h_exit)                # exit velocity, m/s
       
        return (v1, v2, v3, v4)
        
    def getPressureDrops(self, mdot, rho, mu, Pin): #Pin needed only for diagnostic purposes
        
        (v1, v2, v3, v4) = self.getVelocities(mdot, rho, mu)            # velocities at different stations, m/s
        
        Re      = rho*v4*self.Dh/mu                                     # Reynold's number in annulus, dimensionless
        f       = Moody_diagram.getAnnularF(Re, self.OD_shaft, self.ID_shaft, mu, v3, self.rou)
        alfa_t  = 1                                                     # turbulent correction factor, see White p.191
        k_bend  = self.get_kbend(self.OD_shaft, mdot)                   # bend drop coefficient, dimensionless
        
        dp1     = 0.5*rho*alfa_t*(v2**2 - v1**2)                        # velocity loss going from feedlines to manifold, Pa
        dp2     = 0.5*rho*alfa_t*(v3**2 - v2**2)                        # velocity loss going from manifold to annulus, Pa
        dp3     = 0.5*rho*v3**2*f*self.L_shaft/self.Dh                  # friction losses in the annulus, Pa
        dp4     = 0.5*rho*v3**2*k_bend                                  # unrecoverable losses in the bend, Pa
        dp5     = 0.5*rho*alfa_t*(v4**2 - v3**2)                        # velocity loss in the bend, Pa
        
        '''
        print("P_inlet is", '%.1f'%(Pin/psi), "psi")
        print("dp1 (feedline to manifold) is", '%.1f'%(dp1/psi), "psi")
        print("P_manifold is", '%.1f'%((Pin-dp1)/psi), "psi")
        print("dp2 (from manifold to annulus) is", '%.1f'%(dp2/psi), "psi")
        print("P_annulus_in is", '%.1f'%((Pin-dp1-dp2)/psi), "psi")
        print("dp3 (friction losses in annulus) is", '%.1f'%(dp3/psi), "psi")
        print("P_annulus_out is", '%.1f'%((Pin-dp1-dp2-dp3)/psi), "psi")
        print("dp4 (bend losses) is", '%.1f'%(dp4/psi), "psi")
        print("dp5 (velocity losses in the bend) is", '%.1f'%(dp5/psi), "psi")
        print("P_chamber_fuel is", '%.1f'%((Pin-dp1-dp2-dp3-dp4-dp5)/psi), "psi")
        print("")
        print("v1_fuel is", '%.1f'%(v1), "m/s")
        print("v2_fuel is", '%.1f'%(v2), "m/s")
        print("v3_fuel is", '%.1f'%(v3), "m/s")
        print("v4_fuel is", '%.1f'%(v3), "m/s")
        print("v_exit_fuel is", '%.1f'%(v4), "m/s")
        print("")
        print("")
        '''
        
        return (dp1, dp2, dp3, dp4, dp5, dp1 + dp2 + dp3 + dp4 + dp5)
        
    def getPressures(self, mdot, rho, mu, Pin):
        
        (dp1, dp2, dp3, dp4, dp5, dpTot)   = self.getPressureDrops(mdot, rho, mu, Pin)
        P1          = Pin - dp1                         # Fuel manifold
        P2          = P1 - dp2                          # Annulus inlet
        P3          = P2 - dp3                          # Annulus outlet
        P4          = P3 - dp4 -dp5                     # Bend exit
        
        return (Pin, P1, P2, P3, P4)
        
        
class GasOrifice:
    def __init__(self, A, Cd, gamma, Rgas):
        self.A      = A
        self.Cd     = Cd
        self.gamma  = gamma
        self.Rgas   = Rgas
        
    def getMdot(self, Pin, Pexit, Tin):
        if Pin < Pexit:
            print("Pin is", Pin/psi, "psi")
            print("Pexit is", Pexit/psi, "psi")
            print("Warning: Pin < Pexit in gas injector. Try a smaller time step or a larger throat so that chamber doesn't overfill")
            
            return 0.
        if self.isChoked(Pin, Pexit):
            M       = 1     
            #print("injector flow is choked")
        else:
            M       = Flows1D.getIsentropicMFromP(Pin, Pexit, self.gamma)
            #print("injector M is", M)
            #print("gas injector is not choked")
            
        mdot        =  self.Cd*Flows1D.isentropicmdot(self.A, Pin, Tin, self.Rgas, self.gamma, M)
        
        #print("injector Tin is", Tin, "K")
        #print("injector Pin is", Pin/psi, "psi")
        #print("injector Pexit is", Pexit/psi, "psi")
        #print("injector mdot is", mdot, "kg/s")     
        return mdot
        
    def getPressureDrop(self, Pin, Tin, mdot):
        g = self.gamma
        f = lambda Pout: self.getMdot(Pin, Pout, Tin) - mdot
        Pout_solved = opt.brentq(f, 1e-6, Pin)
        
        if self.isChoked(Pin, Pout_solved):
            print("injector flow is choked, cannot solve for pressure drop")
        else:
            return Pin - Pout_solved
            
    def getUpstreamPressure(self, Pout, Tin, mdot):
        test_mdot = self.getMdot(950*psi, Pout, Tin)
        #print("mdot incoming is", mdot, "kg/s")
        g = self.gamma
        f = lambda Pin: self.getMdot(Pin, Pout, Tin) - mdot
        Pin_solved = opt.brentq(f, Pout, 1e10)
        print("pin_solved is", Pin_solved/psi, "psi")
        
        if self.isChoked(Pin_solved, Pout):
            print("injector flow is choked, cannot solve for upstream pressure")
        else:
            return Pin_solved
        
        
    def isChoked(self, Pin, Pexit):
        gamma       = self.gamma
        Pin_chokelimit = Pexit*((gamma+1)/2)**(gamma/(gamma-1))
        return Pin >= Pin_chokelimit
        
    def getTexit(self, Pin, Pexit, Tin):
        M       = self.getMexit(Pin, Pexit)
        return Flows1D.isentropicT(Tin, self.gamma, M)

    def getMexit(self, Pin, Pexit):
        if self.isChoked(Pin, Pexit):
            M       = 1     
            print("injector flow is choked")
        else:
            M       = Flows1D.getIsentropicMFromP(Pin, Pexit, self.gamma)
            #print("injector M is", M)        
        return M
    
    def getUexit(self, Pin, Pexit, Tin):
        Texit   = self.getTexit(Pin, Pexit, Tin)            # exit temp, K
        a       = sqrt(self.gamma*self.Rgas*Texit)          # sonic velocity, m/s
        M       = self.getMexit(Pin, Pexit)                 # Mach number
        return M*a                                          # exit velocity, m/s
    
#does not allow flow from exit to inlet (ie check valve)
class LiquidOrifice:
    #Cd appox 0.62 for square orifice, 0.9 to 0.95 for well-rounded orifice
    def __init__(self, A, Cd):
        self.A = A
        self.Cd = Cd
    
    def getMdot(self, Pinlet, Pexit, rho):
        if Pexit < Pinlet:
            return self.Cd*self.A*sqrt(2*rho*(Pinlet-Pexit))
        else:
            print("Warning: Pout>Pin so flow is zero")
            return 0.
    
    def getPressureDrop(self, mdot, rho):
        return (mdot/(self.Cd*self.A*sqrt(2*rho)))**2

class LiqToLiqandvapourOrifice:
    #according to Fauske: "Determine Two-Phase Flows During Releases"
    def __init__(self, A, Lori, CdL):
        self.A = A
        self.CdL = CdL      # = 1/sqrt(Kloss)
        self.Lori = Lori    # orifice length
    
    def getMdot(self, Pin, Pout, rooL, rooV, cL, L, T):
        if Pout > Pin:
            print("Pin is", Pin)
            print("Pout is", Pout)
            print("Warning: Pout>Pin so flow is zero")
            return 0.
        NE = (L*rooV)**2/(2*(Pin-Pout)*rooL*T*cL) + self.Lori/0.1
        #print("Pin is", Pin)
        #print("Pout is", Pout)
        #print("NE is", NE)
        Gtot = self.CdL*L*rooV/sqrt(T*cL*NE)    # mass flux 
        return Gtot*self.A
    
    def getPressureDrop(self, mdot, rooL, rooV, cL, L, T):  #rearranging above equation for mdot
        a = (L*rooV)**2/(2*rooL)
        b = (self.A*L*rooV*self.CdL/mdot)**2
        return a/(b -T*cL*self.Lori/0.1)
        
class MixedToMixedOrifice:
    #according to Fauske: "Determine Two-Phase Flows During Releases"
    def __init__(self, A, CdL, CdV):
        self.A = A
        self.CdL = CdL
        self.CdV = CdV
    
    def getMdot(self, Pin, Pout, rooL, rooV, cL, L, T, k, Mw, x):
        Gliq = rooV*L/sqrt(T*cL)
        Gvap = self.CdV*Pin*sqrt(Mw/(Runiv*T))*sqrt( k*(2/(k+1))**((k+1)/(k-1)) )
        Gtot = 1/sqrt( (1-x)/Gliq**2 + x/Gvap**2 )
        
        return Gtot*self.A
        
#this class assumes that inlet pressure is the same as P0
#does not allow flow from exit to inlet (ie check valve)
class GasInjector:
    def __init__(self, Astar, T0, gamma, Rgas, Cd):
        self.Astar = Astar
        self.T0 = T0
        self.gamma = gamma
        self.Rgas = Rgas
        self.Cd = Cd

    def getMdot(self, Pinlet, Pexit):   
        if Pexit > Pinlet:
            return 0.
        if(self.isChoked(Pinlet, Pexit)):
            M   = 1
        else:
            M   = Flows1D.getIsentropicMFromP(Pinlet, Pexit, self.gamma)
           
        mdot    = Flows1D.isentropicmdot(self.Astar, Pinlet, self.T0, \
            self.Rgas, self.gamma, M)*self.Cd
        return mdot
        
    def isChoked(self, Pinlet, Pexit):
        P0toChoke = Flows1D.getP0(Pexit, self.gamma, 1)
        return Pinlet >= P0toChoke
        
class VentHole:
    def __init__(self, diameter, gamma, Rgas, Cd):
        self.Astar = pi*diameter**2/4
        self.gamma = gamma
        self.Rgas = Rgas
        self.Cd = Cd

    def getMdot(self, Pinlet, Pexit, T0):   
        if Pexit > Pinlet:
            return 0.
        if(self.isChoked(Pinlet, Pexit)):
            M = 1
        else:
            M = Flows1D.getIsentropicMFromP(Pinlet, Pexit, self.gamma)
        return Flows1D.isentropicmdot(self.Astar, Pinlet, T0, \
            self.Rgas, self.gamma, M)*self.Cd
        
    def isChoked(self, Pinlet, Pexit):
        P0toChoke = Flows1D.getP0(Pexit, self.gamma, 1)
        return Pinlet >= P0toChoke

class HybridCombustionChamber:
    def __init__(self, a, n, r, l, rhoFuel):
        self.a = a
        self.n = n
        self.r = r
        self.l = l
        self.rhoFuel = rhoFuel
    
    def Ab(self):
        return 2*pi*self.r*self.l
    
    def rdot(self, mdotox):
        Gox = mdotox/self.Ab()
        return self.a*Gox**self.n
    
    def getOFratio(self, mdotox):
        mdotFuel = self.rdot(mdotox)*self.Ab()*self.rhoFuel
        return mdotox/mdotFuel

class BipropCombustionChamber:
    def __init__(self, nozzle, Vchamber, Tfire, gammaFire, mbarFire, Pfire, Pambient):
        self.m = Vchamber*Pfire/(Runiv/mbarFire)/Tfire
        self.mbar = mbarFire
        self.nozzle = nozzle
        self.V = Vchamber
        self.T = Tfire
        self.gamma = gammaFire
        self.Pa = Pambient
    
    def get_P_inlet(self):
        #print("kammiopaine on" , (self.m/self.V)*(Runiv/self.mbar)*self.T )
        return(self.m/self.V)*(Runiv/self.mbar)*T.self
    
    def update(self, mdot_ox, mdot_fuel, Pa, dt):
        P = self.get_P_inlet()
        mdot_in = mdot_ox + mdot_fuel
        mdot_out = self.nozzle.getmdot(self.gamma, Runiv/self.mbar, P, self.T, self.Pa)
        #print("mdot_in is", mdot_in)
        #print("mdot_out is", mdot_out)
        self.m = self.m + (mdot_in - mdot_out)*dt

class N2OKeroCombustionChamber:
    def __init__(self, nozzle, Vchamber, Tfire, gammaFire, mbarFire, Pfire, Pambient):
        self.m = Vchamber*Pfire/(Runiv/mbarFire)/Tfire
        self.mbar = mbarFire
        self.nozzle = nozzle
        self.V = Vchamber
        self.T = Tfire
        self.gamma = gammaFire
        self.Pa = Pambient
    
    def get_P_inlet(self):
        #print("kammiopaine on", (self.m/self.V)*(Runiv/self.mbar)*self.T )
        return(self.m/self.V)*(Runiv/self.mbar)*self.T
    
    def update(self, mdot_ox, mdot_fuel, Pa, dt):
        #print("mdot fuel is", mdot_fuel)
        OF_ratio = mdot_ox/mdot_fuel
        self.T = self.get_Tfire(OF_ratio)
        self.gamma = self.get_gamma(OF_ratio)
        self.mbar = self.get_mbar(OF_ratio)
        #print("T=", self.T)
        #print("gamma=", self.gamma)
        #print("MW=", self.mbar)
        P = self.get_P_inlet()
        mdot_in = mdot_ox + mdot_fuel
        mdot_out = self.nozzle.getmdot(self.gamma, Runiv/self.mbar, P, self.T, self.Pa)
        #print("mdot_in is", mdot_in)
        #print("mdot_out is", mdot_out)
        self.m = self.m + (mdot_in - mdot_out)*dt
    
    def get_Tfire(self, OFratio):
        return(1951.206*log(OFratio) - 136.507)
        
    def get_gamma(self, OFratio):
        return(-0.0394684*OFratio + 1.4332)
        
    def get_mbar(self, OFratio):
        return(8.62671*log(OFratio) + 9.57643)
       
class LOX_IPACombustionChamber:
    def __init__(self, nozzle, Vchamber, Tfire, gammaFire, mbarFire, Pfire, Pambient):
        self.m      = Vchamber*Pfire/(Runiv/mbarFire)/Tfire
        self.mbar   = mbarFire
        self.nozzle = nozzle
        self.V      = Vchamber
        self.T      = Tfire
        self.gamma  = gammaFire
        self.Pa     = Pambient
        
        # Interpolation of Tfire, mbar, and gamma from CEA data (extrapolation = assumes closest valid data point)
        pmin,pmax       = 100*psi,500*psi   # psi
        pstep           = 100*psi           # psi
        OFmin, OFmax    = 0.5, 2            # dimensionless 
        OFstep          = 0.125             # dimensionless

        p_vector        = np.linspace(pmin,pmax,(pmax-pmin)/pstep+1)
        OF_vector       = np.linspace(OFmin,OFmax,(OFmax-OFmin)/OFstep+1)
        

        T_grid = np.array([[ 1115, 1160, 1188, 1209, 1225],
       [ 1162, 1211, 1241, 1264 ,1282],
       [ 1396, 1403, 1412, 1422, 1431],
       [ 1803, 1803, 1803, 1804, 1804],
       [ 2171, 2173, 2174, 2174, 2174],
       [ 2488, 2496, 2500, 2503, 2505],
       [ 2741, 2764, 2775, 2782, 2787],
       [ 2931, 2971, 2992, 3006, 3016],
       [ 3061, 3118, 3149, 3171, 3187],
       [ 3143, 3214, 3255, 3282, 3304],
       [ 3193, 3273, 3319, 3352, 3377],
       [ 3221, 3306, 3356, 3392, 3420],
       [ 3235, 3324, 3376, 3413, 3442]])
       
       
        MW_grid = np.array([[14.15, 14.48, 14.70, 14.87, 15.01],
       [ 14.91, 15.15, 15.30, 15.41, 15.50],
       [ 15.04, 15.07, 15.11, 15.16, 15.20],
       [ 16.10, 16.10, 16.10, 16.10, 16.10],
       [ 17.16, 17.16, 17.16, 17.17, 17.17],
       [ 18.18, 18.20, 18.21, 18.21, 18.22],
       [ 19.15, 19.19, 19.21, 19.22, 19.23],
       [ 20.01, 20.08, 20.12, 20.15, 20.17],
       [ 20.77, 20.88, 20.95, 20.99, 21.02],
       [ 21.44, 21.59, 21.68, 21.74, 21.77],
       [ 22.03, 22.20, 22.30, 22.37, 22.42],
       [ 22.56, 22.75, 22.85, 22.93, 22.99],
       [ 23.04, 23.24, 23.35, 23.44, 23.50]])       
       
        gamma_grid = np.array([[ 1.145, 1.148, 1.150, 1.152, 1.153],
       [ 1.188, 1.190, 1.191, 1.192, 1.193],
       [ 1.291, 1.280, 1.270, 1.261, 1.256],
       [ 1.273, 1.274, 1.274, 1.274, 1.274],
       [ 1.248, 1.250, 1.251, 1.252, 1.252],
       [ 1.217, 1.223, 1.223, 1.227, 1.228],
       [ 1.186, 1.194, 1.198, 1.201, 1.203],
       [ 1.161, 1.168, 1.173, 1.176, 1.179],
       [ 1.142, 1.149, 1.153, 1.156, 1.158],
       [ 1.131, 1.136, 1.140, 1.142, 1.144],
       [ 1.125, 1.129, 1.131, 1.133, 1.135],
       [ 1.121, 1.125, 1.127, 1.128, 1.130],
       [ 1.119, 1.122, 1.124, 1.126, 1.127]])
       
        self.T_int      = interpolate.RectBivariateSpline(OF_vector, p_vector, T_grid)     # Note the order of Y and X!!
        self.MW_int     = interpolate.RectBivariateSpline(OF_vector, p_vector, MW_grid)    # Note the order of Y and X!!
        self.gamma_int  = interpolate.RectBivariateSpline(OF_vector, p_vector, gamma_grid) # Note the order of Y and X!!

    
    def get_P_inlet(self):
        #print("kammiopaine on", (self.m/self.V)*(Runiv/self.mbar)*self.T )
        return(self.m/self.V)*(Runiv/self.mbar)*self.T
    
    def update(self, mdot_ox, mdot_fuel, Pa, dt):
        #print("mdot fuel is", mdot_fuel)
        OF_ratio    = mdot_ox/mdot_fuel
        P           = self.get_P_inlet()
        self.T      = self.get_Tfire(OF_ratio, P)
        self.gamma  = self.get_gamma(OF_ratio, P)
        self.mbar   = self.get_mbar(OF_ratio, P)
        #print("T=", self.T)
        #print("gamma=", self.gamma)
        #print("MW=", self.mbar)
        
        mdot_in     = mdot_ox + mdot_fuel
        mdot_out    = self.nozzle.getmdot(self.gamma, Runiv/self.mbar, P, self.T, self.Pa)
        #print("mdot_in is", mdot_in)
        #print("mdot_out is", mdot_out)
        self.m = self.m + (mdot_in - mdot_out)*dt
    
    
        #from CEA code assuming 100-500psi chamber pressure and OF-ratio 0.5-2
        
    def get_Tfire(self, OFratio, pressure):
        return self.T_int.ev(OFratio, pressure)
        
    def get_gamma(self, OFratio, pressure):
        return self.gamma_int.ev(OFratio, pressure)
        
    def get_mbar(self, OFratio, pressure):
        return self.MW_int.ev(OFratio, pressure)
        #print("MW_res is", MW_res, "g/mol")
        
      
class LOXKeroCombustionChamber:
    def __init__(self, nozzle, Vchamber, Tfire, gammaFire, mbarFire, Pfire, Pambient):
        self.m      = Vchamber*Pfire/(Runiv/mbarFire)/Tfire
        self.mbar   = mbarFire
        self.nozzle = nozzle
        self.V      = Vchamber
        self.T      = Tfire
        self.gamma  = gammaFire
        self.Pa     = Pambient
    
    def get_P_inlet(self):
        #print("kammiopaine on", (self.m/self.V)*(Runiv/self.mbar)*self.T )
        return(self.m/self.V)*(Runiv/self.mbar)*self.T
    
    def update(self, mdot_ox, mdot_fuel, Pa, dt):
        #print("mdot fuel is", mdot_fuel)
        OF_ratio    = mdot_ox/mdot_fuel
        self.T      = self.get_Tfire(OF_ratio)
        self.gamma  = self.get_gamma(OF_ratio)
        self.mbar   = self.get_mbar(OF_ratio)
        #print("T=", self.T)
        #print("gamma=", self.gamma)
        #print("MW=", self.mbar)
        P           = self.get_P_inlet()
        mdot_in     = mdot_ox + mdot_fuel
        mdot_out    = self.nozzle.getmdot(self.gamma, Runiv/self.mbar, P, self.T, self.Pa)
        #print("mdot_in is", mdot_in)
        #print("mdot_out is", mdot_out)
        self.m      = self.m + (mdot_in - mdot_out)*dt
    
    
        #from CEA code assuming 1015 psi (=7 MPa) chamber pressure and OFratio 1.5-3
        
    def get_Tfire(self, OFratio):
        return(336*OFratio**3 - 3057.9*OFratio**2 + 9271.3*OFratio - 5664.1)
        
    def get_gamma(self, OFratio):
        return(-0.0160*OFratio**3 + 0.1802*OFratio**2 - 0.6410*OFratio + 1.8658)
        
    def get_mbar(self, OFratio):
        return(0.1778*OFratio**3 - 2.7867*OFratio**2 + 14.4408 *OFratio + 1.6702)            
 
# the difference here is that oxygen enters chamber as a room temp gas, not cryogenic liquid -> higher Tflame        
class GOXKeroCombustionChamber:
    def __init__(self, nozzle, Vchamber, Tfire, gammaFire, mbarFire, Pfire, Pambient, mdot_out):
        self.m          = Vchamber*Pfire/(Runiv/mbarFire)/Tfire
        self.mbar       = mbarFire
        self.nozzle     = nozzle
        self.V          = Vchamber
        self.T          = Tfire
        self.gamma      = gammaFire
        self.Pa         = Pambient
        self.mdot_out   = mdot_out
        
        # Interpolation of Tfire, mbar, and gamma from CEA data (extrapolation = assumes closest valid data point)
        pmin,pmax       = 250*psi,1000*psi   # psi
        pstep           = 250*psi           # psi
        OFmin, OFmax    = 0.5, 5            # dimensionless 
        OFstep          = 0.5               # dimensionless

        p_vector        = np.linspace(pmin,pmax,(pmax-pmin)/pstep+1)
        OF_vector       = [0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 3, 3.5, 4, 4.5, 5]
        

        T_grid = np.array([[ 1202, 1252, 1283, 1306],
       [ 1341, 1390, 1422, 1446],
       [ 1545, 1576, 1599, 1617],
       [ 1975, 1976, 1977, 1978],
       [ 2585, 2593, 2598, 2600],
       [ 3020, 3054, 3071, 3082],
       [ 3289, 3352, 3388, 3411],
       [ 3430, 3518, 3568, 3603],
       [ 3495, 3595, 3654, 3695],
       [ 3524, 3632, 3696, 3742],
       [ 3504, 3611, 3675, 3720],
       [ 3468, 3571, 3632, 3676],
       [ 3426, 3523, 3581, 3623],
       [ 3379, 3472, 3527, 3566]])
       
       
        MW_grid = np.array([[ 17.42, 17.85, 18.13, 18.34 ],
       [15.95, 16.25, 16.45, 16.60],
       [15.37, 15.53, 15.65, 15.75],
       [15.92, 15.92, 15.93, 15.93],
       [17.64, 17.65, 17.66, 17.67],
       [19.19, 19.25, 19.28, 19.30],
       [20.51, 20.63, 20.70, 20.74],
       [21.61, 21.78, 21.88, 21.94],
       [22.52, 22.72, 22.84, 22.93],
       [23.97, 24.21, 24.36, 24.46],
       [25.13, 25.38, 25.53, 25.64],
       [26.08, 26.33, 26.48, 26.59],
       [26.88, 27.13, 27.28, 27.38],
       [27.56, 27.80, 27.94, 28.05]])      
       
        gamma_grid = np.array([[ 1.15, 1.15, 1.15, 1.15],
       [1.20, 1.20, 1.19, 1.19],
       [1.27, 1.26, 1.26, 1.25],
       [1.30, 1.30, 1.30, 1.30],
       [1.24, 1.25, 1.25, 1.25],
       [1.19, 1.20, 1.20, 1.21],
       [1.16, 1.16, 1.17, 1.17],
       [1.14, 1.14, 1.15, 1.15],
       [1.13, 1.13, 1.14, 1.14],
       [1.13, 1.13, 1.13, 1.13],
       [1.12, 1.13, 1.13, 1.13],
       [1.12, 1.13, 1.13, 1.13],
       [1.12, 1.13, 1.13, 1.13],
       [1.12, 1.13, 1.13, 1.13]])
       
        self.T_int      = interpolate.RectBivariateSpline(OF_vector, p_vector, T_grid)     # Note the order of Y and X!!
        self.MW_int     = interpolate.RectBivariateSpline(OF_vector, p_vector, MW_grid)    # Note the order of Y and X!!
        self.gamma_int  = interpolate.RectBivariateSpline(OF_vector, p_vector, gamma_grid) # Note the order of Y and X!!
   
   
    def get_P_inlet(self):
        #print("kammiopaine on", (self.m/self.V)*(Runiv/self.mbar)*self.T )
        return(self.m/self.V)*(Runiv/self.mbar)*self.T
    
    def update(self, mdot_ox, mdot_fuel, Pa, dt):
        #print("mdot fuel is", mdot_fuel)
        OF_ratio    = mdot_ox/mdot_fuel
        mdot_in     = mdot_ox + mdot_fuel   
        P           = self.get_P_inlet()
        self.T      = self.get_Tfire(OF_ratio, P)
        self.gamma  = self.get_gamma(OF_ratio, P)
        self.mbar   = self.get_mbar(OF_ratio, P)
        '''   
        self.m      = self.m + (mdot_in - self.mdot_out)*dt
        P           = self.get_P_inlet()
        self.mdot_out = self.nozzle.getmdot(self.gamma, Runiv/self.mbar, P, self.T, self.Pa)
        '''
        #print("T=", self.T)
        #print("gamma=", self.gamma)
        #print("MW=", self.mbar)
       
        #print("Pcha is", P, "Pa")
        #print("mdot_in is", mdot_in)
        #print("mdot_out is", self.mdot_out)
        
        m_int       = self.m + (mdot_in - self.mdot_out)*dt # mass based on mdot_out at beginning of timestep (too big)
        P_int       = (m_int/self.V)*(Runiv/self.mbar)*self.T        # pressure -"- (too big)   
        mdot_out_int= self.nozzle.getmdot(self.gamma, Runiv/self.mbar, P_int, self.T, self.Pa) #mdot_out -"- (too big)
        mdot_out_avg= (self.mdot_out + mdot_out_int)/2
        
        m_new       = self.m + (mdot_in - mdot_out_avg)*dt
        self.m      = m_new
        self.mdot_out = mdot_out_avg
        
        #from CEA code assuming 250-1000psi chamber pressure and OF-ratio 0.5-5
        
    def get_Tfire(self, OFratio, pressure):
        return self.T_int.ev(OFratio, pressure)
        
    def get_gamma(self, OFratio, pressure):
        return self.gamma_int.ev(OFratio, pressure)
        
    def get_mbar(self, OFratio, pressure):
        return self.MW_int.ev(OFratio, pressure)
        

#assumes a smooth tube and Fanning friction factor
class StraightCylindricalTube:
    def __init__(self, diameter, length, fluidViscosity, fluidDensity):
        self.d      = diameter
        self.l      = length
        self.mu     = fluidViscosity
        self.rho    = fluidDensity
    
    def getPressureDrop(self, mdot):
        Re = 4*mdot/pi/self.d/self.mu
        if Re < 2000:
            f = 16/Re
        elif Re < 20000:
            f = 0.0791/Re**0.25
        else:
            f = 0.046/Re**0.2
        A = pi*self.d**2/4
        v = mdot/self.rho/A
        return 4*f*self.l*self.rho*v**2/self.d/2
    
    def getMdot(self, Pin, Pout):
        if Pin > Pout:
            errorFunc = lambda mdot: (Pin-Pout) - self.getPressureDrop(mdot)
            return opt.fsolve(errorFunc, sqrt(2*self.rho*(Pin-Pout)))[0]
        else:
            errorFunc = lambda mdot: (Pout-Pin) - self.getPressureDrop(mdot)
            return -opt.fsolve(errorFunc, sqrt(2*self.rho*(Pout-Pin)))[0]

#uses Darcy friction factor (4x Fanning friction factor)
class RoughStraightCylindricalTube:
    def __init__(self, diameter, length, roughness, headloss):
        self.d = diameter
        self.l = length
        self.rou = roughness    # defined as epsilon/diameter
        self.headloss = headloss
     
    def getPressureDrop(self, mdot, fluidViscosity, fluidDensity):  
        #print("vis is", fluidViscosity)
        #print("dia is", self.d)
        
        Re = 4*mdot/(pi*self.d*fluidViscosity)
        #print("mdot in tube is", mdot)
        #print("Re is", Re)
        
        #Churchill
        C1 = (-2.457*log( (7/Re)**0.9 + 0.27*self.rou) )**16;
        C2 = (37530/Re)**16;
        f = 8*( (8/Re)**12 + (C1+C2)**-1.5 )**(1/12);
        
        # Chen
        #ff = -2*log10(self.rou/3.7065 - 5.0452/Re*log10(1/2.8257*self.rou**1.1098 + 5.8506/Re**0.8981))
        #f  = (1/ff)**2
        
        #print("phi1 is", C1)
        #print("phi2 is", C2)
        #print("f is", f)
        
        A = pi*self.d**2/4
        v = mdot/fluidDensity/A
        #print("v is", v, "m/s")
        if self.headloss == True:
            return 0.5*fluidDensity*v**2*(f*self.l/self.d + 1) # includes head loss due to velocity increase FROM ZERO!
        else :
            return 0.5*fluidDensity*v**2*(f*self.l/self.d)  # only pressure drop due to friction
        
        
    def getMdot(self, Pin, Pout, fluidViscosity, fluidDensity):
        if Pin > Pout:
            errorFunc = lambda mdot: (Pin-Pout) - self.getPressureDrop(mdot[0], fluidViscosity, fluidDensity)
            return opt.fsolve(errorFunc, sqrt(2*fluidDensity*(Pin-Pout)))[0]
        else:
            errorFunc = lambda mdot: (Pout-Pin) - self.getPressureDrop(mdot, fluidViscosity, fluidDensity)
            return -opt.fsolve(errorFunc, sqrt(2*fluidDensity*(Pout-Pin)))[0]
    
class PressureRegulator:
    def __init__(self, Pout, fluid):
        self.Pout   = Pout
    
    def getTempDrop(self, Pin, Pout, T_in):    #J-T coeff is derivative of temp respect to pressure ->integrate to get Tdrop
        
        def dTdP( T, P): 
            return self.fluid.getJT(T, P)           # derivative (dT/dP)|constant enthalpy

        Ps          = np.linspace(Pin, Pout, 5)     # 5 intervals
        T0          = T_in                          # initial conditions
        Ts          = odeint(dTdP, T0, Ps)          # integration results
        T_out       = Ts[-1]
        
        return T_in - T_out                         # Temp drop [K]
            
class CompressibleFlowSolenoid:         # ISA-75.01.01-2007 valve sizing handbook
    def __init__(self, Cv, fluid):
        self.Cv     = Cv                # valve flow coefficient, defined by orifice diameter and valve type
        self.Fgamma = fluid.gamma/1.4
        self.xT     = 0.7               # characteristic to this type of valve(solenoid = plug control, flow up)
        self.fluid  = fluid
        
    def getMdot(self, Pin, Pout, roo_in):
        n           = 2.73                                  # dimensionless experimental/unit correction factor
        
        if self.Fgamma*self.xT <= (Pin-Pout)/Pin:
            print("Pin_gassole is", Pin/psi, "psi")
            print("Pout_gassole is", Pout/psi, "psi")
            print("Warning: gas solenoid flow is choked")
            Y       = 0.667                                             # choked comperessibility factor
            return self.Cv*n*Y*sqrt(self.Fgamma*self.xT*Pin/1000*roo_in)# mass flow rate [kg/s], eq.12
        else:
            Y       = 1 - (Pin-Pout)/(Pin*3*self.Fgamma*self.xT)        # compressibility factor
            return self.Cv*n*Y*sqrt((Pin-Pout)/1000*roo_in)/3600        # mass flow rate [kg/s], eq.6
            
        
    def getPressureDrop(self, mdot, Pin, roo_in):
        n           = 2.73
        
        errorFunc   = lambda dP: mdot - self.getMdot(Pin, Pin-dP, roo_in)
        init_guess  = 1000/roo_in*(mdot*3600/(self.Cv*n*1))**2           # assuming Y = 1 (incompressible)
        dPsolution  = opt.fsolve(errorFunc, init_guess)[0]
        #print("dPsolution is", dPsolution/psi, "psi")
        
        if self.Fgamma*self.xT <= dPsolution/Pin:
            print("Warning: solenoid flow is choked and results might be wrong")
        return dPsolution
        
    def getTempDrop(self, Pin, Pout, T_in):    #J-T coeff is derivative of temp respect to pressure ->integrate to get Tdrop
        
        def dTdP( T, P): 
            return self.fluid.getJT(T, P)           # derivative (dT/dP)|constant enthalpy

        Ps          = np.linspace(Pin, Pout, 5)     # 5 intervals
        T0          = T_in                          # initial conditions
        Ts          = odeint(dTdP, T0, Ps)          # integration results
        T_out       = Ts[-1]
        
        return T_in - T_out                         # Temp drop [K]
        
class CompressibleFlowCheckValve:
    def __init__(self, Cv, Pcrack, fluid):          # https://www.swagelok.com/downloads/webcatalogs/en/MS-06-84.pdf
        self.Cv     = Cv
        self.Pcrack = Pcrack
        self.gamma  = fluid.gamma 
        self.open   = open
        self.fluid  = fluid
        
    def getMdot(self, Pin, Pout, roo_std, roo_in, T_in):
        N2      = 6950                  # gives volumetric flowrate in STD l/min
        N3      = 0.001/60              # to convert from g/min to kg/s
        roo_air = 1.23                  # density of air at STD conditions, kg/m3
        Tstd    = 293                   # STD temp, K
        Pstd    = 101325                # STD pressure, Pa
        Gg      = roo_std/roo_air       # specific gravity of inlet gas, dimensionless
        dP      = Pin-Pout
        
        #print("Pin cek is", Pin/psi, "psi")
        #print("Pout chek is", Pout/psi, "psi")
        #print("selgamma is", self.gamma)
        
        if Pout/Pin <=  (2/(self.gamma+1))**(self.gamma/(self.gamma-1)):
            print("Warning: check valve flow is choked")
            return 0.471*N2*self.Cv*Pin/atm*sqrt(1/(Gg*T_in))*roo_in*(T_in/Tstd)*(Pstd/Pin)*N3 # mass flow rate [kg/s]
        else:
            Vdot    = N2*self.Cv*Pin/atm*(1-2*dP/(3*Pin))*sqrt(dP/(Pin*Gg*T_in))   # volumetric flow rate, STD l/min
            #print("volum flow rate is", Vdot, "stdL/min")
            mdot    = Vdot*roo_in*(T_in/Tstd)*(Pstd/Pin)*N3                 # mass flow rate in kg/s
            return mdot

    def getPressureDrop(self, mdot, Pin, roo_std, roo_in, T_in):
        N2      = 6950                  # gives volumetric flowrate in STD l/min
        N3      = 0.001/60              # to convert from g/min to kg/s
        roo_air = 1.23                  # density of air at STD conditions, kg/m3
        Tstd    = 293                   # STD temp, K
        Pstd    = 101325                # STD pressure, Pa
        Gg      = roo_std/roo_air       # specific gravity of inlet gas, dimensionless
        n       = 2.73                  # experimental correction factor for SOLENOID valves, used for initial guessing
        
        errorFunc   = lambda dP: mdot - self.getMdot(Pin, Pin-dP, roo_std, roo_in, T_in)
        init_guess  = 1000/roo_in*(mdot*3600/(self.Cv*n*1))**2
        #print("init_guess is", init_guess/psi, "psi")
        dPsolution  = opt.fsolve(errorFunc, init_guess)[0]
        #print("dPsolution is", dPsolution/psi, "psi")
        
        if (Pin-dPsolution)/Pin <=  (2/(self.gamma+1))**(self.gamma/(self.gamma-1)):
            print("Warning: check valve flow is choked and results might be wrong")
        return dPsolution
        
    def getTempDrop(self, Pin, Pout, T_in):    #J-T coeff is derivative of temp respect to pressure ->integrate to get Tdrop
        
        def dTdP( T, P): 
            return self.fluid.getJT(T, P)           # derivative (dT/dP)|constant enthalpy

        Ps          = np.linspace(Pin, Pout, 5)     # 5 intervals
        T0          = T_in                          # initial conditions
        Ts          = odeint(dTdP, T0, Ps)          # integration results
        T_out       = Ts[-1]
        return T_in - T_out                         # Temp drop [K]

class IncompressibleFlowSolenoid:
    def __init__(self, Cv):           # ISA-75.01.01-2007 valve sizing handbook
        self.Cv = Cv                  # defined by orifice diameter and valve type, given by manufacturer
        self.FL = 0.9                 # characteristic to this type of valve (plug control, flow up)
        
    def getMdot(self, Pin, Pout, rooL, P_critical, P_vapor):
        Ff = 0.96 - 0.28*sqrt(P_vapor/P_critical)
        if self.FL**2*(Pin-Ff*P_vapor) <= (Pin-Pout):
            print("Warning: solenoid flow is choked")
        
        #print("Pin is", Pin/psi, "psi")
        #print("Pout is", Pout/psi, "psi")
        
        return self.Cv*rooL/1000*3.78/60*sqrt((Pin-Pout)/psi*1000/rooL)
        
    def getPressureDrop(self, mdot, rooL):
        return (mdot*60000/(self.Cv*rooL*3.78))**2*psi*rooL/1000
    
class IncompressibleFlowMeter:
    def __init__(self, Pdrop):
        self.Pdrop = Pdrop
        
    def getPressureDrop(self):
        return self.Pdrop
        
class CompressibleFlowParticleFilter:
    def __init__(self, Pdrop):
        self.Pdrop = Pdrop
    
    def getPressureDrop(self):
        return self.Pdrop

class IncompressibleFlowParticleFilter:
    def __init__(self, Pdrop):
        self.Pdrop = Pdrop
    
    def getPressureDrop(self):
        return self.Pdrop    
    
class ConvergingDivergingNozzle:
    def __init__(self, Ae, Astar):
        self.Ae = Ae
        self.Astar = Astar
    
    def getP0toChoke(self, gamma, Pa):
        #print("gamma=", gamma)
        return Flows1D.getP0(Pa, gamma, 1)
    
    def getCf(self, P0, Pa, gamma):
        if P0 <= Pa:
            return 0 #for the sake of numerical simulations
        elif P0 < self.getP0toChoke(gamma, Pa):#unchoked=subsonic flow
            print("unchoked")
            Pexit = Pa
            Mexit = Flows1D.getIsentropicMFromP(P0, Pexit, gamma)
            Fth = self.Ae*Mexit**2*gamma*Pa
            return Fth/P0/self.Astar
        else:#choked flow
            Pe = self.getPe(P0, gamma, Pa)
            return sqrt(2*gamma**2/(gamma-1)* \
                (2/(gamma+1))**((gamma+1)/(gamma-1))* \
                (1-(Pe/P0)**((gamma-1)/gamma))) + \
                (Pe - Pa)*self.Ae/P0/self.Astar
    
    def getPe(self, P0, gamma, Pa):
        if P0 < self.getP0toChoke(gamma, Pa):
            return Pa
        else:
            Me = Flows1D.getIsentropicMs(self.Astar, self.Ae, gamma)[1]
            Pe = Flows1D.isentropicP(P0, gamma, Me)
            return Pe
            
    def getMe(self, P0, gamma, Pa):
        if P0 < self.getP0toChoke(gamma, Pa):
            return Flows1D.getIsentropicMs(self.Astar, self.Ae, gamma)[0]
        else:
            Me = Flows1D.getIsentropicMs(self.Astar, self.Ae, gamma)[1]
            return Me        
    
    def getmdot(self, gamma, Rgas, P0, T0, Pa):
        if P0 <= Pa:#for numerical simulations
            return 0
        elif P0 < self.getP0toChoke(gamma, Pa):#unchoked flow case
            #print("we are NOT choked")
            Mexit = Flows1D.getIsentropicMFromP(P0, Pa, gamma)
            return Flows1D.isentropicmdot(self.Ae, P0, T0, Rgas, gamma, Mexit)
        else:#choked flow case
            cstar = Flows1D.getcstar(gamma, Rgas, T0)
            #print("cstar is", cstar)
            return P0*self.Astar/cstar
    
    def getThrust(self, P0, Pa, gamma):
        if P0 <= Pa:#for numerical simulations
            return 0
        elif P0 < self.getP0toChoke(gamma, Pa):#unchoked=subsonic flow
            print("unchoked flow at the throat")
            Pexit = Pa
            Mexit = Flows1D.getIsentropicMFromP(P0, Pexit, gamma)
            Fth = self.Ae*Mexit**2*gamma*Pa
            return Fth
        else:#choked flow
            #print ("choked,going supersonic")
            return P0*self.Astar*self.getCf(P0, Pa, gamma)

class CoolingJacket:
    def __init__(self, ref_mdot, ref_Pdrop):
        self.mdot_ref       = ref_mdot  # kg/s
        self.Pdrop_ref      = ref_Pdrop # Pa  
        
    def getPressureDrop(self, mdot):
        return mdot/self.mdot_ref*self.Pdrop_ref

class N2OFluid:
    def __init__(self):
        '''
        # property equations from http://edge.rit.edu/content/P07106/public/Nox.pdf
        self.TCrit  = 309.57            # K 
        self.rooCrit = 452              # kg/m3
        self.pCrit  = 7251e3            # Pa
        self.gamma  = 1.27
        self.Mw     = 44.013            # kg/kmol
        '''
        self.TCrit   = CP.PropsSI("Tcrit", "NitrousOxide")           # K
        self.rooCrit = CP.PropsSI("rhocrit", "NitrousOxide")         # kg/m3
        self.pCrit   = CP.PropsSI("pcrit", "NitrousOxide")           # Pa
        self.gamma   = 1.27
        self.Mw      = CP.PropsSI("molarmass", "NitrousOxide")*10**3 # kg/kmol
        
    
    def getLiquidEnthalpy(self, fluidTemp):
        '''
        bL = [-200, 116.043, -917.225, 794.779, -589.587]
        Tr = fluidTemp/self.TCrit
        
        enthL=0.
        
        for i in range(0,5):
             enthL += bL[i]*(1-Tr)**(i/3.)
             
        return enthL*1000. #J/kg
        '''
        enthL = CP.PropsSI("H", "T", fluidTemp, "Q", 0, "NitrousOxide")
        return enthL
    
    def getVaporEnthalpy(self, fluidTemp):
        '''
        bV = [-200, 440.055, -459.701, 434.081, -485.338] 
        Tr = fluidTemp/self.TCrit
    
        enthV=0.
        
        for i in range(0,5):
             enthV += bV[i]*(1-Tr)**(i/3.)
        
        return enthV*1000. #J/kg
        '''
        enthV = CP.PropsSI("H", "T", fluidTemp, "Q", 1, "NitrousOxide")
        return enthV
    
    def getLatentHeat(self, fluidTemp):
        '''
        bL = [-200, 116.043, -917.225, 794.779, -589.587]
        bV = [-200, 440.055, -459.701, 434.081, -485.338] 
        Tr = fluidTemp/self.TCrit
       
        enthL=0.
        enthV=0.
        
        for i in range(0,5):
             enthL += bL[i]*(1-Tr)**(i/3.)
             enthV += bV[i]*(1-Tr)**(i/3.)
        
        return (enthV-enthL)*1000. #J/kg
        '''
        enthL = CP.PropsSI("H", "T", fluidTemp, "Q", 0, "NitrousOxide")
        enthV = CP.PropsSI("H", "T", fluidTemp, "Q", 1, "NitrousOxide")
        return enthV-enthL
        
    def getLiquidDensity(self, fluidTemp):
        '''
        bL = [1.72328, -0.8395, 0.5106, -0.10412]
        Tr = fluidTemp/self.TCrit
        exponent=0.
        
        for j in range(0,4):
            exponent += bL[j]*(1-Tr)**((j+1)/3.)
            
        return self.rooCrit*exp(exponent)
        '''
        denL = CP.PropsSI("D", "T", fluidTemp, "Q", 0, "NitrousOxide")
        return denL
        
    def getVaporDensity(self, fluidTemp):
        '''
        bV = [-1.009, -6.28792, 7.50332, -7.90463, 0.629427]
        Tr = fluidTemp/self.TCrit
        exponent=0.
        
        for i in range(0,5):
            exponent += bV[i]*(1/Tr-1)**((i+1)/3.)
            
        return self.rooCrit*exp(exponent)
        '''
        denV = CP.PropsSI("D", "T", fluidTemp, "Q", 1, "NitrousOxide")
        return denV
    
    def getVaporPressure(self, fluidTemp):
        '''
        bV = [-6.71893, 1.35966, -1.3779, -4.051]
        pV = [1, 1.5, 2.5, 5]
        Tr = fluidTemp/self.TCrit
        exponent=0.
         
        for i in range(0,4):
            exponent += bV[i]*(1-Tr)**(pV[i])
            
        return self.pCrit*exp(exponent/Tr)
        '''
        pV = CP.PropsSI("P", "T", fluidTemp, "Q", 1, "NitrousOxide")
        return pV
    
    def getLiquidSpecificHeat(self, fluidTemp):   #isobaric specific heat c_p
        '''
        bL = [2.49973, 0.023454, -3.80136, 13.0945, -14.518]
        pL = [-1, 1, 2, 3]
        Tr = fluidTemp/self.TCrit
        cp=1.
         
        for i in range(1,5):
            cp += bL[i]*(1-Tr)**(pL[i-1])
            
        return cp*bL[0]*1000. #J/kg
        '''
        cp = CP.PropsSI("Cpmass", "T", fluidTemp, "Q", 0, "NitrousOxide")
        return cp
    
    def getLiquidViscosity(self, fluidTemp):   # dynamic viscosity mu
        '''
        theta = (self.TCrit - 5.24)/(fluidTemp - 5.24)
        return 0.0293423/1000*exp( 1.6089*(theta-1)**(1/3) + 2.0439*(theta-1)**(4/3) ) #Ns/m^2
        '''
        mu = CP.PropsSI("viscosity", "T", fluidTemp, "Q", 0, "NitrousOxide")
        return mu

class Kerosene:
    
    def __init__(self):
        self.density    = 800              # kg/m3
        self.P_crit     = 1.8e6            # Pa
        #http://thehuwaldtfamily.org/jtrl/research/Propulsion/Rocket%20Propulsion/Propellants/DLR,%20Comparative%20Study%20of%20Kerosene%20and%20Methane%20Engines.pdf
        self.P_vapor    = 2000             # Pa
        self.mu         = 0.002            # dynamic viscosity, Ns/m2, average at 25C from different sources
        
    def getDensity(self, P, T):
        return self.density
        
        
class IPAFluid:
    
    def __init__(self):
        self.density = 786              # kg/m3
        self.P_crit  = 5.37e6           # Pa
        self.P_vapor = 7000             # Pa, at 300K
        self.mu      = 2.5e-3             # Pas = Ns/m2, at 440psi & ?? K 
                                        # http://www.nist.gov/srd/upload/jpcrd395.pdf, page 1101
    def getDensity(self, P, T):
        return self.density         
        
       
class HeFluid:
    
    def __init__(self):
        self.gamma      = 1.67
        self.mbar       = CP.PropsSI("molarmass", "Helium")*10**3
        self.R          = Runiv/self.mbar
        self.P_crit     = CP.PropsSI("pcrit", "Helium")
        self.T_crit     = CP.PropsSI("Tcrit", "Helium")
        self.rho_crit   = CP.PropsSI("rhocrit", "Helium")
        self.roo_std    = CP.PropsSI("D", "T", 293, "P", 101325, "Helium")
    
    def getDensity(self, pres, Temp):
        density         = CP.PropsSI("D", "T", Temp, "P", pres, "Helium") # density, kg/m3
        #self.checkPhase(pres, Temp)
        return density 
        
    def getViscosity(self, pres, Temp):
        viscosity       = CP.PropsSI("V", "T", Temp, "P", pres, "Helium") # Dynamic viscosity, Pa*s
        #self.checkPhase(pres, Temp)
        return viscosity
        
    def getJT(self, P, T): 
        return CP.PropsSI('d(T)/d(P)|Hmass', 'P' ,P, 'T', T, 'Helium') # Joule-Thomson coeff in [K/Pa]
        
    def checkPhase(self, pres, Temp):
        phase           = CP.PhaseSI("T", Temp, "P", pres, "Helium")
        if phase != 'liquid' and phase != 'supercritical_liquid':
            print("He is", phase, "at this point")
           
        
class LOXFluid:
    
    def __init__(self):
        self.P_crit     = CP.PropsSI("pcrit", "Oxygen")
        self.T_crit     = CP.PropsSI("Tcrit", "Oxygen")
        self.rho_crit   = CP.PropsSI("rhocrit", "Oxygen")
        self.P_vapor    = CP.PropsSI("P", "T", 90, "Q", 1, "Oxygen")
    
    def getDensity(self, pres, Temp):
        density         = CP.PropsSI("D", "T", Temp, "P", pres, "Oxygen") # density, kg/m3
        self.checkPhase(pres, Temp)
        return density 
        
    def getViscosity(self, pres, Temp):
        viscosity       = CP.PropsSI("V", "T", Temp, "P", pres, "Oxygen") # Dynamic viscosity, Pa*s
        self.checkPhase(pres, Temp)
        return viscosity
        
    def getVaporPressure(self, fluidTemp):
        
        pV = CP.PropsSI("P", "T", fluidTemp, "Q", 1, "Oxygen")
        return pV    
        
    def checkPhase(self, pres, Temp):
        phase           = CP.PhaseSI("T", Temp, "P", pres, "Oxygen")
        #print("pres is", pres/psi, "psi")
        #print("temp is", Temp, "K")
        #if phase != 'liquid' and phase != 'supercritical_liquid':
            #print("LOX is", phase, "at this point")
        
        
class GOXFluid:
    
    def __init__(self):
        self.gamma      = 1.4
        self.mbar       = CP.PropsSI("molarmass", "Oxygen")*10**3
        self.R          = Runiv/self.mbar
        self.P_crit     = CP.PropsSI("pcrit", "Oxygen")
        self.T_crit     = CP.PropsSI("Tcrit", "Oxygen")
        self.roo_crit   = CP.PropsSI("rhocrit", "Oxygen")
        self.roo_std    = CP.PropsSI("D", "T", 293, "P", 101325, "Oxygen")
 
    def getDensity(self, P, T):
        density = CP.PropsSI("D", "T", T, "P", P, "Oxygen") # density, kg/m3
        self.checkPhase(T, P)
        return density
        
    def getViscosity(self, P, T):     
        viscosity = CP.PropsSI("viscosity", "T", T, "P", P, "Oxygen") # Dynamic viscosity, Pa*s
        self.checkPhase(P, T)
        return viscosity
        
    def getJT(self, P, T): 
        return CP.PropsSI('d(T)/d(P)|Hmass', 'P', P, 'T', T, 'Oxygen') # Joule-Thomson coeff in [K/Pa]

    def checkPhase(self, pres, Temp):
        phase           = CP.PhaseSI("T", Temp, "P", pres, "Oxygen")
        if phase == 'liquid' or phase == 'supercritical_liquid':
            print("GOX is", phase, "at this point")

class NitrogenFluid:

    def __init__(self):
        self.gamma      = 1.403
        self.mbar       = CP.PropsSI("molarmass", "Nitrogen")*10**3
        self.R          = Runiv/self.mbar
        self.P_crit     = CP.PropsSI("pcrit", "Nitrogen")
        self.T_crit     = CP.PropsSI("Tcrit", "Nitrogen")
        self.roo_crit   = CP.PropsSI("rhocrit", "Nitrogen")
        self.roo_std    = CP.PropsSI("D", "T", 293, "P", 101325, "Nitrogen")
        
        '''
        # Interpolation of Joule-Thomson coefficients
        
        Pvector   = np.array([101.325, 2026.5, 3394.39, 6079.5, 10132.5, 14185.5, 18238.5, 20265])*1000
        Tvector   = np.array([148.15, 160.65, 173.15, 185.65, 198.15, 223.15, 248.15, 273.15, 298.15, 323.15, 348.15, 373.15])
        
        #Joule-Thomson coeffs from http://www.ddbst.com/en/EED/PCP/JTC_C1056.php
        
        JT_grid     = np.array([[ 8.17, 7.62, 6.78, 4.88, 1.46, 0.52, 0.19, 0.07],
       [ 7.10, 6.56, 5.97, 4.82, 2.34, 0.97, 0.51, 0.37],
       [ 6.20, 5.71, 5.28, 4.37, 2.77, 1.41, 0.79, 0.61],
       [ 5.45, 5.03, 4.67, 3.97, 2.79, 1.65, 0.97, 0.75],
       [ 4.81, 4.48, 4.15, 3.59, 2.64, 1.75, 1.08, 0.83],
       [ 3.79, 3.58, 3.33, 2.96, 2.28, 1.67, 1.14, 0.93],
       [ 3.07, 2.89, 2.74, 2.44, 1.95, 1.49, 1.11, 0.94],
       [ 2.54, 2.39, 2.28, 2.01, 1.64, 1.30, 1.01, 0.88],
       [ 2.12, 1.97, 1.88, 1.67, 1.36, 1.09, 0.86, 0.77],
       [ 1.77, 1.63, 1.55, 1.40, 1.13, 0.90, 0.72, 0.65],
       [ 1.49, 1.36, 1.28, 1.15, 0.92, 0.73, 0.57, 0.53],
       [ 1.24, 1.13, 1.06, 0.94, 0.75, 0.57, 0.45, 0.41]])
       
        self.JT_int  = interpolate.RectBivariateSpline(Tvector, Pvector, JT_grid)    # Note the order of Y and X!!
        # Units are [K/MPa]
        '''
        
    def getDensity(self, P, T):
        density     = CP.PropsSI("D", "T", T, "P", P, "Nitrogen")  # density, kg/m3
        return density
    
    def getViscosity(self, P, T):      
        viscosity   = CP.PropsSI("viscosity", "T", T, "P", P, "Nitrogen") # Dynamic viscosity, Pa*s
        return viscosity
        
    def getJT(self, P, T): 
        #return self.JT_int.ev(Temp, Pressure)/1e6   
        return CP.PropsSI('d(T)/d(P)|Hmass','P',P,'T',T,'Nitrogen') # Joule-Thomson coeff in [K/Pa]
        
        
    
class IdealgasTank:

    def __init__(self, fluid, volume, temp, pressure):
        self.fluid  = fluid
        self.Vtank  = volume
        self.Ttank  = temp
        self.Ptank  = pressure
        self.mass   = pressure*volume/(fluid.R*temp)
        self.density= self.mass/volume
        
    def update(self, mdot_out, timestep):
        deltam_out  = mdot_out*timestep
        gamma       = self.fluid.gamma
        m_new       = self.mass - deltam_out
        T_new       = self.Ttank*(gamma*m_new + (2-gamma)*self.mass)/(gamma*self.mass + (2-gamma)*m_new)
        P_new       = m_new*Runiv*T_new/(self.fluid.mbar*self.Vtank)     
        
        # update parameters
        self.mass   = m_new
        self.Ttank  = T_new
        self.Ptank  = P_new  
        
    def getM(self):
        return self.mass   
        
    def getTtank(self):
        return self.Ttank
         
    def getPtank(self):
        return self.Ptank    
            
class LiquidPropellantTank:

    def __init__(self, pressurant, propellant, tankvolume, proptemp, Tpres, Ppres, FF, Pfeed):
        self.pressurant = pressurant
        self.propellant = propellant
        self.Vtank      = tankvolume                            # total tank volume [m3]
        self.Tprop      = proptemp                              # propellant temperature [K]
        self.Tpres      = Tpres                                 # pressurant temperature [K]
        self.Ptank      = Ppres                                 # pressurant pressure [Pa]
        self.Pfeed      = Pfeed                                 # pressurant feed pressure [Pa]
        
        self.FF         = FF                                    # fill fraction is defined as Vpropellant/Vtank
        self.Vprop      = tankvolume*FF                         # propellant volume [m3]
        self.mProp      = self.Vprop*propellant.getDensity(Ppres, proptemp)         # propellant mass [kg]
        self.Vpres      = tankvolume*(1-FF)                     # pressurant volume [m3]
        self.mPres      = Ppres*self.Vpres/(pressurant.R*Tpres)  # pressurant mass [kg]
    
    # This method is the workhorse that calculates new tank parameters (P,temp etc.) when mass flows in and out
    # and upstream pressurant temperature are known_______________________________
    def update (self, T_in, mdot_pres_in, mdot_prop_out, Pfeed, timestep):
       
        #some parameters
        rooProp         = self.propellant.getDensity(self.Ptank, self.Tprop)          # liquid propellant density [kg/m3]
        gamma           = self.pressurant.gamma                 # pressurant ratio of specific heats
        mbar            = self.pressurant.mbar                  # pressurant molecular mass [kg/kmol]
       
        #calculate new pressurant volume after liquid propellant has been ejected
        Vpres_new       = self.Vpres + mdot_prop_out*timestep/rooProp
        #print("Vpres_old is", self.Vpres, "m3")
        #print("Vpres_new is", Vpres_new, "m3")
        
        #calculate intermediate P and T after pressurant has expanded to this new volume (adiabatic expansion)
        Ppres_int       = self.Ptank*(self.Vpres/Vpres_new)**gamma
        Tpres_int       = self.Tpres*(Ppres_int/self.Ptank)**((gamma-1)/gamma)
        #print("Ppres_old is", self.Ptank/psi, "psi")
        #print("Ppres_int is", Ppres_int/psi, "psi")
        #print("Tpres_int is", Tpres_int, "K")
        
        #print("mdot_pres_in is", mdot_pres_in, "kg/s")
        #print("mdot_prop_out is", mdot_prop_out, "kg/s")
        
        #calculate mass increments/decrements (assuming constant pressure difference over timestep)
        deltam_prop_out = timestep*mdot_prop_out
        deltam_pres_in  = timestep*mdot_pres_in

        #calculate new masses and volumes in the tank
        mPres_new       = self.mPres + deltam_pres_in
        mProp_new       = self.mProp - deltam_prop_out
        Vprop_new       = mProp_new/rooProp
        Vpres_new       = self.Vtank - Vprop_new
        #print("mPres_old is", self.mPres, "kg")
        #print("mPres_new is", mPres_new, "kg")
        
        #...and solve for new temp and pressure
        Tpres_new       = self.mPres/mPres_new*Tpres_int + gamma*T_in*(1 - self.mPres/mPres_new)
        Ppres_new       = Tpres_new*mPres_new*Runiv/(Vpres_new*mbar)
        #print("Tpres_new is", Tpres_new, "K")
        #print("Ppres_new is", Ppres_new/psi, "psi")
        #print("Pfeed is", Pfeed/psi, "psi")
        if Ppres_new > Pfeed:
            print("WARNING: tank pressure higher than feed pressure. Try decreasing time step")
        
           
        #update parameters
        self.Ptank      = Ppres_new
        self.Tpres      = Tpres_new
        self.mPres      = mPres_new
        self.mProp      = mProp_new
        self.Vpres      = Vpres_new
        self.Vprop      = Vprop_new
        
        self.FF         = Vprop_new/self.Vtank
        if self.FF<0:
            print("Warning: FF is <0: FF = ",self.FF)
        
    def getPtank(self):
        return self.Ptank

    def getTpres(self):
        return self.Tpres
    
    def getMpres(self):        
        return self.mPres
    
    def getMprop(self):        
        return self.mProp
        
    def getFF(self):
        return self.FF
        
        
class TwoFluidTank:
    
    def __init__(self, oxidizer, fuel, tankvolume, pistonvolume, temp, FF, mfuel, twoPhase):
        self.oxidizer = oxidizer
        self.fuel = fuel
        self.Vtank = tankvolume                                 # total tank volume [m3]
        self.Vpiston = pistonvolume                             # piston volume  [m3]
        self.Ttank = temp                                       # tank temperature [K]
        self.Ptank = oxidizer.getVaporPressure(self.Ttank)      # tank pressure [Pa]
        self.FFox = FF           # fill fraction is defined as VoxLiq/(V_available_to_oxidizer)
        self.twoPhase = twoPhase # using two phase oxidizer ejection? True = both liq and vapor, False=only liquid
        self.mFuel = mfuel                                      # fuel mass [kg]
        self.Vfuel = self.mFuel/fuel.getDensity()               # fuel volume [m3]
        
        self.mLiquid = (self.Vtank-self.Vfuel-self.Vpiston)*self.FFox*oxidizer.getLiquidDensity(self.Ttank)
        self.mVapor = (self.Vtank-self.Vfuel-self.Vpiston)*(1-self.FFox)*oxidizer.getVaporDensity(self.Ttank)
        self.mOx = self.mLiquid + self.mVapor                   # oxidizer mass [kg]
        
        self.v_ox = (self.Vtank-self.Vfuel-self.Vpiston)/(self.mLiquid + self.mVapor)   # oxidizer specific volume [m3/kg]
        rooL = oxidizer.getLiquidDensity(self.Ttank)            # liq oxidizer density [kg/m3]
        rooV = oxidizer.getVaporDensity(self.Ttank)             # vapor oxidizer density [kg/m3]
        Lent = oxidizer.getLiquidEnthalpy(self.Ttank)           # liq oxidizer enthalpy [J/kg]
        Vent = oxidizer.getVaporEnthalpy(self.Ttank)            # vapor oxidizer enthalpy [J/kg]
        
        self.X_ox = (self.v_ox - 1/rooL)/(1/rooV-1/rooL) # vapor mass fraction m_vap/(m_liq + m_vap)
        if self.X_ox < 1e-9:
            self.X_ox = 0
        self.u_ox = self.X_ox*(Vent - self.Ptank/rooV) + (1-self.X_ox)*(Lent - self.Ptank/rooL) # ox internal energy [J/kg]
        
        self.FFoxtrue = (self.mLiquid/rooL)/self.Vtank          # FFoxtrue is VoxLiq/(Vtank)
    
    def _fluidStateErrors(self, solvedTemp, vnewSolver, unewSolver): 
        rooL = self.oxidizer.getLiquidDensity(solvedTemp)
        rooV = self.oxidizer.getVaporDensity(solvedTemp)
        Venth = self.oxidizer.getVaporEnthalpy(solvedTemp)
        Lenth = self.oxidizer.getLiquidEnthalpy(solvedTemp)
        p = self.oxidizer.getVaporPressure(solvedTemp)
        X = (vnewSolver - 1/rooL)/(1/rooV-1/rooL)
        
        usolved = X*(Venth - p/rooV) + (1-X)*(Lenth - p/rooL)
        return unewSolver - usolved         # error function to be minimized when solving for temp and pressure
    
    # This method is the workhorse that calculates new tank parameters (P,temp etc.) when mass flows in and out
    # are known_________________________________________________________________________________________________
    def update (self, mdot_in, hin, mdot_ox_out, mdot_fuel_out, mdot_vent_out, timestep, i):
        rooL    = self.oxidizer.getLiquidDensity(self.Ttank)
        rooV    = self.oxidizer.getVaporDensity(self.Ttank)
        cL      = self.oxidizer.getLiquidSpecificHeat(self.Ttank)
        L       = self.oxidizer.getLatentHeat(self.Ttank)
        rooFuel = self.fuel.getDensity()
        
        #calculate mass increments/decrements (assuming constant pressure difference over timestep)
           
        deltam_in = timestep*mdot_in
        deltam_ox_out = timestep*mdot_ox_out
        deltam_liq_out = (1-self.X_ox)*deltam_ox_out
        deltam_vap_out = self.X_ox*deltam_ox_out
        deltam_fuel_out = timestep*mdot_fuel_out
        deltam_vent_out = timestep*mdot_vent_out

        #calculate new masses in the tank
        mOx_new = self.mOx + deltam_in - deltam_ox_out - deltam_vent_out
        mFuel_new = self.mFuel - deltam_fuel_out
        Vfuel_new = (mFuel_new)/rooFuel
        
        #using first law of thermodynamixs, calculate new oxidizer internal energy (u)...
        if self.twoPhase == True:                                   # ejecting both liquid and vapor ox
            unew_ox = (self.mOx*self.u_ox + deltam_in*hin -\
                deltam_liq_out*self.oxidizer.getLiquidEnthalpy(self.Ttank) -\
                deltam_vap_out*self.oxidizer.getVaporEnthalpy(self.Ttank) -\
                self.Ptank*(deltam_fuel_out/rooFuel + deltam_liq_out/rooL)) /(mOx_new)
        else:                                                       # ejecting liquid ox, venting vapor ox
            unew_ox = (self.mOx*self.u_ox + deltam_in*hin -\
                deltam_ox_out*self.oxidizer.getLiquidEnthalpy(self.Ttank) -\
                deltam_vent_out*self.oxidizer.getVaporEnthalpy(self.Ttank) -\
                self.Ptank*(deltam_fuel_out/rooFuel + deltam_ox_out/rooL + deltam_vent_out/rooV)) /(mOx_new)
        #...and specific volume (v):
        vnew_ox = (self.Vtank-Vfuel_new-self.Vpiston)/(mOx_new)
        
        #...and solve for new pressure and temp
        tempGuess = self.Ttank
        solvedTempArray = opt.fsolve(self._fluidStateErrors, tempGuess, args=(vnew_ox, unew_ox))
        solvedTankTemp = solvedTempArray[0]
        
        #update parameters
        self.u_ox = unew_ox
        self.v_ox = vnew_ox
        self.Ttank = solvedTankTemp
        self.Ptank = self.oxidizer.getVaporPressure(solvedTankTemp)
        self.mOx =  mOx_new
        self.mFuel = mFuel_new
        self.Vfuel = Vfuel_new
        rooLnew = self.oxidizer.getLiquidDensity(solvedTankTemp)
        rooVnew = self.oxidizer.getVaporDensity(solvedTankTemp)
        self.X_ox = (vnew_ox-1/rooLnew)/( 1/rooVnew - 1/rooLnew )
        if self.X_ox<0:
            print("Warning: X is <0: X = ",self.X_ox)
        self.FFox = mOx_new*(1- self.X_ox)/(rooLnew*(self.Vtank - Vfuel_new -self.Vpiston))
        self.FFoxtrue =  mOx_new*(1- self.X_ox)/(rooLnew*(self.Vtank)) 
        
    def getPtank(self):
        return self.Ptank

    def getTtank(self):
        return self.Ttank
    '''
    def getMdot_ox(self, oxidizer, Pout):
        
        rooL = self.oxidizer.getLiquidDensity(self.Ttank)
        rooV = self.oxidizer.getVaporDensity(self.Ttank)
        cL = self.oxidizer.getLiquidSpecificHeat(self.Ttank)
        L = self.oxidizer.getLatentHeat(self.Ttank)
        k = self.oxidizer.gamma
        Mw = self.oxidizer.Mw
        x = self.X_ox
        
        if self.realOri == True and self.twoPhase == True:
            mdot_ox_out = self.ox_orifice.getMdot(self.Ptank, Pout, rooL, rooV, cL, L, self.Ttank, k, Mw, x)
        elif self.realOri == True and self.twoPhase == False:
            mdot_ox_out = self.ox_orifice.getMdot(self.Ptank, Pout, rooL, rooV, cL, L, self.Ttank)
        else:    
            mdot_ox_out = self.ox_orifice.getMdot(self.Ptank, Pout, rooL)
        
        return mdot_ox_out
    
    def getMdot_fuel(self, fuel, Pout):
        mdot_fuel_out = self.fuel_orifice.getMdot(self.Ptank, Pout, self.fuel.getDensity() )
        return mdot_fuel_out    
    '''
    def getMox(self):        
        return self.mOx   
    
    def getMliquid(self):        
        return self.mOx*(1-self.X_ox)
        
    def getMvapour(self):        
        return self.mOx*self.X_ox    
        
    def getMfuel(self):        
        return self.mFuel
    
    def getFFox(self):
        return self.FFox
        
    def getFFoxtrue(self):
        return self.FFoxtrue
        
        
class TwoFluidTank2:
    
    def __init__(self, ox_orifice, fuel_orifice, oxidizer, fuel, tankvolume, temp, FF, mfuel, realOri, twoPhase):
        self.ox_orifice =  ox_orifice
        self.fuel_orifice = fuel_orifice
        self.oxidizer = oxidizer
        self.fuel = fuel
        self.Vtank = tankvolume
        self.Ttank = temp
        self.Ptank = oxidizer.getVaporPressure(self.Ttank)
        self.FFox = FF  #fill fraction is defined as VoxLiq/(V_available_to_oxidizer)
        self.realOri = realOri #using real (Fauske) orifices? True/false
        self.twoPhase = twoPhase #using two phase oxidizer ejection? True/False
        self.mFuel = mfuel
        if mfuel !=0:
            self.fueltank = True
        else:
            self.fueltank = False
        self.Vfuel = self.mFuel/fuel.getDensity()
        
        self.mLiquid = (self.Vtank-self.Vfuel)*self.FFox*oxidizer.getLiquidDensity(self.Ttank)
        self.mVapor = (self.Vtank-self.Vfuel)*(1-self.FFox)*oxidizer.getVaporDensity(self.Ttank)
        self.mOx = self.mLiquid + self.mVapor
        
        self.v_ox = (self.Vtank-self.Vfuel)/(self.mLiquid + self.mVapor)
        rooL = oxidizer.getLiquidDensity(self.Ttank)
        rooV = oxidizer.getVaporDensity(self.Ttank)
        Vent = oxidizer.getVaporEnthalpy(self.Ttank)
        Lent = oxidizer.getLiquidEnthalpy(self.Ttank)
        self.X_ox = (self.v_ox - 1/rooL)/(1/rooV-1/rooL) #vapor mass fraction
        self.u_ox = self.X_ox*(Vent - self.Ptank/rooV) + (1-self.X_ox)*(Lent - self.Ptank/rooL)
        self.FFoxtrue = (self.mLiquid/rooL)/self.Vtank #FFoxtrue is VoxLiq/(Vtank)
        
        self.mdot_oxflow_out = [0]
        self.mdot_liqflow_out = [0]
        self.mdot_fuelflow_out = [0]
    
    def _fluidStateErrors(self, solvedTemp, vnewSolver, unewSolver):
        rooL = self.oxidizer.getLiquidDensity(solvedTemp)
        rooV = self.oxidizer.getVaporDensity(solvedTemp)
        Venth = self.oxidizer.getVaporEnthalpy(solvedTemp)
        Lenth = self.oxidizer.getLiquidEnthalpy(solvedTemp)
        p = self.oxidizer.getVaporPressure(solvedTemp)
        X = (unewSolver-Lenth + p/rooL)/(Venth-p/rooV-Lenth+p/rooL)
        
        vsolved = X*(1/rooV - 1/rooL) + 1/rooL
        
        return vnewSolver - vsolved
        
    
    def adamsIntegrator(self, timestep, rate_array_in):
        #print("ratearray is", rate_array_in)
        if len(rate_array_in) == 1:
            delta = timestep/2*(3*rate_array_in[0] - 0)
        else:
            delta = timestep/2*(3*rate_array_in[-1] - rate_array_in[-2])
        #print("delta is", delta)
        return delta
    
    def update (self, mdot_in_array, hin, Pout, timestep, i):
        rooL = self.oxidizer.getLiquidDensity(self.Ttank)
        rooV = self.oxidizer.getVaporDensity(self.Ttank)
        cL = self.oxidizer.getLiquidSpecificHeat(self.Ttank)
        L = self.oxidizer.getLatentHeat(self.Ttank)
        rooFuel = self.fuel.getDensity()
        
        #calculate mass flow rate (assuming constant pressure difference over timestep)
        if self.realOri == True and self.twoPhase == True:
            mdot_ox_out = self.ox_orifice.getMdot(self.getPtank(), Pout, rooL, rooV, cL, L, self.Ttank,
            self.oxidizer.gamma, self.oxidizer.Mw, self.X_ox)
        elif self.realOri == True and self.twoPhase == False:
            mdot_ox_out = self.ox_orifice.getMdot(self.getPtank(), Pout, rooL, rooV, cL, L, self.Ttank)
        else:  
            mdot_ox_out = self.ox_orifice.getMdot(self.getPtank(), Pout, rooL)
            
        if self.fueltank == True:
            mdot_fuel_out = self.fuel_orifice.getMdot(self.getPtank(), Pout, rooFuel)
        else:
            mdot_fuel_out = 0.    
        
        if i == 0:
            self.mdot_oxflow_out[-1] = mdot_ox_out   
            self.mdot_fuelflow_out[-1] = mdot_fuel_out
        else:
            self.mdot_oxflow_out.append(mdot_ox_out)
            self.mdot_fuelflow_out.append(mdot_fuel_out)
        
        #integrate mass flows to determine new masses
        deltam_in = self.adamsIntegrator(timestep, mdot_in_array)
        deltam_ox_out = self.adamsIntegrator(timestep, self.mdot_oxflow_out)
        deltam_liq_out = (1-self.X_ox)*deltam_ox_out
        deltam_vap_out = self.X_ox*deltam_ox_out
        deltam_fuel_out = self.adamsIntegrator(timestep, self.mdot_fuelflow_out)
        
        #calculate new masses in the tank          
        mOx_new = self.mOx + deltam_in - deltam_ox_out
        mFuel_new = self.mFuel - deltam_fuel_out
        Vfuel_new = (mFuel_new)/rooFuel
        Vox_new = mOx_new*(1-self.X_ox)/rooL + mOx_new*self.X_ox/rooV
        print(i, self.Vtank-Vox_new)
        '''
        if Vox_new > self.Vtank:
           mOx_new = self.Vtank / ((1-self.X_ox)/rooL + self.X_ox/rooV) 
           Vox_new = mOx_new*(1-self.X_ox)/rooL + mOx_new*self.X_ox/rooV
           print(i, self.Vtank-Vox_new)
           '''
        
        #calculate new internal energy and specific volume...
        if self.twoPhase == True:
            unew_ox = (self.mOx*self.u_ox + deltam_in*hin -\
                deltam_liq_out*self.oxidizer.getLiquidEnthalpy(self.Ttank) -\
                deltam_vap_out*self.oxidizer.getVaporEnthalpy(self.Ttank) -\
                self.Ptank*(deltam_fuel_out/rooFuel + deltam_liq_out/rooL)) /(mOx_new)
        else:
            unew_ox = (self.mOx*self.u_ox + deltam_in*hin -\
                deltam_ox_out*self.oxidizer.getLiquidEnthalpy(self.Ttank) -\
                self.Ptank*(deltam_fuel_out/rooFuel + deltam_ox_out/rooL)) /(mOx_new)\
                
        vnew_ox = (self.Vtank-Vfuel_new)/(mOx_new)
        
        #...and solve for new pressure and temp
        tempGuess = self.Ttank
        solvedTempArray = opt.newton(self._fluidStateErrors, tempGuess, args=(vnew_ox, unew_ox))
        solvedTankTemp = solvedTempArray
        #print ('solvedTemp is %fK'%(solvedTankTemp))
        
        #update parameters
        self.u_ox = unew_ox
        self.v_ox = vnew_ox
        self.Ttank = solvedTankTemp
        self.Ptank = self.oxidizer.getVaporPressure(solvedTankTemp)
        self.mOx =  mOx_new
        self.mFuel = mFuel_new
        self.Vfuel = Vfuel_new
        rooLnew = self.oxidizer.getLiquidDensity(solvedTankTemp)
        rooVnew = self.oxidizer.getVaporDensity(solvedTankTemp)
        self.X_ox = (vnew_ox-1/rooLnew)/( 1/rooVnew - 1/rooLnew )
        self.FFox = mOx_new*(1- self.X_ox)/(rooLnew*(self.Vtank - Vfuel_new))
        self.FFoxtrue =  mOx_new*(1- self.X_ox)/(rooLnew*(self.Vtank))
        
        if self.X_ox<0:
            print("X is",self.X_ox)
    
    def getPtank(self):
        return self.Ptank

    def getTtank(self):
        return self.Ttank
    
    def getMdot_ox(self, oxidizer, Pout):
        
        rooL = self.oxidizer.getLiquidDensity(self.Ttank)
        rooV = self.oxidizer.getVaporDensity(self.Ttank)
        cL = self.oxidizer.getLiquidSpecificHeat(self.Ttank)
        L = self.oxidizer.getLatentHeat(self.Ttank)
        k = self.oxidizer.gamma
        Mw = self.oxidizer.Mw
        x = self.X_ox
        
        if self.realOri == True and self.twoPhase == True:
            mdot_ox_out = self.ox_orifice.getMdot(self.Ptank, Pout, rooL, rooV,cL,L,self.Ttank, k, Mw, x)
        elif self.realOri == True and self.twoPhase == False:
            mdot_ox_out = self.ox_orifice.getMdot(self.Ptank, Pout, rooL, rooV, cL, L, self.Ttank)
        else:    
            mdot_ox_out = self.ox_orifice.getMdot(self.Ptank, Pout, rooL)
        
        return mdot_ox_out
        
    def getMdot_fuel(self, fuel, Pout):
        mdot_fuel_out = self.fuel_orifice.getMdot(self.Ptank, Pout, self.fuel.getDensity() )
        return mdot_fuel_out    
        
    def getMox(self):        
        return self.mOx   
    
    def getMliquid(self):        
        return self.mOx*(1-self.X_ox)
        
    def getMvapour(self):        
        return self.mOx*self.X_ox    
        
    def getMfuel(self):        
        return self.mFuel
    
    def getFFox(self):
        return self.FFox
        
    def getFFoxtrue(self):
        return self.FFoxtrue
        