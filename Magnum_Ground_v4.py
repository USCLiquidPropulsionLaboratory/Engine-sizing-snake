## LOX-IPA sim
#@ Author Juha Nieminen

#import sys
#sys.path.insert(0, '/Users/juhanieminen/Documents/adamrocket')

import RocketComponents as rc
from physical_constants import poise, inches, Runiv, gallons, lbm, \
    gearth, atm, psi, lbf
from numpy import pi, linspace, cos, radians, sqrt, exp, log, array, full, ceil, round
from scipy import optimize as opt
import matplotlib.pyplot as plt
from matplotlib import collections as mc
import Flows1D as flows


#DESIGN VARIABLES____________________________________________________________________________________


# nominal parameters
Preg_ox_start       = 470*psi         # regulated pressurant outlet pressure [Pa]
Preg_fu_start       = 550*psi         # regulated pressurant outlet pressure [Pa]

mdot_fuel_nom       = 0.2             # This is only for cooling jacket pressure drop purposes [kg/s]
Pdrop_jacket_nom    = 10*psi          # Cooling jacket pressure drop at mdot_nominal [Pa]
OF_nom              = 1.2             # Oxidizer-to-fuel ratio. This has only effect on initial guesses during solving

# Propellant and pressurant tanks dimensions
Vfueltank       = 0.15              # fuel tank volume [m3]
Voxtank         = 0.168             # ox tank volume [m3]
Vfuelprestank   = 0.142             # fuel pressurant tank volume [m3]
Voxprestank     = 0.142             # ox pressurant tank volume [m3]

# Vent orifices
d_oxtank_vent   = 0.00*inches       # [m]
d_fueltank_vent = 0.00*inches       # [m]
fuel_vent_cd    = 0.7               # fuel vent discharge coefficient
ox_vent_cd      = 0.7               # ox vent discharge coefficient

# Tubing
d_presox_tube   = 1.0*inches        # pressurant tank -> ox tank tube diameter [m]
L_presox_tube   = 0.6               # pressurant tank -> ox tank tube length [m]
d_presfuel_tube = 1.0*inches        # pressurant tank -> fuel tank tube diameter [m]
L_presfuel_tube = 0.6               # pressurant tank -> fuel tank tube length [m]
d_oxtube        = 1.0*inches        # ox tank -> manifold tube diameter [m]
L_oxtube        = 2.4               # ox tank -> manifold tube length [m]
d_fueltube      = 1.0*inches        # fuel tank -> manifold tube diameter [m]
L_fueltube      = 2.4               # fuel tank -> manifold tube length [m]

roughness       = 0.005             # epsilon/diameter, dimensionless

# Valves 
Cv_NC1_F       = 9          # fuel solenoid valve flow coefficient, dimensionless
Cv_NC1_O       = 9          # oxidizer solenoid valve flow coefficient, dimensionless
Cv_NC1_FP      = 9          # fuel pressurant solenoid valve flow coefficient, dimensionless
Cv_NC1_OP      = 9          # ox pressurant solenoid valve flow coefficient, dimensionless
Cv_CV1_FP      = 1.7        # fuel pressurant check valve no.1 flow coefficient, dimensionless
Cv_CV2_FP      = 1.7        # fuel pressurant check valve no.2 flow coefficient, dimensionless
Cv_CV1_OP      = 1.7        # ox pressurant check valve no.1 flow coefficient, dimensionless
Cv_CV2_OP      = 1.7        # ox pressurant check valve no.2 flow coefficient, dimensionless
Pcrack_CV1_FP  = 10*psi     # fuel pressurant check valve no.1 opening pressure [Pa]
Pcrack_CV2_FP  = 10*psi     # fuel pressurant check valve no.2 opening pressure [Pa]
Pcrack_CV1_OP  = 10*psi     # ox pressurant check valve no.1 opening pressure [Pa]
Pcrack_CV2_OP  = 10*psi     # ox pressurant check valve no.2 opening pressure [Pa]

# Pintle injector (Note that there is only CFD data for OD_shaft = [14mm,16mm] )

# Fuel side
d_in_fu         = 0.5*inches                                # injector inlet diameter, m
d_mani_fu       = 0.03                                      # manifold/converging section inlet diameter, m
ID_shaft        = 0.01                                      # fuel annulus ID, m
OD_shaft        = 0.015   #see note above                   # fuel annulus OD, m
L_shaft_fu      = 0.075                                     # Length of annular flow path, m
r_tip           = 0.0105                                    # pintle tip radius, m
A_fu_annulus    = pi/4*(OD_shaft**2 - ID_shaft**2)          # annulus cross section, m^2
h_exit          = A_fu_annulus/(2*pi*r_tip)                 # pintle exit slot height, m
Dh_fu           = OD_shaft - ID_shaft                       # annulus hydraulic diameter, m
rou_fu          = 2e-6/Dh_fu                                # annulus _DIMENSIONLESS_ roughness]
# Ox side
d_in_ox         = 0.5*inches                                # injector inlet diameter (two inlets!)
d_mani_ox       = 0.08                                      # ox manifold diameter upstream from orifice plate, m
Nori            = 8                                         # number of orifices in orifice plate
d_ori           = 3.5e-3                                    # single orifice diameter, m
cd_ori          = 0.7                                       # orifice discharge coefficient in orifice plate, dimensionless
OD_ann          = 0.0227                                    # converging section end/annulus outer diameter, m
ID_ann          = 0.019                                     # central shaft outer diameter, m
L_shaft_ox      = 0.01                                      # Length of annular flow path, m
Dh_ox           = OD_ann - ID_ann                           # annulus hydraulic diameter, m
rou_ox          = 2e-6/Dh_ox                                # annulus _DIMENSIONLESS_ roughness


# Define initial/nominal conditions in the chamber (obtained from CEA code assuming OFratio = 1.2)
TfireInit       = 293                                       # initial flame temperature [K]
Tfire_nom       = 2675                                      # nominal flame temperature [K]
Pfire           = 1*atm                                     # initial chamber pressure [Pa]
gammaFireInit   = 1.16                                      # dimensionless
ga              = gammaFireInit
mbarFireInit    = 19.10                                     # combustion products' initial molecular mass [kg/kmol]
RfireInit       = Runiv/mbarFireInit                        # combustion products' initial specific gas constant [J/kgK]
Pambient        = atm                                       # ambient pressure [Pa]

# Nozzle and chamber

d_nozzleThroat  = 0.0604                                                # throat diameter [m]
A_nozzleThroat  = pi*d_nozzleThroat**2/4                                # throat area [m2] 
area_ratio      = 3.947                                                 # nozzle exit-to-throat area ratio
A_nozzleExit    = area_ratio*A_nozzleThroat                             # nozzle exit area [m2]
d_nozzleExit    = sqrt(4*A_nozzleExit/pi)                               # nozzle exit diameter [m]
            
Dchamber        = 0.1205                                                # chamber diameter [m]
Achamber        = pi*Dchamber**2/4                                      # chamber cross sectional area [m2]
Lchamber        = 0.2763                                                # chamber length [m]
Vchamber        = Achamber*Lchamber                                     # chamber volume [m3] 
Lstar           = Vchamber/A_nozzleThroat                               # chamber characteristic length [m]
Mc_nom          = flows.getIsentropicMs(A_nozzleThroat, Achamber, gammaFireInit)[0] # nominal chamber Mach number

print("throat diameter is", '%.1f'%(d_nozzleThroat*1000), 'mm')
print("exit diameter is", '%.1f'%(d_nozzleExit*1000), 'mm')
print("chamber volume is", '%.5f'%Vchamber, "m3")
print("chamber Lstar is", '%.2f'%Lstar, "m")
print("chamber Mach_nom is", '%.2f'%Mc_nom)

# INITIAL CONDITIONS____________________________________________________________________________________________

#Define initial conditions in the tanks

TfuelpresStart      = 293               # Fuel pressurant temp [K]
PfuelprestankStart  = 4000*psi          # Fuel pressurant tank pressure [Pa]

ToxpresStart        = 293               # Ox pressurant temp [K]
PoxprestankStart    = 4000*psi          # Ox pressurant tank pressure [Pa]

ToxStart        = 90                    # Oxidizer (LOX) temp [K] 
PoxtankStart    = Preg_ox_start -1*psi  # Oxidizer tank pressure [Pa] (-1psi helps convergence on first timestep)
FFoxtankStart   = 0.65                  # Oxidizer tank fill fraction, dimensionless

TfuelStart      = 293                   # Fuel temp [K]
PfueltankStart  = Preg_fu_start -1*psi        # Fuel tank pressure [Pa] (-1psi helps convergence on first timestep)
FFfueltankStart = 0.67                  # Fuel tank fill fraction (Vfuel/Vtank)

# initialize propellants
IPA             = rc.IPAFluid()
LOX             = rc.LOXFluid()
FuelPres        = rc.NitrogenFluid()
OxPres          = rc.HeFluid()

#initialize nozzle and chamber
nozzle          = rc.ConvergingDivergingNozzle(A_nozzleExit, A_nozzleThroat)
chamber         = rc.LOX_IPACombustionChamber(nozzle, Vchamber, TfireInit, ga, mbarFireInit, Pfire, atm)

#initialize injector
fuel_pintle     = rc.MagnumFuelPintle(d_in_fu, d_mani_fu, ID_shaft, OD_shaft, L_shaft_fu, r_tip, h_exit, rou_fu)
ox_pintle       = rc.MagnumOxPintle(d_in_ox, d_mani_ox, d_ori, OD_ann, ID_ann, L_shaft_ox, Nori, cd_ori, rou_ox) 

#initialize pressurant tanks
oxprestank      = rc.IdealgasTank(OxPres, Voxprestank, ToxpresStart, PoxprestankStart)
fuelprestank    = rc.IdealgasTank(FuelPres, Vfuelprestank, TfuelpresStart, PfuelprestankStart)

#initialize propellant tanks
oxtank          = rc.LiquidPropellantTank(OxPres, LOX, Voxtank, ToxStart, ToxpresStart,\
                    PoxtankStart, FFoxtankStart, Preg_ox_start)
fueltank        = rc.LiquidPropellantTank(FuelPres, IPA, Vfueltank, TfuelStart, TfuelpresStart,\
                    PfueltankStart, FFfueltankStart, Preg_fu_start)

#initialize vent holes
fuelVent        = rc.VentHole(d_fueltank_vent, FuelPres.gamma, Runiv/FuelPres.mbar, fuel_vent_cd)
oxVent          = rc.VentHole(d_oxtank_vent, OxPres.gamma, Runiv/OxPres.mbar, ox_vent_cd)

#initialize solenoids
NC1_O           = rc.IncompressibleFlowSolenoid( Cv_NC1_O)
NC1_F           = rc.IncompressibleFlowSolenoid( Cv_NC1_F)
NC1_OP          = rc.CompressibleFlowSolenoid( Cv_NC1_OP, OxPres)
NC1_FP          = rc.CompressibleFlowSolenoid( Cv_NC1_FP, FuelPres)

#initialize check valves
CV1_FP          = rc.CompressibleFlowCheckValve( Cv_CV1_FP, Pcrack_CV1_FP, FuelPres)
CV2_FP          = rc.CompressibleFlowCheckValve( Cv_CV2_FP, Pcrack_CV2_FP, FuelPres)
CV1_OP          = rc.CompressibleFlowCheckValve( Cv_CV1_OP, Pcrack_CV1_OP, OxPres)
CV2_OP          = rc.CompressibleFlowCheckValve( Cv_CV2_OP, Pcrack_CV2_OP, OxPres)

#initialize flow meter
FM1_F           = rc.IncompressibleFlowMeter(10.5*psi)
FM1_O           = rc.IncompressibleFlowMeter(10.5*psi)

#initialize particle filter
PF1_F           = rc.IncompressibleFlowParticleFilter(10*psi)
PF1_O           = rc.IncompressibleFlowParticleFilter(10*psi)

#initialize tubing
ox_tube         = rc.RoughStraightCylindricalTube(d_oxtube, L_oxtube, roughness, True)
fuel_tube       = rc.RoughStraightCylindricalTube(d_fueltube, L_fueltube, roughness, True)
presox_tube     = rc.RoughStraightCylindricalTube(d_presox_tube, L_presox_tube, roughness, True)
presfuel_tube   = rc.RoughStraightCylindricalTube(d_presfuel_tube, L_presfuel_tube, roughness, True)

#initialize cooling jacket
jacket          = rc.CoolingJacket(mdot_fuel_nom, Pdrop_jacket_nom)

#initialize arrays for various data time histories
T_chamber       = [chamber.T]                       # combustion chamber temperature [K]
Pchamber        = [chamber.get_P_inlet()]           # combustion chamber pressure [Pa]
Pexit           = [nozzle.getPe(Pchamber[0], gammaFireInit, Pambient)] # nozzle exit pressure [Pa]
Mexit           = [nozzle.getMe(Pchamber[0], gammaFireInit, Pambient)] # nozzle exit Mach number
cmass           = [chamber.m]                       # resident propellant mass in combustion chamber [kg]
mdot_nozzle     = [nozzle.getmdot(gammaFireInit, RfireInit, chamber.get_P_inlet(), chamber.T, chamber.Pa)] # mass flow out of the nozzle [kg/s]

Poxtank         = [oxtank.getPtank()]               # ox tank pressure [Pa]
Toxtank         = [oxtank.getTpres()]               # pressurant temperature in ox tank [K]
mPresOxtank     = [oxtank.getMpres()]               # pressurant mass in ox tank [kg]
mox             = [oxtank.getMprop()]               # oxidizer mass in tank [kg]
FFoxtank        = [oxtank.getFF()]                  # ox tank fill fraction defined as Vox/(Voxtank)

Pfueltank       = [fueltank.getPtank()]             # fuel tank pressure [Pa]
Tfueltank       = [fueltank.getTpres()]             # pressurant temperature in fuel tank[K]
mPresFueltank   = [fueltank.getMpres()]             # pressurant mass in fuel tank [kg]
mfuel           = [fueltank.getMprop()]             # fuel mass in tank [kg]
FFfueltank      = [fueltank.getFF()]                # fuel tank fill fraction defined as Vfuel/(Vfueltank)

Toxprestank     = [oxprestank.getTtank()]           # temperature in ox pressurant tank [K]
Poxprestank     = [oxprestank.getPtank()]           # pressure in ox pressurant tank [Pa]
moxprestank     = [oxprestank.getM()]               # pressurant mass in ox pressurant tank [Pa]

Tfuelprestank   = [fuelprestank.getTtank()]         # temperature in fuel pressurant tank [K]
Pfuelprestank   = [fuelprestank.getPtank()]         # pressure in fuel pressurant tank [Pa]
mfuelprestank   = [fuelprestank.getM()]             # pressurant mass in fuel pressurant tank [Pa]

time            = [0]                               # time array [s]

mdot_ox         = [0]                               # liq ox mass flow out of the tank [kg/s]
rooOx           = oxtank.propellant.getDensity(PoxtankStart, ToxStart)     # liq ox density, assumed constant [kg/m^3]
P2ox            = [0]                               # ox tank presssure [Pa]
P3ox            = [0]                               # ox flow meter outlet pressure [Pa]
P4ox            = [0]                               # ox solenoid outlet pressure [Pa]  
P5ox            = [0]                               # ox particle filter outlet pressure [Pa]
P6ox            = [0]                               # ox injector inlet pressure [Pa]                             

mdot_fuel       = [0]                               # fuel mass flow out of the tank [kg/s]
rooFuel         = fueltank.propellant.density       # fuel density, assumed constant [kg/m3]
P2fuel          = [0]                               # fuel tank presssure [Pa]
P3fuel          = [0]                               # fuel flow meter outlet pressure [Pa]
P4fuel          = [0]                               # fuel solenoid outlet pressure [Pa] 
P5fuel          = [0]                               # fuel particle filter outlet pressure [Pa]     
P6fuel          = [0]                               # fuel cooling jacket inlet pressure [Pa]
P7fuel          = [0]                               # fuel injector inlet pressure [Pa]

mdot_ox_pres    = [0]                               # ox pressurant mass flow rate [kg/s]
P1pres_ox       = [0]                               # ox pressurant pressure at filter outlet [kg/s]
P2pres_ox       = [Preg_ox_start]                   # ox pressurant pressure at regulator outlet [kg/s]
P3pres_ox       = [0]                               # ox pressurant pressure at solenoid valve outlet [kg/s]
P4pres_ox       = [0]                               # ox pressurant pressure at check valve no.1 outlet [kg/s]
P5pres_ox       = [0]                               # ox pressurant pressure at check valve no.2 outlet [kg/s]

mdot_fuel_pres  = [0]                               # fuel pressurant mass flow rate [kg/s]
P1pres_fuel     = [0]                               # fuel pressurant pressure at filter outlet [kg/s]
P2pres_fuel     = [Preg_fu_start]                   # fuel pressurant pressure at regulator outlet [kg/s]
P3pres_fuel     = [0]                               # fuel pressurant pressure at solenoid valve outlet [kg/s]
P4pres_fuel     = [0]                               # fuel pressurant pressure at check valve no.1 outlet [kg/s]
P5pres_fuel     = [0]                               # fuel pressurant pressure at check valve no.2 outlet [kg/s]

mTotal          = [0]                               # propellant mass in the system [kg]
moxpres         = [moxprestank[0] + mPresOxtank[0]]     # ox pressurant mass [kg]
mfuelpres       = [mfuelprestank[0] + mPresFueltank[0]] # fuel pressurant mass [kg]
OFratio         = [0]                               # oxidizer to fuel mass flow ratio 
Isp             = [0]                               # specific impulse [s]
Thrust          = [nozzle.getThrust(chamber.get_P_inlet(), Pambient, gammaFireInit) ] # rocket thrust [N]

#SIMULATE_______________________________________________________________________________________________________
# using orifices as follows: ejecting GOX from manifold to chamber, fuel liq-to-liq from manifold to chamber

print("")
print("STARTING SIM...")
print("")
print("mOxStart is", '%.2f'%mox[0], "kg")
print("mIPAstart is", mfuel[0], "kg")
print("m_pres_fuel_start is", '%.2f'%mfuelprestank[0], "kg")
print("m_pres_ox_start is", '%.2f'%moxprestank[0], "kg")

# The first step is to solve oxidizer and fuel mass flow rates from the tank to combustion chamber.

# definitions:

# P1ox              = ox pressurant tank pressure
# P2ox              = ox tank pressure
# P3ox              = ox flow meter outlet pressure
# P4ox              = ox solenoid valve outlet pressure
# P5ox              = ox particle filter outlet pressure
# P6ox              = ox injector inlet pressure 
# (P1ox-P2ox)       = regulator+tubing pressure drop,           ox pressurant flow, is solved separately from fuel flow 
# (P2ox-P3ox)       = ox flow meter pressure drop,              eq 1
# (P3ox-P4ox)       = ox solenoid valve pressure drop,          eq 2
# (P4ox-P5ox)       = ox particle filter pressure drop,         eq 3
# (P5ox-P6ox)       = ox tubing pressure drop,                  eq 4
# (P6ox-Pchamber)   = ox injector pressure drop,                eq 5

# P1fuel            = fuel pressurant tank pressure
# P2fuel            = fuel tank pressure
# P3fuel            = fuel flow meter outlet pressure
# P4fuel            = fuel solenoid valve outlet pressure
# P5fuel            = fuel particle filter outlet pressure
# P6fuel            = fuel cooling jacket inlet pressure
# P7fuel            = fuel injector inlet pressure
# (P1fuel-P2fuel)   = regulator+tubing pressure drop,           fuel pressurant flow, is solved separately from fuel flow 
# (P2fuel-P3fuel)   = fuel flow meter pressure drop             eq 1
# (P3fuel-P4fuel)   = fuel solenoid valve pressure drop,        eq 2
# (P4fuel-P5fuel)   = fuel particle filter pressure drop,       eq 3
# (P5fuel-P6fuel)   = fuel tubing pressure drop,                eq 4
# (P6fuel-P7fuel)   = cooling jacket pressure drop,             eq 5
# (P7fuel-Pchamber) = injector pressure drop,                   eq 6

# P1pres_ox             = ox pressurant particle filter outlet pressure
# P2pres_ox             = ox pressurant regulator outlet pressure
# P3pres_ox             = ox pressurant solenoid valve outlet pressure
# P4pres_ox             = ox pressurant check valve no.1 outlet pressure
# P5pres_ox             = ox pressurant check valve no.2 outlet pressure
# P6pres_ox             = ox pressurant tubing outlet = ox tank pressure
# (P1pres_ox-P2pres_ox) = ox pressurant regulator pressure drop
# (P2pres_ox-P3pres_ox) = ox pressurant solenoid valve pressure drop
# (P3pres_ox-P4pres_ox) = ox pressurant check valve no.1 pressure drop
# (P4pres_ox-P5pres_ox) = ox pressurant check valve no.2 pressure drop
# (P5pres_ox-P6pres_ox) = ox pressurant tubing pressure drop


# P1pres_fuel               = fuel pressurant particle filter outlet pressure
# P2pres_fuel               = fuel pressurant regulator outlet pressure
# P3pres_fuel               = fuel pressurant solenoid valve outlet pressure
# P4pres_fuel               = fuel pressurant check valve no.1 outlet pressure
# P5pres_fuel               = fuel pressurant check valve no.2 outlet pressure
# P6pres_fuel               = fuel pressurant tubing outlet = fuel tank pressure
# (P1pres_fuel-P2pres_fuel) = fuel pressurant regulator pressure drop
# (P2pres_fuel-P3pres_fuel) = fuel pressurant solenoid valve pressure drop
# (P3pres_fuel-P4pres_fuel) = fuel pressurant check valve no.1 pressure drop
# (P4pres_fuel-P5pres_fuel) = fuel pressurant check valve no.2 pressure drop
# (P5pres_fuel-P6pres_fuel) = fuel pressurant tubing pressure drop

# In the case of oxidizer, P2 and Pchamber are known, so one must solve for P3, P4, P5 & P6. Third unknown is the mass flow rate. The three equations are injector and tubing pressure drops, and expression for solenoid mass flow rate. They are defined in RocketComponents.py under their respective classes.

# With fuel P2 and Pchamber are known, so one must solve for P3, P4, P5, P6 & P7. Fourth unknown is mass flow rate.

# With ox pressurant, P2 (regulation pressure) and P6 (fuel tank pressure) are known, so one must solve for P3, P4 and P5. The fourth unknown is pressurant mass flow rate. Equations to be solved are pressure drops over the check valve, solenoid valve, and the tubing.

# With fuel pressurant, P2 (regulation pressure) and P6 (fuel tank pressure) are known, so one must solve for P3, P4 and P5. The fourth unknown is pressurant mass flow rate. Equations to be solved are pressure drops over the check valve, solenoid valve, and the tubing.

# fsolve requires sensible initial guesses for all unknowns. They are established by guessing the mass flow rate, because all other pressures trickle down from that.

#time                = [0]

timestep_small      = 5e-6      # seconds, used during initial transient
timestep_nom        = 0.0001    # seconds, used after 0.01 seconds of simulation time
t_transient         = 0.01      # seconds, estimated time of initial transient

t_simulation        = 20        # seconds

if t_simulation <= t_transient:
    simsteps    = int(ceil(t_simulation/timestep_small))
else:
    simsteps    = int(ceil( t_transient/timestep_small + (t_simulation-t_transient)/timestep_nom ))

print("Sim time is", t_simulation, "s, number of simsteps is", simsteps)

oxErrorInt = 0
fuelErrorInt = 0

i=0
for i in range(0,simsteps):
    if time[i] < 0.01:
        timestep    = timestep_small                           # use shorter timestep during initial transient
    else: 
        timestep    = timestep_nom 

    P1ox            = Poxprestank[i]
    P2ox            = Poxtank[i]
    P1fuel          = Pfuelprestank[i]
    P2fuel          = Pfueltank[i]
    Pchamb          = Pchamber[i]
    mu_ox           = LOX.getViscosity(Poxtank[i], Toxtank[i])
    mu_fuel         = IPA.mu
    T_pres_ox       = Toxtank[i]
    mu_pres_ox      = OxPres.getViscosity(Poxtank[i], Toxtank[i])
    roo_pres_ox     = OxPres.getDensity(Poxtank[i], Toxtank[i])
    roo_pres_ox_upstream = OxPres.getDensity(P2pres_ox[i], Toxprestank[i])
    T_pres_fuel     = Tfueltank[i]
    mu_pres_fuel    = FuelPres.getViscosity(Pfueltank[i], Tfueltank[i])
    roo_pres_fuel   = FuelPres.getDensity(Pfueltank[i], Tfueltank[i])
    roo_pres_fuel_upstream = FuelPres.getDensity(P2pres_fuel[i], Tfuelprestank[i])
    
    Pcrit_ox        = LOX.P_crit
    Pvapor_ox       = LOX.P_vapor
    
    if i==0:    # First guesses. Based on choked flow at ox injector
        
        #nominal values for guessing
        Pc_nom          = 350*psi   
        mdot_nominal    = A_nozzleThroat*sqrt(ga)*Pc_nom/sqrt(RfireInit*TfireInit)/(1+(ga-1)/2)**((ga+1)/(2*ga-2))/5  
        
        mdot_ox_guess   = mdot_nominal*OF_nom/(OF_nom+1)
        mdot_fuel_guess = mdot_ox_guess/OF_nom
        print("mdot_ox_guess is", mdot_ox_guess, "kg/s")
        P3ox_guess      = P2ox - FM1_O.getPressureDrop()
        P4ox_guess      = P3ox_guess - NC1_O.getPressureDrop(mdot_ox_guess, rooOx)
        P5ox_guess      = P4ox_guess - PF1_O.getPressureDrop()
        P6ox_guess      = P5ox_guess - ox_tube.getPressureDrop(mdot_ox_guess, mu_ox, rooOx)
        print("P2ox_tank is", P2ox/psi, "psi")
        print("P3ox_guess is", P3ox_guess/psi, "psi")
        print("P4ox_guess is", P4ox_guess/psi, "psi")
        print("P5ox_guess is", P5ox_guess/psi, "psi")
        print("P6ox_guess is", P6ox_guess/psi, "psi")
        print("P_chamber is", Pchamber[i]/psi, "psi")
        
        print("mdot_fuel_guess is", mdot_fuel_guess, "kg/s")
        P3fuel_guess    = P2fuel - FM1_F.getPressureDrop()
        P4fuel_guess    = P3fuel_guess - NC1_F.getPressureDrop(mdot_fuel_guess, rooFuel)
        P5fuel_guess    = P4fuel_guess - PF1_F.getPressureDrop()
        P6fuel_guess    = P5fuel_guess - fuel_tube.getPressureDrop(mdot_fuel_guess, mu_fuel, rooFuel)
        P7fuel_guess    = P6fuel_guess - jacket.getPressureDrop(mdot_fuel_guess)
        print("P2fuel_tank is", P2fuel/psi, "psi")
        print("P3fuel_guess is is", P3fuel_guess/psi, "psi")
        print("P4fuel_guess is is", P4fuel_guess/psi, "psi")
        print("P5fuel_guess is is", P5fuel_guess/psi, "psi")
        print("P6fuel_guess is is", P6fuel_guess/psi, "psi")
        print("P7fuel_guess is is", P7fuel_guess/psi, "psi")
        print("P_chamber is", Pchamber[i]/psi, "psi")
        
        
        
        mdot_ox_pres_guess = mdot_ox_guess*roo_pres_ox/rooOx #volumetric flowrates of ox and pressurant are the same
        P3pres_ox_guess    = Preg_ox_start - NC1_OP.getPressureDrop(mdot_ox_pres_guess, Preg_ox_start, roo_pres_ox)
        P4pres_ox_guess    = P3pres_ox_guess - CV1_OP.getPressureDrop(mdot_ox_pres_guess, \
                                                                          P3pres_ox_guess, \
                                                                          OxPres.roo_std, \
                                                                          roo_pres_ox, \
                                                                          T_pres_ox)
        P5pres_ox_guess    = P4pres_ox_guess - CV2_OP.getPressureDrop(mdot_ox_pres_guess, \
                                                                          P4pres_ox_guess, \
                                                                          OxPres.roo_std, \
                                                                          roo_pres_ox, \
                                                                          T_pres_ox)
        P6pres_ox_guess    = P5pres_ox_guess - presox_tube.getPressureDrop(mdot_ox_pres_guess, mu_pres_ox, roo_pres_ox)
        
        print("P3ox_pres_guess is", P3pres_ox_guess/psi, "psi")
        print("P4ox_pres_guess is", P4pres_ox_guess/psi, "psi")
        print("P5ox_pres_guess is", P5pres_ox_guess/psi, "psi")
        print("P6ox_pres_guess is", P6pres_ox_guess/psi, "psi")
        
        mdot_fuel_pres_guess = mdot_fuel_guess*roo_pres_fuel/rooFuel #volumetric flowrates of fuel and pressurant are the same
        P3pres_fuel_guess    = Preg_fu_start - NC1_FP.getPressureDrop(mdot_fuel_pres_guess, Preg_fu_start, roo_pres_fuel)
        P4pres_fuel_guess    = P3pres_fuel_guess - CV1_FP.getPressureDrop(mdot_fuel_pres_guess, \
                                                                          P3pres_fuel_guess, \
                                                                          FuelPres.roo_std, \
                                                                          roo_pres_fuel, \
                                                                          T_pres_fuel)
        P5pres_fuel_guess    = P4pres_fuel_guess - CV2_FP.getPressureDrop(mdot_fuel_pres_guess, \
                                                                          P4pres_fuel_guess, \
                                                                          FuelPres.roo_std, \
                                                                          roo_pres_fuel, \
                                                                          T_pres_fuel)
        P6pres_fuel_guess    = P5pres_fuel_guess - presfuel_tube.getPressureDrop(mdot_fuel_pres_guess, mu_pres_fuel, roo_pres_fuel)
        
        print("P3fuel_pres_guess is", P3pres_fuel_guess/psi, "psi")
        print("P4fuel_pres_guess is", P4pres_fuel_guess/psi, "psi")
        print("P5fuel_pres_guess is", P5pres_fuel_guess/psi, "psi")
        print("P6fuel_pres_guess is", P6pres_fuel_guess/psi, "psi")
      
    else :      # using solutions from previous timestep
        mdot_ox_guess   = mdot_ox[i-1]
        #P3ox_guess      = P2ox - FM1_O.getPressureDrop()
        #P4ox_guess      = P3ox_guess - NC1_O.getPressureDrop(mdot_ox_guess, rooOx)
        #P5ox_guess      = P4ox_guess - PF1_O.getPressureDrop()
        #P6ox_guess      = P5ox_guess - ox_tube.getPressureDrop(mdot_ox_guess, mu_ox, rooOx)
        P3ox_guess      = P3ox[i-1]
        P4ox_guess      = P4ox[i-1]
        P5ox_guess      = P5ox[i-1]
        P6ox_guess      = P6ox[i-1]
        
        #print("mdot_ox_guess is", mdot_ox_guess)
        #print("P2ox is", P2ox/psi, "psi")
        #print("P3ox_guess is", P3ox_guess/psi, "psi")
        #print("P4ox_guess is", P4ox_guess/psi, "psi")
        #print("P_chamber is", Pchamber[i]/psi, "psi")
        
        
        mdot_fuel_guess = mdot_fuel[i-1]
        #P3fuel_guess    = P2fuel - FM1_F.getPressureDrop()
        #P4fuel_guess    = P3fuel_guess - NC1_F.getPressureDrop(mdot_fuel_guess, rooFuel)
        #P5fuel_guess    = P4fuel_guess - PF1_F.getPressureDrop()
        #P6fuel_guess    = P5fuel_guess - fuel_tube.getPressureDrop(mdot_fuel_guess, mu_fuel, rooFuel)
        #P7fuel_guess    = P6fuel_guess - jacket.getPressureDrop(mdot_fuel_guess)
        #mdot_fuel_guess = mdot_fuel[i-1]*1.0
        P3fuel_guess    = P3fuel[i-1]
        P4fuel_guess    = P4fuel[i-1]
        P5fuel_guess    = P5fuel[i-1]
        P6fuel_guess    = P6fuel[i-1]
        P7fuel_guess    = P7fuel[i-1]
        #print("P2fuel is", P2fuel/psi, "psi")
        #print("P3fuel_guess is is", P3fuel_guess/psi, "psi")
        #print("P4fuel_guess is is", P4fuel_guess/psi, "psi")
        #print("P5fuel_guess is is", P5fuel_guess/psi, "psi")
        #print("P_chamber is", Pchamber[i]/psi, "psi")
        
        mdot_ox_pres_guess = mdot_ox_pres[i-1]
        P3pres_ox_guess    = P3pres_ox[i-1]
        P4pres_ox_guess    = P4pres_ox[i-1]
        P5pres_ox_guess    = P5pres_ox[i-1]
        
        mdot_fuel_pres_guess = mdot_fuel_pres[i-1]
        P3pres_fuel_guess    = P3pres_fuel[i-1]
        P4pres_fuel_guess    = P4pres_fuel[i-1]
        P5pres_fuel_guess    = P5pres_fuel[i-1]
        
    initial_ox_guesses  = [P3ox_guess, P4ox_guess,P5ox_guess, P6ox_guess, mdot_ox_guess]
    initial_fuel_guesses = [P3fuel_guess, P4fuel_guess, P5fuel_guess, P6fuel_guess, P7fuel_guess, mdot_fuel_guess]
    initial_pres_ox_guesses = [P3pres_ox_guess, P4pres_ox_guess, P5pres_ox_guess, mdot_ox_pres_guess]
    initial_pres_fuel_guesses = [P3pres_fuel_guess, P4pres_fuel_guess, P5pres_fuel_guess, mdot_fuel_pres_guess]
    
    
    def oxfunks(U):       # defines the system of equations and unknowns U to be solved
        P3              = U[0]
        P4              = U[1]
        P5              = U[2]
        P6              = U[3]
        mdot            = U[4]
        
        #print("P3 as U0 is", P3/psi, "psi")
        #print("P4 as U1 is", P4/psi, "psi")
        #print("mdot as U2 is", mdot, "kg/s")
        
        out             = [ mdot - NC1_O.getMdot(P3, P4, rooOx, Pcrit_ox, Pvapor_ox) ]
        out.append( P2ox - P3 - FM1_O.getPressureDrop() )
        out.append( P4 - P5 - PF1_O.getPressureDrop() )
        out.append( P5 - P6 - ox_tube.getPressureDrop(mdot, mu_ox, rooOx))
        out.append( P6 - Pchamb - ox_pintle.getPressureDrops(mdot, rooOx, mu_ox, 0*psi)[-1] )
        #print("oxoutti", out)
        return out
    
    ox_solution       = opt.fsolve(oxfunks, initial_ox_guesses) # iterates until finds a satisfying solution or goes bust
    #print("ox solution is", array(ox_solution)/psi )
    mdot_ox_new       = ox_solution[4]
    Pox_intermediate  = oxtank.getPtank()
    Pox_eff           = (Pox_intermediate + P2ox)/2 # average of pressures before and after ejection of ox from tank; incoming Helium will see this 'effective' pressure in the tank
    
    #mdot_ox_pres_new    = presox_tube.getMdot(Preg, oxtank.getPtank(), mu_N2_ox, roo_N2_ox)
    #print("mdot_ox_pres_new is", mdot_ox_pres_new, "kg/s")
   
    
    
    def fuelfunks(U):       # defines the system of equations and unknowns U to be solved
        P3              = U[0]
        P4              = U[1]
        P5              = U[2]
        P6              = U[3]
        P7              = U[4]
        mdot            = U[5]
        #print("U is", U)
        #print("fuelmdot is", mdot)
        out             = [ mdot - NC1_F.getMdot(P3, P4, rooFuel, IPA.P_crit, IPA.P_vapor)  ]
        out.append( P2fuel - P3 - FM1_F.getPressureDrop() )
        out.append( P4 - P5 - PF1_F.getPressureDrop() )
        out.append( P5 - P6 - fuel_tube.getPressureDrop(mdot, mu_fuel, rooFuel) )
        out.append( P6 - P7 - jacket.getPressureDrop(mdot) ) 
        out.append( P7 - Pchamb - fuel_pintle.getPressureDrops(mdot, rooFuel, mu_fuel, 0*psi)[-1] )
        
        #print("fueloutti", out)
        return out
    
    fuel_solution       = opt.fsolve(fuelfunks, initial_fuel_guesses)  
    #print("fuel solution is", array(fuel_solution)/psi )
    mdot_fuel_new       = fuel_solution[5]
    Pfuel_intermediate  = fueltank.getPtank()
    Pfuel_eff           = (Pfuel_intermediate + P2fuel)/2 # average of pressures before and after ejection of fuel from tank; incoming Nitrogen will see this 'effective' pressure in the tank
    
    def presoxfunks(U):      # defines the system of equations and unknowns U to be solved
        P3              = U[0]
        P4              = U[1]
        P5              = U[2]
        mdot            = U[3]
        
        #print("P2pres_ox_i is", P2pres_ox[i]/psi, "psi")
        out = [mdot - NC1_OP.getMdot(P2pres_ox[i], P3, roo_pres_ox)]
        out.append(mdot - CV1_OP.getMdot(P3, P4, OxPres.roo_std, roo_pres_ox, T_pres_ox))
        out.append(mdot - CV2_OP.getMdot(P4, P5, OxPres.roo_std, roo_pres_ox, T_pres_ox))
        #out.append(mdot - presox_tube.getMdot(P5, Pox_eff, mu_pres_ox, roo_pres_ox))
        out.append(P5 - Pox_eff - presox_tube.getPressureDrop(mdot, mu_pres_ox, roo_pres_ox))
        
        #print("presox_outti", out)
        return out
    
    presox_solution       = opt.fsolve(presoxfunks, initial_pres_ox_guesses) 
    #print("presox solution is", array(presox_solution)/psi )
    mdot_ox_pres_new      = presox_solution[3]
    
    def presfuelfunks(U):      # defines the system of equations and unknowns U to be solved
        P3              = U[0]
        P4              = U[1]
        P5              = U[2]
        mdot            = U[3]
        
        out = [mdot - NC1_FP.getMdot(P2pres_fuel[i], P3, roo_pres_fuel)]
        out.append(mdot - CV1_FP.getMdot(P3, P4, FuelPres.roo_std, roo_pres_fuel, T_pres_fuel))
        out.append(mdot - CV2_FP.getMdot(P4, P5, FuelPres.roo_std, roo_pres_fuel, T_pres_fuel))
        #out.append(mdot - presfuel_tube.getMdot(P5, Pfuel_eff, mu_pres_fuel, roo_pres_fuel))
        out.append(P5 - Pfuel_eff - presfuel_tube.getPressureDrop(mdot, mu_pres_fuel, roo_pres_fuel))
        
        return out
    
    presfuel_solution       = opt.fsolve(presfuelfunks, initial_pres_fuel_guesses) 
    #print("presfuel solution is", array(presfuel_solution)/psi )
    mdot_fuel_pres_new      = presfuel_solution[3]
    
        
    # Now that mass flow rates out have been solved, intermediate states of the prop tanks can be established (only prop ejection):
    oxtank.update(Toxprestank[i], 0 , mdot_ox_new, P2pres_ox[i], timestep) #T_in, mdot_pres_in, mdot_prop_out, Pfeed, timestep, i):
    fueltank.update(Tfuelprestank[i], 0, mdot_fuel_new, P2pres_fuel[i], timestep) 
    
    # Determine final conditions in prop tanks (only pressurant inflow)
    oxtank.update(Toxprestank[i], mdot_ox_pres_new, 0, P2pres_ox[i], timestep)
    fueltank.update(Tfuelprestank[i], mdot_fuel_pres_new, 0, P2pres_fuel[i], timestep)
    
    # ...and pressurant tanks
    oxprestank.update(mdot_ox_pres_new, timestep)
    fuelprestank.update(mdot_fuel_pres_new, timestep)

    # Check if OFratio exceeds 3.5. If so, stop simulation
    if (mdot_ox_new/mdot_fuel_new) > 10:
        print("OF ratio > 10, terminate")
        break
    
    # Update chamber parameters:
    chamber.update(mdot_ox_new, mdot_fuel_new, Pambient, timestep) # mdot_ox_in, mdot_fuel_in, Pambient, timestep


    # Check if ox or fuel tank will empty during this timestep. If so, stop simulation.
    
    if oxtank.getMprop() < 0:
        print("Ox tank empty after", i, " iterations, ie", i*timestep, "seconds")
        print("remaining fuel", mfuel[i], "kg")
        print("remaining ox prs", moxprestank[i], "kg,", "i.e.", moxprestank[i]/moxprestank[0]*100, " % of initial amount")
        print("remaining fuel prs", mfuelprestank[i], "kg,", "i.e.", mfuelprestank[i]/mfuelprestank[0]*100, " % of initial amount")
        break 
            
    if fueltank.getMprop() < 0:
        print("Fuel tank empty after", i, " iterations, ie", i*timestep, "seconds")
        print("remaining ox", mox[i], "kg")
        print("remaining ox prs", moxprestank[i], "kg,", "i.e.", moxprestank[i]/moxprestank[0]*100, " % of initial amount")
        print("remaining fuel prs", mfuelprestank[i], "kg,", "i.e.", mfuelprestank[i]/mfuelprestank[0]*100, " % of initial amount")
        break
        
    if oxprestank.getPtank() < 1000*psi:
        print("Out of ox pressurant after", i, " iterations, ie", i*timestep, "seconds")
        print("remaining fuel", mfuel[i], "kg")
        print("remaining ox", mox[i], "kg")
        print("remaining fuel prs", mfuelprestank[i], "kg,", "i.e.", mfuelprestank[i]/mfuelprestank[0]*100, " % of initial amount")
        break
        
    if fuelprestank.getPtank() < 1000*psi:
        print("Out of fuel pressurant after", i, " iterations, ie", i*timestep, "seconds")
        print("remaining fuel", mfuel[i], "kg")
        print("remaining ox", mox[i], "kg")
        print("remaining ox prs", oxprestank[i], "kg,", "i.e.", moxprestank[i]/moxprestank[0]*100, " % of initial amount")
        break
        
    #update mass flow time histories. These are values during the CURRENT time step.
    if i==0:
        P3ox            = [ox_solution[0]]
        P4ox            = [ox_solution[1]]
        P5ox            = [ox_solution[2]]
        P6ox            = [ox_solution[3]]
        mdot_ox         = [ox_solution[4]]
        
        P3fuel          = [fuel_solution[0]]
        P4fuel          = [fuel_solution[1]]
        P5fuel          = [fuel_solution[2]]
        P6fuel          = [fuel_solution[3]]
        P7fuel          = [fuel_solution[4]]
        mdot_fuel       = [fuel_solution[5]]
        
        OFratio         = [ mdot_ox[0]/mdot_fuel[0] ]
        
        P3pres_ox       = [presox_solution[0]]
        P4pres_ox       = [presox_solution[1]]
        P5pres_ox       = [presox_solution[2]]
        mdot_ox_pres    = [presox_solution[3]]

        P3pres_fuel     = [presfuel_solution[0]]
        P4pres_fuel     = [presfuel_solution[1]]
        P5pres_fuel     = [presfuel_solution[2]]
        mdot_fuel_pres  = [presfuel_solution[3]]
        
    else:
        
        P3ox.append( ox_solution[0])
        P4ox.append( ox_solution[1])
        P5ox.append( ox_solution[2])
        P6ox.append( ox_solution[3])
        mdot_ox.append( ox_solution[4])
        
        P3fuel.append( fuel_solution[0])
        P4fuel.append( fuel_solution[1])
        P5fuel.append( fuel_solution[2])
        P6fuel.append( fuel_solution[3])
        P7fuel.append( fuel_solution[4])
        mdot_fuel.append( fuel_solution[5])
        
        #print("i is= ", i)
        OFratio.append( mdot_ox[i]/mdot_fuel[i])
        
        P3pres_ox.append( presox_solution[0])
        P4pres_ox.append( presox_solution[1])
        P5pres_ox.append( presox_solution[2])
        mdot_ox_pres.append( presox_solution[3])

        P3pres_fuel.append( presfuel_solution[0])
        P4pres_fuel.append( presfuel_solution[1])
        P5pres_fuel.append( presfuel_solution[2])
        mdot_fuel_pres.append( presfuel_solution[3])
    
    #update the rest of the time histories. System will have these values during the NEXT time step. 
      
    Poxtank.append( oxtank.getPtank())
    Toxtank.append( oxtank.getTpres())
    mPresOxtank.append( oxtank.getMpres())
    mox.append( oxtank.getMprop())
    FFoxtank.append( oxtank.getFF())
    
    Pfueltank.append( fueltank.getPtank())
    Tfueltank.append( fueltank.getTpres())
    mPresFueltank.append( fueltank.getMpres())
    mfuel.append( fueltank.getMprop())
    FFfueltank.append( fueltank.getFF())
    
    Toxprestank.append( oxprestank.getTtank())
    Poxprestank.append( oxprestank.getPtank())
    moxprestank.append( oxprestank.getM())
    #mdot_ox_pres.append( mdot_ox_pres_new)

    Tfuelprestank.append( fuelprestank.getTtank())
    Pfuelprestank.append( fuelprestank.getPtank())
    mfuelprestank.append( fuelprestank.getM())
    #mdot_fuel_pres.append( mdot_fuel_pres_new)
        
    Pchamber.append( chamber.get_P_inlet() )
    Pexit.append( nozzle.getPe(Pchamber[i+1], chamber.gamma, Pambient) )
    Mexit.append( nozzle.getMe(Pchamber[i+1], chamber.gamma, Pambient) )
    cmass.append( chamber.m)
    mdot_nozzle.append( nozzle.getmdot(chamber.gamma, Runiv/chamber.mbar, chamber.get_P_inlet(),\
                        chamber.T, chamber.Pa) )
    Thrust.append( nozzle.getThrust(chamber.get_P_inlet(), Pambient, chamber.gamma) )
    T_chamber.append( chamber.T)
    Isp.append( Thrust[i+1]/(mdot_ox[i] + mdot_fuel[i])/9.81 )
    
    mTotal.append(mox[i+1] + mfuel[i+1] + cmass[i+1] + mdot_nozzle[i]*timestep )
    moxpres.append( moxprestank[i+1] + mPresOxtank[i+1] )
    mfuelpres.append( mfuelprestank[i+1] + mPresFueltank[i+1] )
    time.append( time[i]+timestep )
    
    # Here the PI(Proportional and Integral) control theory is used
    oxErrorInt      = oxErrorInt + (Poxtank[0]-Poxtank[-1])*timestep    # Integral term for Preg_ox
    fuelErrorInt    = fuelErrorInt + (Pfueltank[0]-Pfueltank[-1])*timestep
    
    #if (time[-1]*1000)%25 < timestep*1000: 
    #PI-controller operates on every single timestep)
    Preg_ox_new     = Preg_ox_start + 0.1*(Poxtank[0]-Poxtank[-1]) + oxErrorInt
    Preg_fuel_new   = Preg_fu_start + 0.1*(Pfueltank[0]-Pfueltank[-1]) + fuelErrorInt
    
    P2pres_ox.append (Preg_ox_new)
    P2pres_fuel.append (Preg_fuel_new)
    
    i+=1
    
    #print("i=",i)
    if i%1000 == 0:
        print("i=",i)
    '''
    print('i =', '%d, '%i, 'time =', '%.2fs'%time[-1], \
            Preg_ox/psi, (Poxtank[0]-Poxtank[-1])/psi, \
            Preg_fu/psi, (Pfueltank[0]-Pfueltank[-1])/psi, \
            oxErrorInt/psi, fuelErrorInt/psi)
    '''
            
    
# Evaluate and Print some values

Vinj_ox     = ox_pintle.getVelocities(mdot_ox[-1], rooOx, mu_ox)[-1]               # Ox injection velocity, m/s
Vinj_fu     = fuel_pintle.getVelocities(mdot_fuel[-1], rooFuel, mu_fuel)[-1]       # Fuel injection velocity, m/s
fire        = chamber.get_Tfire(1.5, 2e6)

print("")
print("mdot_nozzle steady state is",            '%.3f'%mdot_nozzle[-1], "kg/s")
print("SS thrust is",                           '%.1f'%Thrust[-1], "N")
print("SS Isp is",                              '%.1f'%Isp[-1], "s")
print("SS T_chamber is",                        '%.1f'%T_chamber[-1], "K")
print("SS P_chamber is",                        '%.1f'%(Pchamber[-1]/psi), "psi")
print("SS P_exit is",                           '%.3f'%(Pexit[-1]/atm), "atm")
print("SS thrust coeff is", '%.3f'%nozzle.getCf(Pchamber[-1], atm, chamber.get_gamma(OFratio[-1],Pchamber[-1] )) )
print("SS mdot_pres_fuel is",                   '%.3f'%mdot_fuel_pres[-1], "kg/s")
print("SS mdot_pres_ox is",                     '%.3f'%mdot_ox_pres[-1], "kg/s")
print("SS pres_fuel flow rate is",              '%.3f'%(mdot_fuel_pres[-1]/roo_pres_fuel*1000/3.78*60), "GPM")
print("SS pres_ox flow rate is",                '%.3f'%(mdot_ox_pres[-1]/roo_pres_ox*1000/3.78*60), "GPM")
print("SS mdot_ox is",                          '%.3f'%mdot_ox[-1], "kg/s")
print("SS mdot_fuel is",                        '%.3f'%mdot_fuel[-1], "kg/s")
print("SS O/F ratio is",                        '%.3f'%OFratio[-1])
print("SS ox tube velocity is",                 '%.1f'%(mdot_ox[-1]/(rooOx*pi*d_oxtube**2/4)), "m/s")
print("SS fuel tube velocity is",               '%.1f'%(mdot_fuel[-1]/(rooFuel*pi*d_fueltube**2/4)), "m/s")
print("SS ox injection velocity is",            '%.1f'%(Vinj_ox), "m/s")
print("SS fuel injection velocity is",          '%.1f'%(Vinj_fu), "m/s")
print("Momentum ratio is", '%.3f'%(Vinj_fu*mdot_fuel[-1]/(Vinj_ox*mdot_ox[-1])))
print("SS ox injector P_drop is",               '%.1f'%((P4ox[-1]-Pchamber[-1])/psi), "psi, ie.", '%.1f'%((P4ox[-1]-Pchamber[-1])/Pchamber[-1]*100), "% of Pchamber")
print("SS fuel injector P_drop",                '%.1f'%((P5fuel[-1]-Pchamber[-1])/psi), "psi,ie, "'%.1f'%((P5fuel[-1]-Pchamber[-1])/Pchamber[-1]*100), "% of Pchamber")

print("")
print("SS ox pres line mass flow rate is",            '%.3f'%mdot_ox_pres[-1], "kg/s")
print("SS ox pres line pressure at RG1-He outlet is", '%.2f'%(P2pres_ox[-1]/psi), "psi")
print("SS ox pres line pressure at NC1-He outlet is", '%.2f'%(P3pres_ox[-1]/psi), "psi")
print("SS ox pres line pressure at CV1-He outlet is", '%.2f'%(P4pres_ox[-1]/psi), "psi")
print("SS ox pres line pressure at CV2-He outlet is", '%.2f'%(P5pres_ox[-1]/psi), "psi")
print("SS pressure drop across NC1-He is", '%.2f'%((P2pres_ox[-1]-P3pres_ox[-1])/psi), "psi")
print("SS pressure drop across CV1-He is", '%.2f'%((P3pres_ox[-1]-P4pres_ox[-1])/psi), "psi")
print("SS pressure drop across CV2-He is", '%.2f'%((P4pres_ox[-1]-P5pres_ox[-1])/psi), "psi")

print("")
print("SS fuel pres line mass flow rate is",           '%.3f'%mdot_fuel_pres[-1], "kg/s")
print("SS fuel pres line pressure at RG1-N outlet is", '%.2f'%(P2pres_fuel[-1]/psi), "psi")
print("SS fuel pres line pressure at NC1-N outlet is", '%.2f'%(P3pres_fuel[-1]/psi), "psi")
print("SS fuel pres line pressure at CV1-N outlet is", '%.2f'%(P4pres_fuel[-1]/psi), "psi")
print("SS fuel pres line pressure at CV2-N outlet is", '%.2f'%(P5pres_fuel[-1]/psi), "psi")
print("SS pressure drop across NC1-N is", '%.2f'%((P2pres_fuel[-1]-P3pres_fuel[-1])/psi), "psi")
print("SS pressure drop across CV1-N is", '%.2f'%((P3pres_fuel[-1]-P4pres_fuel[-1])/psi), "psi")
print("SS pressure drop across CV2-N is", '%.2f'%((P4pres_fuel[-1]-P5pres_fuel[-1])/psi), "psi")

print("")
print("SS pressure drop across cooling jacket is", '%.2f'%(jacket.getPressureDrop(mdot_fuel[-1])/psi), "psi")


print("")
print("bend drop in fuel inj is", '%.1f'%(fuel_pintle.getPressureDrops(mdot_fuel[-1], rooFuel,mu_fuel, 0*psi)[3]/psi), "psi")
print("fuel injector k_bend =", '%.3f'%( fuel_pintle.get_kbend(fuel_pintle.OD_shaft, mdot_fuel[-1])))
print("")
print("Pinj_in_fuel is", '%.1f'%(fuel_pintle.getPressures(mdot_fuel[-1], rooFuel,mu_fuel, P5fuel[-1])[0]/psi), "psi")
print("Pfuel_manifold is", '%.1f'%(fuel_pintle.getPressures(mdot_fuel[-1], rooFuel,mu_fuel, P5fuel[-1])[1]/psi), "psi")
print("Pfuel_annulus_in is", '%.1f'%(fuel_pintle.getPressures(mdot_fuel[-1], rooFuel,mu_fuel, P5fuel[-1])[2]/psi), "psi")
print("Pfuel_annulus_out is", '%.1f'%(fuel_pintle.getPressures(mdot_fuel[-1], rooFuel,mu_fuel, P5fuel[-1])[3]/psi), "psi")
print("Pfuel_bend_exit is", '%.1f'%(fuel_pintle.getPressures(mdot_fuel[-1], rooFuel,mu_fuel, P5fuel[-1])[4]/psi), "psi")
print("")
print("Pinj_in_ox is", '%.1f'%(ox_pintle.getPressures(mdot_ox[-1], rooOx,mu_ox, P4ox[-1])[0]/psi), "psi")
print("Pox_manifold is", '%.1f'%(ox_pintle.getPressures(mdot_ox[-1], rooOx,mu_ox, P4ox[-1])[1]/psi), "psi")
print("Pox_converging_in is", '%.1f'%(ox_pintle.getPressures(mdot_ox[-1], rooOx,mu_ox, P4ox[-1])[2]/psi), "psi")
print("Pox_annulus_in is", '%.1f'%(ox_pintle.getPressures(mdot_ox[-1], rooOx,mu_ox, P4ox[-1])[3]/psi), "psi")
print("Pox_annulus_exit is", '%.1f'%(ox_pintle.getPressures(mdot_ox[-1], rooOx,mu_ox, P4ox[-1])[4]/psi), "psi")
print("")
print("v_fuel_manifold is", '%.2f'%fuel_pintle.getVelocities(mdot_fuel[-1], rooFuel,mu_fuel)[1], "m/s")
print("v_fuel_annulus is", '%.2f'%fuel_pintle.getVelocities(mdot_fuel[-1], rooFuel,mu_fuel)[2], "m/s")
print("v_fuel_injection is", '%.2f'%fuel_pintle.getVelocities(mdot_fuel[-1], rooFuel,mu_fuel)[3], "m/s")
print("")
print("v_ox_manifold is", '%.2f'%ox_pintle.getVelocities(mdot_ox[-1], rooOx, mu_ox)[1], "m/s")
print("v_ox_ori is", '%.2f'%ox_pintle.getVelocities(mdot_ox[-1], rooOx, mu_ox)[2], "m/s")
print("v_ox_manifold after orifices is", '%.2f'%ox_pintle.getVelocities(mdot_ox[-1], rooOx, mu_ox)[3], "m/s")
print("v_ox_injection", '%.2f'%ox_pintle.getVelocities(mdot_ox[-1], rooOx, mu_ox)[4], "m/s")

# following time histories are one element shorter than the rest, so the last calculated value will be duplicated to match the length of other time histories.

P3ox.append( ox_solution[0])
P4ox.append( ox_solution[1])
P5ox.append( ox_solution[2])
P6ox.append( ox_solution[3])
mdot_ox.append( ox_solution[4])

P3fuel.append( fuel_solution[0])
P4fuel.append( fuel_solution[1])
P5fuel.append( fuel_solution[2])
P6fuel.append( fuel_solution[3])
P7fuel.append( fuel_solution[4])
mdot_fuel.append( fuel_solution[5])

P3pres_ox.append( presox_solution[0])
P4pres_ox.append( presox_solution[1])
P5pres_ox.append( presox_solution[2])
mdot_ox_pres.append( presox_solution[3])

P3pres_fuel.append( presfuel_solution[0])
P4pres_fuel.append( presfuel_solution[1])
P5pres_fuel.append( presfuel_solution[2])
mdot_fuel_pres.append( presfuel_solution[3])

OFratio.append( mdot_ox[i]/mdot_fuel[i])

# plot time histories
plt.ion()

plt.figure(1)
plt.plot(time,array(Poxprestank)/psi, label='pressurant tank')
plt.figure(1)
plt.plot(time,array(P3pres_ox)/psi, label='pressurant solenoid valve out')
plt.figure(1)
plt.plot(time,array(P4pres_ox)/psi, label='pressurant check valve 1 out')
plt.figure(1)
plt.plot(time,array(P5pres_ox)/psi, label='pressurant check valve 2 out')
plt.figure(1)
plt.plot(time,array(Poxtank)/psi, label='ox tank')
plt.figure(1)
plt.plot(time,array(P3ox)/psi, label='ox flow meter out')
plt.figure(1)
plt.plot(time,array(P4ox)/psi, label='ox solenoid valve out')
plt.figure(1)
plt.plot(time,array(P5ox)/psi, label='ox particle filter out')
plt.figure(1)
plt.plot(time,array(P6ox)/psi, label='ox injector in')
plt.figure(1)
plt.plot(time,array(Pchamber)/psi, label='chamber')
plt.figure(1)
plt.plot(time,array(Pexit)/psi, label='exit')
plt.title('Ox pressures')
plt.legend( loc='upper right')
plt.xlabel('Time [s]')
plt.ylabel('psi')
plt.show()

plt.figure(2)
plt.plot(time,array(Pfuelprestank)/psi, label='pressurant tank')
plt.figure(2)
plt.plot(time,array(P3pres_fuel)/psi, label='pressurant solenoid valve out')
plt.figure(2)
plt.plot(time,array(P4pres_fuel)/psi, label='pressurant check valve 1 out')
plt.figure(2)
plt.plot(time,array(P5pres_fuel)/psi, label='pressurant check valve 2 out')
plt.figure(2)
plt.plot(time,array(Pfueltank)/psi, label='fuel tank')
plt.figure(2)
plt.plot(time,array(P3fuel)/psi, label='fuel flow meter out')
plt.figure(2)
plt.plot(time,array(P4fuel)/psi, label='fuel solenoid valve out')
plt.figure(2)
plt.plot(time,array(P5fuel)/psi, label='fuel particle filter out')
plt.figure(2)
plt.plot(time,array(P6fuel)/psi, label='fuel cooling jacket in')
plt.figure(2)
plt.plot(time,array(P7fuel)/psi, label='fuel injector in')
plt.figure(2)
plt.plot(time,array(Pchamber)/psi, label='chamber')
plt.figure(2)
plt.plot(time,array(Pexit)/psi, label='exit')
plt.title('Fuel pressures')
plt.legend( loc='upper right')
plt.xlabel('Time [s]')
plt.ylabel('psi')
plt.show()

plt.figure(3)
plt.plot(time,Toxtank, label='ox tank')
plt.figure(3)
plt.plot(time,Toxprestank, label='ox pressurant tank')
plt.figure(3)
plt.plot(time,Tfueltank, label='fuel tank')
plt.figure(3)
plt.plot(time,Tfuelprestank, label='fuel pressurant tank')
plt.title('Tank temperatures')
plt.legend( loc='upper right')
plt.xlabel('Time [s]')
plt.ylabel('K')
plt.show()

plt.figure(4)
plt.plot(time,mdot_ox, label='ox mdot')
plt.figure(4)
plt.plot(time,mdot_fuel, label='fuel mdot')
plt.figure(4)
plt.plot(time,mdot_ox_pres, label='ox pressurant mdot')
plt.figure(4)
plt.plot(time,mdot_fuel_pres, label='fuel pressurant mdot')
plt.figure(4)
plt.plot(time,mdot_nozzle, label='nozzle mdot')
plt.title('Mass flows')
plt.xlabel('Time [s]')
plt.ylabel('kg/s')
plt.legend( loc='upper right')
plt.show()

plt.figure(5)
plt.plot(time,FFoxtank, label='ox tank')
plt.figure(5)
plt.plot(time,FFfueltank, label='fuel tank')
plt.title('Fill fractions in the tanks (Vprop_/Vtank)')
plt.xlabel('Time [s]')
plt.ylabel('')
plt.legend( loc='upper right')
plt.show()

plt.figure(6)
plt.plot(time, OFratio)
plt.title('O/F ratio')
plt.xlabel('Time [s]')
plt.ylabel('')
plt.show()

plt.figure(7)
plt.plot(time,mox, label='ox')
plt.figure(7)
plt.plot(time,mfuel, label='fuel')
plt.figure(7)
plt.plot(time,moxprestank, label='ox pressurant in pressurant tank')
plt.figure(7)
plt.plot(time,mfuelprestank, label='fuel pressurant in pressurant tank')
plt.figure(7)
plt.plot(time,mPresOxtank, label='ox pressurant in ox tank')
plt.figure(7)
plt.plot(time,mPresFueltank, label='fuel pressurant in fuel tank')
plt.figure(7)
plt.plot(time,moxpres, label='total ox pressurant')
plt.figure(7)
plt.plot(time,mfuelpres, label='total fuel pressurant')
plt.title('Fluid masses')
plt.xlabel('Time [s]')
plt.ylabel('kg')
plt.legend( loc='upper right')
plt.show()

plt.figure(8)
plt.plot(time, cmass)
plt.title('Resident mass in chamber')
plt.xlabel('Time [s]')
plt.ylabel('kg')
plt.show()

plt.figure(9)
plt.plot(time, Thrust)
plt.title('Thrust')
plt.xlabel('Time [s]')
plt.ylabel('N')
plt.show()

plt.figure(10)
plt.plot(time, Isp)
plt.title('Isp')
plt.xlabel('Time [s]')
plt.ylabel('s')
plt.show()

plt.figure(11)
plt.plot(time, T_chamber)
plt.title('T chamber')
plt.xlabel('Time [s]')
plt.ylabel('K')
plt.show()

plt.figure(12)
plt.plot(time, Mexit)
plt.title('Exit Mach number')
plt.xlabel('Time [s]')
plt.ylabel('-')
plt.show()

plt.figure(13)
y1  = Poxprestank[-1]/psi
y2  = P2pres_ox[-1]/psi
y3  = P3pres_ox[-1]/psi
y4  = P4pres_ox[-1]/psi
y5  = P5pres_ox[-1]/psi
y6  = Poxtank[-1]/psi
y7  = P3ox[-1]/psi
y8  = P4ox[-1]/psi
y9  = P5ox[-1]/psi
y10 = P6ox[-1]/psi
y11 = Pchamber[-1]/psi
plt.plot( [0, 1],   [y1, y1],   linewidth=2, label="pressurant tank")
plt.plot( [1, 2],   [y1, y2],   linewidth=2, label="pressurant regulator")
plt.plot( [2, 3],   [y2, y3],   linewidth=2, label="pressurant solenoid valve")
plt.plot( [3, 4],   [y3, y4],   linewidth=2, label="pressurant check valve 1")
plt.plot( [4, 5],   [y4, y5],   linewidth=2, label="pressurant check valve 2")
plt.plot( [5, 6],   [y5, y6],   linewidth=2, label="pressurant tubing")
plt.plot( [6, 7],   [y6, y6],   linewidth=2, label="ox tank")
plt.plot( [7, 8],   [y6, y7],   linewidth=2, label="ox flow meter")
plt.plot( [8, 9],   [y7, y8],   linewidth=2, label="ox solenoid valve")
plt.plot( [9, 10],  [y8, y9],   linewidth=2, label="ox particle filter")
plt.plot( [10, 11], [y9, y10],  linewidth=2, label="ox piping")
plt.plot( [11, 12], [y10, y11], linewidth=2, label="ox injector")
plt.plot( [12, 13], [y11, y11], linewidth=2, label="chamber")
plt.title('Ox line pressures at end of burn')
plt.ylabel('psi')
plt.legend( loc='upper right')
plt.show()

plt.figure(14)
y1  = Pfuelprestank[-1]/psi
y2  = P2pres_fuel[-1]/psi
y3  = P3pres_fuel[-1]/psi
y4  = P4pres_fuel[-1]/psi
y5  = P5pres_fuel[-1]/psi
y6  = Pfueltank[-1]/psi
y7  = P3fuel[-1]/psi
y8  = P4fuel[-1]/psi
y9  = P5fuel[-1]/psi
y10 = P6fuel[-1]/psi
y11 = P7fuel[-1]/psi
y12 = Pchamber[-1]/psi
plt.plot( [0, 1],   [y1, y1],   linewidth=2, label="pressurant tank")
plt.plot( [1, 2],   [y1, y2],   linewidth=2, label="pressurant regulator")
plt.plot( [2, 3],   [y2, y3],   linewidth=2, label="pressurant solenoid valve")
plt.plot( [3, 4],   [y3, y4],   linewidth=2, label="pressurant check valve 1")
plt.plot( [4, 5],   [y4, y5],   linewidth=2, label="pressurant check valve 2")
plt.plot( [5, 6],   [y5, y6],   linewidth=2, label="pressurant tubing")
plt.plot( [6, 7],   [y6, y6],   linewidth=2, label="fuel tank")
plt.plot( [7, 8],   [y6, y7],   linewidth=2, label="fuel flow meter")
plt.plot( [8, 9],   [y7, y8],   linewidth=2, label="fuel solenoid valve")
plt.plot( [9, 10],  [y8, y9],   linewidth=2, label="fuel particle filter")
plt.plot( [10, 11], [y9, y10],  linewidth=2, label="fuel piping")
plt.plot( [11, 12], [y10, y11], linewidth=2, label="fuel cooling jacket")
plt.plot( [12, 13], [y11, y12], linewidth=2, label="fuel injector")
plt.plot( [13, 14], [y12, y12], linewidth=2, label="chamber")
plt.title('Fuel line pressures at end of burn')
plt.ylabel('psi')
plt.legend( loc='upper right')
plt.show()

plt.figure(15)
plt.plot(time, array(Poxprestank)/psi, label='ox pressurant tank')
plt.plot(time, array(Pfuelprestank)/psi, label='fuel pressurant tank')
plt.title('Pressurant Tank Pressure Time History')
plt.xlabel('Time [s]')
plt.ylabel('psi')
plt.legend(loc='upper right')
plt.show()

plt.figure(16)
plt.plot(time, Toxprestank, label='ox pressurant tank')
plt.plot(time, Tfuelprestank, label='fuel pressurant tank')
plt.title('Pressurant Tank Temperature Time History')
plt.xlabel('Time [s]')
plt.ylabel('K')
plt.legend(loc='upper right')
plt.show()

plt.figure(17)
plt.plot(time, array(Poxtank)/psi, label='ox tank')
plt.plot(time, array(Pfueltank)/psi, label='fuel tank')
plt.title('Propellant Tank Pressure Time History')
plt.xlabel('Time [s]')
plt.ylabel('psi')
plt.legend(loc='upper right')
plt.show()

plt.figure(18)
plt.plot(time, Toxtank, label='ox tank')
plt.plot(time, Tfueltank, label='fuel tank')
plt.title('Propellant Tank Temperature Time History')
plt.xlabel('Time [s]')
plt.ylabel('K')
plt.legend(loc='upper right')
plt.show()

plt.figure(19)
y1  = P2pres_ox[-1]/psi
y2  = P3pres_ox[-1]/psi
y3  = P4pres_ox[-1]/psi
y4  = P5pres_ox[-1]/psi
y5  = Poxtank[-1]/psi
y6  = P3ox[-1]/psi
y7  = P4ox[-1]/psi
y8  = P5ox[-1]/psi
y9  = P6ox[-1]/psi
y10 = Pchamber[-1]/psi
plt.plot( [0, 1],   [y1, y2],   linewidth=2, label="pressurant solenoid valve")
plt.plot( [1, 2],   [y2, y3],   linewidth=2, label="pressurant check valve 1")
plt.plot( [2, 3],   [y3, y4],   linewidth=2, label="pressurant check valve 2")
plt.plot( [3, 4],   [y4, y5],   linewidth=2, label="pressurant tubing")
plt.plot( [4, 5],   [y5, y5],   linewidth=2, label="ox tank")
plt.plot( [5, 6],   [y5, y6],   linewidth=2, label="ox flow meter")
plt.plot( [6, 7],   [y6, y7],   linewidth=2, label="ox solenoid valve")
plt.plot( [7, 8],   [y7, y8],   linewidth=2, label="ox particle filter")
plt.plot( [8, 9],   [y8, y9],   linewidth=2, label="ox piping")
plt.plot( [9, 10],  [y9, y10],  linewidth=2, label="ox injector")
plt.plot( [10, 11], [y10, y10], linewidth=2, label="chamber")
plt.title('Ox line pressures downstream of regulator at end of burn')
plt.ylabel('psi')
plt.legend( loc='lower left')
plt.show()

plt.figure(20)
y1  = P2pres_fuel[-1]/psi
y2  = P3pres_fuel[-1]/psi
y3  = P4pres_fuel[-1]/psi
y4  = P5pres_fuel[-1]/psi
y5  = Pfueltank[-1]/psi
y6  = P3fuel[-1]/psi
y7  = P4fuel[-1]/psi
y8  = P5fuel[-1]/psi
y9  = P6fuel[-1]/psi
y10 = P7fuel[-1]/psi
y11 = Pchamber[-1]/psi
plt.plot( [0, 1],   [y1, y2],   linewidth=2, label="pressurant solenoid valve")
plt.plot( [1, 2],   [y2, y3],   linewidth=2, label="pressurant check valve 1")
plt.plot( [2, 3],   [y3, y4],   linewidth=2, label="pressurant check valve 2")
plt.plot( [3, 4],   [y4, y5],   linewidth=2, label="pressurant tubing")
plt.plot( [4, 5],   [y5, y5],   linewidth=2, label="fuel tank")
plt.plot( [5, 6],   [y5, y6],   linewidth=2, label="fuel flow meter")
plt.plot( [6, 7],   [y6, y7],   linewidth=2, label="fuel solenoid valve")
plt.plot( [7, 8],   [y7, y8],   linewidth=2, label="fuel particle filter")
plt.plot( [8, 9],   [y8, y9],   linewidth=2, label="fuel piping")
plt.plot( [9, 10],  [y9, y10],  linewidth=2, label="fuel cooling jacket")
plt.plot( [10, 11], [y10, y11], linewidth=2, label="fuel injector")
plt.plot( [11, 12], [y11, y11], linewidth=2, label="chamber")
plt.title('Fuel line pressures downstream of regulator at end of burn')
plt.ylabel('psi')
plt.legend( loc='lower left')
plt.show()

plt.figure(21)
plt.title( "PI-controller related pressures")
plt.plot(time, array(P2pres_fuel)/psi, label = "Fuel pressurant reg" )
plt.plot(time, array(Pfueltank)/psi, label = "Fuel tank" )
plt.plot(time, array(P2pres_ox)/psi, label = "Ox pressurant reg" )
plt.plot(time, array(Poxtank)/psi, label = "Ox tank" )
plt.xlabel('Time [s]')
plt.ylabel('psi')
plt.legend( loc='upper left')
plt.show()