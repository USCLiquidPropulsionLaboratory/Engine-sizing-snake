## GOX-kerosene sim
#@ Author Juha Nieminen

#import sys
#sys.path.insert(0, '/Users/juhanieminen/Documents/adamrocket')

import RocketComponents as rc
from physical_constants import poise, inches, Runiv, gallons, lbm, \
    gearth, atm, psi, lbf
from numpy import pi, linspace, cos, radians, sqrt, exp, log, array, full, ceil
from scipy import optimize as opt
import matplotlib.pyplot as plt
from matplotlib import collections as mc
import Flows1D as flows


#DESIGN VARIABLES____________________________________________________________________________________

# nominal parameters
Preg_ox          = 1100*psi         # regulated GOX outlet pressure [Pa]
Preg_N2          = 1060*psi         # regulated N2 outlet pressure [Pa]

mdot_fuel_nom   = 0.2               # This is only for cooling jacket pressure drop purposes [kg/s]
Pdrop_jacket_nom= 1*psi             # Cooling jacket pressure drop at mdot_nominal [Pa]
OF_nom          = 2.25              # Oxidizer-to-fuel ratio. This has only effect on initial guesses during solving

# Pressurant tank dimensions
Vprestank       = 0.050             # N2 pressurant tank volume [m3]

# Propellant tank dimensions
Vfueltank       = 0.006             # fuel tank volume
Voxtank         = 0.050             # ox tank volume [m3]

# Tubing
d_presfuel_tube = 1.0*inches        # pressurant tank -> fuel tank tube diameter [m]
L_presfuel_tube = 0.5               # pressurant tank -> fuel tank tube length [m]

d_oxtube        = 0.87*inches       # ox tank -> manifold tube diameter [m]
L_oxtube        = 2.4               # ox tank -> manifold tube length [m]
d_fueltube      = 0.87*inches       # fuel tank -> manifold tube diameter [m]
L_fueltube      = 3.0               # fuel tank -> manifold tube length [m]

roughness       = 0.005             # epsilon/diameter, dimensionless

# Valves
Cv_ox_check     = 4.7               # oxidizer check valve flow coefficient, dimensionless
Pcrack_ox_check = 10*psi            # oxidizer check valve opening pressure [Pa]
Cv_pres_check   = 4.7               # nitrogen check valve flow coefficient, dimensionless
Pcrack_pres_check  = 10*psi         # nitrogen check valve opening pressure [Pa]
Cv_pres_valve   = 9                 # nitrogen solenoid valve flow coefficient, dimensionless
Cv_ox_valve     = 9                 # oxidizer solenoid valve flow coefficient, dimensionless 
Cv_fuel_valve   = 9                 # fuel solenoid valve flow coefficient, dimensionless

# Injector

cd_oxInjector   = 0.767                                                 # orifice discharge coefficient
diameter_oxInjectorHoles = 2.54e-3 #number xx drill                     # ox orifice diameter [m]
#length_oxHole   = 0.005                                                # ox orifice length [m]
numOxInjectorHoles = 24                                                 # number of ox orifices in the injector
area_oxInjector = numOxInjectorHoles*pi*diameter_oxInjectorHoles**2/4   # total ox flow area [m2]

cd_fuelInjector = 0.767                                                 # orifice discharge coefficient
diameter_fuelInjectorHoles = 0.508e-3 #number xx drill                      # fuel orifice diameter [m]
numFuelHoles    = 64                                                    # number of fuel orifices in the injector
area_fuelInjector = numFuelHoles*pi*diameter_fuelInjectorHoles**2/4     # total fuel flow area [m2]


# Define initial/nominal conditions in the chamber (obtained from CEA code assuming OFratio = 2.25)
TfireInit       = 293                                                   # initial flame temperature [K]
Pfire           = 1*atm                                                 # initial chamber pressure [Pa]
gammaFireInit   = 1.148                                                 # dimensionless
ga              = gammaFireInit
mbarFireInit    = 21.87                              # combustion products' initial molecular mass [kg/kmol]
RfireInit       = Runiv/mbarFireInit                 # combustion products' initial specific gas constant [J/kgK]
Pambient        = atm                                                   # ambient pressure [Pa]

# Nozzle and chamber

d_nozzleThroat  = 1.0*inches                                            # throat diameter [m]
A_nozzleThroat  = pi*d_nozzleThroat**2/4                                # throat area [m2] 
area_ratio      = 7.46                                                  # nozzle exit-to-throat area ratio
A_nozzleExit    = area_ratio*A_nozzleThroat                             # nozzle exit area [m2]
d_nozzleExit    = sqrt(4*A_nozzleExit/pi)                               # nozzle exit diameter [m]
            
Dchamber        = 0.08                                                  # chamber diameter [m]
Achamber        = pi*Dchamber**2/4                                      # chamber cross sectional area [m2]
Lchamber        = 0.14                                                  # chamber length [m]
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

TfuelPresStart  = 293                   # Fuel pressurant (=nitrogen) temp [K]
FFfueltankStart = 0.5                   # Fuel tank fill fraction (Vfuel/Vtank)
PfuelPrestankStart  = 2216*psi - Preg_N2*Vfueltank*FFfueltankStart/Vprestank     # Fuel pressurant tank pressure once fueltank has been pressurized [Pa]

ToxStart        = 293                   # Oxidizer (GOX) temp [K] 
PoxtankStart    = 2216*psi              # Oxidizer tank pressure [Pa]
TfuelStart      = 293                   # Fuel temp [K]
PfueltankStart  = Preg_N2 -10*psi       # Fuel tank pressure [Pa] (-10psi helps convergence on first timestep)

# initialize propellants
nitrogen        = rc.NitrogenFluid()
GOX             = rc.GOXFluid()
kerosene        = rc.Kerosene()

#initialize nozzle and chamber
nozzle          = rc.ConvergingDivergingNozzle(A_nozzleExit, A_nozzleThroat)
mdot_init_noz   = nozzle.getmdot(gammaFireInit, GOX.R, Pfire, TfireInit, atm)
chamber         = rc.GOXKeroCombustionChamber(nozzle, Vchamber, TfireInit, ga, mbarFireInit, Pfire, atm, mdot_init_noz)

#initialize injector orifices
ox_orifice      = rc.GasOrifice(area_oxInjector, cd_oxInjector, GOX.gamma, GOX.R)
fuel_orifice    = rc.LiquidOrifice(area_fuelInjector, cd_fuelInjector )

#initialize pressurant tanks
fuelprestank    = rc.IdealgasTank(nitrogen, Vprestank, TfuelPresStart, PfuelPrestankStart)

#initialize propellant tanks
oxtank          = rc.IdealgasTank(GOX, Voxtank, ToxStart, PoxtankStart)
fueltank        = rc.LiquidPropellantTank(nitrogen, kerosene, Vfueltank, TfuelStart, TfuelPresStart,\
                    PfueltankStart, FFfueltankStart, Preg_N2)

#initialize pressure regulators
N2_regu         = rc.PressureRegulator(Preg_N2, nitrogen)
ox_regu         = rc.PressureRegulator(Preg_ox, GOX)

#initialize solenoids
fuelSole        = rc.IncompressibleFlowSolenoid( Cv_fuel_valve)
oxSole          = rc.CompressibleFlowSolenoid( Cv_ox_valve, GOX)
presSole        = rc.CompressibleFlowSolenoid( Cv_pres_valve, nitrogen)

#initialize check valves
ox_check        = rc.CompressibleFlowCheckValve( Cv_ox_check, Pcrack_ox_check, GOX)
pres_check      = rc.CompressibleFlowCheckValve( Cv_pres_check, Pcrack_pres_check, nitrogen)

#initialize tubing
ox_tube         = rc.RoughStraightCylindricalTube(d_oxtube, L_oxtube, roughness, True)
fuel_tube       = rc.RoughStraightCylindricalTube(d_fueltube, L_fueltube, roughness, True)
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
Toxtank         = [oxtank.getTtank()]               # ox tank temperature [K]
mox             = [oxtank.getM()]                   # oxidizer mass in tank [kg]

Pfueltank       = [fueltank.getPtank()]             # fuel tank pressure [Pa]
Tfueltank       = [fueltank.getTpres()]             # pressurant temperature in fuel tank[K]
mPresFueltank   = [fueltank.getMpres()]             # pressurant mass in fuel tank [kg]
mfuel           = [fueltank.getMprop()]             # fuel mass in tank [kg]
FFfueltank      = [fueltank.getFF()]                # fuel tank fill fraction defined as Vfuel/(Vfueltank)

TfuelPres       = [fuelprestank.getTtank()]         # temperature in fuel pressurant tank [K]
PfuelPres       = [fuelprestank.getPtank()]         # pressure in fuel pressurant tank [Pa]
mfuelPres       = [fuelprestank.getM()]             # pressurant mass in fuel pressurant tank [Pa]

time            = [0]                               # time array [s]

mdot_ox         = [0]                               # ox mass flow out of the tank [kg/s]
P1ox            = [0]                               # ox tank presssure [Pa]
P2ox            = [0]                               # ox regulator outlet pressure [Pa]
P3ox            = [0]                               # ox check valve outlet pressure [Pa]      
P4ox            = [0]                               # ox flow solenoid outlet pressure [Pa]
P5ox            = [0]                               # ox injector inlet pressure [Pa]

T1ox            = [0]                               # ox tank temp [K]
T2ox            = [0]                               # ox regulator output temp [K]
T3ox            = [0]                               # ox check valve outlet temp [K]
T4ox            = [0]                               # ox flow solenoid outlet temp [K]
T5ox            = [0]                               # ox injector inlet temp [K]

mdot_fuel       = [0]                               # fuel mass flow out of the tank [kg/s]
rooFuel         = fueltank.propellant.density       # fuel density, assumed constant [kg/m3]
P1fuel          = [0]                               # fuel tank presssure [Pa]
P2fuel          = [0]                               # fuel solenoid outlet pressure [Pa]      
P3fuel          = [0]                               # fuel cooling jacket inlet pressure [Pa]
P4fuel          = [0]                               # fuel injector inlet pressure [Pa]

mdot_fuel_pres  = [0]                               # fuel pressurant mass flow rate [kg/s]
P3pres          = [0]                               # pressurant pressure at check valve outlet [kg/s]
P4pres          = [0]                               # pressurant pressure at solenoid valve outlet [kg/s]

mTotal          = [0]                               # propellant mass in the system [kg]
mprs            = [mfuelPres[0]+mPresFueltank[0]]   # pressurant mass in the system [kg]
OFratio         = [0]                               # oxidizer to fuel mass flow ratio 
Isp             = [0]                               # specific impulse [s]
Thrust          = [nozzle.getThrust(chamber.get_P_inlet(), Pambient, gammaFireInit) ] # rocket thrust [N]

#SIMULATE_______________________________________________________________________________________________________
# using orifices as follows: ejecting GOX from manifold to chamber, fuel liq-to-liq from manifold to chamber

print("")
print("STARTING SIM...")
print("")
print("mOxStart is", '%.2f'%mox[0], "kg")
print("mKerostart is", mfuel[0], "kg")
print("mN2start in N2 tank is", '%.2f'%mfuelPres[0], "kg")
print("mN2start in fuel tank is", '%.2f'%(fueltank.getMpres()), "kg")

# The first step is to solve oxidizer and fuel mass flow rates from the tank to combustion chamber.

# definitions:

# P1ox              = GOX tank pressure
# P2ox              = regulation pressure
# P3ox              = check valve outlet pressure
# P4ox              = ox valve outlet pressure
# P5ox              = injector inlet, pressure 
# (P1ox-P2ox)       = regulator pressure drop,                  known constant 
# (P2ox-P3ox)       = ox check valve pressure drop,             eq 1
# (P3ox-P4ox)       = ox flow solenoid pressure drop,           eq 2
# (P4ox-P5ox)       = ox tubing pressure drop,                  eq 3
# (P5ox-Pchamber)   = ox injector pressure drop,                eq 4

# P1pres            = Nitrogen tank pressure
# P2pres            = Regulation pressure
# P3pres            = Check valve outlet pressure
# P4pres            = Nitrogen solenoid outlet
# P5pres            = Nitrogen tubing outlet = fuel tank pressure
# (P2pres-P3pres)   = Nitrogen check valve pressure drop
# (P3pres-P4pres)   = Nitrogen solenoid valve pressure drop
# (P4pres-P5pres)   = Nitrogen tubing pressure drop

# P1fuel            = fuel tank pressure
# P2fuel            = fuel valve outlet pressure
# P3fuel            = cooling jacket inlet pressure
# P4fuel            = injector inlet pressure
# (P1fuel-P2fuel)   = fuel valve pressure drop,                 eq1
# (P2fuel-P3fuel)   = fuel tubing pressure drop,                eq2
# (P3fuel-P4fuel)   = cooling jacket pressure drop,             eq3
# (P4fuel-Pchamber) = injector pressure drop,                   eq4

# In the case of oxidizer, P2 and Pchamber are known, so one must solve for P3, P4, and P5. Fourth unknown is the mass flow rate. The four equations are check valve/solenoid/tubing/injector pressure drops. These equations are defined in oxfunks method below, and underlying physics are in RocketComponents.py under their respective classes.

# With pressurant, P2 (regulation pressure) and P5 (fuel tank pressure) are known, so one must solve for P3 and P4. The third unknown is pressurant mass flow rate. Equations to be solved are pressure drops over the check valve, solenoid valve, and the tubing.

# With fuel P1 and Pchamber are known, so one must solve for P2, P3, and P4. Fourth unknown is mass flow rate.

# fsolve requires sensible initial guesses for all unknowns. They are established by guessing the mass flow rate, because all other pressures trickle down from that.


timestep_small      = 5e-6      # seconds, used during initial transient
timestep_nom        = 0.0001    # seconds, used after 0.01 seconds of simulation time
t_transient         = 0.01      # seconds, estimated time of initial transient

t_simulation        = 3         # seconds

if t_simulation <= t_transient:
    simsteps    = int(ceil(t_simulation/timestep_small))
else:
    simsteps    = int(ceil( t_transient/timestep_small + (t_simulation-t_transient)/timestep_nom ))

print("Sim time is", t_simulation, "s, number of simsteps is", simsteps)

i=0
for i in range(0, simsteps):
    if time[i] < 0.01:
        timestep    = 1e-5                             # use shorter timestep during initial transient
    else: timestep    = timestep_nom                   # proceed with nominal timestep
#while True:
    #print("i=", i) 
    P1ox            = Poxtank[i]
    P2ox            = Preg_ox    
    P1fuel          = Pfueltank[i]
    Pchamb          = Pchamber[i]
    mu_ox           = GOX.getViscosity(Preg_ox, Toxtank[i])
    roo_ox          = GOX.getDensity(Preg_ox, Toxtank[i])
    Tox             = Toxtank[i]
    Tpres           = Tfueltank[i]
    mu_fuel         = kerosene.mu
    mu_N2_fuel      = nitrogen.getViscosity(Preg_N2, Tfueltank[i])
    roo_N2_fuel     = nitrogen.getDensity(Preg_N2, Tfueltank[i])
    
    if i==0:    # First guesses. Based on choked flow at ox injector (multplied by 0.7 to adjust for better convergence)
        
        mdot_ox_guess   = ox_orifice.getMdot(Preg_ox, Pfire, Tox)*0.7
        P3ox_guess      = P2ox - ox_check.getPressureDrop(mdot_ox_guess, P2ox, GOX.roo_std, roo_ox, Tox)
        P4ox_guess      = P3ox_guess - oxSole.getPressureDrop(mdot_ox_guess, P3ox_guess, roo_ox)
        P5ox_guess      = P4ox_guess - ox_tube.getPressureDrop(mdot_ox_guess, mu_ox, roo_ox)
        
        #print("mdot_ox_guess is", mdot_ox_guess)
        #print("P2ox is", P2ox/psi, "psi")
        #print("P3ox_guess is", P3ox_guess/psi, "psi")
        #print("P4ox_guess is", P4ox_guess/psi, "psi")
        #print("P5ox_guess is", P5ox_guess/psi, "psi")
        #print("P_chamber is", Pchamber[i]/psi, "psi")
        
        mdot_fuel_guess = mdot_ox_guess/OF_nom
        P2fuel_guess    = P1fuel - fuelSole.getPressureDrop(mdot_fuel_guess, rooFuel)
        P3fuel_guess    = P2fuel_guess - fuel_tube.getPressureDrop(mdot_fuel_guess, mu_fuel, rooFuel)
        P4fuel_guess    = P3fuel_guess - jacket.getPressureDrop(mdot_fuel_guess)
        
        mdot_pres_guess = mdot_fuel_guess*roo_N2_fuel/rooFuel #volumetric flowrates of fuel and pressurant are the same
        P3pres_guess    = Preg_N2 - pres_check.getPressureDrop(mdot_pres_guess, Preg_N2, nitrogen.roo_std, roo_N2_fuel, Tpres)
        P4pres_guess    = P3pres_guess - presSole.getPressureDrop(mdot_pres_guess, P3pres_guess, roo_N2_fuel)
        P5pres_guess    = P4pres_guess - presfuel_tube.getPressureDrop(mdot_pres_guess, mu_N2_fuel, roo_N2_fuel)
        
        #print("mdot_pres_guess is is", mdot_pres_guess, "kg/s")
        #print("P3pres_guess is is", P3pres_guess/psi, "psi")
        #print("P4pres_guess is is", P4pres_guess/psi, "psi")
        #print("P5pres_guess is is", P5pres_guess/psi, "psi")
        
        
        #print("mdot_fuel_guess is", mdot_fuel_guess)
        #print("P2fuel is", P2fuel/psi, "psi")
        #print("P3fuel_guess is is", P3fuel_guess/psi, "psi")
        #print("P4fuel_guess is is", P4fuel_guess/psi, "psi")
        #print("P5fuel_guess is is", P5fuel_guess/psi, "psi")
        #print("P_chamber is", Pchamber[i]/psi, "psi")
      
    else :      # guesses for further steps. Use values from previous timestep
        mdot_ox_guess   = mdot_ox[i-1] #ox_orifice.getMdot(Preg_ox, Pchamb, Tox)
        #P3ox_guess      = P2ox - oxSole.getPressureDrop(mdot_ox_guess, P2ox,roo_ox)
        #P4ox_guess      = P3ox_guess - ox_tube.getPressureDrop(mdot_ox_guess, mu_ox, roo_ox)
        P3ox_guess      = P3ox[i-1]
        P4ox_guess      = P4ox[i-1]
        P5ox_guess      = P5ox[i-1]
        #print("mdot_ox_guess is", mdot_ox_guess)
        #print("P2ox is", P2ox/psi, "psi")
        #print("P3ox_guess is", P3ox_guess/psi, "psi")
        #print("P4ox_guess is", P4ox_guess/psi, "psi")
        #print("P_chamber is", Pchamber[i]/psi, "psi")
        
        mdot_fuel_guess  = mdot_fuel[i-1] #mdot_ox_guess/OF_nom*1
        P2fuel_guess    = P2fuel[i-1]
        P3fuel_guess    = P3fuel[i-1]
        P4fuel_guess    = P4fuel[i-1]
        #print("P2fuel is", P2fuel/psi, "psi")
        #print("P3fuel_guess is is", P3fuel_guess/psi, "psi")
        #print("P4fuel_guess is is", P4fuel_guess/psi, "psi")
        #print("P_chamber is", Pchamber[i]/psi, "psi")
        
        mdot_pres_guess = mdot_fuel_pres[i-1]
        P3pres_guess    = P3pres[i-1]
        P4pres_guess    = P4pres[i-1]
        
       
    initial_ox_guesses  = [P3ox_guess, P4ox_guess, P5ox_guess, mdot_ox_guess]
    initial_fuel_guesses= [P2fuel_guess, P3fuel_guess, P4fuel_guess, mdot_fuel_guess]
    initial_pres_guesses= [P3pres_guess, P4pres_guess, mdot_pres_guess]
    
    def oxfunks(U):       # defines the system of equations and unknowns U to be solved
        P3              = U[0]
        P4              = U[1]
        P5              = U[2]
        mdot            = U[3]
        
        #print("nyt TAALLA")
        
        #print("P3 as U0 is", P3/psi, "psi")
        #print("P4 as U1 is", P4/psi, "psi")
        #print("P5 as U2 is", P5/psi, "psi")
        #print("mdot as U3 is", mdot, "kg/s")
        
        #print("mdot is", mdot, "kg/s")
        #print("P4ox is", P4/psi, "psi")
        #print("Pchamb is", Pchamb/psi, "psi")
        #out             = [ P2ox - P3 - ox_check.getPressureDrop(mdot, P2ox, GOX.roo_std, roo_ox, Tox) ]
        out             = [ mdot - ox_check.getMdot(P2ox, P3, GOX.roo_std, roo_ox, Tox) ]
        out.append( P3 - P4 - oxSole.getPressureDrop( mdot, P3, roo_ox) )
        out.append( P4 - P5 - ox_tube.getPressureDrop(mdot, mu_ox, roo_ox) )
        out.append( mdot - ox_orifice.getMdot(P5, Pchamb, Tox) )
        
        #print("oxoutti", out)
        return out
    
    ox_solution         = opt.fsolve(oxfunks, initial_ox_guesses) # iterates until finds a solution or goes bust
    #print("ox solution is", ox_solution)

    mdot_ox_new         = ox_solution[3]
    #print("mdot_ox_nyyy is", mdot_ox_new, "kg/s")

    
    def fuelfunks(U):       # defines the system of equations and unknowns U to be solved
        P2              = U[0]
        P3              = U[1]
        P4              = U[2]
        mdot            = U[3]
        #print("U is", U)
        #print("fuelmdot is", mdot)
        out             = [ mdot - fuelSole.getMdot(P1fuel, P2, rooFuel, kerosene.P_crit, kerosene.P_vapor)  ]
        out.append( P2 - P3 - fuel_tube.getPressureDrop(mdot, mu_fuel, rooFuel) )
        out.append( P3 - P4 - jacket.getPressureDrop(mdot) ) 
        out.append( P4 - Pchamb - fuel_orifice.getPressureDrop(mdot, rooFuel) )
        
        #print("fueloutti", out)
        return out
    
    fuel_solution       = opt.fsolve(fuelfunks, initial_fuel_guesses)  
    #print("fuel solution is", fuel_solution) 
    mdot_fuel_new       = fuel_solution[3]
        

    # Now that fuel mass flow rate out has been solved, intermediate state (=no N2 inflow yet) of the fuel tank can be established:
    fueltank.update(TfuelPres[i], 0, mdot_fuel_new, timestep) 
    Pfuel_intermediate  = fueltank.getPtank()
    Pfuel_eff           = (Pfuel_intermediate + P1fuel)/2 # average of pressures before and after ejection of fuel from tank; incoming nitrogen will see this 'effective' pressure in the tank
    
    # Next, nitrogen flow into the void created by ejected fuel is calculated
    
    def presfunks(U):      # defines the system of equations and unknowns U to be solved
        P3              = U[0]
        P4              = U[1]
        mdot            = U[2]
        
        out             = [mdot - pres_check.getMdot(Preg_N2, P3, nitrogen.roo_std, roo_N2_fuel, Tpres) ]
        #out.append( P3 - P4 - presSole.getPressureDrop(mdot, P3, roo_N2_fuel) )
        out.append( mdot - presSole.getMdot(P3, P4, roo_N2_fuel) )
        #out.append( P4 - Pfuel_eff - presfuel_tube.getPressureDrop(mdot, mu_N2_fuel, roo_N2_fuel) )
        out.append( mdot - presfuel_tube.getMdot(P4, Pfuel_eff, mu_N2_fuel, roo_N2_fuel) )
        
        #print("presoutti", out)
        return out
    
    pres_solution       = opt.fsolve(presfunks, initial_pres_guesses) 
    #print("pres solution is", pres_solution)
    mdot_pres_new       = pres_solution[2]
    #print("mdot_pres_new is", mdot_pres_new, "kg/s")
    
    # Determine final conditions in prop tanks now that N2 inflow has been determined
    oxtank.update(mdot_ox_new, timestep)
    fueltank.update(TfuelPres[i], mdot_pres_new, 0, timestep)
    
    # ...and fuel pressurant tank 
    fuelprestank.update(mdot_pres_new, timestep)
    
    # Check if OFratio is within limits. If not, stop simulation (no CEA data beyond OFratio 0.5-3.0)
    if (mdot_ox_new/mdot_fuel_new) < 0.5 or (mdot_ox_new/mdot_fuel_new) > 3.0:
        print("OF ratio out of range, terminate")
        print("mdot_ox_new is", mdot_ox_new, "kg/s")
        print("mdot_fuel_new is", mdot_fuel_new, "kg/s")
        
        break
    
    # Update chamber parameters:
    chamber.update(mdot_ox_new, mdot_fuel_new, Pambient, timestep) # mdot_ox_in, mdot_fuel_in, Pambient, timestep

    # Check if ox or fuel tank will empty during this timestep. If so, stop simulation.
    
    if oxtank.getPtank() < Preg_ox:
        print("Ox tank reached regulation pressure (=empty) after", i, " iterations, ie", i*timestep, "seconds")
        print("remaining fuel", mfuel[i], "kg")
        print("remaining fuel prs", mfuelPres[i], "kg,", "i.e.", mfuelPres[i]/mfuelPres[0]*100, " % of initial amount")
        break
        
    if fueltank.getMprop() < 0:
        print("Fuel tank empty after", i, " iterations, ie", i*timestep, "seconds")
        print("remaining GOX", mox[i], "kg")
        print("remaining fuel prs", mfuelPres[i], "kg,", "i.e.", mfuelPres[i]/mfuelPres[0]*100, " % of initial amount")
        break
        
    if fuelprestank.getPtank() < Preg_N2:
        print("Out of fuel pressurant after", i, " iterations, ie", i*timestep, "seconds")
        print("remaining fuel", mfuel[i], "kg")
        print("remaining GOX", mox[i], "kg")
        break
        
    #update mass flow time histories. These are values during the CURRENT time step.
    if i==0:
        P3ox            = [ox_solution[0]]
        P4ox            = [ox_solution[1]]
        P5ox            = [ox_solution[2]]
        mdot_ox         = [ox_solution[3]]
        P2fuel          = [fuel_solution[0]]
        P3fuel          = [fuel_solution[1]]
        P4fuel          = [fuel_solution[2]]
        mdot_fuel       = [fuel_solution[3]]
        P3pres          = [pres_solution[0]]
        P4pres          = [pres_solution[1]]
        mdot_fuel_pres  = [pres_solution[2]]
        OFratio         = [ mdot_ox[0]/mdot_fuel[0] ]
    else:
        
        P3ox.append( ox_solution[0])
        P4ox.append( ox_solution[1])
        P5ox.append( ox_solution[2])
        mdot_ox.append( ox_solution[3])
        P2fuel.append( fuel_solution[0])
        P3fuel.append( fuel_solution[1])
        P4fuel.append( fuel_solution[2])
        mdot_fuel.append( fuel_solution[3])
        P3pres.append( pres_solution[0])
        P4pres.append( pres_solution[1])
        mdot_fuel_pres.append( pres_solution[2])
        #print("i is= ", i)
        OFratio.append( mdot_ox[i]/mdot_fuel[i])
    
    #update the rest of the time histories. System will have these values during the NEXT time step.
      
    Poxtank.append( oxtank.getPtank())
    Toxtank.append( oxtank.getTtank())
    mox.append( oxtank.getM())
    
    Pfueltank.append( fueltank.getPtank())
    Tfueltank.append( fueltank.getTpres())
    mPresFueltank.append( fueltank.getMpres())
    mfuel.append( fueltank.getMprop())
    FFfueltank.append( fueltank.getFF())
    TfuelPres.append( fuelprestank.getTtank())
    PfuelPres.append( fuelprestank.getPtank())
    mfuelPres.append( fuelprestank.getM())
    #mdot_fuel_pres.append( mdot_pres_new)
        
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
    mprs.append( mPresFueltank[i+1] + mfuelPres[i+1] )
    time.append( time[i]+timestep ) 
    
    i+=1
    
# Print some values

print("")
print("mdot_nozzle steady state (end of sim) is", '%.3f'%mdot_nozzle[-1], "kg/s")
print("SS thrust is", '%.1f'%Thrust[-1], "N")
print("SS Isp is", '%.1f'%Isp[-1], "s")
print("SS T_chamber is",'%.1f'%T_chamber[-1], "K")
print("SS P_chamber is", '%.1f'%(Pchamber[-1]/psi), "psi")
print("SS P_exit is", '%.3f'%(Pexit[-1]/atm), "atm")
print("SS thrust coeff is", '%.3f'%nozzle.getCf(Pchamber[-1], atm, chamber.get_gamma(OFratio[-1])) )
print("SS mdot_N2 is", '%.3f'%mdot_fuel_pres[-1], "kg/s")
print("SS N2 flow rate is", '%.3f'%(mdot_fuel_pres[-1]/roo_N2_fuel*1000/3.78*60), "GPM")
print("SS mdot_ox is", '%.3f'%mdot_ox[-1], "kg/s")
print("SS mdot_fuel is", '%.3f'%mdot_fuel[-1], "kg/s")
print("SS O/F ratio is", '%.3f'%OFratio[-1])
print("SS ox tube velocity is", '%.1f'%(mdot_ox[-1]/(roo_ox*pi*d_oxtube**2/4)), "m/s")
print("SS fuel tube velocity is", '%.1f'%(mdot_fuel[-1]/(rooFuel*pi*d_fueltube**2/4)), "m/s")
print("SS ox injection velocity is", '%.1f'%(mdot_ox[-1]/(roo_ox*pi*diameter_oxInjectorHoles**2/4*numOxInjectorHoles)), "m/s")
print("SS fuel injection velocity is", '%.1f'%(mdot_fuel[-1]/(rooFuel*pi*diameter_fuelInjectorHoles**2/4*numFuelHoles)), "m/s")
print("SS ox injector P_drop", '%.1f'%((P4ox[-1]-Pchamber[-1])/Pchamber[-1]*100), "% of Pchamber")
print("SS fuel injector P_drop", '%.1f'%((P4fuel[-1]-Pchamber[-1])/Pchamber[-1]*100), "% of Pchamber")
print("")

# See what check valves are doing
dP_ox_check = (Preg_ox - P3ox[-1])
dP_N2_check = (Preg_N2 - P3pres[-1])

if  dP_ox_check < ox_check.Pcrack:
    print("Warning: Pressure drop over ox check valve (",'%.1f'%(dP_ox_check/psi),"psi) is less than its cracking pressure (",ox_check.Pcrack/psi,"psi) and will remain shut")
else:
    print("Ox check valve pressure drop is", '%.1f'%(dP_ox_check/psi), "psi, enough to keep it flowing")
    
if  dP_N2_check < ox_check.Pcrack:
    print("Warning: Pressure drop over N2 check valve(",'%.1f'%(dP_N2_check/psi),"psi) is less than its cracking pressure (",pres_check.Pcrack/psi,"psi) and will remain shut")
else:
    print("N2 check valve pressure drop is", '%.1f'%(dP_N2_check/psi), "psi, enough to keep it flowing")
      

# following time histories are one element shorter than the rest, so the last calculated value will be duplicated to match the length of other time histories.

P3ox.append( ox_solution[0])
P4ox.append( ox_solution[1])
P5ox.append(  ox_solution[2])
mdot_ox.append( ox_solution[3])
P2fuel.append( fuel_solution[0])
P3fuel.append( fuel_solution[1])
P4fuel.append( fuel_solution[2])
mdot_fuel.append( fuel_solution[3])
P3pres.append( pres_solution[0])
P4pres.append( pres_solution[1])
mdot_fuel_pres.append( pres_solution[2])

OFratio.append( mdot_ox[i]/mdot_fuel[i])

# plot time histories
plt.ion()

Preg_ox_array = full((1, len(time)), Preg_ox/psi)
plt.figure(1)
plt.plot(time, array(Poxtank)/psi, label='ox tank')
plt.figure(1)
plt.plot(time, Preg_ox_array.T, label="P_regulation")
plt.figure(1)
plt.plot(time,array(P3ox)/psi, label='Pcheck_out')
plt.figure(1)
plt.plot(time,array(P4ox)/psi, label='Psolenoid_out')
plt.figure(1)
plt.plot(time,array(P5ox)/psi, label='Pinj_in')
plt.figure(1)
plt.plot(time,array(Pchamber)/psi, label='Pchamber')
plt.figure(1)
plt.plot(time,array(Pexit)/psi, label='Pexit')
plt.title('Ox pressures')
plt.legend( loc='upper right')
plt.xlabel('Time [s]')
plt.ylabel('psia')
plt.show()

Preg_N2_array = full((1, len(time)), Preg_N2/psi)
plt.figure(2)
plt.plot(time, array(PfuelPres)/psi, label='fuelpres tank')
plt.figure(2)
plt.plot(time, Preg_N2_array.T, label="P_regulation")
plt.figure(2)
plt.plot(time,array(P3pres)/psi, label='N2 check valve out')
plt.figure(2)
plt.plot(time,array(P4pres)/psi, label='N2 solenoid valve out')
plt.figure(2)
plt.plot(time,array(Pfueltank)/psi, label='fuel tank')
plt.figure(2)
plt.plot(time,array(P2fuel)/psi, label='Pvalve_out')
plt.figure(2)
plt.plot(time,array(P3fuel)/psi, label='Pjacket_in')
plt.figure(2)
plt.plot(time,array(P4fuel)/psi, label='Pinj_in')
plt.figure(2)
plt.plot(time,array(Pchamber)/psi, label='Pchamber')
plt.figure(2)
plt.plot(time,array(Pexit)/psi, label='Pexit')
plt.title('Fuel pressures')
plt.legend( loc='upper right')
plt.xlabel('Time [s]')
plt.ylabel('Psia')
plt.show()

plt.figure(3)
plt.plot(time,Toxtank, label='Ox tank')
plt.figure(3)
plt.plot(time,Tfueltank, label='Fuel tank')
plt.figure(3)
plt.plot(time,TfuelPres, label='fuel pressurant tank')
plt.title('Tank temperatures')
plt.legend( loc='lower left')
plt.xlabel('Time [s]')
plt.ylabel('K')
plt.show()

plt.figure(4)
plt.plot(time,mdot_ox, label='mdot_ox')
plt.figure(4)
plt.plot(time,mdot_fuel, label='mdot_fuel')
plt.figure(4)
plt.plot(time,mdot_nozzle, label='mdot_nozzle')
plt.figure(4)
plt.plot(time,mdot_fuel_pres, label='mdot_fuel_pres')
plt.title('Mass flows')
plt.xlabel('Time [s]')
plt.ylabel('kg/s')
plt.legend( loc='upper right')
plt.show()

plt.figure(5)
plt.plot(time,FFfueltank, label='fuel tank')
plt.title('Fill fractions in fuel tank (Vfuel_/Vtank)')
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
plt.plot(time,mox, label='GOX')
plt.figure(7)
plt.plot(time,mfuel, label='fuel')
plt.figure(7)
plt.plot(time,mfuelPres, label='fuel pressurant')
plt.figure(7)
plt.plot(time,mPresFueltank, label='pressurant in fuel tank')
plt.figure(7)
plt.plot(time,mprs, label='total pressurant')
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
y1  = PfuelPres[-1]/psi
y2  = Preg_N2/psi
y3  = P3pres[-1]/psi
y4  = P4pres[-1]/psi
y5  = Pfueltank[-1]/psi
y6  = P2fuel[-1]/psi
y7  = P3fuel[-1]/psi
y8  = P4fuel[-1]/psi
y9  = Pchamber[-1]/psi
plt.plot( [0, 1], [y1, y1], linewidth=2, label="Pressurant tank")
plt.plot( [1, 2], [y1, y2], linewidth=2, label="Regulator")
plt.plot( [2, 3], [y2, y3], linewidth=2, label="Check valve")
plt.plot( [3, 4], [y3, y4], linewidth=2, label="Pressurant solenoid")
plt.plot( [4, 5], [y4, y5], linewidth=2, label="Pressurant tubing")
plt.plot( [5, 6], [y5, y5], linewidth=2, label="Fuel tank")
plt.plot( [6, 7], [y5, y6], linewidth=2, label="Fuel solenoid")
plt.plot( [7, 8], [y6, y7], linewidth=2, label="Piping")
plt.plot( [8, 9], [y7, y8], linewidth=2, label="Cooling jacket")
plt.plot( [9, 10], [y8, y9], linewidth=2, label="Fuel injector")
plt.plot( [10, 11], [y9, y9], linewidth=2, label="Chamber")
plt.title('Fuel line pressures at end of burn')
plt.ylabel('psi')
plt.legend( loc='upper right')

plt.figure(14)
y1  = Poxtank[-1]/psi
y2  = Preg_ox/psi
y3  = P3ox[-1]/psi
y4  = P4ox[-1]/psi
y5  = P5ox[-1]/psi
y6  = Pchamber[-1]/psi
plt.plot( [0, 1], [y1, y1], linewidth=2, label="Ox tank")
plt.plot( [1, 2], [y1, y2], linewidth=2, label="Regulator")
plt.plot( [2, 3], [y2, y3], linewidth=2, label="Check valve")
plt.plot( [3, 4], [y3, y4], linewidth=2, label="Ox  solenoid")
plt.plot( [4, 5], [y4, y5], linewidth=2, label="Tubing")
plt.plot( [5, 6], [y5, y6], linewidth=2, label="Ox injector")
plt.plot( [6, 7], [y6, y6], linewidth=2, label="Chamber")
plt.title('Ox line pressures at end of burn')
plt.ylabel('psi')
plt.legend( loc='upper right')

