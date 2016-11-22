## LOX-IPA sim
#@ Author Juha Nieminen

import sys
sys.path.insert(0, '/Users/juhanieminen/Documents/adamrocket')

import RocketComponents as rc
from physical_constants import poise, inches, Runiv, gallons, lbm, \
    gearth, atm, psi, lbf
from numpy import pi, linspace, cos, radians, sqrt, exp, log, array
from scipy import optimize as opt
import matplotlib.pyplot as plt
import Flows1D as flows


#DESIGN VARIABLES____________________________________________________________________________________

# nominal parameters
Preg            = 450*psi          # regulated pressurant outlet pressure [Pa]

mdot_fuel_nom   = 0.2               # This is only for cooling jacket pressure drop purposes [kg/s]
Pdrop_jacket_nom= 1*psi             # Cooling jacket pressure drop at mdot_nominal [Pa]
OF_nom          = 2.25              # Oxidizer-to-fuel ratio. This has only effect on initial guesses during solving

# Pressurant tank dimensions
Vprestank       = 0.08              # N2 pressurant tank volume [m3]

# Propellant tank dimensions
Vfueltank       = 0.006             # fuel tank volume
Voxtank         = 0.025             # ox tank volume [m3]

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
Cv_ox_valve     = 17                # oxidizer valve characteristic parameter, dimensionless 
Cv_fuel_valve   = 17                # fuel valve characteristic parameter, dimensionless

# Injector

cd_oxInjector   = 0.767                                                 # orifice discharge coefficient
diameter_oxInjectorHoles = 2.0e-3 #number xx drill                        # ox orifice diameter [m]
#length_oxHole   = 0.005                                                # ox orifice length [m]
numOxInjectorHoles = 28                                                 # number of ox orifices in the injector
area_oxInjector = numOxInjectorHoles*pi*diameter_oxInjectorHoles**2/4   # total ox flow area [m2]

cd_fuelInjector = 0.767                                                 # orifice discharge coefficient
diameter_fuelInjectorHoles = 1.7e-3 #number xx drill                    # fuel orifice diameter [m]
numFuelHoles    = 40                                                    # number of fuel orifices in the injector
area_fuelInjector = numFuelHoles*pi*diameter_fuelInjectorHoles**2/4     # total fuel flow area [m2]

# Define initial/nominal conditions in the chamber (obtained from CEA code assuming OFratio = 1.2)
TfireInit       = 293                                                   # initial flame temperature [K]
Tfire_nom       = 2675                                                  # nominal flame temperature [K]
Pfire           = 1*atm                                                 # initial chamber pressure [Pa]
gammaFireInit   = 1.16                                                  # dimensionless
ga              = gammaFireInit
mbarFireInit    = 19.10                              # combustion products' initial molecular mass [kg/kmol]
RfireInit       = Runiv/mbarFireInit                 # combustion products' initial specific gas constant [J/kgK]
Pambient        = atm                                                   # ambient pressure [Pa]

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

TfuelPresStart  = 293                   # Fuel pressurant (=nitrogen) temp [K]
PfuelPrestankStart  = 3000*psi          # Fuel pressurant tank pressure [Pa]
ToxPresStart    = 293                   # Ox pressurant (=nitrogen) temp [K]
PoxPrestankStart  = 3000*psi            # Ox pressurant tank pressure [Pa]

ToxStart        = 90                    # Oxidizer (LOX) temp [K] 
PoxtankStart    = Preg                  # Oxidizer tank pressure [Pa]
TfuelStart      = 293                   # Fuel temp [K]
PfueltankStart  = Preg                  # Fuel tank pressure [Pa]

FFoxtankStart   = 0.5                   # oxidizer tank fill fraction, dimensionless
FFfueltankStart = 0.5                   # fuel tank fill fraction (Vfuel/Vtank)

# initialize propellants
IPA             = rc.IPAFluid()
nitrogen        = rc.NitrogenFluid()
LOX             = rc.LOXFluid()

#initialize nozzle and chamber
nozzle          = rc.ConvergingDivergingNozzle(A_nozzleExit, A_nozzleThroat)
chamber         = rc.LOX_IPACombustionChamber(nozzle, Vchamber, TfireInit, ga, mbarFireInit, Pfire, atm)

#initialize injector orifices
ox_orifice      = rc.LiquidOrifice(area_oxInjector, cd_oxInjector )
fuel_orifice    = rc.LiquidOrifice(area_fuelInjector, cd_fuelInjector )

#initialize pressurant tanks
oxprestank      = rc.IdealgasTank(nitrogen, Vprestank, ToxPresStart, PoxPrestankStart)
fuelprestank    = rc.IdealgasTank(nitrogen, Vprestank, TfuelPresStart, PfuelPrestankStart)

#initialize propellant tanks
oxtank          = rc.LiquidPropellantTank(nitrogen, LOX, Voxtank, ToxStart, ToxPresStart,\
                    PoxtankStart, FFoxtankStart, Preg)
fueltank        = rc.LiquidPropellantTank(nitrogen, IPA, Vfueltank, TfuelStart, TfuelPresStart,\
                    PfueltankStart, FFfueltankStart, Preg)

#initialize vent holes
fuelVent        = rc.VentHole(d_fueltank_vent, nitrogen.gamma, Runiv/nitrogen.mbar, fuel_vent_cd)
oxVent          = rc.VentHole(d_oxtank_vent, nitrogen.gamma, Runiv/nitrogen.mbar, ox_vent_cd)

#initialize solenoids
oxSole          = rc.IncompressibleFlowSolenoid( Cv_ox_valve)
fuelSole        = rc.IncompressibleFlowSolenoid( Cv_fuel_valve)

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

ToxPres         = [oxprestank.getTtank()]           # temperature in ox pressurant tank [K]
PoxPres         = [oxprestank.getPtank()]           # pressure in ox pressurant tank [Pa]
moxPres         = [oxprestank.getM()]               # pressurant mass in ox pressurant tank [Pa]
mdot_ox_pres    = [0]                               # ox pressurant mass flow rate [kg/s]

Pfueltank       = [fueltank.getPtank()]             # fuel tank pressure [Pa]
Tfueltank       = [fueltank.getTpres()]             # pressurant temperature in fuel tank[K]
mPresFueltank   = [fueltank.getMpres()]             # pressurant mass in fuel tank [kg]
mfuel           = [fueltank.getMprop()]             # fuel mass in tank [kg]
FFfueltank      = [fueltank.getFF()]                # fuel tank fill fraction defined as Vfuel/(Vfueltank)

TfuelPres       = [fuelprestank.getTtank()]         # temperature in fuel pressurant tank [K]
PfuelPres       = [fuelprestank.getPtank()]         # pressure in fuel pressurant tank [Pa]
mfuelPres       = [fuelprestank.getM()]             # pressurant mass in fuel pressurant tank [Pa]
mdot_fuel_pres  = [0]                               # fuel pressurant mass flow rate [kg/s]

mdot_ox         = [0]                               # liq ox mass flow out of the tank [kg/s]
rooOx           = oxtank.propellant.density         # liq ox density, assumed constant [kg/m^3]
P2ox            = [0]                               # ox tank presssure [Pa]
P3ox            = [0]                               # ox solenoid outlet pressure [Pa]      
P4ox            = [0]                               # ox cooling jacket inlet pressure [Pa]
P5ox            = [0]                               # ox injector inlet pressure [Pa]

mdot_fuel       = [0]                               # fuel mass flow out of the tank [kg/s]
rooFuel         = fueltank.propellant.density       # fuel density, assumed constant [kg/m3]
P2fuel          = [0]                               # fuel tank presssure [Pa]
P3fuel          = [0]                               # fuel solenoid outlet pressure [Pa]      
P4fuel          = [0]                               # fuel cooling jacket inlet pressure [Pa]
P5fuel          = [0]                               # fuel injector inlet pressure [Pa]

mTotal          = [0]                               # propellant mass in the system [kg]
mprs            = [moxPres[0]+mfuelPres[0]+mPresOxtank[0]+mPresFueltank[0]]         # pressurant mass [kg]
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
print("mN2_fuel_start is", '%.2f'%mfuelPres[0], "kg")
print("mN2_ox_start is", '%.2f'%moxPres[0], "kg")

# The first step is to solve oxidizer and fuel mass flow rates from the tank to combustion chamber.

# definitions:

# P1ox              = nitrogen tank pressure
# P2ox              = fuel tank pressure
# P3ox              = ox valve outlet pressure
# P4ox              = injector inlet, pressure 
# (P1ox-P2ox)       = regulator+tubing pressure drop,           nitrogen flow, is solved separately from fuel flow 
# (P2ox-P3ox)       = ox valve pressure drop,                   eq 1
# (P3ox-P4ox)       = ox tubing pressure drop,                  eq 2
# (P4ox-Pchamber)   = injector pressure drop,                   eq 3

# P1fuel            = nitrogen tank pressure
# P2fuel            = fuel tank pressure
# P3fuel            = fuel valve outlet pressure
# P4fuel            = cooling jacket inlet pressure
# P5fuel            = injector inlet pressure
# (P1fuel-P2fuel)   = regulator+tubing pressure drop,           nitrogen flow, is solved separately from fuel flow 
# (P2fuel-P3fuel)   = fuel valve pressure drop,                 eq1
# (P3fuel-P4fuel)   = fuel tubing pressure drop,                eq2
# (P4fuel-P5fuel)   = cooling jacket pressure drop,             eq3
# (P5fuel-Pchamber) = injector pressure drop,                   eq4

# In the case of oxidizer, P2 and Pchamber are known, so one must solve for P3 & P4. Third unknown is the mass flow rate. The three equations are injector and tubing pressure drops, and expression for solenoid mass flow rate. They are defined in RocketComponents.py under their respective classes.

# With fuel P2 and Pchamber are known, so one must solve for P3, P4, and P5. Fourth unknown is mass flow rate.

# fsolve requires sensible initial guesses for all unknowns. They are established by guessing the mass flow rate, because all other pressures trickle down from that.

time                = [0]
timestep_nom        = 0.0001    # seconds

i=0
for i in range(0,2000):
    if time[i] < 0.01:
        timestep    = 5e-6                             # use shorter timestep during initial transient
    else: timestep    = timestep_nom 
#while True:
    #print("i=", i) 
    P1ox            = PoxPres[i]
    P2ox            = Poxtank[i]
    P1fuel          = PfuelPres[i]
    P2fuel          = Pfueltank[i]
    Pchamb          = Pchamber[i]
    mu_ox           = LOX.mu
    mu_fuel         = IPA.mu 
    mu_N2_ox        = nitrogen.getViscosity(Preg, Toxtank[i])
    roo_N2_ox       = nitrogen.getDensity(Preg, Toxtank[i])
    mu_N2_fuel      = nitrogen.getViscosity(Preg, Tfueltank[i])
    roo_N2_fuel     = nitrogen.getDensity(Preg, Tfueltank[i])
    
    Pcrit_ox        = LOX.P_crit
    Pvapor_ox       = LOX.P_vapor
    
    if i==0:    # First guesses. Based on choked flow at ox injector
        
        #nominal values for guessing
        Pc_nom          = 350*psi   
        mdot_nominal    = A_nozzleThroat*sqrt(ga)*Pc_nom/sqrt(RfireInit*TfireInit)/(1+(ga-1)/2)**((ga+1)/(2*ga-2))/5  
        
        mdot_ox_guess   = mdot_nominal*OF_nom/(OF_nom+1)
        mdot_fuel_guess = mdot_ox_guess/OF_nom
        print("mdot_ox_guess is", mdot_ox_guess)
        P3ox_guess      = P2ox - oxSole.getPressureDrop(mdot_ox_guess, rooOx)
        P4ox_guess      = P3ox_guess - ox_tube.getPressureDrop(mdot_ox_guess, mu_ox, rooOx)
        print("P2ox is", P2ox/psi, "psi")
        print("P3ox_guess is", P3ox_guess/psi, "psi")
        print("P4ox_guess is", P4ox_guess/psi, "psi")
        print("P_chamber is", Pchamber[i]/psi, "psi")
        
        print("mdot_fuel_guess is", mdot_fuel_guess)
        P3fuel_guess    = P2fuel - fuelSole.getPressureDrop(mdot_fuel_guess, rooFuel)
        P4fuel_guess    = P3fuel_guess - fuel_tube.getPressureDrop(mdot_fuel_guess, mu_fuel, rooFuel)
        P5fuel_guess    = P4fuel_guess - jacket.getPressureDrop(mdot_fuel_guess)
        print("P2fuel is", P2fuel/psi, "psi")
        print("P3fuel_guess is is", P3fuel_guess/psi, "psi")
        print("P4fuel_guess is is", P4fuel_guess/psi, "psi")
        print("P5fuel_guess is is", P5fuel_guess/psi, "psi")
        print("P_chamber is", Pchamber[i]/psi, "psi")
      
    else :      # using solutions from previous timestep
        mdot_ox_guess   = mdot_nominal*OF_nom/(OF_nom+1)
        P3ox_guess      = P2ox - oxSole.getPressureDrop(mdot_ox_guess, rooOx)
        P4ox_guess      = P3ox_guess - ox_tube.getPressureDrop(mdot_ox_guess, mu_ox, rooOx)
        #P3ox_guess      = P3ox[i-1]
        #P4ox_guess      = P4ox[i-1]
        #print("mdot_ox_guess is", mdot_ox_guess)
        #print("P2ox is", P2ox/psi, "psi")
        #print("P3ox_guess is", P3ox_guess/psi, "psi")
        #print("P4ox_guess is", P4ox_guess/psi, "psi")
        #print("P_chamber is", Pchamber[i]/psi, "psi")
        
        
        mdot_fuel_guess = mdot_ox_guess/OF_nom*0.5
        P3fuel_guess    = P2fuel - fuelSole.getPressureDrop(mdot_fuel_guess, rooFuel)
        P4fuel_guess    = P3fuel_guess - fuel_tube.getPressureDrop(mdot_fuel_guess, mu_fuel, rooFuel)
        P5fuel_guess    = P4fuel_guess - jacket.getPressureDrop(mdot_fuel_guess)
        #mdot_fuel_guess = mdot_fuel[i-1]*1.0
        #P3fuel_guess    = P3fuel[i-1]
        #P4fuel_guess    = P4fuel[i-1]
        #P5fuel_guess    = P5fuel[i-1]
        #print("P2fuel is", P2fuel/psi, "psi")
        #print("P3fuel_guess is is", P3fuel_guess/psi, "psi")
        #print("P4fuel_guess is is", P4fuel_guess/psi, "psi")
        #print("P5fuel_guess is is", P5fuel_guess/psi, "psi")
        #print("P_chamber is", Pchamber[i]/psi, "psi")
        
    initial_ox_guesses  = [P3ox_guess, P4ox_guess, mdot_ox_guess]
    initial_fuel_guesses= [P3fuel_guess, P4fuel_guess, P5fuel_guess, mdot_fuel_guess]
    
    def oxfunks(U):       # defines the system of equations and unknowns U to be solved
        P3              = U[0]
        P4              = U[1]
        mdot            = U[2]
        
        #print("P3 as U0 is", P3/psi, "psi")
        #print("P4 as U1 is", P4/psi, "psi")
        #print("mdot as U2 is", mdot, "kg/s")
        
        out             = [ mdot - oxSole.getMdot(P2ox, P3, rooOx, Pcrit_ox, Pvapor_ox) ]
        out.append( P3 - P4 - ox_tube.getPressureDrop(mdot, mu_ox, rooOx) )
        out.append( P4 - Pchamb - ox_orifice.getPressureDrop(mdot, rooOx) )
        #print("oxoutti", out)
        return out
    
    ox_solution         = opt.fsolve(oxfunks, initial_ox_guesses) # iterates until finds a satisfying solution or goes bust
    #print("ox solution is", ox_solution)
    mdot_ox_new         = ox_solution[2]
    
    #mdot_ox_pres_new    = presox_tube.getMdot(Preg, oxtank.getPtank(), mu_N2_ox, roo_N2_ox)
    #print("mdot_ox_pres_new is", mdot_ox_pres_new, "kg/s")
    #oxtank.update(ToxPres[i], mdot_ox_pres_new, 0, timestep)
    #oxprestank.update(mdot_ox_pres_new, timestep)
    
    
    def fuelfunks(U):       # defines the system of equations and unknowns U to be solved
        P3              = U[0]
        P4              = U[1]
        P5              = U[2]
        mdot            = U[3]
        #print("U is", U)
        #print("fuelmdot is", mdot)
        out             = [ mdot - fuelSole.getMdot(P2fuel, P3, rooFuel, IPA.P_crit, IPA.P_vapor)  ]
        out.append( P3 - P4 - fuel_tube.getPressureDrop(mdot, mu_fuel, rooFuel) )
        out.append( P4 - P5 - jacket.getPressureDrop(mdot) ) 
        out.append( P5 - Pchamb - fuel_orifice.getPressureDrop(mdot, rooFuel) )
        
        #print("fueloutti", out)
        return out
    
    fuel_solution       = opt.fsolve(fuelfunks, initial_fuel_guesses)  
    #print("fuel solution is", fuel_solution) 
    mdot_fuel_new       = fuel_solution[3]
        
    # Now that mass flow rates out have been solved, intermediate states of the prop tanks can be established:
    oxtank.update(ToxPres[i], 0 , mdot_ox_new, timestep) #T_in, mdot_pres_in, mdot_prop_out, timestep, i):
    fueltank.update(TfuelPres[i], 0, mdot_fuel_new, timestep) 
    
    # Get pressurant mass flow rates in:
    mdot_ox_pres_new    = presox_tube.getMdot(Preg, oxtank.getPtank(), mu_N2_ox, roo_N2_ox)
    mdot_fuel_pres_new  = presfuel_tube.getMdot(Preg, fueltank.getPtank(), mu_N2_fuel, roo_N2_fuel)
    
    # Determine final conditions in prop tanks
    oxtank.update(ToxPres[i], mdot_ox_pres_new, 0, timestep)
    fueltank.update(TfuelPres[i], mdot_fuel_pres_new, 0, timestep)
    
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
        print("remaining ox prs", moxPres[i], "kg,", "i.e.", moxPres[i]/moxPres[0]*100, " % of initial amount")
        print("remaining fuel prs", fuelPres[i], "kg,", "i.e.", mfuelPres[i]/mfuelPres[0]*100, " % of initial amount")
        break
        
    if oxprestank.getPtank() < 1000*psi:
        print("Out of ox pressurant after", i, " iterations, ie", i*timestep, "seconds")
        print("remaining fuel", mfuel[i], "kg")
        print("remaining LOX", moxl[i], "kg")
        print("remaining fuel prs", fuelPres[i], "kg,", "i.e.", mfuelPres[i]/mfuelPres[0]*100, " % of initial amount")
        break 
            
    if fueltank.getMprop() < 0:
        print("Fuel tank empty after", i, " iterations, ie", i*timestep, "seconds")
        print("remaining LOX", mox[i], "kg")
        print("remaining ox prs", moxPres[i], "kg,", "i.e.", moxPres[i]/moxPres[0]*100, " % of initial amount")
        print("remaining fuel prs", fuelPres[i], "kg,", "i.e.", mfuelPres[i]/mfuelPres[0]*100, " % of initial amount")
        break
        
    if fuelprestank.getPtank() < 1000*psi:
        print("Out of fuel pressurant after", i, " iterations, ie", i*timestep, "seconds")
        print("remaining fuel", mfuel[i], "kg")
        print("remaining LOX", moxl[i], "kg")
        print("remaining ox prs", oxPres[i], "kg,", "i.e.", moxPres[i]/moxPres[0]*100, " % of initial amount")
        break
        
    #update mass flow time histories. These are values during the CURRENT time step.
    if i==0:
        P3ox            = [ox_solution[0]]
        P4ox            = [ox_solution[1]]
        mdot_ox         = [ox_solution[2]]
        P3fuel          = [fuel_solution[0]]
        P4fuel          = [fuel_solution[1]]
        P5fuel          = [fuel_solution[2]]
        mdot_fuel       = [fuel_solution[3]]
        OFratio         = [ mdot_ox[0]/mdot_fuel[0] ]
    else:
        
        P3ox.append( ox_solution[0])
        P4ox.append( ox_solution[1])
        mdot_ox.append( ox_solution[2])
        P3fuel.append( fuel_solution[0])
        P4fuel.append( fuel_solution[1])
        P5fuel.append( fuel_solution[2])
        mdot_fuel.append( fuel_solution[3])
        #print("i is= ", i)
        OFratio.append( mdot_ox[i]/mdot_fuel[i])
    
    #update the rest of the time histories. System will have these values during the NEXT time step.
      
    Poxtank.append( oxtank.getPtank())
    Toxtank.append( oxtank.getTpres())
    mPresOxtank.append( oxtank.getMpres())
    mox.append( oxtank.getMprop())
    FFoxtank.append( oxtank.getFF())
    ToxPres.append( oxprestank.getTtank())
    PoxPres.append( oxprestank.getPtank())
    moxPres.append( oxprestank.getM())
    mdot_ox_pres.append( mdot_ox_pres_new)
    
    Pfueltank.append( fueltank.getPtank())
    Tfueltank.append( fueltank.getTpres())
    mPresFueltank.append( fueltank.getMpres())
    mfuel.append( fueltank.getMprop())
    FFfueltank.append( fueltank.getFF())
    TfuelPres.append( fuelprestank.getTtank())
    PfuelPres.append( fuelprestank.getPtank())
    mfuelPres.append( fuelprestank.getM())
    mdot_fuel_pres.append( mdot_fuel_pres_new)
        
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
print("mdot_nozzle steady state is", '%.3f'%mdot_nozzle[-1], "kg/s")
print("SS thrust is", '%.1f'%Thrust[-1], "N")
print("SS Isp is", '%.1f'%Isp[-1], "s")
print("SS T_chamber is",'%.1f'%T_chamber[-1], "K")
print("SS P_chamber is", '%.1f'%(Pchamber[-1]/psi), "psi")
print("SS P_exit is", '%.3f'%(Pexit[-1]/atm), "atm")
print("SS thrust coeff is", '%.3f'%nozzle.getCf(Pchamber[-1], atm, chamber.get_gamma(OFratio[-1],Pchamber[-1] )) )
print("SS mdot_N2_fuel is", '%.3f'%mdot_fuel_pres[-1], "kg/s")
print("SS mdot_N2_ox is", '%.3f'%mdot_ox_pres[-1], "kg/s")
print("SS N2_fuel flow rate is", '%.3f'%(mdot_fuel_pres[-1]/roo_N2_fuel*1000/3.78*60), "GPM")
print("SS N2_ox flow rate is", '%.3f'%(mdot_ox_pres[-1]/roo_N2_ox*1000/3.78*60), "GPM")
print("SS mdot_ox is", '%.3f'%mdot_ox[-1], "kg/s")
print("SS mdot_fuel is", '%.3f'%mdot_fuel[-1], "kg/s")
print("SS O/F ratio is", '%.3f'%OFratio[-1])
print("SS ox tube velocity is", '%.1f'%(mdot_ox[-1]/(rooOx*pi*d_oxtube**2/4)), "m/s")
print("SS fuel tube velocity is", '%.1f'%(mdot_fuel[-1]/(rooFuel*pi*d_fueltube**2/4)), "m/s")
print("SS ox injection velocity is", '%.1f'%(mdot_ox[-1]/(rooOx*pi*diameter_oxInjectorHoles**2/4*numOxInjectorHoles)), "m/s")
print("SS fuel injection velocity is", '%.1f'%(mdot_fuel[-1]/(rooFuel*pi*diameter_fuelInjectorHoles**2/4*numFuelHoles)), "m/s")
print("SS ox injector P_drop", '%.1f'%((P4ox[-1]-Pchamber[-1])/Pchamber[-1]*100), "% of Pchamber")
print("SS fuel injector P_drop", '%.1f'%((P5fuel[-1]-Pchamber[-1])/Pchamber[-1]*100), "% of Pchamber")

# following time histories are one element shorter than the rest, so the last calculated value will be duplicated to match the length of other time histories.

P3ox.append( ox_solution[0])
P4ox.append( ox_solution[1])
mdot_ox.append( ox_solution[2])
P3fuel.append( fuel_solution[0])
P4fuel.append( fuel_solution[1])
P5fuel.append( fuel_solution[2])
mdot_fuel.append( fuel_solution[3])

OFratio.append( mdot_ox[i]/mdot_fuel[i])

# plot time histories

plt.figure(1)
plt.plot(time,array(PoxPres)/psi, label='oxpres tank')
plt.figure(1)
plt.plot(time,array(Poxtank)/psi, label='ox tank')
plt.figure(1)
plt.plot(time,array(P3ox)/psi, label='Pvalve_out')
plt.figure(1)
plt.plot(time,array(P4ox)/psi, label='Pinj_in')
plt.figure(1)
plt.plot(time,array(Pchamber)/psi, label='Pchamber')
plt.figure(1)
plt.plot(time,array(Pexit)/psi, label='Pexit')
plt.title('Ox pressures')
plt.legend( loc='upper right')
plt.xlabel('Time [s]')
plt.ylabel('psia')
plt.show()

plt.figure(2)
plt.plot(time,array(PfuelPres)/psi, label='fuelpres tank')
plt.figure(2)
plt.plot(time,array(Pfueltank)/psi, label='fuel tank')
plt.figure(2)
plt.plot(time,array(P3fuel)/psi, label='Pvalve_out')
plt.figure(2)
plt.plot(time,array(P4fuel)/psi, label='Pjacket_in')
plt.figure(2)
plt.plot(time,array(P5fuel)/psi, label='Pinj_in')
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
plt.plot(time,ToxPres, label='ox pressurant tank')
plt.figure(3)
plt.plot(time,Tfueltank, label='Fuel tank')
plt.figure(3)
plt.plot(time,TfuelPres, label='fuel pressurant tank')
plt.title('Tank temperatures')
plt.legend( loc='upper right')
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
plt.plot(time,mdot_ox_pres, label='mdot_ox_pres')
plt.figure(4)
plt.plot(time,mdot_fuel_pres, label='mdot_fuel_pres')
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
plt.plot(time,mox, label='LOX')
plt.figure(7)
plt.plot(time,moxPres, label='ox pressurant')
plt.figure(7)
plt.plot(time,mfuel, label='fuel')
plt.figure(7)
plt.plot(time,mfuelPres, label='fuel pressurant')
plt.figure(7)
plt.plot(time,mPresOxtank, label='pressurant in LOX tank')
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
