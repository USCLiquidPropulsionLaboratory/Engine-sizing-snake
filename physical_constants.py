##Useful physical constants and unit conversions.
#Note: This file is all in mks.
#@author Alexander Adams
#edited for psi=6894.757 Juha Nieminen 5/11/1014

h = 6.62606957e-34 #planck's constant
kb = 1.3806488e-23#boltzman constant
Runiv = 8.3144621e3#J/kmolK universal gass constant
Navo = 6.022e23#avagadros number
G = 6.67384e-11#gravitational constant
sigma = 5.67e-8#stephen boltzmann constant
epsilon0 = 8.85e-12#C^2/Nm^2 permativity of free space/electric constnat
mu0 = 1.25663706e-6#permeability of free space/magnetic constant
c = 299792458. #speed of light
gearth = 9.80665#m/s^2

##########################
#unit convertion factors##
##########################
poise = 0.1
#to use, multiply something in non mks by its conversion to factor to get
#in mks. For example 9*eV will give the value of 9eV in J
eV = 1.602176565e-19 #electron volt
qe = eV#fundamental charge
#distance units
nmi = 1852.#nautical miles
miles = 1609.
ft = 0.3048#feet
inches = ft/12
#pressure units
atm = 101325. #atmosphere
psi = 6894.757#pounds per square inch
mmHg = 133.322387415 #mm mercury
inHg = 3386.#inches mercury
#force units
lbf = 4.448#pound force
#mass units
lbm = 0.4536#pound mass
amu = 1.660468e-27#atomic mass unit
me = 9.1093821545e-31#mass of electron
#volume units
liters = 1/1000.
gallons = 0.00378541
#units of time
minutes = 60.
hours = 60*minutes
sidereal_days =  86164.0905
jyear = 86400*365.25#julian year
