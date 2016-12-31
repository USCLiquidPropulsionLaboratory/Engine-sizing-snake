## Functions to obtain friction factor from Moody diagram
#@ Author Juha Nieminen

from numpy import sqrt, log

def getDhCorrectionFactor(OD,ID):

    # Correction factor for hydraulic diameter, Dh, for annular flow channels
    # D_eff = Dh/ksi

    a       = OD/2
    b       = ID/2
    return (a-b)**2*(a**2-b**2)/(a**4-b**4-((a**2-b**2)**2/log(a/b)))
    
def getAnnularF(Re, OD, ID, mu, v, rou):

    ksi     = getDhCorrectionFactor(OD,ID)
    Dh      = OD-ID                         # Hydraulic diameter, m
    Dh_eff  = Dh/ksi                        # Effective hydraulic diameter, m
    rou_eff = rou*ksi                       # DIMENSIONLESS effective surface roughness, i.e. roughness[m]/(Dh[m]/ksi)
    Re_eff  = Re/ksi                        # Effective Reynold's number
    #print("Re is", Re)
    #print("Re_eff is", Re_eff)
    #print("ksi is", ksi)
    #print("rou is", rou)
    #print("rou_eff is", rou_eff)

    C1      = (-2.457*log( (7/Re_eff)**0.9 + 0.27*rou_eff) )**16
    C2      = (37530/Re_eff)**16
    f       = 8*( (8/Re_eff)**12 + (C1+C2)**-1.5 )**(1/12)   # Darcy-Weisbach friction factor, dimensionless
    #print("f is", f)
    
    return f

