# delta G stuff for review article
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


# General parameters
T_low = 273 + 20
T_high = 273 + 120
P_low = 0.15 # bar
P_high = 1. # bar

# no cyclization
dHn = -70 # kJ
dSn = -.2 # kJ/mol K

# yes cyclization
dHy = -90 # kJ
dSy = -.26 # kJ/mol K

# extreme small effect
dHxs = -99 # kJ
dSxs = -0.232 # kJ/mol K

# extreme large effect
dHxl = -78 # kJ
dSxl = -0.288 # kJ/mol K


##### AUXILIARY #####
def make_dG(dH, dS):
    def dG(T):
        return dH - T * dS
    return dG

def cyclic_capacity(T_low, T_high, P_low, P_high, dH, dS, N=1, verbose=False):
    """ Assumes 1 mole of compound in a 1 L volume.
        T_: low and high temperature ends (K)
        P_: low and high CO2 partial pressure ends (bar)
        dH: Total binding (free CO2 -> bound CO2) enthalpy (kJ/mol)
        dS: Total binding entropy (kJ/mol K)
        N: binding_ratio, how many molecules of sorbent per molecule of CO2
            Assumes one molecule gets CO2 bound, and then N-1 others are inactivated
        Returns: Molar cyclic capacity (mol CO2/mol substance)"""
    # calculate equilibrium constants at T_low and T_high
    R = .008314 # kJ/mol K
    dG = make_dG(dH, dS)
    # G = -RTlnK
    K_low = np.exp(-dG(T_low) / (R*T_low))
    K_high = np.exp(-dG(T_high) / (R*T_high))
    # Solve for amount of CO2 dissolved. Example cases:
    # 1:1
    # K*P_CO2 = [product (spent molecules)] / [reactant (active molecules)]
    # N:1
    # K*P_CO2 = [product]^N / [reactant]^N, e.g. [RNHCOO-][RNH3+] / [RNH2]^2 for N=2
    p_r_low = np.power(P_low * K_low, 1/N) # ratio of [product] / [reactant]
    p_r_high = np.power(P_high * K_high, 1/N)
    # Where [product] + [reactant] = 1
    product_low = 1 / (1 + 1/p_r_low)
    product_high = 1 / (1 + 1/p_r_high)
    # Convert from [product] to [CO2], then take difference
    capacity = (product_low - product_high) / N
    # Printouts
    if verbose:
        print(
            f"Desorption K: {K_high}\nAbsorption K: {K_low}\nLean product concentration: {product_low}\nRich product concentration: {product_high}\nCyclic capacity: {capacity}"
        )
    return capacity
    
if __name__ == "__main__":

    ##### PLOTTING DELTA G #####
    Ts = np.linspace(0, 500, 3)

    fn = make_dG(dHn, dSn)
    fy = make_dG(dHy, dSy)

    fxs = make_dG(dHxs, dSxs)
    fxl = make_dG(dHxl, dSxl)

    plt.tight_layout()

    plt.axhline(y=0, color='k', linestyle='dotted')
    plt.axvspan(T_low, T_high, alpha=0.3, color='lightgray')

    plt.plot(Ts, fn(Ts), label='no H-bond')
    plt.plot(Ts, fy(Ts), label='with H-bond')
    plt.plot(Ts, fxs(Ts), label='strong H-bond small $\Delta S$', linestyle='dotted')
    plt.plot(Ts, fxl(Ts), label='weak H-bond large $\Delta S$', linestyle='dotted')

    plt.fill_between(Ts, fxs(Ts), fxl(Ts), color='orange', alpha=0.2)

    plt.title("Approximate $\Delta G$ of Binding vs. Temperature")
    plt.xlabel("Temperature (K)")
    plt.ylabel("$\Delta G$ (kJ)")

    plt.legend(loc='lower right')

    plt.show()


    ##### SENSITIVITY ANALYSIS: CYCLIC CAPACITY v. H-BONDING DELTA H and DELTA S #####
    # cyclic_capacity(T_low, T_high, P_low, P_high, dHy, dSy, verbose=1)
    grid_n = 120
    Hvals = np.linspace(-110, -60, grid_n)
    Svals = np.linspace(-0.3, -0.18, grid_n)
     
    # Evaluate cyclic_capacity for each H, S pair
    # yes, this is inefficient
    Hgrid, Sgrid = np.meshgrid(Hvals, Svals)
    Cgrid = np.zeros((grid_n, grid_n))
    for i in range(grid_n):
        for j in range(grid_n):
            dH = Hgrid[i][j]
            dS = Sgrid[i][j]
            Cgrid[i][j] = cyclic_capacity(T_low, T_high, P_low, P_high, dH, dS)


    fig, ax = plt.subplots(dpi=120)
    ax.pcolormesh(Hgrid, Sgrid, Cgrid, shading='gouraud', alpha=1)
    CS = ax.contour(Hgrid, Sgrid, Cgrid, colors='w')
    ax.clabel(CS, inline=True, fontsize=10)
    ax.plot(dHn, dSn, label='no H-bond', marker='o')
    ax.plot(dHy, dSy, label='H-bond', marker='o')
    ax.set_xlabel("$\Delta H$ (kJ/mol)")
    ax.set_ylabel("$\Delta S$ (kJ/mol$\cdot$K)")
    ax.legend(loc='lower right')
    ax.set_title(f"Molar Cyclic Capacity {T_low-273}-{T_high-273}$^\circ$C @ {P_low}-{P_high} bar ")
    fig.tight_layout()
    plt.show()
