### CODE START ###

# Printing the start message
print("Running the script...")

# --- Importing required Python libraries ---
import numpy as np
from scipy.optimize import fsolve, curve_fit
from scipy.integrate import solve_ivp
import pandas as pd # To export the data to a csv file

# Avoid displaying warning for the ndarray combination due to the implementation of NaClO pulses
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


# --- Variables input ---

# Initial conditions of the chemical system:
Mn0 = 0.018771 # Initial Mn concentration
HOCl0 = 2.32e-2 # Initial HOCl concentration (2mL NaClO solution added at once)
pH0 = 1.50 # Initial pH

# Kinetic constants according to the mechanism proposed, in the order: k0, k1, k2
k = [
    14, 
    1.30e3,
    5.00e4
] 

# Total time to be considered during the modeling (in min):
simulation_time = 120

# Loading the .csv file with the experimental data for pH vs time:
pH_exp_data = np.genfromtxt("pH_experimental_data_fast.csv",dtype=None, skip_header=1, delimiter=",")

pH_time = np.array([pH_exp_data[n][0] for n in range(len(pH_exp_data))])
pH_value = np.array([pH_exp_data[n][1] for n in range(len(pH_exp_data))])

# Point in time (in min) where the pH behavior suddenly flattens
pH_cutoff_point = 1


# --- Fitting the pH measurements to a function ---
def piecewise_linear(t, m1, m2, t0=0.0001, H0=10**(-1.50), *args):
    """ Defines a piecewise function of 2 linear steps relating [H+] concentration with time.
    Since the pH data shows a clear cutoff point after {pH_cutoff_point} min, the linear steps are:

    H(t) = H0 + m1*t    if t < {pH_cutoff_point}
    H(t) = H0 + m2*t    if t > {pH_cutoff_point}
        
    Parameters
    ----------
        t: array_like
            time element used by scipy.curve_fit routine
        m1: float
            slope of the first linear function (e.g. [H+]_1/t)
        m2: float
            slope of the first linear function (e.g. [H+]_2/t)
        t0: float
            initial time value (used as initial guess for curve fitting)
        H0: float
            initial [H+] concentration (used as initial guess for curve fitting)  
    Returns
    -------
        out: float
            [H+] concentration at time t calculated using the piecewise function
    """

    return np.piecewise(t, [t < pH_cutoff_point+0.1, t > pH_cutoff_point+0.1], [lambda t:m1*t + H0-m1*t0, lambda t:m2*t + H0-m2*t0])

# Change pH measurements into H+ concentrations
H_conc = [10**(-pH) for pH in pH_value]

# Fitting the parameters m1, m2, t0, H0 to our data: H_conc and pH_time
# The fitting returns an ndarray containing:
#     H_fit: A list with the fitted parameters in order
#     H_fit_error: A list containing the covariance matrix for the parameters
H_fit, H_fit_cov = curve_fit(piecewise_linear, pH_time, H_conc)

# Printing the fitting results
print(f"Parameters for H+ function: m1= {H_fit[0]}, m2= {H_fit[1]} \n\n",
    'Standard deviation:', 'sigma_m1=' , np.sqrt(np.diag(H_fit_cov)[0]),', sigma_m2= ', np.sqrt(np.diag(H_fit_cov)[1]), '\n\n',
    f"Covariance matrix (all parameters): {H_fit_cov} \n")


# --- Solving the kinetic ODE system for Mn2+, HOCl and H+ ---
def f(t, y, k, H_fit, Mn0, *args):
    """ Defines the system of kinetic ODEs that need to be solved.
    
    Parameters
    ----------
        t: array_like
            time element used by the solver
        y: list
            reagents which are being solved, with y[0]:Mn2+, y[1]:HOCl, y[2]:H+
        k: list 
            kinetic constants for each mechanism step, with k[0]:k0, k[1]:k1, k[2]:k2

    Returns
    -----------
        dydt: list
            Contains the kinetic ODEs for the reagents, defined as follows:    
            dydt[0] : -d[Mn2+]/dt = r[Mn2+]
            dydt[1] : -d[HOCl]/dt = r[HOCl]
            dydt[2] : -d[H+]/dt = r[H+]
    """   
   
    dydt = [
            # For Mn2+:
            (-1* # Negative sign due to Mn2+ consumption
                (
                k[0]*y[0]*y[1] # Formation of the initial MnO2 seeds by direct HOCl oxidation
                +k[1]*y[0]*(Mn0-y[0])*y[1] # Autocatalytic MnO2 formation 
                
                # Redissolution of the MnO-HOCl complex
                -k[2]*(
                    (Mn0-y[0])
                    *y[1]
                    *np.piecewise(t, [t < pH_cutoff_point, t > pH_cutoff_point], [H_fit[0]*t + H_fit[3]-H_fit[0]*H_fit[2], H_fit[1]*t + H_fit[3]-H_fit[1]*H_fit[2]])                    
                    )
                )               
            ),
            
            # For HOCl: 
            (-1* # Negative sign due to HOCl consumption
                (k[0]*y[0]*y[1]) # Formation of the initial MnO2 seeds by direct oxidation                
            ),
            
            # For H+:
            (+1* # Positive sign due to H+ being approximated by H_fit function
                (np.piecewise(t, [t < pH_cutoff_point, t > pH_cutoff_point], [lambda t:H_fit[0], lambda t:H_fit[1]])) # Differential form of the H_fit function
            )       
        ]
    return dydt

# Solving the ODE system
y_initial = [Mn0, HOCl0, 10**(-pH0)] #List of the initial concentrations of Mn2+, HOCl and H+ respectively (at t=0 min)

tspan = np.linspace(0, simulation_time, simulation_time+1) # Provides the total time to be modeled

# Using scipy.integrate.solve_ivp function to integrate numerically
ODEsolution = solve_ivp(lambda t, y: f(t, y, k, H_fit, Mn0), 
                [tspan[0], tspan[-1]], y_initial, t_eval=tspan, rtol = 1e-5)


# --- Exporting the calculated concentrations to a Pandas Dataframe and a csv file ---
concentrations = pd.DataFrame(
    {'time [min]':ODEsolution.t, 
    'Model Mn conc [M]':ODEsolution.y[0], 
    'Model HOCl conc [M]':ODEsolution.y[1], 
    'Model H+ conc [M]':ODEsolution.y[2]}
    )

# Output of the CSV file
output_filename = 'Mn_oxidation_fast_HOCl_concentrations.csv'
concentrations.to_csv(output_filename, sep='\t', index=False)

# Printing information message
print(f"Output of the {output_filename} file completed")

### END CODE ###