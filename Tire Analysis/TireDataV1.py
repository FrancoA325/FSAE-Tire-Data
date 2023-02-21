# Welcome to my tire data analysis code. This fits trimmed ttc data to a simplified pacejka model
# Look at my github repo for the matlab script that chops up the tire data

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.optimize import curve_fit
import numpy as np
from numpy import arange
from sympy import *
sns.set_style("ticks")

# These headers are grafted onto the csv you input. they match the matlab script output
headers=['ET','FX','FY','FZ','SA','P','TSTI','TSTC','TSTO','AMBTMP','IA'];

# Input the chopped CSV from the matlab script and make sure your path and name is correct

df = pd.read_csv('C:\CODING\PYTHONCODING\SlicedRUNS\A2356FIXED11.csv', delimiter = ',', header=None, names = headers)

# df.to_csv('C:\\FSAE EV\\Vehicle Dynamics\\TIRES DINK DONKERY\\pandasOUT\\df.csv',sep = ',',index = False)

# This loop iterates through the tire data csv and pulls the entire row at forces and pressures
for rows in df:

    df1 = df.loc[(df.FZ >= -55.0) & (df.FZ <= 45.0) & (df['P'] >= 11.5) & (df['P'] <= 12.5) & (df['IA'] >= 0.5) & (df['IA'] <= 2.0)]

    df2 = df.loc[(df.FZ >= -105.0) & (df.FZ <= -95.0) & (df['P'] >= 11.5) & (df['P'] <= 12.5) & (df['IA'] >= 0.5) & (df['IA'] <= 2.0)]

    df3 = df.loc[(df.FZ >= -155.0) & (df.FZ <= -145.0) & (df['P'] >= 11.5) & (df['P'] <= 12.5) & (df['IA'] >= 0.5) & (df['IA'] <= 2.0)] 

    df4 = df.loc[(df.FZ >= -205.0) & (df.FZ <= -195.0) & (df['P'] >= 11.5) & (df['P'] <= 12.5) & (df['IA'] >= 0.5) & (df['IA'] <= 2.0)]

df1.to_csv('C:\\FSAE EV\\Vehicle Dynamics\\TIRES DINK DONKERY\\pandasOUT\\df1.csv',sep = '\t',index = False)
df3 = df3.iloc[1000:]

df3.to_csv('C:\\FSAE EV\\Vehicle Dynamics\\TIRES DINK DONKERY\\pandasOUT\\df1.csv',sep = '\t',index = False)

# Graphing data
# The four different pacejka fitting algorithms are to make the loading times shorter and so that the curves fit a bit better that if there was
# just one model
def pacejka1(a, B, C, D, E):
    return D * np.sin(C * np.arctan(B * a - E * (B * a - np.arctan(B * a))))

def fit_pacejka1(a, Fy):
    initial_guess = [0.2, 2.18, 55, -2.5]  # initial guess for B, C, D, and E
    popt, _ = curve_fit(pacejka1, a, Fy, p0=initial_guess, maxfev= 100000)
    return popt

def pacejka2(a, B, C, D, E):
    return D * np.sin(C * np.arctan(B * a - E * (B * a - np.arctan(B * a))))

def fit_pacejka2(a, Fy):
    initial_guess = [0.2, 2.18, 100, -2.5]  # initial guess for B, C, D, and E
    popt, _ = curve_fit(pacejka2, a, Fy, p0=initial_guess, maxfev= 100000)
    return popt

def pacejka3(a, B, C, D, E):
    return D * np.sin(C * np.arctan(B * a - E * (B * a - np.arctan(B * a))))

def fit_pacejka3(a, Fy):
    initial_guess = [0.2, 2.18, 150, -2.5]  # initial guess for B, C, D, and E
    popt, _ = curve_fit(pacejka3, a, Fy, p0=initial_guess, maxfev= 100000)
    return popt

def pacejka4(a, B, C, D, E):
    return D * np.sin(C * np.arctan(B * a - E * (B * a - np.arctan(B * a))))

def fit_pacejka4(a, Fy):
    initial_guess = [0.2, 2.18, 200, -2.5]  # initial guess for B, C, D, and E
    popt, _ = curve_fit(pacejka4, a, Fy, p0=initial_guess, maxfev= 100000)
    return popt

# This is where the data from the loop is fit to the pacejka model

# Pulls out the slip angle and lateral force values to be fit
a = df1["SA"].values
Fy = df1["FY"].values
# Popt is the fitting of the model and outputs the coeffiecents 
# the coeff are saved into their own variable to used on the cornering stiffness calculations
popt = fit_pacejka1(a, Fy)
print(popt)
popt1 = popt

a_fit = np.linspace(min(a), max(a), num=200)
Fy_fit = pacejka1(a_fit, *popt)
# saving the fit data into its own dataframe to be used later
plt.plot(a_fit, Fy_fit, label='50 lbs')

a = df2["SA"].values
Fy = df2["FY"].values

popt = fit_pacejka2(a, Fy)
print(popt)
popt2 = popt

a_fit = np.linspace(min(a), max(a), num=200)
Fy_fit = pacejka2(a_fit, *popt)
plt.plot(a_fit, Fy_fit, label='100 lbs')

a = df3["SA"].values
Fy = df3["FY"].values

popt = fit_pacejka3(a, Fy)
print(popt)
popt3 = popt

a_fit = np.linspace(min(a), max(a), num=200)
Fy_fit = pacejka3(a_fit, *popt)
plt.plot(a_fit, Fy_fit, label='150 lbs')

a = df4["SA"].values
Fy = df4["FY"].values

popt = fit_pacejka4(a, Fy)
print(popt)
popt4 = popt

a_fit = np.linspace(min(a), max(a), num=200)
Fy_fit = pacejka4(a_fit, *popt)
plt.plot(a_fit, Fy_fit, label='200 lbs')

# final SA FY plot formatting and saving to an image. you want to make sure that your path is correct

plt.xlabel('Slip angle (Deg)')
plt.ylabel('Lateral force (lbs)')
plt.legend()
plt.savefig('C:\CODING\PYTHONCODING\OUTPUT plots\LateralForceSA.jpg')
plt.show()

# Calulations and plotting data for cornering stiffness(the slope of the lateral force slip angle)
# Dataframes were made of each of the curve fit point sets these are what are used to make the Cornering stiffness calculations

def Cornering_Stiffness(a, B, C, D, E):
    term1 = D * C * np.cos(C * np.arctan(B * a - E * (B * a - np.arctan(B * a))))
    term2 = B - E * (2 * B * a / (1 + (B * a)**2))
    return term1 * term2

CornerStiff1 = Cornering_Stiffness(0.0, popt1[0], popt1[1], popt1[2], popt1[3])
CornerStiff2 = Cornering_Stiffness(0.0, popt2[0], popt2[1], popt2[2], popt2[3])
CornerStiff3 = Cornering_Stiffness(0.0, popt3[0], popt3[1], popt3[2], popt3[3])
CornerStiff4 = Cornering_Stiffness(0.0, popt4[0], popt4[1], popt4[2], popt4[3])
CSdf = pd.DataFrame({"FZ" : [50,100,150,200], "CS" :[CornerStiff1, CornerStiff2, CornerStiff3, CornerStiff4]})
sns.pointplot(data = CSdf.abs(), x = 'FZ', y = 'CS')
plt.xlabel("Vertical Force")
plt.ylabel("Cornering Stiffness")
plt.legend()
plt.savefig('C:\CODING\PYTHONCODING\OUTPUT plots\CorneringStiffness.jpg')
plt.show()

# df1.to_csv('C:\\FSAE EV\\Vehicle Dynamics\\TIRES DINK DONKERY\\pandasOUT\\df1.csv',sep = '\t',index = False)
# df2.to_csv('C:\\FSAE EV\\Vehicle Dynamics\\TIRES DINK DONKERY\\pandasOUT\\df2.csv',sep = '\t',index = False)
# df3.to_csv('C:\\FSAE EV\\Vehicle Dynamics\\TIRES DINK DONKERY\\pandasOUT\\df3.csv',sep = '\t',index = False)
# df4.to_csv('C:\\FSAE EV\\Vehicle Dynamics\\TIRES DINK DONKERY\\pandasOUT\\df4.csv',sep = '\t',index = False)