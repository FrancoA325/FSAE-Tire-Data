# Welcome to my tire data analysis code. This fits trimmed ttc data to a simplified pacejka model
# Look at my github repo for the matlab script that chops up the tire data

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.optimize import curve_fit
import numpy as np
from numpy import arange
from sympy import *
from scipy.signal import savgol_filter 
import xlsxwriter
sns.set_style("ticks")

# These headers are grafted onto the csv you input. they match the matlab script output
def lighten_color(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

headers=['ET','FX','FY','FZ','SA','P','TSTI','TSTC','TSTO','AMBTMP','IA']

# Input the chopped CSV from the matlab script and make sure your path and name is correct

df = pd.read_csv('path of the cornering file', delimiter = ',', header=None, names = headers)

writer = pd.ExcelWriter('path that you want the excel output to be', engine = 'xlsxwriter')
workbook = writer.book


# This loop iterates through the tire data csv and pulls the entire row at forces and pressures
for rows in df:
    dfLatLoads = df.loc[(df['P'] >= 11.5) & (df['P'] <= 12.5) & (df['IA'] >= -1.0) & (df['IA'] <= -1.0)]
# 0 IA data scrape

    # df1 = df.loc[(df['FZ'] >= -55.0) & (df['FZ'] <= -45.0) & (df['P'] >= 11.5) & (df['P'] <= 12.5) & (df['IA'] >= -1.0) & (df['IA'] <= 1.0)]

    # df2 = df.loc[(df['FZ'] >= -105.0) & (df['FZ'] <= -95.0) & (df['P'] >= 11.5) & (df['P'] <= 12.5) & (df['IA'] >= -1.0) & (df['IA'] <= 1.0)]

    # df3 = df.loc[(df['FZ'] >= -155.0) & (df['FZ'] <= -145.0) & (df['P'] >= 11.5) & (df['P'] <= 12.5) & (df['IA'] >= -1.0) & (df['IA'] <= 1.0)] 

    # df4 = df.loc[(df['FZ'] >= -205.0) & (df['FZ'] <= -195.0) & (df['P'] >= 11.5) & (df['P'] <= 12.5) & (df['IA'] >= -1.0) & (df['IA'] <= 1.0)]

    # df5 = df.loc[(df['FZ'] >= -255.0) & (df['FZ'] <= -245.0) & (df['P'] >= 11.5) & (df['P'] <= 12.5) & (df['IA'] >= -1.0) & (df['IA'] <= 1.0)]

#  2 IA data scrape

    # df1 = df.loc[(df['FZ'] >= -55.0) & (df['FZ'] <= -45.0) & (df['P'] >= 11.5) & (df['P'] <= 12.5) & (df['IA'] >= 1.0) & (df['IA'] <= 3.0)]

    # df2 = df.loc[(df['FZ'] >= -105.0) & (df['FZ'] <= -95.0) & (df['P'] >= 11.5) & (df['P'] <= 12.5) & (df['IA'] >= 1.0) & (df['IA'] <= 3.0)]

    # df3 = df.loc[(df['FZ'] >= -155.0) & (df['FZ'] <= -145.0) & (df['P'] >= 11.5) & (df['P'] <= 12.5) & (df['IA'] >= 1.0) & (df['IA'] <= 3.0)] 

    # df4 = df.loc[(df['FZ'] >= -205.0) & (df['FZ'] <= -195.0) & (df['P'] >= 11.5) & (df['P'] <= 12.5) & (df['IA'] >= 1.0) & (df['IA'] <= 3.0)]

    # df5 = df.loc[(df['FZ'] >= -255.0) & (df['FZ'] <= -245.0) & (df['P'] >= 11.5) & (df['P'] <= 12.5) & (df['IA'] >= 1.0) & (df['IA'] <= 3.0)]

#  4 IA data scrape

    df1 = df.loc[(df['FZ'] >= -55.0) & (df['FZ'] <= -45.0) & (df['P'] >= 11.5) & (df['P'] <= 12.5) & (df['IA'] >= 3.0) & (df['IA'] <= 5.0)]

    df2 = df.loc[(df['FZ'] >= -105.0) & (df['FZ'] <= -95.0) & (df['P'] >= 11.5) & (df['P'] <= 12.5) & (df['IA'] >= 3.0) & (df['IA'] <= 5.0)]

    df3 = df.loc[(df['FZ'] >= -155.0) & (df['FZ'] <= -145.0) & (df['P'] >= 11.5) & (df['P'] <= 12.5) & (df['IA'] >= 3.0) & (df['IA'] <= 5.0)] 

    df4 = df.loc[(df['FZ'] >= -205.0) & (df['FZ'] <= -195.0) & (df['P'] >= 11.5) & (df['P'] <= 12.5) & (df['IA'] >= 3.0) & (df['IA'] <= 5.0)]

    df5 = df.loc[(df['FZ'] >= -255.0) & (df['FZ'] <= -245.0) & (df['P'] >= 11.5) & (df['P'] <= 12.5) & (df['IA'] >= 3.0) & (df['IA'] <= 5.0)]

# df1 = df1.iloc[500:]
# df2 = df2.iloc[500:]
# df3 = df3.iloc[500:]
# df4 = df4.iloc[500:]
# df5 = df5.iloc[500:]
df1.to_csv('this was a path that checks to see if the loop is pulling data properly',sep = '\t',index = False)
# this is for checking what loads are tested
plt.plot(dfLatLoads['ET'].values, dfLatLoads['FZ'].values)
plt.show()

# df3 = df3.iloc[1000:]
# Graphing data
# The four different pacejka fitting algorithms are to make the loading times shorter and so that the curves fit a bit better that if there was
# just one model
def pacejka1(a, B, C, D, E):
    return D * np.sin(C * np.arctan(B * a - E * (B * a - np.arctan(B * a))))

def fit_pacejka1(a, Fy):
    initial_guess = [0.33, 0.93, 50, 0.29]  # initial guess for B, C, D, and E
    popt, _ = curve_fit(pacejka1, a, Fy, p0=initial_guess, maxfev= 100000)
    return popt

def pacejka2(a, B, C, D, E):
    return D * np.sin(C * np.arctan(B * a - E * (B * a - np.arctan(B * a))))

def fit_pacejka2(a, Fy):
    initial_guess = [0.25, 1.03, 100, 0.3018]  # initial guess for B, C, D, and E
    popt, _ = curve_fit(pacejka2, a, Fy, p0=initial_guess, maxfev= 100000)
    return popt

def pacejka3(a, B, C, D, E):
    return D * np.sin(C * np.arctan(B * a - E * (B * a - np.arctan(B * a))))

def fit_pacejka3(a, Fy):
    initial_guess = [0.165, 1.47, 150, 0.3727]  # initial guess for B, C, D, and E
    popt, _ = curve_fit(pacejka3, a, Fy, p0=initial_guess, maxfev= 100000)
    return popt

def pacejka4(a, B, C, D, E):
    return D * np.sin(C * np.arctan(B * a - E * (B * a - np.arctan(B * a))))

def fit_pacejka4(a, Fy):
    initial_guess = [0.187, 0.88, 200, -3.26]  # initial guess for B, C, D, and E
    popt, _ = curve_fit(pacejka4, a, Fy, p0=initial_guess, maxfev= 100000)
    return popt

def pacejka5(a, B, C, D, E):
    return D * np.sin(C * np.arctan(B * a - E * (B * a - np.arctan(B * a))))

def fit_pacejka5(a, Fy):
    initial_guess = [0.18, .00464, 250, -7.09]  # initial guess for B, C, D, and E
    popt, _ = curve_fit(pacejka5, a, Fy, p0=initial_guess, maxfev= 1000000)
    return popt


# This is where the data from the loop is fit to the pacejka model

# Pulls out the slip angle and lateral force values to be fit
a = df1["SA"].values
Fy = df1["FY"].values * (-1)


# the coeff are saved into their own variable to used on the cornering stiffness calculations
# fitting of the model and output of the coeffiecents 
popt = fit_pacejka1(a, Fy)
popt1 = popt
# Plots the fit data 
a_fit1 = np.linspace(min(a), max(a), num=200)
Fy_fit1 = pacejka1(a_fit1, *popt)
plt.plot(a_fit1, Fy_fit1, label='50 lbs', color = 'blue')
plt.plot(a_fit1, Fy_fit1 * 0.65, label='50 lbs scaled', color = lighten_color('blue', 0.4))

a = df2["SA"].values
Fy = df2["FY"].values * (-1)

popt = fit_pacejka2(a, Fy)
popt2 = popt

a_fit2 = np.linspace(min(a), max(a), num=200)
Fy_fit2 = pacejka2(a_fit2, *popt)
plt.plot(a_fit2, Fy_fit2, label='100 lbs', color = 'purple')
plt.plot(a_fit2, Fy_fit2 * 0.65, label='100 lbs scaled', color = lighten_color('purple', 0.4))

a = df3["SA"].values
Fy = df3["FY"].values * (-1)

popt = fit_pacejka3(a, Fy)
popt3 = popt

a_fit3 = np.linspace(min(a), max(a), num=200)
Fy_fit3 = pacejka3(a_fit3, *popt)
plt.plot(a_fit3, Fy_fit3, label='150 lbs', color = 'green')
plt.plot(a_fit3, Fy_fit3 * 0.65, label='150 lbs scaled', color = lighten_color('green', 0.4))

a = df4["SA"].values
Fy = df4["FY"].values * (-1)

popt = fit_pacejka4(a, Fy)
popt4 = popt

a_fit4 = np.linspace(min(a), max(a), num=200)
Fy_fit4 = pacejka4(a_fit4, *popt)
plt.plot(a_fit4, Fy_fit4, label='200 lbs', color = 'orange')
plt.plot(a_fit4, Fy_fit4 * 0.65, label='200 lbs scaled', color = lighten_color('orange', 0.4))

a = df5["SA"].values
Fy = df5["FY"].values * (-1)

popt = fit_pacejka5(a, Fy)
popt5 = popt

a_fit5 = np.linspace(min(a), max(a), num=200)
Fy_fit5 = pacejka5(a_fit5, *popt)
plt.plot(a_fit5, Fy_fit5, label='250 lbs', color = 'red')
plt.plot(a_fit5, Fy_fit5 * 0.65, label='250 lbs scaled', color = lighten_color('red', 0.4))


LatData = {"SA 50" : a_fit1, "FY 50" : Fy_fit1* 0.65,"SA 100" : a_fit2, "FY 100" : Fy_fit2* 0.65,"SA 150" : a_fit3, "FY 150" : Fy_fit3* 0.65, 
          "SA 200" : a_fit4, "FY 200" : Fy_fit4 * 0.65 ,"SA 250" : a_fit5, "FY 250" : Fy_fit5* 0.65}
df_FYSA = pd.DataFrame.from_dict(LatData, orient='index')
df_FYSA = df_FYSA.transpose()
df_FYSA.to_excel(writer, sheet_name = 'LatForce VS SlipAng ADJ', index = False)

plt.xlabel('Slip angle (Deg)')
plt.ylabel('Lateral force (lbs)')
plt.grid()
plt.legend()
plt.savefig('file path for image saveing')
Worksheet1 = writer.sheets['LatForce VS SlipAng ADJ']
Worksheet1.insert_image('A1', 'file path of the above image for posing in excel')
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
CornerStiff5 = Cornering_Stiffness(0.0, popt5[0], popt5[1], popt5[2], popt5[3])
CSdf = pd.DataFrame({"FZ" : [50,100,150,200,250], "CS" :[CornerStiff1, CornerStiff2, CornerStiff3,CornerStiff4,CornerStiff5]})
CSdf2 = pd.DataFrame({"FZ ADJ" : [50,100,150,200,250], "CS" :[CornerStiff1 * 0.65, CornerStiff2 * 0.65, CornerStiff3 * 0.65,CornerStiff4 * 0.65,CornerStiff5 * 0.65]})

sns.pointplot(data = CSdf.abs(), x = 'FZ', y = 'CS')
sns.pointplot(data = CSdf2.abs(), x = 'FZ ADJ', y = 'CS', color = lighten_color('blue', 0.4))

CSdf2.to_excel(writer, sheet_name = 'CorStiff VS VertLoad ADJ', startrow = 0, startcol = 0, index = False)

plt.xlabel("Vertical Force")
plt.ylabel("Cornering Stiffness")
plt.grid()
plt.savefig('file path for image saveing')
Worksheet2 = writer.sheets['CorStiff VS VertLoad ADJ']
Worksheet2.insert_image('A1', 'Cornering stiff png file path for excel')
plt.show()


headers2=['ET','FX','FZ','SA','P','TSTI','TSTC','TSTO','AMBTMP','IA','SL']

# Input the chopped CSV from the matlab script and make sure your path and name is correct

dfLong = pd.read_csv('drive brake csv path from matlab', delimiter = ',', header=None, names = headers2)

for rows in dfLong:

    dfLongLoads = dfLong.loc[(dfLong['P'] >= 11.5) & (dfLong['P'] <= 12.5) & (dfLong['IA'] >= -1.0) & (dfLong['IA'] <= 1.0)]

# 0 IA data scrape

    # dfLong1 = dfLong.loc[(dfLong['FZ'] >= -55.0) & (dfLong['FZ'] <= -45.0) & (dfLong['P'] >= 11.5) & (dfLong['P'] <= 12.5) & (dfLong['IA'] >= -1.0) & (dfLong['IA'] <= 1.0)]
    
    # dfLong3 = dfLong.loc[(dfLong['FZ'] >= -155.0) & (dfLong['FZ'] <= -145.0) & (dfLong['P'] >= 11.5) & (dfLong['P'] <= 12.5) & (dfLong['IA'] >= -1.0) & (dfLong['IA'] <= 1.0)]

    # dfLong4 = dfLong.loc[(dfLong['FZ'] >= -205.0) & (dfLong['FZ'] <= -195.0) & (dfLong['P'] >= 11.5) & (dfLong['P'] <= 12.5) & (dfLong['IA'] >= -1.0) & (dfLong['IA'] <= 1.0)]
    
    # dfLong5 = dfLong.loc[(dfLong['FZ'] >= -255.0) & (dfLong['FZ'] <= -145.0) & (dfLong['P'] >= 11.5) & (dfLong['P'] <= 12.5) & (dfLong['IA'] >= -1.0) & (dfLong['IA'] <= 1.0)]

# 2 IA data scrape

    # dfLong1 = dfLong.loc[(dfLong['FZ'] >= -55.0) & (dfLong['FZ'] <= -45.0) & (dfLong['P'] >= 11.5) & (dfLong['P'] <= 12.5) & (dfLong['IA'] >= 1.0) & (dfLong['IA'] <= 3.0)]
    
    # dfLong3 = dfLong.loc[(dfLong['FZ'] >= -155.0) & (dfLong['FZ'] <= -145.0) & (dfLong['P'] >= 11.5) & (dfLong['P'] <= 12.5) & (dfLong['IA'] >= 1.0) & (dfLong['IA'] <= 3.0)]

    # dfLong4 = dfLong.loc[(dfLong['FZ'] >= -205.0) & (dfLong['FZ'] <= -195.0) & (dfLong['P'] >= 11.5) & (dfLong['P'] <= 12.5) & (dfLong['IA'] >= 1.0) & (dfLong['IA'] <= 3.0)]
    
    # dfLong5 = dfLong.loc[(dfLong['FZ'] >= -255.0) & (dfLong['FZ'] <= -145.0) & (dfLong['P'] >= 11.5) & (dfLong['P'] <= 12.5) & (dfLong['IA'] >= 1.0) & (dfLong['IA'] <= 3.0)]

# 4 IA data scrape

    dfLong1 = dfLong.loc[(dfLong['FZ'] >= -55.0) & (dfLong['FZ'] <= -45.0) & (dfLong['P'] >= 11.5) & (dfLong['P'] <= 12.5) & (dfLong['IA'] >= 3.0) & (dfLong['IA'] <= 5.0)]
    
    dfLong3 = dfLong.loc[(dfLong['FZ'] >= -155.0) & (dfLong['FZ'] <= -145.0) & (dfLong['P'] >= 11.5) & (dfLong['P'] <= 12.5) & (dfLong['IA'] >= 3.0) & (dfLong['IA'] <= 5.0)]

    dfLong4 = dfLong.loc[(dfLong['FZ'] >= -205.0) & (dfLong['FZ'] <= -195.0) & (dfLong['P'] >= 11.5) & (dfLong['P'] <= 12.5) & (dfLong['IA'] >= 3.0) & (dfLong['IA'] <= 5.0)]
    
    dfLong5 = dfLong.loc[(dfLong['FZ'] >= -255.0) & (dfLong['FZ'] <= -145.0) & (dfLong['P'] >= 11.5) & (dfLong['P'] <= 12.5) & (dfLong['IA'] >= 3.0) & (dfLong['IA'] <= 5.0)]

dfLong1.to_csv('same loop test as the first one dont really need this',sep = '\t',index = False)  
dfLong1 = dfLong1.iloc[500:]
plt.plot(dfLongLoads['ET'].values, dfLongLoads['IA'].values)
plt.show()

def pacejkaLong1(Sl, B, C, D, E):
    return D * np.sin(C * np.arctan(B * (Sl) - E * (B * (Sl) - np.arctan(B * (Sl)))))

def fit_pacejkaLong1(Sl, Fx):
    initial_guess = [0.99, 13.855, 55, 29.42]  # initial guess for B, C, D, E
    poptL, _ = curve_fit(pacejkaLong1, Sl, Fx, p0=initial_guess, maxfev= 1000000)
    return poptL

def pacejkaLong3(Sl, B, C, D, E):
    return D * np.sin(C * np.arctan(B * (Sl) - E * (B * (Sl) - np.arctan(B * (Sl)))))

def fit_pacejkaLong3(Sl, Fx):
    initial_guess = [11.97, .00612, 150, 1.01]  # initial guess for B, C, D, E
    poptL, _ = curve_fit(pacejkaLong3, Sl, Fx, p0=initial_guess, maxfev= 1000000)
    return poptL

def pacejkaLong4(Sl, B, C, D, E):
    return D * np.sin(C * np.arctan(B * (Sl) - E * (B * (Sl) - np.arctan(B * (Sl)))))

def fit_pacejkaLong4(Sl, Fx):
    initial_guess = [11.35, .006, 200, 1.13]  # initial guess for B, C, D, E
    poptL, _ = curve_fit(pacejkaLong4, Sl, Fx, p0=initial_guess, maxfev= 1000000)
    return poptL

def pacejkaLong5(Sl, B, C, D, E):
    return D * np.sin(C * np.arctan(B * (Sl) - E * (B * (Sl) - np.arctan(B * (Sl))))) 

def fit_pacejkaLong5(Sl, Fx):
    initial_guess = [11.19, .0079, 250, 1.159]  # initial guess for B, C, D, E
    poptL, _ = curve_fit(pacejkaLong5, Sl, Fx, p0=initial_guess, maxfev= 1000000)
    return poptL



# Pulls out the slip angle and lateral force values to be fit
Sl = dfLong1["SL"].values 
Fx = dfLong1["FX"].values
# the coeff are saved into their own variable to used on the cornering stiffness calculations
# fitting of the model and output of the coeffiecents 
poptL = fit_pacejkaLong1(Sl, Fx)
print(poptL)
# Plots the fit data 
Sl_fit = np.linspace(min(Sl), max(Sl), num=200)
Fx_fit = pacejkaLong1(Sl_fit, *poptL)
Fx_fit_50 = Fx_fit
Sl_fit_50 = Sl_fit
plt.plot(Sl_fit, Fx_fit, label='50 lbs', color = 'blue')
plt.plot(Sl_fit, Fx_fit * 0.65, label='50 lbs scaled',color = lighten_color('blue', 0.4))

Sl = dfLong3["SL"].values 
Fx = dfLong3["FX"].values

poptL = fit_pacejkaLong3(Sl, Fx)
print(poptL)
Sl_fit = np.linspace(min(Sl), max(Sl), num=200)
Fx_fit = pacejkaLong3(Sl_fit, *poptL)
Fx_fit_150 = Fx_fit
Sl_fit_150 = Sl_fit
plt.plot(Sl_fit, Fx_fit, label='150 lbs', color = 'green')
plt.plot(Sl_fit, Fx_fit * 0.65, label='150 lbs scaled', color = lighten_color('green', 0.4))


Sl = dfLong4["SL"].values 
Fx = dfLong4["FX"].values

poptL = fit_pacejkaLong4(Sl, Fx)
print(poptL)
Sl_fit = np.linspace(min(Sl), max(Sl), num=200)
Fx_fit = pacejkaLong4(Sl_fit, *poptL)
Fx_fit_200 = Fx_fit
Sl_fit_200 = Sl_fit
plt.plot(Sl_fit, Fx_fit, label='200 lbs', color = 'orange')
plt.plot(Sl_fit, Fx_fit * 0.65, label='200 lbs scale', color = lighten_color('orange', 0.4))


Sl = dfLong5["SL"].values 
Fx = dfLong5["FX"].values

poptL = fit_pacejkaLong5(Sl, Fx)
print(poptL)
Sl_fit = np.linspace(min(Sl), max(Sl), num=200)
Fx_fit = pacejkaLong5(Sl_fit, *poptL)
Fx_fit_250 = Fx_fit
Sl_fit_250 = Sl_fit
plt.plot(Sl_fit, Fx_fit, label='250 lbs', color = 'red')
plt.plot(Sl_fit, Fx_fit * 0.65, label='250 lbs scaled', color = lighten_color('red', 0.4))

LatData = {"SL 50" : Sl_fit_50, "FX 50" : Fx_fit_50* 0.65,"SL 150" : Sl_fit_150, "FX 150" : Fx_fit_150* 0.65, 
          "SL 200" : Sl_fit_200, "FX 200" : Fx_fit_200* 0.65,"SL 250" : Sl_fit_250, "FX 250" : Fx_fit_250* 0.65}
df_LatData = pd.DataFrame.from_dict(LatData, orient='index')
df_LatData = df_LatData.transpose()
df_LatData.to_excel(writer, sheet_name = 'LongForce VS SlipRat ADJ', index = False)

plt.xlabel('Slip Ratio')
plt.ylabel('Longitudinal Force')
plt.grid()
plt.legend()
plt.savefig('more image file path saving')
Worksheet3 = writer.sheets['LongForce VS SlipRat ADJ']
Worksheet3.insert_image('A1', 'excel image upload file path')
plt.show()

mu_50 = Fx_fit_50 / 50
mu_150 = Fx_fit_150 / 150
mu_200 = Fx_fit_200 / 200
mu_250 = Fx_fit_250 / 250

plt.plot(Sl_fit_50, mu_50, label='50 lbs', color = 'blue')
plt.plot(Sl_fit_50, mu_50 * 0.65, label='50 lbs Scaled', color = lighten_color('blue', 0.4))
plt.plot(Sl_fit_150, mu_150, label='150 lbs', color = 'green')
plt.plot(Sl_fit_150, mu_150 * 0.65, label='150 lbs Scaled', color = lighten_color('green', 0.4))
plt.plot(Sl_fit_200, mu_200, label='200 lbs', color = 'orange')
plt.plot(Sl_fit_200, mu_200 * 0.65, label='200 lbs Scaled', color = lighten_color('orange', 0.4))
plt.plot(Sl_fit_250, mu_250, label='250 lbs', color = 'red')
plt.plot(Sl_fit_250, mu_250 * 0.65, label='250 lbs Scaled', color = lighten_color('red', 0.4))

MuFSL = {"SL 50" : Sl_fit_50, "Mu 50" : mu_50 * 0.65,"SL 150" : Sl_fit_150, "Mu 150" : mu_150 * 0.65, 
          "SL 200" : Sl_fit_200, "Mu 200" : mu_200 * 0.65,"SL 250" : Sl_fit_250, "Mu 250" : mu_250 * 0.65}
df_MuFSL = pd.DataFrame.from_dict(MuFSL, orient='index')
df_MuFSL = df_MuFSL.transpose()
df_MuFSL.to_excel(writer, sheet_name = 'TireFrict VS SlipRat ADJ', index = False)
writer.close()

plt.xlabel('Slip Ratio')
plt.ylabel('Friction Coefficient')
plt.grid()
plt.legend()
plt.savefig('image file path save for friction coe')
Worksheet4 = writer.sheets['TireFrict VS SlipRat ADJ']
Worksheet4.insert_image('A1', 'friction coe image path for excel upload')
plt.show()
