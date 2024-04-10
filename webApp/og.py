#-----------------------------------------------------------------------------
#  Copyright (C) 2023  Francisco H. Antunez
#
#  Distributed under the terms of the MIT and Personal License. The full license is in
#  the file LICENSE, distributed as part of this software.
#-----------------------------------------------------------------------------

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.optimize import curve_fit
import numpy as np
from numpy import arange
from scipy.signal import savgol_filter 
import xlsxwriter
import scipy.io as sio
import csv
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog

import tkinter as tk
from tkinter import ttk, filedialog
import os

# Simple Tkninter GUI that I am looking to update and improve, This section will take file inputs, tire choices, and output folders.

class Interface:
    def __init__(self, master):
        self.master = master
        self.master.title("File Selection Interface")

        self.file_lat = tk.StringVar()
        self.file_long = tk.StringVar()
        self.file_out = tk.StringVar()

        self.create_widgets()

    def create_widgets(self):
        ttk.Button(self.master, text="Lateral File", command=self.lateralFile).grid(column=100, row=10)
        ttk.Label(self.master, textvariable=self.file_lat, font=('Calibri 12')).grid(column=100, row=20)

        ttk.Button(self.master, text="Longitudinal File", command=self.longFile).grid(column=100, row=30)
        ttk.Label(self.master, textvariable=self.file_long, font=('Calibri 12')).grid(column=100, row=40)

        ttk.Button(self.master, text="Output Folder", command=self.outputFolder).grid(column=100, row=50)
        ttk.Label(self.master, textvariable=self.file_out, font=('Calibri 12')).grid(column=100, row=60)

        ttk.Label(self.master, text="Camber Choice 0, 2, or 4", font=('Calibri 12')).grid(column=100, row=90)
        self.camber_entry = tk.Entry(self.master, width=15)
        self.camber_entry.grid(column=100, row=100)

        ttk.Label(self.master, text="Pressure Choice 10, 12, or 14", font=('Calibri 12')).grid(column=100, row=110)
        self.psi_entry = tk.Entry(self.master, width=15)
        self.psi_entry.grid(column=100, row=130)

        ttk.Label(self.master, text="Tire Output File Name", font=('Calibri 12')).grid(column=100, row=150)
        self.out_name_entry = tk.Entry(self.master, width=15)
        self.out_name_entry.grid(column=100, row=170)

        ttk.Button(self.master, text="Set Choices", command=self.setChoices).grid(column=100, row=190)

    def lateralFile(self):
        self.file_lat.set(filedialog.askopenfilename())

    def longFile(self):
        self.file_long.set(filedialog.askopenfilename())

    def outputFolder(self):
        self.file_out.set(filedialog.askdirectory())

    def setChoices(self):
        self.camber = int(self.camber_entry.get())
        self.psi = int(self.psi_entry.get())
        self.out_name = self.out_name_entry.get()

        self.master.destroy()

if __name__ == "__main__":
    interface = tk.Tk()
    app = Interface(interface)
    interface.mainloop()

# Accessing variables outside the class for the gui
lateral = app.file_lat.get()
longitude = app.file_long.get()
output = app.file_out.get()

camber = app.camber
psi = app.psi
outName= app.out_name

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

# This is the lateral load file input. download the respective .mat cornering file from the ttc 
# the data channels can be changed but they are hard coded for theses calculations.

LatLoads = sio.loadmat(lateral)
def Convert(string):
    li = list(string.split(","))
    out = []
    for item in li:
        out.append(float(item))
    return out

ETin = LatLoads['ET'].tolist()
ETin = str(ETin)
ETin = ETin.replace("[","")
ETin = ETin.replace("]","")
ETin = ETin.replace(" ","")
ETin = Convert(ETin)

FXin = LatLoads['FX'].tolist()
FXin = str(FXin)
FXin = FXin.replace("[","")
FXin = FXin.replace("]","")
FXin = FXin.replace(" ","")
FXin = Convert(FXin)

FYin = LatLoads['FY'].tolist()
FYin = str(FYin)
FYin = FYin.replace("[","")
FYin = FYin.replace("]","")
FYin = FYin.replace(" ","")
FYin = Convert(FYin)

FZin = LatLoads['FZ'].tolist()
FZin = str(FZin)
FZin = FZin.replace("[","")
FZin = FZin.replace("]","")
FZin = FZin.replace(" ","")
FZin = Convert(FZin)

SAin = LatLoads['SA'].tolist()
SAin = str(SAin)
SAin = SAin.replace("[","")
SAin = SAin.replace("]","")
SAin = SAin.replace(" ","")
SAin = Convert(SAin)

Pin = LatLoads['P'].tolist()
Pin = str(Pin)
Pin = Pin.replace("[","")
Pin = Pin.replace("]","")
Pin = Pin.replace(" ","")
Pin = Convert(Pin)

TSTIin = LatLoads['TSTI'].tolist()
TSTIin = str(TSTIin)
TSTIin = TSTIin.replace("[","")
TSTIin = TSTIin.replace("]","")
TSTIin = TSTIin.replace(" ","")
TSTIin = Convert(TSTIin)

TSTCin = LatLoads['TSTC'].tolist()
TSTCin = str(TSTCin)
TSTCin = TSTCin.replace("[","")
TSTCin = TSTCin.replace("]","")
TSTCin = TSTCin.replace(" ","")
TSTCin = Convert(TSTCin)

TSTOin = LatLoads['TSTO'].tolist()
TSTOin = str(TSTOin)
TSTOin = TSTOin.replace("[","")
TSTOin = TSTOin.replace("]","")
TSTOin = TSTOin.replace(" ","")
TSTOin = Convert(TSTOin)

AMBTMPin = LatLoads['AMBTMP'].tolist()
AMBTMPin = str(AMBTMPin)
AMBTMPin = AMBTMPin.replace("[","")
AMBTMPin = AMBTMPin.replace("]","")
AMBTMPin = AMBTMPin.replace(" ","")
AMBTMPin = Convert(AMBTMPin)

IAin = LatLoads['IA'].tolist()
IAin = str(IAin)
IAin = IAin.replace("[","")
IAin = IAin.replace("]","")
IAin = IAin.replace(" ","")
IAin = Convert(IAin)

d = {'ET' : ETin, 'FX' : FXin, 'FY' : FYin, 'FZ' : FZin, 'SA' : SAin, 'P' : Pin, 'TSTI' : TSTIin,
      'TSTC' : TSTCin, 'TSTO' : TSTOin, 'AMBTMP' : AMBTMPin, 'IA' : IAin}

df = pd.DataFrame(data=d)

# df.to_csv(output + '/df.csv',sep = ',',index = False)
# creates the output file for the tire data to be used later
writer = pd.ExcelWriter(output + '/' + outName + '.xlsx', engine = 'xlsxwriter')
workbook = writer.book


# This loop iterates through the tire data csv and pulls the entire row at forces and pressures that the user choose at the start
for rows in df:
    
    if psi == 10:
    # 0 IA data scrape
        if camber == 0:
            df1 = df.loc[(df['FZ'] >= -55.0) & (df['FZ'] <= -45.0) & (df['P'] >= 9.5) & (df['P'] <= 10.5) & (df['IA'] >= -1.0) & (df['IA'] <= 1.0)]

            df2 = df.loc[(df['FZ'] >= -105.0) & (df['FZ'] <= -95.0) & (df['P'] >= 9.5) & (df['P'] <= 10.5) & (df['IA'] >= -1.0) & (df['IA'] <= 1.0)]

            df3 = df.loc[(df['FZ'] >= -155.0) & (df['FZ'] <= -145.0) & (df['P'] >= 9.5) & (df['P'] <= 10.5) & (df['IA'] >= -1.0) & (df['IA'] <= 1.0)] 

            df4 = df.loc[(df['FZ'] >= -205.0) & (df['FZ'] <= -195.0) & (df['P'] >= 9.5) & (df['P'] <= 10.5) & (df['IA'] >= -1.0) & (df['IA'] <= 1.0)]

            df5 = df.loc[(df['FZ'] >= -255.0) & (df['FZ'] <= -245.0) & (df['P'] >= 9.5) & (df['P'] <= 10.5) & (df['IA'] >= -1.0) & (df['IA'] <= 1.0)]

    #  2 IA data scrape
        if camber == 2:
            df1 = df.loc[(df['FZ'] >= -55.0) & (df['FZ'] <= -45.0) & (df['P'] >= 9.5) & (df['P'] <= 10.5) & (df['IA'] >= 1.0) & (df['IA'] <= 3.0)]

            df2 = df.loc[(df['FZ'] >= -105.0) & (df['FZ'] <= -95.0) & (df['P'] >= 9.5) & (df['P'] <= 10.5) & (df['IA'] >= 1.0) & (df['IA'] <= 3.0)]

            df3 = df.loc[(df['FZ'] >= -155.0) & (df['FZ'] <= -145.0) & (df['P'] >= 9.5) & (df['P'] <= 10.5) & (df['IA'] >= 1.0) & (df['IA'] <= 3.0)] 

            df4 = df.loc[(df['FZ'] >= -205.0) & (df['FZ'] <= -195.0) & (df['P'] >= 9.5) & (df['P'] <= 10.5) & (df['IA'] >= 1.0) & (df['IA'] <= 3.0)]

            df5 = df.loc[(df['FZ'] >= -255.0) & (df['FZ'] <= -245.0) & (df['P'] >= 9.5) & (df['P'] <= 10.5) & (df['IA'] >= 1.0) & (df['IA'] <= 3.0)]

    #  4 IA data scrape
        if camber == 4:
            df1 = df.loc[(df['FZ'] >= -55.0) & (df['FZ'] <= -45.0) & (df['P'] >= 9.5) & (df['P'] <= 10.5) & (df['IA'] >= 3.0) & (df['IA'] <= 5.0)]

            df2 = df.loc[(df['FZ'] >= -105.0) & (df['FZ'] <= -95.0) & (df['P'] >= 9.5) & (df['P'] <= 10.5) & (df['IA'] >= 3.0) & (df['IA'] <= 5.0)]

            df3 = df.loc[(df['FZ'] >= -155.0) & (df['FZ'] <= -145.0) & (df['P'] >= 9.5) & (df['P'] <= 10.5) & (df['IA'] >= 3.0) & (df['IA'] <= 5.0)] 

            df4 = df.loc[(df['FZ'] >= -205.0) & (df['FZ'] <= -195.0) & (df['P'] >= 9.5) & (df['P'] <= 10.5) & (df['IA'] >= 3.0) & (df['IA'] <= 5.0)]

            df5 = df.loc[(df['FZ'] >= -255.0) & (df['FZ'] <= -245.0) & (df['P'] >= 9.5) & (df['P'] <= 10.5) & (df['IA'] >= 3.0) & (df['IA'] <= 5.0)]
    if psi == 12:
    # 0 IA data scrape
        if camber == 0:
            df1 = df.loc[(df['FZ'] >= -55.0) & (df['FZ'] <= -45.0) & (df['P'] >= 11.5) & (df['P'] <= 12.5) & (df['IA'] >= -1.0) & (df['IA'] <= 1.0)]

            df2 = df.loc[(df['FZ'] >= -105.0) & (df['FZ'] <= -95.0) & (df['P'] >= 11.5) & (df['P'] <= 12.5) & (df['IA'] >= -1.0) & (df['IA'] <= 1.0)]

            df3 = df.loc[(df['FZ'] >= -155.0) & (df['FZ'] <= -145.0) & (df['P'] >= 11.5) & (df['P'] <= 12.5) & (df['IA'] >= -1.0) & (df['IA'] <= 1.0)] 

            df4 = df.loc[(df['FZ'] >= -205.0) & (df['FZ'] <= -195.0) & (df['P'] >= 11.5) & (df['P'] <= 12.5) & (df['IA'] >= -1.0) & (df['IA'] <= 1.0)]

            df5 = df.loc[(df['FZ'] >= -255.0) & (df['FZ'] <= -245.0) & (df['P'] >= 11.5) & (df['P'] <= 12.5) & (df['IA'] >= -1.0) & (df['IA'] <= 1.0)]

    #  2 IA data scrape
        if camber == 2:
            df1 = df.loc[(df['FZ'] >= -55.0) & (df['FZ'] <= -45.0) & (df['P'] >= 11.5) & (df['P'] <= 12.5) & (df['IA'] >= 1.0) & (df['IA'] <= 3.0)]

            df2 = df.loc[(df['FZ'] >= -105.0) & (df['FZ'] <= -95.0) & (df['P'] >= 11.5) & (df['P'] <= 12.5) & (df['IA'] >= 1.0) & (df['IA'] <= 3.0)]

            df3 = df.loc[(df['FZ'] >= -155.0) & (df['FZ'] <= -145.0) & (df['P'] >= 11.5) & (df['P'] <= 12.5) & (df['IA'] >= 1.0) & (df['IA'] <= 3.0)] 

            df4 = df.loc[(df['FZ'] >= -205.0) & (df['FZ'] <= -195.0) & (df['P'] >= 11.5) & (df['P'] <= 12.5) & (df['IA'] >= 1.0) & (df['IA'] <= 3.0)]

            df5 = df.loc[(df['FZ'] >= -255.0) & (df['FZ'] <= -245.0) & (df['P'] >= 11.5) & (df['P'] <= 12.5) & (df['IA'] >= 1.0) & (df['IA'] <= 3.0)]

    #  4 IA data scrape
        if camber == 4:
            df1 = df.loc[(df['FZ'] >= -55.0) & (df['FZ'] <= -45.0) & (df['P'] >= 11.5) & (df['P'] <= 12.5) & (df['IA'] >= 3.0) & (df['IA'] <= 5.0)]

            df2 = df.loc[(df['FZ'] >= -105.0) & (df['FZ'] <= -95.0) & (df['P'] >= 11.5) & (df['P'] <= 12.5) & (df['IA'] >= 3.0) & (df['IA'] <= 5.0)]

            df3 = df.loc[(df['FZ'] >= -155.0) & (df['FZ'] <= -145.0) & (df['P'] >= 11.5) & (df['P'] <= 12.5) & (df['IA'] >= 3.0) & (df['IA'] <= 5.0)] 

            df4 = df.loc[(df['FZ'] >= -205.0) & (df['FZ'] <= -195.0) & (df['P'] >= 11.5) & (df['P'] <= 12.5) & (df['IA'] >= 3.0) & (df['IA'] <= 5.0)]

            df5 = df.loc[(df['FZ'] >= -255.0) & (df['FZ'] <= -245.0) & (df['P'] >= 11.5) & (df['P'] <= 12.5) & (df['IA'] >= 3.0) & (df['IA'] <= 5.0)]
    
    if psi == 14:
    # 0 IA data scrape
        if camber == 0:
            df1 = df.loc[(df['FZ'] >= -55.0) & (df['FZ'] <= -45.0) & (df['P'] >= 13.5) & (df['P'] <= 14.5) & (df['IA'] >= -1.0) & (df['IA'] <= 1.0)]

            df2 = df.loc[(df['FZ'] >= -105.0) & (df['FZ'] <= -95.0) & (df['P'] >= 13.5) & (df['P'] <= 14.5) & (df['IA'] >= -1.0) & (df['IA'] <= 1.0)]

            df3 = df.loc[(df['FZ'] >= -155.0) & (df['FZ'] <= -145.0) & (df['P'] >= 13.5) & (df['P'] <= 14.5) & (df['IA'] >= -1.0) & (df['IA'] <= 1.0)] 

            df4 = df.loc[(df['FZ'] >= -205.0) & (df['FZ'] <= -195.0) & (df['P'] >= 13.5) & (df['P'] <= 14.5) & (df['IA'] >= -1.0) & (df['IA'] <= 1.0)]

            df5 = df.loc[(df['FZ'] >= -255.0) & (df['FZ'] <= -245.0) & (df['P'] >= 13.5) & (df['P'] <= 14.5) & (df['IA'] >= -1.0) & (df['IA'] <= 1.0)]

    #  2 IA data scrape
        if camber == 2:
            df1 = df.loc[(df['FZ'] >= -55.0) & (df['FZ'] <= -45.0) & (df['P'] >= 13.5) & (df['P'] <= 14.5) & (df['IA'] >= 1.0) & (df['IA'] <= 3.0)]

            df2 = df.loc[(df['FZ'] >= -105.0) & (df['FZ'] <= -95.0) & (df['P'] >= 13.5) & (df['P'] <= 14.5) & (df['IA'] >= 1.0) & (df['IA'] <= 3.0)]

            df3 = df.loc[(df['FZ'] >= -155.0) & (df['FZ'] <= -145.0) & (df['P'] >= 13.5) & (df['P'] <= 14.5) & (df['IA'] >= 1.0) & (df['IA'] <= 3.0)] 

            df4 = df.loc[(df['FZ'] >= -205.0) & (df['FZ'] <= -195.0) & (df['P'] >= 13.5) & (df['P'] <= 14.5) & (df['IA'] >= 1.0) & (df['IA'] <= 3.0)]

            df5 = df.loc[(df['FZ'] >= -255.0) & (df['FZ'] <= -245.0) & (df['P'] >= 13.5) & (df['P'] <= 14.5) & (df['IA'] >= 1.0) & (df['IA'] <= 3.0)]

    #  4 IA data scrape
        if camber == 4:
            df1 = df.loc[(df['FZ'] >= -55.0) & (df['FZ'] <= -45.0) & (df['P'] >= 13.5) & (df['P'] <= 14.5) & (df['IA'] >= 3.0) & (df['IA'] <= 5.0)]

            df2 = df.loc[(df['FZ'] >= -105.0) & (df['FZ'] <= -95.0) & (df['P'] >= 13.5) & (df['P'] <= 14.5) & (df['IA'] >= 3.0) & (df['IA'] <= 5.0)]

            df3 = df.loc[(df['FZ'] >= -155.0) & (df['FZ'] <= -145.0) & (df['P'] >= 13.5) & (df['P'] <= 14.5) & (df['IA'] >= 3.0) & (df['IA'] <= 5.0)] 

            df4 = df.loc[(df['FZ'] >= -205.0) & (df['FZ'] <= -195.0) & (df['P'] >= 13.5) & (df['P'] <= 14.5) & (df['IA'] >= 3.0) & (df['IA'] <= 5.0)]

            df5 = df.loc[(df['FZ'] >= -255.0) & (df['FZ'] <= -245.0) & (df['P'] >= 13.5) & (df['P'] <= 14.5) & (df['IA'] >= 3.0) & (df['IA'] <= 5.0)]

# This section cleans noise from the data, The format of the runs changes between tests and this can be changed to clean that.
df1 = df1.iloc[800:]
df2 = df2.iloc[800:]
df3 = df3.iloc[500:]
df4 = df4.iloc[500:]
df5 = df5.iloc[500:]

# The four different pacejka fitting algorithms are to make the loading times shorter and so that the curves fit a bit better that if there was
# just one model, This is a relativly simple model however it works fine for the FSAE application
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
Fy_fit_50 = Fy_fit1
plt.plot(a_fit1, Fy_fit1, label='50 lbs', color = 'blue')
plt.plot(a_fit1, Fy_fit1 * 0.65, label='50 lbs scaled', color = lighten_color('blue', 0.4))

a = df2["SA"].values
Fy = df2["FY"].values * (-1)

popt = fit_pacejka2(a, Fy)
popt2 = popt

a_fit2 = np.linspace(min(a), max(a), num=200)
Fy_fit2 = pacejka2(a_fit2, *popt)
Fy_fit_100 = Fy_fit2
plt.plot(a_fit2, Fy_fit2, label='100 lbs', color = 'purple')
plt.plot(a_fit2, Fy_fit2 * 0.65, label='100 lbs scaled', color = lighten_color('purple', 0.4))

a = df3["SA"].values
Fy = df3["FY"].values * (-1)

popt = fit_pacejka3(a, Fy)
popt3 = popt

a_fit3 = np.linspace(min(a), max(a), num=200)
Fy_fit3 = pacejka3(a_fit3, *popt)
Fy_fit_150 = Fy_fit3
plt.plot(a_fit3, Fy_fit3, label='150 lbs', color = 'green')
plt.plot(a_fit3, Fy_fit3 * 0.65, label='150 lbs scaled', color = lighten_color('green', 0.4))

a = df4["SA"].values
Fy = df4["FY"].values * (-1)

popt = fit_pacejka4(a, Fy)
popt4 = popt

a_fit4 = np.linspace(min(a), max(a), num=200)
Fy_fit4 = pacejka4(a_fit4, *popt)
Fy_fit_200 = Fy_fit4
plt.plot(a_fit4, Fy_fit4, label='200 lbs', color = 'orange')
plt.plot(a_fit4, Fy_fit4 * 0.65, label='200 lbs scaled', color = lighten_color('orange', 0.4))

a = df5["SA"].values
Fy = df5["FY"].values * (-1)

popt = fit_pacejka5(a, Fy)
popt5 = popt

a_fit5 = np.linspace(min(a), max(a), num=200)
Fy_fit5 = pacejka5(a_fit5, *popt)
Fy_fit_250 = Fy_fit5
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
plt.savefig(output + '/LateralForceSA.png')
Worksheet1 = writer.sheets['LatForce VS SlipAng ADJ']
Worksheet1.insert_image('A1', output + '/LateralForceSA.png')
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

plt.xlabel("Vertical Force (lbs)")
plt.ylabel("Cornering Stiffness (lbs/deg)")
plt.grid()
plt.savefig(output + '/CorneringStiffness.png')
Worksheet2 = writer.sheets['CorStiff VS VertLoad ADJ']
Worksheet2.insert_image('A1', output + '/CorneringStiffness.png')
plt.show()

muy_50 = Fy_fit_50 / 50
muy_100 = Fy_fit_100 / 100
muy_150 = Fy_fit_150 / 150
muy_200 = Fy_fit_200 / 200
muy_250 = Fy_fit_250 / 250

plt.plot(a_fit1, muy_50, label='50 lbs', color = 'blue')
plt.plot(a_fit1, muy_50 * 0.65, label='50 lbs Scaled', color = lighten_color('blue', 0.4))
plt.plot(a_fit2, muy_100, label='100 lbs', color = 'green')
plt.plot(a_fit2, muy_100 * 0.65, label='100 lbs Scaled', color = lighten_color('green', 0.4))
plt.plot(a_fit3, muy_150, label='150 lbs', color = 'orange')
plt.plot(a_fit3, muy_150 * 0.65, label='150 lbs Scaled', color = lighten_color('orange', 0.4))
plt.plot(a_fit4, muy_200, label='200 lbs', color = 'red')
plt.plot(a_fit4, muy_200 * 0.65, label='200 lbs Scaled', color = lighten_color('red', 0.4))
plt.plot(a_fit5, muy_250, label='250 lbs', color = 'yellow')
plt.plot(a_fit5, muy_250 * 0.65, label='250 lbs Scaled', color = lighten_color('yellow', 0.4))

Muy = {"SA @ 50" : a_fit1, "Muy 50" : muy_50 * 0.65,"SA @ 100" : a_fit2, "Muy @ 100" : muy_100 * 0.65, "SA @ 150" : a_fit3, "Muy @ 150" : muy_150 * 0.65, 
          "SA @ 200" : a_fit4, "Muy 200" : muy_200 * 0.65,"SA @ 250" : a_fit5, "Muy @ 250" : muy_250 * 0.65}

df_Muy = pd.DataFrame.from_dict(Muy, orient='index')
df_Muy = df_Muy.transpose()
df_Muy.to_excel(writer, sheet_name = 'FrictCoef VS SlipAng ADJ', index = False)

plt.xlabel('Slip Angle (deg)')
plt.ylabel('Friction Coefficient (muy)')
plt.grid()
plt.legend()
plt.savefig(output + '/frictionMuySA.png')
Worksheet3 = writer.sheets['FrictCoef VS SlipAng ADJ']
Worksheet3.insert_image('A1', output + '/frictionMuySA.png')
plt.show()

headers2=['ET','FX','FZ','SA','P','TSTI','TSTC','TSTO','AMBTMP','IA','SL']

# where the longitudinal calculations are done. Like the lateral choices for camber, IA, and pressure are set in the gui
LongLoads = sio.loadmat(longitude)
def Convert(string):
    li = list(string.split(","))
    out = []
    for item in li:
        out.append(float(item))
    return out

ETin = LongLoads['ET'].tolist()
ETin = str(ETin)
ETin = ETin.replace("[","")
ETin = ETin.replace("]","")
ETin = ETin.replace(" ","")
ETin = Convert(ETin)

FXin = LongLoads['FX'].tolist()
FXin = str(FXin)
FXin = FXin.replace("[","")
FXin = FXin.replace("]","")
FXin = FXin.replace(" ","")
FXin = Convert(FXin)

FZin = LongLoads['FZ'].tolist()
FZin = str(FZin)
FZin = FZin.replace("[","")
FZin = FZin.replace("]","")
FZin = FZin.replace(" ","")
FZin = Convert(FZin)

SAin = LongLoads['SA'].tolist()
SAin = str(SAin)
SAin = SAin.replace("[","")
SAin = SAin.replace("]","")
SAin = SAin.replace(" ","")
SAin = Convert(SAin)

Pin = LongLoads['P'].tolist()
Pin = str(Pin)
Pin = Pin.replace("[","")
Pin = Pin.replace("]","")
Pin = Pin.replace(" ","")
Pin = Convert(Pin)

TSTIin = LongLoads['TSTI'].tolist()
TSTIin = str(TSTIin)
TSTIin = TSTIin.replace("[","")
TSTIin = TSTIin.replace("]","")
TSTIin = TSTIin.replace(" ","")
TSTIin = Convert(TSTIin)

TSTCin = LongLoads['TSTC'].tolist()
TSTCin = str(TSTCin)
TSTCin = TSTCin.replace("[","")
TSTCin = TSTCin.replace("]","")
TSTCin = TSTCin.replace(" ","")
TSTCin = Convert(TSTCin)

TSTOin = LongLoads['TSTO'].tolist()
TSTOin = str(TSTOin)
TSTOin = TSTOin.replace("[","")
TSTOin = TSTOin.replace("]","")
TSTOin = TSTOin.replace(" ","")
TSTOin = Convert(TSTOin)

AMBTMPin = LongLoads['AMBTMP'].tolist()
AMBTMPin = str(AMBTMPin)
AMBTMPin = AMBTMPin.replace("[","")
AMBTMPin = AMBTMPin.replace("]","")
AMBTMPin = AMBTMPin.replace(" ","")
AMBTMPin = Convert(AMBTMPin)

IAin = LongLoads['IA'].tolist()
IAin = str(IAin)
IAin = IAin.replace("[","")
IAin = IAin.replace("]","")
IAin = IAin.replace(" ","")
IAin = Convert(IAin)

SLin = LongLoads['SL'].tolist()
SLin = str(SLin)
SLin = SLin.replace("[","")
SLin = SLin.replace("]","")
SLin = SLin.replace(" ","")
SLin = Convert(SLin)

d1 = {'ET' : ETin, 'FX' : FXin, 'FZ' : FZin, 'SA' : SAin, 'P' : Pin, 'TSTI' : TSTIin,
      'TSTC' : TSTCin, 'TSTO' : TSTOin, 'AMBTMP' : AMBTMPin, 'IA' : IAin, 'SL' : SLin}

dfLong = pd.DataFrame(data=d1)

# Similar to the lateral files, the data is sorted out based on the choices made in the gui

for rows in dfLong:

    # dfLongLoads = dfLong.loc[(dfLong['P'] >= 11.5) & (dfLong['P'] <= 12.5) & (dfLong['IA'] >= -1.0) & (dfLong['IA'] <= 1.0)]
    if psi == 10:
    # 0 IA data scrape
        if camber == 0:
            dfLong1 = dfLong.loc[(dfLong['FZ'] >= -55.0) & (dfLong['FZ'] <= -45.0) & (dfLong['P'] >= 9.5) & (dfLong['P'] <= 10.5) & (dfLong['IA'] >= -1.0) & (dfLong['IA'] <= 1.0)]
            
            dfLong3 = dfLong.loc[(dfLong['FZ'] >= -155.0) & (dfLong['FZ'] <= -145.0) & (dfLong['P'] >= 9.5) & (dfLong['P'] <= 10.5) & (dfLong['IA'] >= -1.0) & (dfLong['IA'] <= 1.0)]

            dfLong4 = dfLong.loc[(dfLong['FZ'] >= -205.0) & (dfLong['FZ'] <= -195.0) & (dfLong['P'] >= 9.5) & (dfLong['P'] <= 10.5) & (dfLong['IA'] >= -1.0) & (dfLong['IA'] <= 1.0)]
            
            dfLong5 = dfLong.loc[(dfLong['FZ'] >= -255.0) & (dfLong['FZ'] <= -145.0) & (dfLong['P'] >= 9.5) & (dfLong['P'] <= 10.5) & (dfLong['IA'] >= -1.0) & (dfLong['IA'] <= 1.0)]

    # 2 IA data scrape
        if camber == 2:
            dfLong1 = dfLong.loc[(dfLong['FZ'] >= -55.0) & (dfLong['FZ'] <= -45.0) & (dfLong['P'] >= 9.5) & (dfLong['P'] <= 10.5) & (dfLong['IA'] >= 1.0) & (dfLong['IA'] <= 3.0)]
            
            dfLong3 = dfLong.loc[(dfLong['FZ'] >= -155.0) & (dfLong['FZ'] <= -145.0) & (dfLong['P'] >= 9.5) & (dfLong['P'] <= 10.5) & (dfLong['IA'] >= 1.0) & (dfLong['IA'] <= 3.0)]

            dfLong4 = dfLong.loc[(dfLong['FZ'] >= -205.0) & (dfLong['FZ'] <= -195.0) & (dfLong['P'] >= 9.5) & (dfLong['P'] <= 10.5) & (dfLong['IA'] >= 1.0) & (dfLong['IA'] <= 3.0)]
            
            dfLong5 = dfLong.loc[(dfLong['FZ'] >= -255.0) & (dfLong['FZ'] <= -145.0) & (dfLong['P'] >= 9.5) & (dfLong['P'] <= 10.5) & (dfLong['IA'] >= 1.0) & (dfLong['IA'] <= 3.0)]

    # 4 IA data scrape
        if camber == 4:
            dfLong1 = dfLong.loc[(dfLong['FZ'] >= -55.0) & (dfLong['FZ'] <= -45.0) & (dfLong['P'] >= 9.5) & (dfLong['P'] <= 10.5) & (dfLong['IA'] >= 3.0) & (dfLong['IA'] <= 5.0)]
            
            dfLong3 = dfLong.loc[(dfLong['FZ'] >= -155.0) & (dfLong['FZ'] <= -145.0) & (dfLong['P'] >= 9.5) & (dfLong['P'] <= 10.5) & (dfLong['IA'] >= 3.0) & (dfLong['IA'] <= 5.0)]

            dfLong4 = dfLong.loc[(dfLong['FZ'] >= -205.0) & (dfLong['FZ'] <= -195.0) & (dfLong['P'] >= 9.5) & (dfLong['P'] <= 10.5) & (dfLong['IA'] >= 3.0) & (dfLong['IA'] <= 5.0)]
            
            dfLong5 = dfLong.loc[(dfLong['FZ'] >= -255.0) & (dfLong['FZ'] <= -145.0) & (dfLong['P'] >= 9.5) & (dfLong['P'] <= 10.5) & (dfLong['IA'] >= 3.0) & (dfLong['IA'] <= 5.0)]
    if psi == 12:
    # 0 IA data scrape
        if camber == 0:
            dfLong1 = dfLong.loc[(dfLong['FZ'] >= -55.0) & (dfLong['FZ'] <= -45.0) & (dfLong['P'] >= 11.5) & (dfLong['P'] <= 12.5) & (dfLong['IA'] >= -1.0) & (dfLong['IA'] <= 1.0)]
            
            dfLong3 = dfLong.loc[(dfLong['FZ'] >= -155.0) & (dfLong['FZ'] <= -145.0) & (dfLong['P'] >= 11.5) & (dfLong['P'] <= 12.5) & (dfLong['IA'] >= -1.0) & (dfLong['IA'] <= 1.0)]

            dfLong4 = dfLong.loc[(dfLong['FZ'] >= -205.0) & (dfLong['FZ'] <= -195.0) & (dfLong['P'] >= 11.5) & (dfLong['P'] <= 12.5) & (dfLong['IA'] >= -1.0) & (dfLong['IA'] <= 1.0)]
            
            dfLong5 = dfLong.loc[(dfLong['FZ'] >= -255.0) & (dfLong['FZ'] <= -145.0) & (dfLong['P'] >= 11.5) & (dfLong['P'] <= 12.5) & (dfLong['IA'] >= -1.0) & (dfLong['IA'] <= 1.0)]

    # 2 IA data scrape
        if camber == 2:
            dfLong1 = dfLong.loc[(dfLong['FZ'] >= -55.0) & (dfLong['FZ'] <= -45.0) & (dfLong['P'] >= 11.5) & (dfLong['P'] <= 12.5) & (dfLong['IA'] >= 1.0) & (dfLong['IA'] <= 3.0)]
            
            dfLong3 = dfLong.loc[(dfLong['FZ'] >= -155.0) & (dfLong['FZ'] <= -145.0) & (dfLong['P'] >= 11.5) & (dfLong['P'] <= 12.5) & (dfLong['IA'] >= 1.0) & (dfLong['IA'] <= 3.0)]

            dfLong4 = dfLong.loc[(dfLong['FZ'] >= -205.0) & (dfLong['FZ'] <= -195.0) & (dfLong['P'] >= 11.5) & (dfLong['P'] <= 12.5) & (dfLong['IA'] >= 1.0) & (dfLong['IA'] <= 3.0)]
            
            dfLong5 = dfLong.loc[(dfLong['FZ'] >= -255.0) & (dfLong['FZ'] <= -145.0) & (dfLong['P'] >= 11.5) & (dfLong['P'] <= 12.5) & (dfLong['IA'] >= 1.0) & (dfLong['IA'] <= 3.0)]

    # 4 IA data scrape
        if camber == 4:
            dfLong1 = dfLong.loc[(dfLong['FZ'] >= -55.0) & (dfLong['FZ'] <= -45.0) & (dfLong['P'] >= 11.5) & (dfLong['P'] <= 12.5) & (dfLong['IA'] >= 3.0) & (dfLong['IA'] <= 5.0)]
            
            dfLong3 = dfLong.loc[(dfLong['FZ'] >= -155.0) & (dfLong['FZ'] <= -145.0) & (dfLong['P'] >= 11.5) & (dfLong['P'] <= 12.5) & (dfLong['IA'] >= 3.0) & (dfLong['IA'] <= 5.0)]

            dfLong4 = dfLong.loc[(dfLong['FZ'] >= -205.0) & (dfLong['FZ'] <= -195.0) & (dfLong['P'] >= 11.5) & (dfLong['P'] <= 12.5) & (dfLong['IA'] >= 3.0) & (dfLong['IA'] <= 5.0)]
            
            dfLong5 = dfLong.loc[(dfLong['FZ'] >= -255.0) & (dfLong['FZ'] <= -145.0) & (dfLong['P'] >= 11.5) & (dfLong['P'] <= 12.5) & (dfLong['IA'] >= 3.0) & (dfLong['IA'] <= 5.0)]

    if psi == 14:
    # 0 IA data scrape
        if camber == 0:
            dfLong1 = dfLong.loc[(dfLong['FZ'] >= -55.0) & (dfLong['FZ'] <= -45.0) & (dfLong['P'] >= 13.5) & (dfLong['P'] <= 14.5) & (dfLong['IA'] >= -1.0) & (dfLong['IA'] <= 1.0)]
            
            dfLong3 = dfLong.loc[(dfLong['FZ'] >= -155.0) & (dfLong['FZ'] <= -145.0) & (dfLong['P'] >= 13.5) & (dfLong['P'] <= 14.5) & (dfLong['IA'] >= -1.0) & (dfLong['IA'] <= 1.0)]

            dfLong4 = dfLong.loc[(dfLong['FZ'] >= -205.0) & (dfLong['FZ'] <= -195.0) & (dfLong['P'] >= 13.5) & (dfLong['P'] <= 14.5) & (dfLong['IA'] >= -1.0) & (dfLong['IA'] <= 1.0)]
            
            dfLong5 = dfLong.loc[(dfLong['FZ'] >= -255.0) & (dfLong['FZ'] <= -145.0) & (dfLong['P'] >= 13.5) & (dfLong['P'] <= 14.5) & (dfLong['IA'] >= -1.0) & (dfLong['IA'] <= 1.0)]

    # 2 IA data scrape
        if camber == 2:
            dfLong1 = dfLong.loc[(dfLong['FZ'] >= -55.0) & (dfLong['FZ'] <= -45.0) & (dfLong['P'] >= 13.5) & (dfLong['P'] <= 14.5) & (dfLong['IA'] >= 1.0) & (dfLong['IA'] <= 3.0)]
            
            dfLong3 = dfLong.loc[(dfLong['FZ'] >= -155.0) & (dfLong['FZ'] <= -145.0) & (dfLong['P'] >= 13.5) & (dfLong['P'] <= 14.5) & (dfLong['IA'] >= 1.0) & (dfLong['IA'] <= 3.0)]

            dfLong4 = dfLong.loc[(dfLong['FZ'] >= -205.0) & (dfLong['FZ'] <= -195.0) & (dfLong['P'] >= 13.5) & (dfLong['P'] <= 14.5) & (dfLong['IA'] >= 1.0) & (dfLong['IA'] <= 3.0)]
            
            dfLong5 = dfLong.loc[(dfLong['FZ'] >= -255.0) & (dfLong['FZ'] <= -145.0) & (dfLong['P'] >= 13.5) & (dfLong['P'] <= 14.5) & (dfLong['IA'] >= 1.0) & (dfLong['IA'] <= 3.0)]

    # 4 IA data scrape
        if camber == 4:
            dfLong1 = dfLong.loc[(dfLong['FZ'] >= -55.0) & (dfLong['FZ'] <= -45.0) & (dfLong['P'] >= 13.5) & (dfLong['P'] <= 14.5) & (dfLong['IA'] >= 3.0) & (dfLong['IA'] <= 5.0)]
            
            dfLong3 = dfLong.loc[(dfLong['FZ'] >= -155.0) & (dfLong['FZ'] <= -145.0) & (dfLong['P'] >= 13.5) & (dfLong['P'] <= 14.5) & (dfLong['IA'] >= 3.0) & (dfLong['IA'] <= 5.0)]

            dfLong4 = dfLong.loc[(dfLong['FZ'] >= -205.0) & (dfLong['FZ'] <= -195.0) & (dfLong['P'] >= 13.5) & (dfLong['P'] <= 14.5) & (dfLong['IA'] >= 3.0) & (dfLong['IA'] <= 5.0)]
            
            dfLong5 = dfLong.loc[(dfLong['FZ'] >= -255.0) & (dfLong['FZ'] <= -145.0) & (dfLong['P'] >= 13.5) & (dfLong['P'] <= 14.5) & (dfLong['IA'] >= 3.0) & (dfLong['IA'] <= 5.0)]

# there is no section where the data is cleaned up here it might be needed if you encounter a file with noise 

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



# Pulls out the slip ratio and lateral force values to be fit
Sl = dfLong1["SL"].values 
Fx = dfLong1["FX"].values

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
plt.savefig(output + '/LongSR.png')
Worksheet4 = writer.sheets['LongForce VS SlipRat ADJ']
Worksheet4.insert_image('A1', output + '/LongSR.png')
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


plt.xlabel('Slip Ratio')
plt.ylabel('Friction Coefficient')
plt.grid()
plt.legend()
plt.savefig(output + '/FrictionForceSlip.png')
Worksheet5 = writer.sheets['TireFrict VS SlipRat ADJ']
Worksheet5.insert_image('A1', output + '/FrictionForceSlip.png')
plt.show()

writer.close()