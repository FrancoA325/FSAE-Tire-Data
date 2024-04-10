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
import streamlit as st
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog

import tkinter as tk
from tkinter import ttk, filedialog
import os

def main():
    matlabIn = st.file_uploader("Tire File Input:", type=['mat'])

    if matlabIn is not None:
        inFile = sio.loadmat(matlabIn)

        option = st.selectbox(
            'Select an option:',
            ('Cornering', 'Drive/Brake')
        )

    else:
        st.write("Please upload a MATLAB file.")
        
    def Convert(string):
        li = list(string.split(","))
        out = []
        for item in li:
            out.append(float(item))
        return out
    
    def filterData():
        
        ETin = inFile['ET'].tolist()
        ETin = str(ETin)
        ETin = ETin.replace("[","")
        ETin = ETin.replace("]","")
        ETin = ETin.replace(" ","")
        ETin = Convert(ETin)

        FXin = inFile['FX'].tolist()
        FXin = str(FXin)
        FXin = FXin.replace("[","")
        FXin = FXin.replace("]","")
        FXin = FXin.replace(" ","")
        FXin = Convert(FXin)

        FYin = inFile['FY'].tolist()
        FYin = str(FYin)
        FYin = FYin.replace("[","")
        FYin = FYin.replace("]","")
        FYin = FYin.replace(" ","")
        FYin = Convert(FYin)

        FZin = inFile['FZ'].tolist()
        FZin = str(FZin)
        FZin = FZin.replace("[","")
        FZin = FZin.replace("]","")
        FZin = FZin.replace(" ","")
        FZin = Convert(FZin)

        SAin = inFile['SA'].tolist()
        SAin = str(SAin)
        SAin = SAin.replace("[","")
        SAin = SAin.replace("]","")
        SAin = SAin.replace(" ","")
        SAin = Convert(SAin)

        Pin = inFile['P'].tolist()
        Pin = str(Pin)
        Pin = Pin.replace("[","")
        Pin = Pin.replace("]","")
        Pin = Pin.replace(" ","")
        Pin = Convert(Pin)

        TSTIin = inFile['TSTI'].tolist()
        TSTIin = str(TSTIin)
        TSTIin = TSTIin.replace("[","")
        TSTIin = TSTIin.replace("]","")
        TSTIin = TSTIin.replace(" ","")
        TSTIin = Convert(TSTIin)

        TSTCin = inFile['TSTC'].tolist()
        TSTCin = str(TSTCin)
        TSTCin = TSTCin.replace("[","")
        TSTCin = TSTCin.replace("]","")
        TSTCin = TSTCin.replace(" ","")
        TSTCin = Convert(TSTCin)

        TSTOin = inFile['TSTO'].tolist()
        TSTOin = str(TSTOin)
        TSTOin = TSTOin.replace("[","")
        TSTOin = TSTOin.replace("]","")
        TSTOin = TSTOin.replace(" ","")
        TSTOin = Convert(TSTOin)

        AMBTMPin = inFile['AMBTMP'].tolist()
        AMBTMPin = str(AMBTMPin)
        AMBTMPin = AMBTMPin.replace("[","")
        AMBTMPin = AMBTMPin.replace("]","")
        AMBTMPin = AMBTMPin.replace(" ","")
        AMBTMPin = Convert(AMBTMPin)

        IAin = inFile['IA'].tolist()
        IAin = str(IAin)
        IAin = IAin.replace("[","")
        IAin = IAin.replace("]","")
        IAin = IAin.replace(" ","")
        IAin = Convert(IAin)
        
        SLin = inFile['SL'].tolist()
        SLin = str(SLin)
        SLin = SLin.replace("[","")
        SLin = SLin.replace("]","")
        SLin = SLin.replace(" ","")
        SLin = Convert(SLin)

        d = {'ET' : ETin, 'FX' : FXin, 'FY' : FYin, 'FZ' : FZin, 'SA' : SAin, 'P' : Pin, 'TSTI' : TSTIin,
            'TSTC' : TSTCin, 'TSTO' : TSTOin, 'AMBTMP' : AMBTMPin, 'IA' : IAin, 'SL' : SLin}

        tireMatFilter = pd.DataFrame(data=d)
        
if __name__ == "__main__":
    
    if st.button("Start App"):
        main()
