import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.interpolate import UnivariateSpline

global df2


headers=['ET','FX','FY','FZ','SA','P','TSTI','TSTC','TSTO','AMBTMP','IA'];

df = pd.read_csv('C:\CODING\PYTHONCODING\SlicedRUNS\A2356FIXED12test.csv', delimiter = ',', header=None, names = headers)

for rows in df:
    df1 = df.loc[(df.FZ.abs() >= 45.0) & (df.FZ.abs() <= 55.0) & (df['P'] >= 11.0) & (df['P'] <= 12.0)]

    df2 = df.loc[(df.FZ.abs() >= 95.0) & (df.FZ.abs()<= 105.0) & (df['P'] >= 11.0) & (df['P'] <= 12.0)]

    df3 = df.loc[(df.FZ.abs() >= 145.0) & (df.FZ.abs() <= 155.0) & (df['P'] >= 11.0) & (df['P'] <= 12.0)] 

    df4 = df.loc[(df.FZ.abs() >= 190.0) & (df.FZ.abs() <= 210.0) & (df['P'] >= 11.0) & (df['P'] <= 12.0)]


df1 = df1.iloc[:3000]
df1.FY.interpolate(method = 'nearest')
plt.figure()
ax = sns.lineplot(x = df1.SA, y = df1.FY)
ax.set(xlabel = 'Slip Angle', ylabel = 'Lateral Force')
plt.subplot()
df2 = df2.iloc[:3000]
df2.FY.interpolate(method = 'cubic')
ax2 = sns.lineplot(x = df2.SA, y = df2.FY)
plt.subplot()
df3 = df3.iloc[:3000]
df3.FY.interpolate(method = 'cubic')
ax3 = sns.lineplot(x = df3.SA, y = df3.FY)
plt.subplot()
df4 = df4.iloc[:3000]
df4.FY.interpolate(method = 'cubic')
ax4 = sns.lineplot(x = df4.SA, y = df4.FY)
plt.show()

# for rows in df:
#     df1 = df.loc[(df['TSTI'] >= 100.0) & (df['TSTI'] <= 200.0) & (df['FZ'] >= -55.0) & (df['FZ'] <= -45.0) & (df['P'] >= 11.0) & (df['P'] <= 12.0)]

#     df2 = df.loc[(df['TSTC'] >= 100.0) & (df['TSTC'] <= 200.0) & (df['FZ'] >= -55.0) & (df['FZ'] <= -45.0) & (df['P'] >= 11.0) & (df['P'] <= 12.0)]

#     df3 = df.loc[(df['TSTO'] >= 100.0) & (df['TSTO'] <= 200.0) &(df['FZ'] >= -55.0) & (df['FZ'] <= -45.0) & (df['P'] >= 11.0) & (df['P'] <= 12.0)] 


# plt.figure()
# ax4 = sns.lineplot(x = df1.ET, y = df1.FZ.abs())
# ax4.set(xlabel = "Time", ylabel = "Vertical Load")
# plt.show()

# df1.to_csv('C:\\FSAE EV\\Vehicle Dynamics\\TIRES DINK DONKERY\\pandasOUT\\df1.csv',sep = '\t',index = False)
# df2.to_csv('C:\\FSAE EV\\Vehicle Dynamics\\TIRES DINK DONKERY\\pandasOUT\\df2.csv',sep = '\t',index = False)
# df3.to_csv('C:\\FSAE EV\\Vehicle Dynamics\\TIRES DINK DONKERY\\pandasOUT\\df3.csv',sep = '\t',index = False)
# df4.to_csv('C:\\FSAE EV\\Vehicle Dynamics\\TIRES DINK DONKERY\\pandasOUT\\df4.csv',sep = '\t',index = False)