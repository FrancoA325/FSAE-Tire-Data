import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.interpolate import UnivariateSpline

global df2


headers=['SA', 'FY', 'FZ'];
headers2 = ['SA','FY'];
df = pd.read_csv('C:\\FSAE EV\\Vehicle Dynamics\\TIRES DINK DONKERY\\Optimum-Mindstorm-master\\ballz.csv', delimiter = ',', header=None, names = headers)
for rows in df:
    df1 = df.loc[(df['FZ'] >= -55.0) & (df['FZ'] <= -45.0)]

    df2 = df.loc[(df['FZ'] >= -105.0) & (df['FZ'] <= -95.0)]

    df3 = df.loc[(df['FZ'] >= -155.0) & (df['FZ'] <= -145.0)] 

    df4 = df.loc[(df['FZ'] >= -210.0) & (df['FZ'] <= -190.0)]

df1.to_csv('C:\\FSAE EV\\Vehicle Dynamics\\TIRES DINK DONKERY\\pandasOUT\\df1.csv',sep = '\t',index = False)
df2.to_csv('C:\\FSAE EV\\Vehicle Dynamics\\TIRES DINK DONKERY\\pandasOUT\\df2.csv',sep = '\t',index = False)
df3.to_csv('C:\\FSAE EV\\Vehicle Dynamics\\TIRES DINK DONKERY\\pandasOUT\\df3.csv',sep = '\t',index = False)
df4.to_csv('C:\\FSAE EV\\Vehicle Dynamics\\TIRES DINK DONKERY\\pandasOUT\\df4.csv',sep = '\t',index = False)

# df2.head(3000), df3.head(3000), df4.head(3000)
df1 = df1.iloc[:3000]
df1.FY.interpolate(method = 'cubic')
plt.figure()
ax = sns.lineplot(x = df1.SA, y = df1.FY)
ax.set(xlabel = 'Slip Angle', ylabel = 'Lateral Force')
plt.clt()
df2 = df2.iloc[:3000]
df2.FY.interpolate(method = 'cubic')
ax1 = sns.lineplot(x = df2.SA, y = df2.FY)
plt.show()
