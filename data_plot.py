from anyio import ClosedResourceError
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression

my_dataset = pd.read_excel('ferrous detection of EMPA.xlsx', sheet_name='test3')



fig = plt.figure(figsize=(11,5))
ax1= fig.add_subplot(1,2,1)
ax1.set_title("Lalpha")
for (mineral, clr) in [("hematite","blue"),("magnetite","pink"),("Di","orange"),("ChDi","green"),("Bu","red"),("chromite","black")]:
    my_data = my_dataset[(my_dataset.name == mineral)]
    ax1.scatter(my_data.Lα_intensity,my_data.ferrous_total)
"""
def new_func(my_dataset):
    features = my_dataset.Lα_intensity.values.reshape(-1,1)
    target = my_dataset.ferrous_proportion
    regression = LinearRegression()
    model = regression.fit(features,target)
    model.plot()

new_func(my_dataset)
"""
ax1.set_xlabel('Lα_intensity')
ax1.set_ylabel('ferrous_proportion')
ax1.legend()

"""
b1, b0, rho_value, p_value, std_err = st.linregress(my_dataset.ferrous_proportion, my_dataset.Lα_intensity)
x = np.linspace(my_dataset.ferrous_proportion.min(),my_dataset.ferrous_proportion.max())
y = b0 + b1*x
ax1.plot(x, y, linewidth=1, color=clr, linestyle='--', 
         label=r"fit param.: $\beta_0$ = " + '{:.1f}'.format(b0) 
         + r" - $\beta_1$ = "  + '{:.1f}'.format(b1) + r" - $r_{xy}^{2}$ = " 
         + '{:.2f}'.format(rho_value**2))
"""
"""
popt1, pcov1 = curve_fit(func, my_dataset.ferrous_proportion, my_dataset.Lα_intensity, method='trf', bounds=([0.8,0,0],[1.3,1000,1000]))
x1 = np.linspace(my_dataset.ferrous_proportion.min(),my_dataset.ferrous_proportion.max())
y1 = func(x1,popt1[0],popt1[1], popt1[2])
ax1.plot(x1,y1, color='#ff464a', linewidth=2, linestyle ='--', 
         label=r'$r_0$ = ' + '{:.3f}'.format(popt1[0]) 
         + r', $D_0$ = ' + '{:.0f}'.format(popt1[1]) 
         + ', E = ' + '{:.0f}'.format(popt1[2]))

"""

ax2 = fig.add_subplot(1,2,2)
for (mineral, clr) in [("hematite","blue"),("magnetite","pink"),("Di","orange"),("ChDi","green"),("Bu","red"),("chromite","black")]:
    my_data = my_dataset[(my_dataset.name == mineral)]
    ax2.scatter(my_data.Lα_intensity/my_data.Lβ_intensity,my_data.ferrous_total,  marker='o', edgecolor='k', color=clr)

ax2.set_xlabel('Lα/Lβ')
ax2.set_ylabel('ferrous_proportion')
ax2.legend()
"""
b1, b0, rho_value, p_value, std_err = st.linregress(my_dataset.ferrous_proportion, my_dataset.Lα_intensity/my_dataset.Lβ_intensity)
x = np.linspace(my_dataset.ferrous_proportion.min(),my_dataset.ferrous_proportion.max())
y = b0 + b1*x
ax2.plot(x, y, linewidth=1, color='#ff464a', linestyle='--', 
         label=r"fit param.: $\beta_0$ = " + '{:.1f}'.format(b0) 
         + r" - $\beta_1$ = "  + '{:.1f}'.format(b1) + r" - $r_{xy}^{2}$ = " 
         + '{:.2f}'.format(rho_value**2))

popt2, pcov2 = curve_fit(func, my_dataset.ferrous_proportion, my_dataset.Lα_intensity/my_dataset.Lβ_intensity, 
                method='lm', p0=(1.1,100,100))
x2 = np.linspace(my_dataset.ferrous_proportion.min(),my_dataset.ferrous_proportion.max())
y2 = func(x2,popt2[0],popt2[1], popt2[2])
ax2.plot(x2,y2, color='#4881e9', linewidth=2, linestyle ='--', 
         label=r'$r_0$ = ' + '{:.3f}'.format(popt2[0]) + r', $D_0$ = ' + '{:.0f}'.format(popt2[1]) + ', E = ' + '{:.0f}'.format(popt2[2]))
"""



plt.show()






