# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# read excel
my_dataset = pd.read_excel('ferrous detection of EMPA.xlsx', sheet_name='test3')

#definitions
# define name
name = my_dataset.name

# define intensity
Lα_intensity = my_dataset.Lα_intensity
Lβ_intensity = my_dataset.Lβ_intensity

# define peak_position
Lα_peak_position = my_dataset.Lα_peak_position
Lβ_peak_position = my_dataset.Lβ_peak_position
Lα_area_intensity = my_dataset.Lα_area_intensity
Lβ_area_intensity = my_dataset.Lβ_area_intensity

#define asymmetric index
Lα_asymmetric_index = my_dataset.Lα_asymmetric_index
Lβ_asymmetric_index = my_dataset.Lβ_asymmetric_index

#define position shift
Lα_peak_shift = my_dataset.Lα_peak_shift
Lβ_peak_shift = my_dataset.Lβ_peak_shift
Kβ_peak_shift = my_dataset.Kβ_peak_shift

#define colors
my_color = ["blue","red","yellow","green","pink","brown"]


fig = plt.figure(figsize=(20,16))
ax1 = fig.add_subplot(2,3,1)
ax2 = fig.add_subplot(2,3,2)
ax3 = fig.add_subplot(2,3,3)
ax4 = fig.add_subplot(2,3,4)
ax5 = fig.add_subplot(2,3,5)
ax6 = fig.add_subplot(2,3,6)
"""
my_data = my_dataset[(my_dataset.name == "hematite")]
ax.scatter(my_data.peak,my_data.Lα_intensity/my_data.Lβ_intensity,color = "blue")
my_data = my_dataset[(my_dataset.name == "magnetite")]
ax.scatter(my_data.peak,my_data.Lα_intensity/my_data.Lβ_intensity,color = "red",alpha = 0.1)
my_data = my_dataset[(my_dataset.name == "Di")]
ax.scatter(my_data.peak,my_data.Lα_intensity/my_data.Lβ_intensity,color = "orange")
my_data = my_dataset[(my_dataset.name == "ChDi")]
ax.scatter(my_data.peak,my_data.Lα_intensity/my_data.Lβ_intensity,color = "green",alpha = 0.5)
ax.set_xlabel(r"L$\alpha$/ L$\beta$")
ax.set_ylabel("peak position")
plt.show()
"""



## the ferric proportion and intensity field
for (mineral, clr) in [("hematite","blue"),("magnetite","pink"),("Di","orange"),("ChDi","green"),("Bu","red"),("chromite","black")]:
    my_data = my_dataset[(my_dataset.name == mineral)]
    ax1.scatter(my_data.Kα_intensity,my_data.Lα_peak_shift,color = clr, label = mineral ,alpha = 0.8)
ax1.set_xlabel(r"K$\alpha$")
ax1.set_ylabel(r"L$\alpha$ peak shift")
ax1.legend()

for (mineral, clr) in [("hematite","blue"),("magnetite","pink"),("Di","orange"),("ChDi","green"),("Bu","red"),("chromite","black")]:
    my_data = my_dataset[(my_dataset.name == mineral)]
    ax2.scatter(my_data.Kα_intensity,my_data.Lβ_peak_shift,color = clr, label = mineral ,alpha = 0.8)
ax2.set_xlabel(r"K$\alpha$")
ax2.set_ylabel(r"L$\beta$ peak shift")
ax2.legend()

for (mineral, clr) in [("hematite","blue"),("magnetite","pink"),("Di","orange"),("ChDi","green"),("Bu","red"),("chromite","black")]:
    my_data = my_dataset[(my_dataset.name == mineral)]
    ax3.scatter(my_data.Kα_intensity,my_data.Kβ_peak_shift,color = clr, label = mineral ,alpha = 0.8)
ax3.set_xlabel(r"K$\alpha$")
ax3.set_ylabel(r"K$\beta$ peak shift")
ax3.legend()


for (mineral, clr) in [("hematite","blue"),("magnetite","pink"),("Di","orange"),("ChDi","green"),("Bu","red"),("chromite","black")]:
    my_data = my_dataset[(my_dataset.name == mineral)]
    ax4.scatter(my_data.Lβ_peak_shift , my_data.Lα_peak_shift,color = clr, label = mineral ,alpha = 0.8)
ax4.set_xlabel(r"L$\beta$")
ax4.set_ylabel(r"L$\alpha$")
ax4.legend()

for (mineral, clr) in [("hematite","blue"),("magnetite","pink"),("Di","orange"),("ChDi","green"),("Bu","red"),("chromite","black")]:
    my_data = my_dataset[(my_dataset.name == mineral)]
    ax5.scatter(my_data.Lα_peak_shift , my_data.ferrous_total,color = clr, label = mineral ,alpha = 0.8)
ax5.set_xlabel(r"L$\alpha$")
ax5.set_ylabel(r"ferrous_total")
ax5.legend()

for (mineral, clr) in [("hematite","blue"),("magnetite","pink"),("Di","orange"),("ChDi","green"),("Bu","red"),("chromite","black")]:
    my_data = my_dataset[(my_dataset.name == mineral)]
    ax6.scatter(my_data.Lβ_peak_shift , my_data.ferrous_total,color = clr, label = mineral ,alpha = 0.8)
ax6.set_xlabel(r"L$\alpha$")
ax6.set_ylabel(r"ferrous_total")
ax6.legend()
plt.show()

"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

my_dataset = pd.read_excel('ferrous detection of EMPA.xlsx', sheet_name='test3')

fig, ax = plt.subplots()
colors = ["blue","red","yellow","green","pink","brown"]
sns.set_palette(sns.color_palette(colors))
ax = sns.boxplot(x="name", y="Kβ_asymmetric_index", data=my_dataset)




plt.show()


"""


