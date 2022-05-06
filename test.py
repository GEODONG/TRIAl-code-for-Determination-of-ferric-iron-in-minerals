"""
author: ZDH
date: 2022-5-5
purpose: modeling the peaks of the iron
"""
"""
------------------------------------------------------------------------importing libraries--------------------------------------------------------------------------
"""
from ast import Constant
from audioop import findmax
from os import times
from turtle import color
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
from pyparsing import alphanums
from scipy import signal
from scipy.signal import find_peaks
from sympy import CoercionFailed, jacobi_normalized
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.interpolate import UnivariateSpline
"""
-----------------------------------------------------------------------setting up the constant-------------------------------------------------------------------------
"""
#setting ups for basic values
#peak_position
ferrous_position = 1.75
ferric_position = 1.755  
#peak shape: the standard variation δ
sig1 = math.sqrt(0.0001) 
sig2 = math.sqrt(0.0001) 
#the step
scale_factor = 0.000001 

#spectrum range
start_position = 1.7
end_position = 1.8

#movement
motion = 0
motion_variation = 0.001

#function parameters
i=1
cycle_times_for_conceptual_fig = 8
cycle_times_for_modeling = 800
al = 0.9
ratio = 0

"""
--------------------------------------------------------------setting up the conceptual graphics--------------------------------------------------------------------
"""
#create a figure
fig, axes = plt.subplots(1)

#generating the basic x axis
#int(0.1/scale_factor) refers to the number of steps
t = np.linspace(start_position,end_position, int(0.1/scale_factor), endpoint=False)#create data space

for m in range(0,cycle_times_for_conceptual_fig):
    #ferrous peak
    y1 = np.exp(-(t - ferrous_position) ** 2 / (2 * sig1 ** 2)) / (math.sqrt(2 * math.pi) * sig1) 
    #ferric peak
    y2 = ratio*np.exp(-(t - ferric_position) ** 2 / (2 * sig2 ** 2)) / (math.sqrt(2 * math.pi) * sig2)   
    #random noise
    #random1 = np.random.normal(loc=y1, scale=4, size=None)
    #random2 = np.random.normal(loc=y2, scale=4, size=None)
    y3 = y1 + y2 #+random2+random1
    ratio += 0.5
    peaks, _ = find_peaks(y3)
    core_value = peaks*scale_factor + start_position
    plt.plot(core_value, y3[peaks], "x")
    plt.vlines(x = core_value, ymin=0, ymax= y3[peaks] ,color = "C1")
    #adjusting the alpha
    al = al/1.2
    axes.plot(t, y1 , color='c',alpha = 1)
    axes.plot(t, y2 , color='b',alpha = al)
    axes.plot(t, y3, color='r',alpha = 1)
axes.set_xlabel("wave length/nm")
axes.set_ylabel("Intesity")
axes.legend()

"""
----------------------------------------------------------------setting up the modeling graphics---------------------------------------------------------------------
"""
#reset the ratio
ratio = 0

#read the excel
my_dataset = pd.read_excel('ferrous detection of EMPA.xlsx', sheet_name='test3')

#create modeling result
fig = plt.figure(figsize=(25,18))
ax1 = fig.add_subplot(3,1,1)
ax2 = fig.add_subplot(3,1,2)
ax3 = fig.add_subplot(3,1,3)

#calculation to find peak
for m in range(0,cycle_times_for_modeling):
    y1 = np.exp(-(t - ferrous_position) ** 2 / (2 * sig1 ** 2)) / (math.sqrt(2 * math.pi) * sig1)     #ferrous peak
    y2 = ratio*np.exp(-(t - ferric_position) ** 2 / (2 * sig2 ** 2)) / (math.sqrt(2 * math.pi) * sig2)   #ferric peak
    random1 = np.random.normal(loc=y1, scale=4, size=None)#noise 1
    random2 = np.random.normal(loc=y2, scale=4, size=None)#noise 2
    y3 = y1 + y2 #+random2+random1
    #ratio calculation
    ferric_ratio = ratio/(ratio+1)
    ratio += 0.01     
    #motion += motion_variation
    peaks, _ = find_peaks(y3)
    core_value = peaks*scale_factor + start_position
    #find symmetry index
    spline = UnivariateSpline(t, y3-np.max(y3)/2, s=0)
    r1, r2 = spline.roots()
    right_side_width = r2 - core_value
    left_side_width = core_value - r1
    asymetry_index = right_side_width / left_side_width
    #output
    print(i)
    i+=1
    print(core_value)
    #output the feasibility 
    ax1.scatter(y3[peaks], max(y1)+max(y2),color='c')
    x = np.linspace(0,1200,100)
    y=x 
    ax1.plot(x,y,'red',linestyle='--')
    #output modeling asymetry index
    ax2.scatter(asymetry_index,ferric_ratio,color='c',alpha = 1)
    #output modeling peak position
    ax3.scatter(core_value,ferric_ratio,color='c',alpha = 1)

    #plt.axvspan(r1, r2, facecolor='g', alpha=0.1)  
    #plt.hlines(y=properties["width_heights"], xmin=properties["left_ips"], xmax=properties["right_ips"], color = "C1")  
    #axes.axvline(y3[yyyd == 0], color='blue', label='max', linewidth=1,alpha=0.8)
plt.grid(True)
"""
----------------------------------------------------------casting the experiment data--------------------------------------------------------------------------
"""
#cast experiment data of peak asymmetry
for (mineral, clr) in [("hematite","blue"),("magnetite","pink"),("Di","orange"),("ChDi","green"),("Bu","red"),("chromite","black")]:
    my_data = my_dataset[(my_dataset.name == mineral)]
    ax2.scatter(my_data.Lα_asymmetric_index, my_data.ferric_proportion, color = clr, label = mineral, alpha = 0.8)
    ax2.axvspan(xmin=min(my_data.Lα_asymmetric_index),xmax=max(my_data.Lα_asymmetric_index),ymin=0,ymax=max(my_data.ferric_proportion),color=clr,alpha = 0.2)
#cast experiment data of peak position
ax3.vlines(x = ferrous_position, ymin = 0,ymax = 1,colors="b",label="ferrous",alpha=0.8)
ax3.vlines(x = ferric_position, ymin = 0,ymax = 1,colors="r",label="ferric",alpha=0.8)
for (mineral, clr) in [("hematite","blue"),("magnetite","pink"),("Di","orange"),("ChDi","green"),("Bu","red"),("chromite","black")]:
    my_data = my_dataset[(my_dataset.name == mineral)]
    ax3.scatter(my_data.Lα_peak_position, my_data.ferric_proportion, color = clr, label = mineral, alpha = 0.8)
    ax3.axvspan(xmin=min(my_data.Lα_peak_position),xmax=max(my_data.Lα_peak_position),ymin=0,ymax=max(my_data.ferric_proportion),color=clr,alpha = 0.2)

#modification of figures
ax1.set_xlabel("Integrated peak intensity")
ax1.set_ylabel("The sum")
ax1.legend()

ax2.set_xlabel("asymmetry index")
ax2.set_ylabel("ferric ratio")
ax2.legend()

ax3.set_xlabel("peak position")
ax3.set_ylabel("ferric ratio")
ax3.legend()

plt.show()