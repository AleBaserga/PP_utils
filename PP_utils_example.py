# -*- coding: utf-8 -*-
"""
This is an example script of usage of PP_utils_module

@author: Alessandro
"""

# Used libraries
import numpy as np
import matplotlib.pyplot as plt
import os

# Our library
import PP_utils_module as utilsPP

# Close all previous plots
plt.close("all")
    
#%% Load File

path = r"C:\Users\aless\Downloads\AleMatteo\251007"
path = r"C:\Users\aless\Downloads\AleMatteo"
os.chdir(path)

file_vect = "d25100719_average"
file_vect = "d25100602_average"
loadPath = file_vect + ".dat"

# Let's use load the data as a class
data = utilsPP.load_dat(loadPath, asClass= True)
print(data)

# Or let's load it as single files
data_s = utilsPP.load_dat(loadPath, asClass= False)
t = data_s[0]
wl = data_s[1]
map_data = data_s[2]

#%% Manipulate data to cut relevant information

# Cut spectra
wl_l = [500, 600]
wl_cut, map_cut = utilsPP.cut_spectra(wl, map_data, wl_l)

data_cut = data.cut_spectra_class(wl_l)
print(data_cut)

# bkg cleaning
t_bkg = -400
map_cut_2 = utilsPP.remove_bkg(t, map_cut, t_bkg)

data_bkg_free = data_cut.remove_bkg_class(t_bkg)

# smooting
map_cut_2 = utilsPP.smooth_2d(map_cut_2, 3, 2)

# plot maps
fig, ax, c = utilsPP.plot_map(t, wl_cut, map_cut_2)
ax.axvline(x = t[utilsPP.find_in_vector(t, t_bkg)], color="black")

fig, ax, c = data_bkg_free.plot_map_class()
ax.axvline(x = t[utilsPP.find_in_vector(t, t_bkg)], color="red")

# plot spectras
delays_to_plot = [1000, 5000, 10000, 100000]
fig, ax = utilsPP.plot_spectra(t, wl_cut, map_cut_2, delays_to_plot)
fig, ax = data_bkg_free.plot_spectra_class(delays_to_plot)

# plot dynamics
wls_to_plot = [510, 530, 580]
fig, ax = utilsPP.plot_dynamics(t, wl_cut, map_cut_2, wls_to_plot)
fig, ax = data_bkg_free.plot_dynamics_class(wls_to_plot)


#%% Extract Spectrum and maximas
ts = [1.5, 10, 20]
ts = np.array(ts)
ts = ts * 1000

index_maximas, values_maximas, i_taken = utilsPP.find_abs_max_spectra(t,wl_cut, map_cut, ts)

fig, ax = utilsPP.plot_spectra(t, wl_cut, map_cut, ts)

colors = utilsPP.create_diverging_colormap(len(index_maximas), 'plasma')

for i in range(len(index_maximas)):
    i_m = index_maximas[i]
    v_m = values_maximas[i]
    i_n = i_taken[i]
    t_n = t[i_n]
    ax.axvline(x=wl_cut[i_m], color = colors[i], linestyle='--', linewidth=2, label=f"V_max {t_n} ps")
    ax.axhline(y=v_m, color = colors[i], linestyle='-', linewidth=2, label=f"V_max {t_n} ps")
    
    ax.legend([f"V_max {t[i_taken[i]]} ps" for i in range(len(index_maximas))])
