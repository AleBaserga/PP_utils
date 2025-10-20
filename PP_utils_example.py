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

fig, ax, c = utilsPP.plot_map(t, wl_cut, map_cut_2)
ax.axvline(x = t[utilsPP.find_in_vector(t, t_bkg)], color="black")

fig, ax, c = data_bkg_free.plot_map_class()
ax.axvline(x = t[utilsPP.find_in_vector(t, t_bkg)], color="red")


#utilsPP.plot_spectra(t, wl_cut, map_cut_2, [1000, 5000, 10000, 100000])

#%% Extract Spectrum and maximas
ts = [1, 10, 20]
ts = np.array(ts)
ts = ts * 1000

index_maximas, values_maximas, i_taken = find_abs_max_spectra(t,wl_cut, map_cut, ts)

fig, ax = plot_spectra(t, wl_cut, map_cut, ts)

colors = create_diverging_colormap(len(index_maximas), 'plasma')

for i in range(len(index_maximas)):
    i_m = index_maximas[i]
    v_m = values_maximas[i]
    i_n = i_taken[i]
    t_n = t[i_n]
    ax.axvline(x=wl_cut[i_m], color = colors[i], linestyle='--', linewidth=2, label=f"V_max {t_n} ps")
    ax.axhline(y=v_m, color = colors[i], linestyle='-', linewidth=2, label=f"V_max {t_n} ps")
    
    ax.legend([f"V_max {t[i_taken[i]]} ps" for i in range(len(index_maximas))])


#%% dataset
path = r"C:\Users\aless\Downloads\AleMatteo\251007"
path = r"C:\Users\aless\Downloads\AleMatteo"

os.chdir(path)

file_vect = ["d25100713_average", "d25100714_average", "d25100715_average", "d25100716_average", "d25100717_average", "d25100718_average",
            "d25100719_average", "d25100720_average", "d25100721_average", "d25100722_average", "d25100723_average", "d25100724_average", 
            "d25100725_average", "d25100726_average", "d25100727_average", "d25100728_average", "d25100729_average", "d25100730_average"]

power_vector = [4, 2, 1, 0.5, 0.25, 0.13, 0.08, 0.76, 1.5, 3.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 20.0, 30.0 ]
"""
file_vect = ["C:/Users/matte/OneDrive/Desktop/BRIXNER/AleMatteo/251007/d25100713_average.dat", 
             "C:/Users/matte/OneDrive/Desktop/BRIXNER/AleMatteo/251007/d25100714_average.dat", 
             "C:/Users/matte/OneDrive/Desktop/BRIXNER/AleMatteo/251007/d25100715_average.dat", 
             "C:/Users/matte/OneDrive/Desktop/BRIXNER/AleMatteo/251007/d25100716_average.dat", 
             "C:/Users/matte/OneDrive/Desktop/BRIXNER/AleMatteo/251007/d25100717_average.dat",
             "C:/Users/matte/OneDrive/Desktop/BRIXNER/AleMatteo/251007/d25100718_average.dat",
             "C:/Users/matte/OneDrive/Desktop/BRIXNER/AleMatteo/251007/d25100719_average.dat",
             "C:/Users/matte/OneDrive/Desktop/BRIXNER/AleMatteo/251007/d25100720_average.dat", 
             "C:/Users/matte/OneDrive/Desktop/BRIXNER/AleMatteo/251007/d25100721_average.dat", 
             "C:/Users/matte/OneDrive/Desktop/BRIXNER/AleMatteo/251007/d25100722_average.dat", 
             "C:/Users/matte/OneDrive/Desktop/BRIXNER/AleMatteo/251007/d25100723_average.dat"]
"""

#file_vect = np.array[file_vect]
file_vect = ["d25100602_average", "d25100603_average", "d25100604_average", "d25100605_average", "d25100606_average", "d25100607_average", "d25100608_average"]
power_vector = [2, 4, 8, 16.5, 1, 0.5, 0.2]

A = power_vector
B = file_vect
# Sort both lists based on A
A_sorted, B_sorted = zip(*sorted(zip(A, B)))

# Convert back to lists if needed
power_vector = list(A_sorted)
file_vect = list(B_sorted)

wl_lim = [620, 730]
wl_lim = [500, 600]
t_pick = [1000, 5000, 10000, 500000, 100000, 500000, 900000]
index_maximas_mat, values_maximas_mat, i_taken_mat = find_abs_max_multiple_files(file_vect, wl_lim, t_pick)

for i in range(len(t_pick)):
    fig, ax = plt.subplots(1, 1)
    ax.plot(power_vector, values_maximas_mat[:, i], "o")
    ax.set_title(f"{t_pick[i]} fs delay")
    ax.set_xlabel("Power (uW)")
    ax.set_ylabel("Max signal (a.u.)")

t_to_find_peak = 5000
dyn_max_mat, i_taken_mat = find_abs_max_dyn_multiple_files(file_vect, wl_l, t_to_find_peak)

colors = create_diverging_colormap(len(file_vect), 'plasma')


fig, ax = plt.subplots(1, 1)
for i in range(len(file_vect)):
    ax.plot(t, dyn_max_mat[i, :], color = colors[i], label = f"{power_vector[i]} uW")
ax.legend()
ax.set_title("Signal")
ax.set_xlabel("Delay (fs)")
ax.set_ylabel("Signal (a.u.)")

i_norm = find_in_vector(t, 5000)
fig, ax = plt.subplots(1, 1)
for i in range(len(file_vect)):
    ax.plot(t, dyn_max_mat[i, :] / dyn_max_mat[i, i_norm], color = colors[i], label = f"{power_vector[i]} uW")
ax.legend()
ax.set_title("Signal")
ax.set_xlabel("Delay (fs)")
ax.set_ylabel("Signal (a.u.)")