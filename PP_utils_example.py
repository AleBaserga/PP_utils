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

# load single file
path = r"C:\Users\aless\OneDrive - Politecnico di Milano\PhD_backup\Experiments\NonLinear_PP\Data\AleMatteo Stratus Long\d251009\PM6"
os.chdir(path)

file_vect = "d25100924_average"
loadPath = file_vect + ".dat"


# Let's use load the data as a class
data = utilsPP.load_dat(loadPath, asClass= True, decimal=",", transpose_dataset=True)
print(data)


# Or let's load it as single files
data_s = utilsPP.load_dat(loadPath, asClass= False, decimal=",", transpose_dataset=True)
t = data_s[0]
wl = data_s[1]
map_data = data_s[2]

#%% load a famiy of files
base_dir = r"C:\Users\aless\OneDrive - Politecnico di Milano\PhD_backup\Experiments\NonLinear_PP\Data\AleMatteo Stratus Long\d251009\PM6"
base_file = "d25100924.dat"

# Find all related files
related = utilsPP.find_related_files(base_dir, base_file)
print("Related files:", related)

# Load and stack
t, wl, stacked, files_used = utilsPP.load_and_stack_related_maps(base_dir, base_file, decimal=",", transpose_dataset=True)
print("Stacked shape:", stacked.shape)

# Cut
wl_l = [500, 740]
wl_cut, stacked_cut = utilsPP.cut_spectra_stacked(wl, stacked, wl_l)

# plot all measurements in single graph
fig, ax = utilsPP.plot_dynamics_stack(t, wl_cut, stacked_cut, wl_choice=[630.0], show_mean_std = True)

# find the spikes
spike_mask, detected_indices, wl_idx = utilsPP.detect_spikes_stack_at_wl(stacked_cut, wl_cut, 530.0, window=11, thresh=15.0, min_distance=1)

# plot overlay with spikes marked
fig2, ax2 = utilsPP.plot_spike_mask_overlay(t, wl_cut, stacked_cut, spike_mask, wl_choice=630)

# clean the spikes
cleaned = utilsPP.replace_spikes_stack_with_median_spectrum(stacked_cut, spike_mask)

# show the difference
before_m = utilsPP.mean_stack(stacked_cut)
after_m = utilsPP.mean_stack(cleaned)

fig, ax = plt.subplots(figsize=(8, 3))
wl_p = utilsPP.find_in_vector(wl_cut, 530)
ax.plot(t, before_m[wl_p], label="before spike det")
ax.plot(t, after_m[wl_p], label="after spike det")
ax.set_xlabel("Delay (fs)")
ax.set_ylabel("dTT (%)")
ax.legend()

wl = wl_cut
map_data = after_m


#%% Manipulate data to cut relevant information

# Cut spectra
wl_l = [500, 740]
wl_cut, map_cut = utilsPP.cut_spectra(wl, map_data, wl_l)

data_cut = data.cut_spectra_class(wl_l)
print(data_cut)

# bkg cleaning
t_bkg = -1000
map_cut_2 = utilsPP.remove_bkg(t, map_cut, t_bkg)

data_bkg_free = data_cut.remove_bkg_class(t_bkg)

# smooting
map_cut_2 = utilsPP.smooth_along_axis(map_cut_2, axis=0, method="uniform", window=5)
#map_cut_2 = utilsPP.smooth_2d(map_cut_2, 0, 0)
#map_cut_2, k, s = utilsPP.svd_denoise(map_cut_2)

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
wls_to_plot = [580, 635]
fig, ax = utilsPP.plot_dynamics(t, wl_cut, map_cut_2, wls_to_plot)
fig, ax = data_bkg_free.plot_dynamics_class(wls_to_plot)

# plot lin-log
fig, (ax_lin, ax_log), (c_lin, c_log) = utilsPP.plot_map_linear_log(
    t, wl_cut, map_cut_2,
    t_split=5000,          # split between linear and log region
    cmap_use="PuOr_r",
    clims="auto"
)

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

#%% track maximumx
wl_s = [620, 660]

max_vals, wl_max, idxs, t_out = utilsPP.track_maxima_fulltimeline(
    wl_cut, t, map_cut_2,
    wl_search=wl_s,
    t_start=1000,   # pick seed near t=100
    t_stop=60,   # t_stop earlier than t_start
    maxSteps=6
)

# plot dynamics
fig, ax = plt.subplots(1, 1, figsize=(8,3))
ax.plot(t, max_vals, label = "Max signal", color = "red")
ax.set_xlabel("Delay (fs)")
ax.set_ylabel("dTT (%)")

# assuming wl, t, map_data are defined and track_maxima_fulltimeline is loaded:
fig, ax, line, scat, cbar = utilsPP.plot_tracked_wavelength_vs_time(
    wl_cut, t, map_cut_2,
    wl_search=wl_s,
    t_start=1000,   # pick seed near t=100
    t_stop=500,   # t_stop earlier than t_start
    maxSteps=6,
    show_map=True, cmap='PuOr_r', figsize=(10,6),
    title='Tracked wavelength vs time'
)
plt.show()

#%% see fluence dependence

# define your routine in a function
def find_abs_max_dyn_multiple_files(path_folder, file_name_vector, wl_l, t_to_find_peak, t_bkg):
    
    for i in range(len(file_name_vector)):
        
        base_file = file_name_vector[i]
        
        # Load and stack
        t, wl, stacked, files_used = utilsPP.load_and_stack_related_maps(path_folder, base_file)

        # Cut
        wl_cut, stacked_cut = utilsPP.cut_spectra_stacked(wl, stacked, wl_l)

        # find the spikes
        spike_mask, detected_indices, wl_idx = utilsPP.detect_spikes_stack_at_wl(stacked_cut, wl_cut, 530.0, window=11, thresh=15.0, min_distance=1)
        
        """
        # plot overlay with spikes marked
        fig2, ax2 = utilsPP.plot_spike_mask_overlay(t, wl_cut, stacked_cut, spike_mask, wl_choice=630)
        """
        
        # clean the spikes
        cleaned = utilsPP.replace_spikes_stack_with_median_spectrum(stacked_cut, spike_mask)        
        
        map_cut = utilsPP.mean_stack(cleaned)
        
        #bkg
        map_cut = utilsPP.remove_bkg(t, map_cut, t_bkg)
        
        #Extract Spectrum and maximas
        i_max, values_maximas, i_t_taken = utilsPP.find_abs_max_spectra(t, wl_cut, map_cut, t_to_find_peak)
        
        #Extract dyns
        dyn = map_cut[i_max, :]
        
        if i == 0:
            shape_res = (len(file_name_vector), dyn.shape[1])
            dyn_max_mat = np.zeros(shape_res, dtype=np.float64)
            i_taken_mat = np.zeros((len(file_name_vector), 1), dtype=np.float64)
        
        #append the results
        dyn_max_mat[i, :] = dyn
        i_taken_mat[i, :] = i_max
        
    return dyn_max_mat, i_taken_mat

# define a list of your files and powers
path_folder = r"C:\Users\aless\OneDrive - Politecnico di Milano\PhD_backup\Experiments\NonLinear_PP\Data\AleMatteo Stratus Long\d251009\PM6"
#file_name_vector = ["d25100909", "d25100913", "d25100913", "d25100913","d25100913"]
file_seed = "d251009"
file_nums = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 24]
file_name_vector = utilsPP.generate_string_list(file_seed, file_nums)
powers = [2.35, 1.55, 4.0, 10, 16, 12, 4, 13.657, 8.0, 2.343, 1.072, 14.928, 6.0, 2.0, 0.4]

# sort it from low to high power
file_name_vector, powers = utilsPP.sort_two_lists(file_name_vector, powers)

# call the loading function
t_bkg = -1000
wl_l = [500, 740]
t_to_find_peak = [2000]

dyn_max_mat, i_taken_mat = find_abs_max_dyn_multiple_files(path_folder, file_name_vector, wl_l, t_to_find_peak, t_bkg)

# plot results

fig, ax = plt.subplots(figsize=(8, 3))
colors = utilsPP.create_diverging_colormap(dyn_max_mat.shape[0], 'plasma')

max_vect = []
for i in range(dyn_max_mat.shape[0]):
    dyn = dyn_max_mat[i]
    
    my = max(abs(dyn))
    max_vect.append(my)
    
    y = dyn / my
    
    ax.plot(t, y, label=f"power = {powers[i]:.2f} uW", color=colors[i])

ax.set_xlabel("Delay (fs)")
ax.set_ylabel("dTT (norm)")
ax.legend()


fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(powers, max_vect, "o-", color="red")
ax.set_xlabel("Power (uW)")
ax.set_ylabel("dTT (%)")