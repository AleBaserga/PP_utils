# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 07:50:49 2025

@author: matte
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

plt.close("all")

def unpack_df(dataframe):
    df_np = dataframe.to_numpy()
    t = df_np[1:,0]
    wl = df_np[0, 1:]
    map_data = df_np[1:, 1:]
    return t, wl, map_data

# Usful Functions
def find_in_vector(vect, value):
    index = np.argmin(np.abs(vect-value))
    return index

def find_in_vector_multiple(vect, values):
    
    indexs = []
    if isinstance(values, int):
            index = find_in_vector(vect, values)
            indexs.append(index)
    else:
        for value in values:
            index = find_in_vector(vect, value)
            indexs.append(index)
        
            
    return indexs

def extract_spectr(t, map_matrix, values_to_extract):
    index_extract = find_in_vector_multiple(t, values_to_extract)
    return map_matrix[index_extract, :], index_extract

def extract_dyn(wl_array, map_matrix, values_to_extract):
    index_extract = find_in_vector(wl_array, values_to_extract)
    return map_matrix[:, index_extract], index_extract

def cut_spectra(wl, map_mat, wl_lims):
    wl_lims = np.sort(wl_lims)
    idx_min = find_in_vector(wl, wl_lims[0])
    idx_max = find_in_vector(wl, wl_lims[1])

    # Taglia wl e map_data + plotta cinetica
    wl_cut = wl[idx_min:idx_max+1]
    map_cut = map_mat[:, idx_min:idx_max+1]
    
    return wl_cut, map_cut

def plot_spectra(t, wl, map_mat, ts):
    
    spectra, i_taken = extract_spectr(t, map_mat, ts)
    
    colors = create_diverging_colormap(spectra.shape[0], 'plasma')
    
    fig, ax = plt.subplots(1, 1, figsize=(8,3))
    
    for i in range(spectra.shape[0]):
        spectrum = spectra[i,:]
        t_c = t[i_taken[i]]
        ax.plot(wl, spectrum, label = f' {t_c} ps', color = colors[i])
        
    ax.set_xlabel("Lunghezza d'onda (nm)")
    ax.set_ylabel("Intensità")
    ax.legend()
    #ax.set_title("Spettro al tempo 12.5 s")
    plt.show()
    return fig, ax

def find_abs_max(vect):
    
    abs_vect = np.abs(vect)
    i_max = np.argmax(abs_vect)
    
    return i_max, vect[i_max]

def find_abs_max_spectra(t, wl, map_mat, t_find):
    
    spectra, i_taken = extract_spectr(t, map_mat, t_find)
    index_maximas = []
    values_maximas = []
    
    for i in range(spectra.shape[0]):
        spectrum = spectra[i,:]
        i_m, v_m = find_abs_max(spectrum)
        
        index_maximas.append(i_m)
        values_maximas.append(v_m)
        
    return index_maximas, values_maximas, i_taken

def create_diverging_colormap(n_colors: int, cmap_name: str = 'coolwarm'):
    """
    Create a diverging colormap with specified number of colors.
    
    Parameters
    ----------
    n_colors : int
        Number of colors needed
    cmap_name : str, optional
        Name of matplotlib colormap
    
    Returns
    -------
    List
        List of colors
    """
    n_colors = n_colors +1
    cmap = plt.get_cmap(cmap_name)
    return [cmap(i / (n_colors - 1)) for i in range(n_colors)]

def find_abs_max_multiple_files(file_path_vector, wl_l, t_to_find):
    
    shape_res = (len(file_path_vector), len(t_to_find))
    index_maximas_mat = np.zeros(shape_res, dtype=np.float64)
    values_maximas_mat = np.zeros_like(index_maximas_mat, dtype=np.float64)
    i_taken_mat = np.zeros_like(index_maximas_mat, dtype=np.float64)
    
    for i in range(len(file_path_vector)):
        file_path = file_path_vector[i]
        file_path = file_path + ".dat"
        
        #load file
        try:
            df = pd.read_csv(file_path, sep="\t", header=None, decimal=",")
        except Exception as e:
            print(f"Errore nel file {file_path}: {e}")
            continue
        
        #Unpack Dataframe
        t, wl, map_data = unpack_df(df)
        
        # cut dataset in region
        wl_cut, map_cut = cut_spectra(wl, map_data, wl_l)
        
        #bkg
        map_cut = remove_bkg(t, map_cut, -400)
        
        #Extract Spectrum and maximas
        index_maximas, values_maximas, i_taken = find_abs_max_spectra(t, wl_cut, map_cut, t_to_find)
        
        #append the results
        index_maximas_mat[i, :] = index_maximas
        values_maximas_mat[i, :] = values_maximas
        i_taken_mat[i, :] = i_taken
        
    return index_maximas_mat, values_maximas_mat, i_taken_mat

def remove_bkg(t, map_local, t_bkg):
    
    i_bkg = find_in_vector(t, t_bkg)
    
    map_bkg = map_local[1:i_bkg, :]
    bkg = np.mean(map_bkg, axis= 0)
    
    map_bkg_free = map_local - bkg
    
    return map_bkg_free
    
def find_abs_max_dyn_multiple_files(file_path_vector, wl_l, t_to_find_peak):
    
    file_path = file_path_vector[0]
    file_path = file_path + ".dat"
    
    #load file
    df = pd.read_csv(file_path, sep="\t", header=None, decimal=",")
    
    #Unpack Dataframe
    t, wl, map_data = unpack_df(df)
    
    shape_res = (len(file_path_vector), map_data.shape[0])
    dyn_max_mat = np.zeros(shape_res, dtype=np.float64)
    i_taken_mat = np.zeros((len(file_path_vector), 1), dtype=np.float64)
    
    for i in range(len(file_path_vector)):
        file_path = file_path_vector[i]
        file_path = file_path + ".dat"
        
        #load file
        try:
            df = pd.read_csv(file_path, sep="\t", header=None, decimal=",")
        except Exception as e:
            print(f"Errore nel file {file_path}: {e}")
            continue
        
        #Unpack Dataframe
        t, wl, map_data = unpack_df(df)
        
        # cut dataset in region
        wl_cut, map_cut = cut_spectra(wl, map_data, wl_l)
        
        #bkg
        map_cut = remove_bkg(t, map_cut, -400)
        
        #Extract Spectrum and maximas
        i_max, values_maximas, i_t_taken = find_abs_max_spectra(t, wl_cut, map_cut, t_to_find_peak)
        
        dyn = map_cut[:, i_max]
        #append the results
        dyn_max_mat[i, :] = np.transpose(dyn)
        i_taken_mat[i, :] = i_max
        
    return dyn_max_mat, i_taken_mat
    
#%% Load File

path = r"C:\Users\aless\Downloads\AleMatteo\251007"
path = r"C:\Users\aless\Downloads\AleMatteo"
os.chdir(path)

file_vect = "d25100719_average"
file_vect = "d25100602_average"
loadPath = file_vect + ".dat"

df = pd.read_csv(loadPath, sep="\t", header=None, decimal=",")

#Unpack Dataframe
t, wl, map_data = unpack_df(df)

#%% Choose the correct window

wl_l = [500, 600]
wl_cut, map_cut = cut_spectra(wl, map_data, wl_l)

map_cut_2 = remove_bkg(t, map_cut, -400)

fig, ax = plt.subplots(1, 1)
c = ax.pcolormesh(wl_cut, t, map_cut, shading="auto", cmap = "turbo", vmin = -0.005, vmax = 0.005)

ax.axhline(y = t[24], color="black")
cb = plt.colorbar(c, ax=ax)
cb.set_label("dTT")
ax.set_xlabel("Time")
ax.set_ylabel("Wl")

plot_spectra(t, wl_cut, map_cut_2, [1000, 5000, 10000, 100000])

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