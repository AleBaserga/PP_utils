# -*- coding: utf-8 -*-
"""
This is an module that allows one to analyse, manipulate and plot transient
absortion datasets

@author: Alessandro
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Loading functions
def load_as_df(loadPath, transpose_dataset=False, decimal=".", sep="\t"):
    df = pd.read_csv(loadPath, sep=sep, header=None, decimal=decimal)
    
    if transpose_dataset:
        df = df.transpose()
        
    return df

def unpack_df(dataframe):
    df_np = dataframe.to_numpy()
    wl = df_np[1:,0]
    t = df_np[0, 1:]
    map_data = df_np[1:, 1:]
    return t, wl, map_data

def load_dat(loadPath, asClass= True,transpose_dataset=False, decimal=".", sep="\t"):
    df = load_as_df(loadPath, transpose_dataset = True, decimal=",")

    # Unpack Dataframe
    t, wl, map_data = unpack_df(df)
    
    if asClass:
        return PP_data(t, wl, map_data)
    else:
        return t, wl, map_data

# Miscellaneous Useful Functions
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

def find_abs_max(vect):
    
    abs_vect = np.abs(vect)
    i_max = np.argmax(abs_vect)
    
    return i_max, vect[i_max]


# Data extraction and manipulation

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
    map_cut = map_mat[idx_min:idx_max+1, :]
    
    return wl_cut, map_cut

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

def remove_bkg(t, map_local, t_bkg):
    
    i_bkg = find_in_vector(t, t_bkg)
    
    map_bkg = map_local[:, 1:i_bkg]
    bkg = np.mean(map_bkg, axis= 1)
    bkg = bkg.reshape(-1, 1)
    
    map_bkg_free = map_local - bkg
    
    return map_bkg_free

# Plotting functions 

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

def plot_map(t, wl, map_mat):
    
    fig, ax = plt.subplots(1, 1)
    c = ax.pcolormesh(t, wl, map_mat, shading="auto", cmap = "turbo", vmin = -0.005, vmax = 0.005)
    
    cb = plt.colorbar(c, ax=ax)
    cb.set_label("dTT")
    ax.set_xlabel("Delay (fs)")
    ax.set_ylabel("Wavelength (nm)")
    
    return fig, ax, c

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
    

#%% let's try to use a call of PP data

class PP_data:
    """
    Object to store and manipulate pump–probe data.

    Attributes
    ----------
    t : np.ndarray
        Time vector.
    wl : np.ndarray
        Wavelength vector.
    map : np.ndarray
        2D or 3D matrix of data values (e.g., ΔT/T).
    """

    def __init__(self, t: np.ndarray, wl: np.ndarray, map_data: np.ndarray):
        self.t = np.asarray(t, dtype=float)
        self.wl = np.asarray(wl, dtype=float)
        self.map = np.asarray(map_data, dtype=float)

    def cut_spectra_class(self, wl_lims):
        """
        Cut wavelength and map data between wavelength limits.

        Parameters
        ----------
        wl_lims : list or tuple of 2 floats
            [wl_min, wl_max] defining the spectral range to cut.

        Returns
        -------
        new_data : PP_data
            A new PP_data object containing the cropped wavelength
            and corresponding section of the map.
        """
        
        wl = self.wl
        map_data = self.map
        
        wl_cut, map_cut = cut_spectra(wl, map_data, wl_lims)
        
        cutted_data = PP_data(self.t, wl_cut, map_cut)
        return cutted_data
    
    def remove_bkg_class(self, t_bkg):
        """
        TODO: fix this description
        Cut wavelength and map data between wavelength limits.

        Parameters
        ----------
        wl_lims : list or tuple of 2 floats
            [wl_min, wl_max] defining the spectral range to cut.

        Returns
        -------
        new_data : PP_data
            A new PP_data object containing the cropped wavelength
            and corresponding section of the map.
        """
        
        wl = self.wl
        t = self.t
        map_data = self.map
        
        map_bkg_free = remove_bkg(t, map_data, t_bkg)
        
        data_bkg_free = PP_data(t, wl, map_bkg_free)
        
        return data_bkg_free
    
    def plot_map_class(self):
        """
        TODO: fix this description
        """
        
        wl = self.wl
        t = self.t
        map_data = self.map
        
        fig, ax, c = plot_map(t, wl, map_data)
        
        return fig, ax, c

    def __repr__(self):
        return f"PP_data(t={self.t.shape}, wl={self.wl.shape}, map={self.map.shape})"