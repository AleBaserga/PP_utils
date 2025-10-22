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
from scipy.ndimage import uniform_filter

# Loading functions
def load_as_df(loadPath, transpose_dataset=False, decimal=".", sep="\t"):
    """
    TODO: fix this description
    """
    df = pd.read_csv(loadPath, sep=sep, header=None, decimal=decimal)
    
    if transpose_dataset:
        df = df.transpose()
        
    return df

def unpack_df(dataframe, makePercentage=True):
    """
    TODO: fix this description
    """
    
    df_np = dataframe.to_numpy()
    wl = df_np[1:,0]
    t = df_np[0, 1:]
    map_data = df_np[1:, 1:]
    
    if makePercentage:
        map_data = map_data * 100
        
    return t, wl, map_data

def load_dat(loadPath, asClass= True,transpose_dataset=False, decimal=".", sep="\t",  makePercentage=True):
    """
    TODO: fix this description
    """
    
    df = load_as_df(loadPath, transpose_dataset = True, decimal=",")

    # Unpack Dataframe
    t, wl, map_data = unpack_df(df, makePercentage)
    
    if asClass:
        return PP_data(t, wl, map_data)
    else:
        return t, wl, map_data

# Miscellaneous Useful Functions
def find_in_vector(vect, value):
    """
    TODO: fix this description
    """
    
    index = np.argmin(np.abs(vect-value))
    return index

def find_in_vector_multiple(vect, values):
    """
    TODO: fix this description
    """
    
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
    """
    TODO: fix this description
    """
    abs_vect = np.abs(vect)
    i_max = np.argmax(abs_vect)
    
    return i_max, vect[i_max]

# Denoising 

def smooth_2d(array: np.ndarray, p: int, r: int) -> np.ndarray:
    """
    Smooth a 2D NumPy array along both dimensions using a uniform moving average.

    Parameters
    ----------
    array : np.ndarray
        Input 2D matrix (shape n×m).
    p : int
        Smoothing radius (or window size) along axis 0 (rows).
    r : int
        Smoothing radius (or window size) along axis 1 (columns).

    Returns
    -------
    smoothed : np.ndarray
        Smoothed 2D array of the same shape as input.
    """
    if array.ndim != 2:
        raise ValueError("Input array must be 2D.")

    # Ensure window sizes are odd so the filter is centered
    size = (max(1, 2*p+1), max(1, 2*r+1))
    
    smoothed = uniform_filter(array, size=size, mode="reflect")
    return smoothed


def _choose_rank_from_singulars(s, method="energy", energy=0.99, tol=1e-2):
    """
    Decide rank from singular values s (1D array, descending).
    """
    s = np.asarray(s, dtype=float)
    if s.size == 0:
        return 0

    if method == "energy":
        # cumulative energy fraction
        energy_s = np.cumsum(s**2) / np.sum(s**2)
        k = np.searchsorted(energy_s, energy) + 1  # +1 because searchsorted returns index where energy_s[idx] >= energy
        k = max(1, min(k, s.size))
        return int(k)

    if method == "threshold":
        smax = s[0]
        keep = np.where(s >= tol * smax)[0]
        if keep.size == 0:
            return 1
        return int(keep.size)

    if method == "gap":
        # Find largest relative drop in log singular values
        # Avoid zeros by adding small epsilon
        eps = np.finfo(float).eps
        logs = np.log(s + eps)
        diffs = -np.diff(logs)  # positive if drop
        if diffs.size == 0:
            return 1
        idx = np.argmax(diffs)  # cut after this index
        k = idx + 1
        k = max(1, min(k, s.size))
        return int(k)

    raise ValueError(f"Unknown rank selection method: {method}")


def svd_denoise(matrix, rank=None, method="energy", energy=0.99, tol=1e-2, return_uv=False):
    """
    SVD denoising (rank truncation) of a 2D matrix.

    Parameters
    ----------
    matrix : array-like, shape (n, m)
        Input 2D array (real or complex).
    rank : int or None, optional
        Number of singular values to keep. If None (default), auto-select using `method`.
    method : {"energy", "threshold", "gap"}, optional
        If rank is None, strategy to pick rank:
          - "energy": keep smallest k with cumulative energy >= `energy`.
          - "threshold": keep singular values >= tol * s_max.
          - "gap": choose elbow by largest drop in log singular values.
        Default: "energy".
    energy : float in (0,1), optional
        Energy fraction for "energy" method. Default 0.99.
    tol : float, optional
        Relative threshold for "threshold" method (default 1e-2).
    return_uv : bool, optional
        If True return (denoised, chosen_rank, s, U, Vh), else return (denoised, chosen_rank, s).

    Returns
    -------
    denoised : ndarray, shape (n, m)
        Reconstructed matrix after keeping only `rank` singular values.
    chosen_rank : int
        The integer rank used.
    s : ndarray
        Singular values (descending).
    (optionally) U, Vh : arrays returned when return_uv=True.

    Notes
    -----
    - Uses economy SVD (np.linalg.svd with full_matrices=False).
    - Works with real or complex-valued matrices.
    - Default automatic strategy ('energy' with energy=0.99) is a reasonable default for denoising.
    """
    A = np.asarray(matrix)
    if A.ndim != 2:
        raise ValueError("Input `matrix` must be 2D")

    # economy SVD
    U, s, Vh = np.linalg.svd(A, full_matrices=False)  # U: (n,k), s: (k,), Vh: (k,m)

    # choose rank
    if rank is None:
        chosen_rank = _choose_rank_from_singulars(s, method=method, energy=energy, tol=tol)
    else:
        chosen_rank = int(rank)
        if chosen_rank < 1:
            chosen_rank = 1
        chosen_rank = min(chosen_rank, s.size)

    # reconstruct using truncated SVD
    # build S_k as diagonal of top-k singulars
    k = chosen_rank
    Uk = U[:, :k]
    sk = s[:k]
    Vhk = Vh[:k, :]

    # efficient reconstruction: Uk * diag(sk) * Vhk
    # compute Uk * diag(sk) using broadcasting
    denoised = (Uk * sk[np.newaxis, :]) @ Vhk

    if return_uv:
        return denoised, chosen_rank, s, U, Vh
    return denoised, chosen_rank, s


# Data extraction and manipulation

def extract_spectra(t, map_matrix, values_to_extract):
    """
    TODO: fix this description
    """
    index_extract = find_in_vector_multiple(t, values_to_extract)
    return map_matrix[:, index_extract], index_extract

def extract_dyns(wl_array, map_matrix, values_to_extract):
    """
    TODO: fix this description
    """
    index_extract = find_in_vector_multiple(wl_array, values_to_extract)
    return map_matrix[index_extract, :], index_extract

def cut_spectra(wl, map_mat, wl_lims):
    """
    TODO: fix this description
    """
    wl_lims = np.sort(wl_lims)
    idx_min = find_in_vector(wl, wl_lims[0])
    idx_max = find_in_vector(wl, wl_lims[1])

    # Taglia wl e map_data + plotta cinetica
    wl_cut = wl[idx_min:idx_max+1]
    map_cut = map_mat[idx_min:idx_max+1, :]
    
    return wl_cut, map_cut

def find_abs_max_spectra(t, wl, map_mat, t_find):
    """
    TODO: fix this description
    """
    spectra, i_taken = extract_spectra(t, map_mat, t_find)
    index_maximas = []
    values_maximas = []
    
    for i in range(spectra.shape[1]):
        spectrum = spectra[:,i]
        i_m, v_m = find_abs_max(spectrum)
        
        index_maximas.append(i_m)
        values_maximas.append(v_m)
        
    return index_maximas, values_maximas, i_taken

def remove_bkg(t, map_local, t_bkg):
    """
    TODO: fix this description
    """
    
    i_bkg = find_in_vector(t, t_bkg)
    
    map_bkg = map_local[:, 1:i_bkg]
    bkg = np.mean(map_bkg, axis= 1)
    bkg = bkg.reshape(-1, 1)
    
    map_bkg_free = map_local - bkg
    
    return map_bkg_free

# Plotting functions 

def plot_spectra(t, wl, map_mat, ts):
    """
    TODO: fix this description
    """
    
    spectra, i_taken = extract_spectra(t, map_mat, ts)
    
    colors = create_diverging_colormap(spectra.shape[1], 'plasma')
    
    fig, ax = plt.subplots(1, 1, figsize=(8,3))
    
    for i in range(spectra.shape[1]):
        spectrum = spectra[:, i]
        t_c = t[i_taken[i]]
        ax.plot(wl, spectrum, label = f' {t_c:.2f} ps', color = colors[i])
        
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("dTT (%)")
    
    ax.set_xlim([min(wl), max(wl)])
    ax.legend()
    
    plt.show()
    return fig, ax

def plot_dynamics(t, wl, map_mat, wls):
    """
    TODO: fix this description
    """
    
    dynamics, i_taken = extract_dyns(wl, map_mat, wls)
    
    colors = create_diverging_colormap(dynamics.shape[0], 'plasma')
    
    fig, ax = plt.subplots(1, 1, figsize=(8,3))
    
    for i in range(dynamics.shape[0]):
        dynamic = dynamics[i, :]
        wl_c = wl[i_taken[i]]
        ax.plot(t, dynamic, label = f' {wl_c:.2f} nm', color = colors[i])
        
    ax.set_xlabel("Delay (fs)")
    ax.set_ylabel("dTT (%)")
    
    ax.set_xlim([min(t), max(t)])
    ax.legend()
    
    plt.show()
    return fig, ax

def compute_clims_auto(matrix):
    """
    TODO: fix this description
    """
    vmax = np.nanmax(np.abs(matrix)) * 0.9
    vmin = -vmax
    return vmin, vmax

def plot_map(t, wl, map_mat, cmap_use = "PuOr_r", clims = "auto"):
    """
    TODO: fix this description
    """
    
    # Handle color limits
    if isinstance(clims, str) and clims.lower() == "auto":
        # --- AUTO MODE ---
        # Example: use ±0.6 × max(abs(values))
        vmin_use, vmax_use = compute_clims_auto(map_mat)
    else:
        # --- MANUAL MODE ---
        clims = np.asarray(clims).flatten()
        if clims.size != 2:
            raise ValueError("clims must be 'auto' or a sequence of two numbers [vmin, vmax].")

        vmin_use, vmax_use = np.sort(clims)  # ensure vmin < vmax
    
    fig, ax = plt.subplots(1, 1)
    c = ax.pcolormesh(t, wl, map_mat, shading="auto", cmap = cmap_use, vmin = vmin_use, vmax = vmax_use)
    
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

# deprecated
def find_abs_max_multiple_files(file_path_vector, wl_l, t_to_find):
    """
    TODO: fix this description
    """
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

# deprecated    
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
    
    def extract_dyns_class(self, wls):
        
        wl = self.wl
        t = self.t
        map_data = self.map
        
        return extract_dyns(wl, map_data, wls)
    
    def extract_spectra_class(self, tc):
        
        wl = self.wl
        t = self.t
        map_data = self.map
        
        return extract_spectra(t, map_data, tc)
        
    def smooth_2d_class(self, p: int, r: int):
        wl = self.wl
        t = self.t
        map_data = self.map
        
        smooth_map = smooth_2d(map_data, p, r)
        
        return PP_data(t, wl, smooth_map)
    
        
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
        
        return PP_data(self.t, wl_cut, map_cut)
    
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
        
    def plot_spectra_class(self, TS):
        """
        TODO: fix this description
        """
        
        wl = self.wl
        t = self.t
        map_data = self.map
        
        return plot_spectra(t, wl, map_data, TS)
    
    def plot_dynamics_class(self, wls):
        """
        TODO: fix this description
        """
        
        wl = self.wl
        t = self.t
        map_data = self.map
        
        return plot_dynamics(t, wl, map_data, wls)

    def plot_map_class(self, cmap_use = "PuOr_r", clims = "auto"):
        """
        TODO: fix this description
        """
        
        wl = self.wl
        t = self.t
        map_data = self.map
        
        return plot_map(t, wl, map_data, cmap_use, clims)

        
    def __repr__(self):
        return f"PP_data(t={self.t.shape}, wl={self.wl.shape}, map={self.map.shape})"