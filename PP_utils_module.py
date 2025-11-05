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
import re
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import Sequence, Optional, Tuple, List, Iterable

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

        
def load_dat(loadPath, asClass= True, transpose_dataset=False, decimal=".", sep=r'\s+|\t+',  makePercentage=True):
    """
    TODO: fix this description
    """
    
    df = load_as_df(loadPath, transpose_dataset = transpose_dataset, decimal=decimal)

    # Unpack Dataframe
    t, wl, map_data = unpack_df(df, makePercentage)
    
    if asClass:
        return PP_data(t, wl, map_data)
    else:
        return t, wl, map_data

            
def find_related_files(base_dir: str, base_filename: str) -> list[str]:
    """
    Find all files in `base_dir` related to a given base file name.
    Example:
        base_filename = "d25093003.dat"
        returns files like:
            ["d25093003_1.dat", "d25093003_22.dat", "d25093003_99.dat"]
    """
    # Split base name and extension
    base_name, ext = os.path.splitext(base_filename)
    pattern = re.compile(rf"^{re.escape(base_name)}_(\d+){re.escape(ext)}$", re.IGNORECASE)

    # List and filter
    related = [f for f in os.listdir(base_dir) if pattern.match(f)]
    # Sort numerically by the number after the underscore
    related.sort(key=lambda f: int(pattern.match(f).group(1)))

    return related

        
def load_and_stack_related_maps(base_dir: str, base_filename: str, discard_last = True, transpose_dataset=False, decimal=".", sep="\t",  makePercentage=True, **load_kwargs) -> np.ndarray:
    """
    Load and stack all map_data arrays from files matching base_filename pattern.

    Parameters
    ----------
    base_dir : str
        Directory containing the files.
    base_filename : str
        Reference filename (e.g. "d25093003.dat").
    load_kwargs : dict
        Additional arguments to pass to load_func.

    Returns
    -------
    stacked_maps : np.ndarray
        3D array of stacked map_data along first axis (N, n_wl, n_t).
    file_list : list[str]
        List of files used (full paths).
    """
    # --- find all related files ---
    related_files = find_related_files(base_dir, base_filename)
    if not related_files:
        raise FileNotFoundError(f"No related files found for {base_filename} in {base_dir}")
        
    if discard_last:
        related_files = related_files[0:-1]
    
    maps = []
    file_list = []
    for fname in related_files:
        path = os.path.join(base_dir, fname)
        data = load_dat(path, asClass= False, transpose_dataset = transpose_dataset, decimal = decimal, sep = sep, makePercentage = makePercentage)
        
        if isinstance(data, tuple):
            t, wl, map_data = data
        else:
            # Assume it's a PP_data-like class with attribute `.map`
            map_data = data.map
        maps.append(np.array(map_data, dtype=np.float64))
        file_list.append(path)

    # --- stack along first axis ---
    stacked_maps = np.stack(maps, axis=0)
    return t, wl, stacked_maps, file_list

        
def mean_stack(stacked):
    """
    TODO: fix this description
    """
    
    return np.mean(stacked, axis=0)

        
def generate_string_list(base: str, numbers: list[int]) -> list[str]:
    """
    Generate a list of strings formed by concatenating a base string
    with zero-padded two-digit numbers.

    Example:
    --------
    base = "d251009"
    numbers = [1, 13]
    -> ["d25100901", "d25100913"]
    """
    return [f"{base}{num:02d}.dat" for num in numbers]
        
        
# Miscellaneous Useful Functions
def find_in_vector(vect, value):
    """
    TODO: fix this description
    """
    
    index = np.argmin(np.abs(vect-value))
    return index

        
def _find_nearest_index(vec, value):
    vec = np.asarray(vec)
    return int(np.argmin(np.abs(vec - value)))

        
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

        
def calculate_fluence(powers, f_r, diameter):
    """
    TODO: fix this description
    power -> uW
    f_r -> Hz
    diameter -> um
    """
    diameter = diameter/10000; #change from um to cm
    area = np.pi() * diameter^2 / 4;

    fluences = 2 * powers / (f_r * area)

    return fluences

        
def sort_two_lists(list1, list2):
    """
    TODO: fix this description
    this sorts parallely the two list using ascending order of list2
    """
    list1_sorted, list2_sorted = zip(*sorted(zip(list1, list2), key=lambda x: x[1]))
    list1_sorted = list(list1_sorted)
    list2_sorted = list(list2_sorted)
    
    return list1_sorted, list2_sorted


def shift_x_to_y_half(x, y, which='first', tol=0.0):
    """
    Shift x so that the position where y == 0.5 becomes x == 0.

    Parameters
    ----------
    x : array-like, shape (n,)
        x coordinates (not required to be equally spaced).
    y : array-like, shape (n,)
        y values corresponding to x.
    which : {'first', 'last'}, optional
        If more than one crossing exists, choose the 'first' (left-most) or 'last' (right-most).
        Default 'first'.
    tol : float, optional
        Tolerance used when considering a sample equal to 0.5 (default 0.0). Useful for numerical noise.

    Returns
    -------
    x_shifted : np.ndarray, shape (n,)
        The input x translated so that the x corresponding to y==0.5 is 0.
    x0 : float
        The interpolated x position where y == 0.5 (before shifting).
    used_idx : tuple (i, i+1) or (i, )
        Indices used to compute the crossing. If exact match, returns (i,). If interpolated between samples i and i+1, returns (i, i+1).

    Raises
    ------
    ValueError
        If no crossing / point with y==0.5 is found.

    Notes
    -----
    Uses linear interpolation between the two samples surrounding the crossing.
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be 1D arrays")
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same length")

    n = x.shape[0]
    target = 0.5

    # handle exact matches (within tol)
    exact_mask = np.isfinite(y) & (np.abs(y - target) <= tol)
    exact_idxs = np.nonzero(exact_mask)[0]
    if exact_idxs.size > 0:
        if which == 'first':
            i = int(exact_idxs[0])
        else:
            i = int(exact_idxs[-1])
        x0 = x[i]
        used_idx = (i,)
        return x - x0, float(x0), used_idx

    # build sign of (y - target), ignoring NaNs
    diff = y - target
    finite = np.isfinite(diff)
    sign = np.sign(diff)
    # where sign changes between consecutive finite points -> crossing
    crossings = []
    # find indices i where pair (i,i+1) crosses target
    for i in range(n - 1):
        if not (finite[i] and finite[i+1]):
            continue
        # if signs are opposite or one is zero (handled above) -> crossing in interval
        if sign[i] == 0 or sign[i+1] == 0:
            # already caught exact matches earlier, skip
            continue
        if sign[i] * sign[i+1] < 0:
            crossings.append(i)

    if len(crossings) == 0:
        raise ValueError("No crossing of y==0.5 found in the data.")

    if which == 'first':
        i = int(crossings[0])
    else:
        i = int(crossings[-1])

    # linear interpolation between (x[i], y[i]) and (x[i+1], y[i+1])
    x_i, x_ip1 = x[i], x[i+1]
    y_i, y_ip1 = y[i], y[i+1]

    # Guard against equal y values (shouldn't happen because we checked sign change)
    if y_ip1 == y_i:
        # fallback: pick midpoint x
        x0 = 0.5 * (x_i + x_ip1)
    else:
        t = (target - y_i) / (y_ip1 - y_i)  # fraction between i and i+1
        x0 = x_i + t * (x_ip1 - x_i)

    used_idx = (i, i+1)
    return x - float(x0), float(x0), used_idx


# Denoising 

try:
    from scipy import ndimage as ndi
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

def _make_exponential_kernel(length, decay):
    """
    Create a causal exponential kernel of given length.
    decay controls how fast the weights decay:
        kernel[i] = exp(-i / decay) for i = 0..length-1
    The kernel is normalized to sum=1.
    """
    if length <= 0:
        raise ValueError("length must be > 0")
    pos = np.arange(length, dtype=float)
    k = np.exp(-pos / float(decay))
    k /= k.sum()
    return k

        
def smooth_along_axis(arr, axis=-1, method="uniform", window=5, sigma=None, decay=None, mode="reflect"):
    """
    Smooth a 2D or 3D array along a chosen axis.

    Parameters
    ----------
    arr : ndarray
        Input array of ndim == 2 or 3.
    axis : int, optional
        Axis along which to smooth (default -1).
    method : str, optional
        One of {"uniform", "gaussian", "exponential"}.
          - "uniform": moving average of length `window`.
          - "gaussian": gaussian smoothing with std `sigma` (if sigma is None a rule-of-thumb from window is used).
          - "exponential": exponentially weighted moving average using `decay` or `window`.
    window : int, optional
        Window length for "uniform" or for "exponential" if `decay` is None.
        Must be a positive integer. Default 5.
    sigma : float or None, optional
        Standard deviation for Gaussian smoothing. If None and method=="gaussian",
        sigma will be set to max(0.5, window/3).
    decay : float or None, optional
        Decay constant for exponential kernel. If None and method=="exponential",
        decay defaults to window/3.
    mode : str, optional
        Boundary mode for convolution (same semantics as scipy.ndimage.convolve1d):
        'reflect', 'constant', 'nearest', 'mirror', 'wrap'. Default 'reflect'.

    Returns
    -------
    out : ndarray
        Smoothed array, same shape as `arr` and dtype float64.

    Notes
    -----
    - The function only smooths along one axis; other axes are left unchanged.
    - For large inputs, prefer SciPy because it's vectorized and fast.
    """
    arr = np.asarray(arr)
    if arr.ndim not in (2, 3):
        raise ValueError("arr must be 2D or 3D")

    axis = int(axis)
    axis = axis if axis >= 0 else arr.ndim + axis
    if axis < 0 or axis >= arr.ndim:
        raise IndexError("axis out of range")

    if method not in ("uniform", "gaussian", "exponential"):
        raise ValueError("method must be 'uniform', 'gaussian' or 'exponential'")

    # Validate params
    if method in ("uniform", "exponential"):
        window = int(window)
        if window < 1:
            raise ValueError("window must be >= 1")

    if method == "gaussian":
        if sigma is None:
            sigma = max(0.5, window / 3.0)
        sigma = float(sigma)

    if method == "exponential":
        if decay is None:
            decay = max(0.1, float(window) / 3.0)
        decay = float(decay)

    # Convert to float64 for numerical stability
    out = arr.astype(np.float64, copy=True)

    # Helper: apply 1D convolution along axis using kernel k
    def _convolve1d_numpy(a, k, axis, mode):
        """
        Perform 1D convolution along axis using numpy's fftless approach:
        - pad using reflect/constant etc (we implement reflect and constant; others we map to reflect)
        - use sliding window via stride tricks if small, otherwise fallback to np.apply_along_axis
        This is not as fast as scipy but robust.
        """
        from numpy.lib.stride_tricks import sliding_window_view

        if mode not in ("reflect", "constant"):
            # for simplicity map other modes to reflect
            pad_mode = "reflect"
        else:
            pad_mode = mode

        half = (k.size - 1) // 2  # we will pad symmetric to preserve centered conv for uniform/gauss
        # Build pad widths tuple for np.pad
        pad_width = [(0, 0)] * a.ndim
        pad_width[axis] = (half, half)
        a_pad = np.pad(a, pad_width=pad_width, mode=pad_mode)

        # Use sliding window view for vectorized conv if available
        try:
            sw = sliding_window_view(a_pad, window_shape=k.size, axis=axis)
            # sw shape: same as a plus last axis=k.size
            # multiply and sum over last axis
            # move kernel to an axis with correct broadcast shape
            # we want to multiply along the last axis and sum
            out_conv = np.tensordot(sw, k[::-1], axes=([sw.ndim - 1], [0]))
            # tensordot yields shape of sw without last axis; that equals a.shape
            return out_conv
        except Exception:
            # fallback to apply along axis
            def conv1d_vec(v):
                return np.convolve(v, k, mode="same")
            return np.apply_along_axis(conv1d_vec, axis, a)

    # Choose method implementation
    if method == "uniform":
        # uniform kernel of length window
        k = np.ones(window, dtype=float) / float(window)
        if _HAS_SCIPY:
            out = ndi.convolve1d(out, weights=k, axis=axis, mode=mode, origin=0)
        else:
            out = _convolve1d_numpy(out, k, axis, mode)

    elif method == "gaussian":
        if _HAS_SCIPY:
            # gaussian_filter1d is vectorized for N-D arrays
            out = ndi.gaussian_filter1d(out, sigma=sigma, axis=axis, mode=mode)
        else:
            # approximate gaussian kernel with window length
            length = max(3, int(np.ceil(sigma * 8)))  # 8 sigma rule
            if length % 2 == 0:
                length += 1
            x = np.arange(length) - (length - 1) / 2.0
            k = np.exp(-(x**2) / (2.0 * sigma**2))
            k /= k.sum()
            out = _convolve1d_numpy(out, k, axis, mode)

    else:  # exponential
        # Use causal kernel centered around current point; build symmetric kernel so convolution is centered
        # build length window kernel (we create a causal then center it)
        length = int(window)
        k = _make_exponential_kernel(length, decay)
        # For a centered replacement of causal EWMA we reverse the kernel so highest weight is at center
        # If you prefer true causal EWMA (only past samples), replace with appropriate convolution
        k_centered = np.concatenate((k[length//2 + 1:][::-1], k[:length//2 + 1]))
        k_centered = k_centered / k_centered.sum()
        if _HAS_SCIPY:
            out = ndi.convolve1d(out, weights=k_centered, axis=axis, mode=mode, origin=0)
        else:
            out = _convolve1d_numpy(out, k_centered, axis, mode)

    return out

        
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
        
    if p == 0 and r == 0:
        return array.copy()

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

# Dechirping (not yet tried)
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.signal import correlate

def _try_unpack(dataFrame_scans):
    """
    Accept either a tuple (t, wl, scans) or a custom dataFrame that needs a
    user-provided 'dataCleaner_unpack' function. If the latter is present in the
    global namespace it will be called.
    """
    if isinstance(dataFrame_scans, (tuple, list)) and len(dataFrame_scans) == 3:
        return dataFrame_scans
    # allow user to provide a dataCleaner_unpack in the module where they call this
    if 'dataCleaner_unpack' in globals():
        return globals()['dataCleaner_unpack'](dataFrame_scans)
    raise ValueError("dataFrame_scans must be (t, wl, scans) or provide dataCleaner_unpack()")

def _nearest_int_index(arr, val):
    return int(np.argmin(np.abs(arr - val)))

def _smooth_vector(v, window=5):
    # simple symmetric moving median smoothing for robustness
    from scipy.ndimage import median_filter
    return median_filter(v, size=max(1, int(window)))

def _auto_chirp_derivative(avg_map, t, wl, smooth_win=5, center_ref=None):
    """
    Estimate chirp by locating maximum absolute temporal derivative per wavelength.
    Returns ch (time-shifts for each wl).
    """
    # avg_map shape: (n_wl, n_t)
    n_wl, n_t = avg_map.shape
    dt = np.diff(t)
    if not np.allclose(dt, dt[0]):
        # If non-uniform time axis, compute derivative with np.gradient
        deriv = np.abs(np.gradient(avg_map, t, axis=1))
    else:
        deriv = np.abs(np.diff(avg_map, axis=1)) / dt[0]
        # pad to same shape
        deriv = np.pad(deriv, ((0,0),(0,1)), mode='edge')

    # For each wl find index of max derivative
    idx_max = np.argmax(deriv, axis=1)

    # Convert indices to times
    ch_times = t[idx_max]

    # Optionally choose a reference: central wl or provided
    if center_ref is None:
        # subtract median so that chirp is relative to median time
        ch_times = ch_times - np.median(ch_times)
    else:
        ch_times = ch_times - center_ref

    # Smooth chirp (avoid large local jumps)
    ch_times = _smooth_vector(ch_times, window=max(1, smooth_win))

    # Return time offsets: here we return absolute map shifts (ch_times)
    return ch_times

def _auto_chirp_xcorr(avg_map, t, wl, max_lag_fraction=0.2, upsample=1, reference_wl_index=None, smooth_win=5):
    """
    Estimate chirp by cross-correlation: for each wavelength row correlate with
    a reference row and find lag of maximum correlation. The lag (in samples)
    is converted to time shift and returned.
    """
    n_wl, n_t = avg_map.shape
    dt = t[1] - t[0] if n_t>1 else 1.0
    if reference_wl_index is None:
        reference_wl_index = n_wl // 2

    ref = avg_map[reference_wl_index].astype(float)
    # optionally normalize to zero mean
    ref = (ref - np.nanmean(ref))
    ch = np.zeros(n_wl, dtype=float)

    max_lag = int(np.ceil(max_lag_fraction * n_t))

    for i in range(n_wl):
        row = avg_map[i].astype(float)
        row = (row - np.nanmean(row))
        # compute cross-correlation (full)
        corr = correlate(row, ref, mode='full')
        lags = np.arange(- (len(row) - 1), len(ref))
        # restrict search around zero
        center = np.argmin(np.abs(lags))
        i_min = max(0, center - max_lag)
        i_max = min(len(corr), center + max_lag + 1)
        sub_corr = corr[i_min:i_max]
        sub_lags = lags[i_min:i_max]
        # find peak
        peak_idx = np.argmax(sub_corr)
        lag = sub_lags[peak_idx]
        ch[i] = -lag * dt  # negative lag: row needs to be shifted by -lag*dt to align to ref

    # subtract median so that shifts are relative
    ch = ch - np.median(ch)
    ch = _smooth_vector(ch, window=max(1, smooth_win))
    return ch

def dataCleaner_dechirp_v3_py(
    dataFrame_scans,
    graphic_mode: bool = True,
    iterative_mode: bool = False,
    index_t_min: Optional[int] = None,
    index_t_max: Optional[int] = None,
    correction_factor: Optional[Iterable[float]] = None,
    cmap: str = "viridis",
    auto_mode: str = "derivative",   # 'derivative' or 'xcorr' or 'manual'
    auto_params: Optional[dict] = None,
    spline_smoothing: float = 0.0,   # smoothing factor for UnivariateSpline (0 -> interpolating)
    show_plots: bool = True
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    """
    Python version of dataCleaner_dechirp_v3 with interactive/manual and automatic modes.

    Parameters
    ----------
    dataFrame_scans : (t, wl, scans) or custom object unpackable by dataCleaner_unpack()
        t : 1D array (n_t,)
        wl : 1D array (n_wl,)
        scans : 3D array (n_wl, n_t, n_scans) OR (n_t, n_wl, n_scans) — shapes are auto-detected.
    graphic_mode : bool
        If True, interactive clicking is used (ginput). If False and not manual, the automatic method is used.
    iterative_mode : bool
        If True, repeat interactive selection until user confirms.
    index_t_min / index_t_max : optional ints
        Range in t indices to display when clicking (1-based indices like MATLAB).
    correction_factor : 2-tuple for vmin/vmax scaling like in your MATLAB code.
    cmap : colormap for plotting.
    auto_mode : 'derivative'|'xcorr'|'manual' — which automatic strategy to use if not graphic_mode.
    auto_params : dict of parameters for automatic mode (e.g. {'smooth_win':5, 'max_lag_fraction':0.2})
    spline_smoothing : float smoothing factor for UnivariateSpline (s parameter)
    show_plots : whether to show the various figures (False useful in headless runs).

    Returns
    -------
    dechirped_dataFrame_avg : (t_extended, wl, dechirped_pp_avg)   # (same form as pack)
    chirp_data : ndarray shape (n_wl+1, n_t+1) (first row t, first column wl, body old_pp_avg)
    """

    # Unpack inputs
    t, wl, scans = _try_unpack(dataFrame_scans)

    # Normalize shapes: we expect scans shape (n_wl, n_t, n_scans)
    scans = np.asarray(scans)
    if scans.ndim != 3:
        raise ValueError("scans must be 3D (n_wl, n_t, n_scans) or convertible to that shape")

    # Detect orientation: if shape is (n_t, n_wl, n_scans) swap axes
    n0, n1, n2 = scans.shape
    n_wl, n_t = len(wl), len(t)
    if (n0 == n_t and n1 == n_wl):
        scans = np.transpose(scans, (1, 0, 2))

    # average across scans
    old_pp_avg = np.nanmean(scans, axis=2)   # shape (n_wl, n_t)

    # set defaults
    if index_t_min is None:
        index_t_min = 0
    else:
        index_t_min = max(0, int(index_t_min) - 1)  # convert MATLAB 1-based to 0-based

    if index_t_max is None:
        index_t_max = n_t - 1
    else:
        index_t_max = min(n_t - 1, int(index_t_max) - 1)

    if correction_factor is None:
        correction_factor = (1.0, 1.0)

    if auto_params is None:
        auto_params = {}

    noexit_confirmed = True
    ch = None

    # interactive or automatic chirp selection loop
    while noexit_confirmed:
        if graphic_mode and auto_mode == "manual":
            # present the average map and let user click points
            fig, ax = plt.subplots(figsize=(8, 4))
            im = ax.pcolormesh(t[index_t_min:index_t_max+1],
                               wl,
                               old_pp_avg[:, index_t_min:index_t_max+1],
                               shading='auto', cmap=cmap)
            ax.set_xlabel("Delay")
            ax.set_ylabel("Wavelength")
            plt.colorbar(im, ax=ax)
            ax.set_title("Click points along the chirp (close window when done)")
            if show_plots:
                plt.show(block=False)
            print("Click points along the chirp line (press Enter or close the figure when done)...")
            pts = plt.ginput(n=-1, timeout=0)  # list of (x,y)
            if len(pts) < 2:
                raise RuntimeError("Need at least two points for chirp estimation")
            xs = np.array([p[0] for p in pts])
            ys = np.array([p[1] for p in pts])
            # Fit spline of x(t) as function of wl: x = f(wl)
            # we will create ch(wl_i) = f(wl_i)
            spline = UnivariateSpline(ys, xs, s=spline_smoothing)
            ch = spline(wl)
        elif graphic_mode and auto_mode in ('derivative', 'xcorr'):
            # Show map then run automatic method but allow user to inspect
            fig, ax = plt.subplots(figsize=(8, 4))
            im = ax.pcolormesh(t[index_t_min:index_t_max+1],
                               wl,
                               old_pp_avg[:, index_t_min:index_t_max+1],
                               shading='auto', cmap=cmap)
            plt.colorbar(im, ax=ax)
            ax.set_xlabel("Delay")
            ax.set_ylabel("Wavelength")
            ax.set_title(f"Automatic chirp estimate: {auto_mode}")
            if show_plots:
                plt.show(block=False)

            if auto_mode == "derivative":
                ch = _auto_chirp_derivative(old_pp_avg, t, wl,
                                            smooth_win=auto_params.get('smooth_win', 5),
                                            center_ref=auto_params.get('center_ref', None))
            else:
                ch = _auto_chirp_xcorr(old_pp_avg, t, wl,
                                       max_lag_fraction=auto_params.get('max_lag_fraction', 0.2),
                                       smooth_win=auto_params.get('smooth_win', 5))

            # overlay chirp
            ax.plot(ch, wl, 'k--', lw=1.5)
            if show_plots:
                plt.draw()
        else:
            # Non-graphical automatic selection (or graphic_mode=False)
            if auto_mode == 'manual':
                raise ValueError("graphic_mode must be True for manual mode")
            if auto_mode == 'derivative':
                ch = _auto_chirp_derivative(old_pp_avg, t, wl, smooth_win=auto_params.get('smooth_win', 5))
            else:
                ch = _auto_chirp_xcorr(old_pp_avg, t, wl, max_lag_fraction=auto_params.get('max_lag_fraction', 0.2),
                                       smooth_win=auto_params.get('smooth_win', 5))

        # Show an estimated chirp over the map for inspection
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        im2 = ax2.pcolormesh(t[index_t_min:index_t_max+1],
                             wl,
                             old_pp_avg[:, index_t_min:index_t_max+1],
                             shading='auto', cmap=cmap)
        plt.colorbar(im2, ax=ax2)
        ax2.plot(ch, wl, 'k--', lw=1.5)
        ax2.set_xlabel("Delay")
        ax2.set_ylabel("Wavelength")
        ax2.set_title("Estimated chirp (inspect)")
        if show_plots:
            plt.show(block=False)

        if iterative_mode:
            userkey = input(' Press Enter to confirm or any other key+Enter to repeat selection: ')
            if userkey == '':
                noexit_confirmed = False
            else:
                noexit_confirmed = True
        else:
            noexit_confirmed = False

    # At this point ch is a vector of length n_wl giving time offsets (in same units as t)
    ch = np.asarray(ch, dtype=float).flatten()
    if ch.shape[0] != len(wl):
        # try to interpolate or resample ch to match wl
        ch = np.interp(wl, np.linspace(wl.min(), wl.max(), num=ch.shape[0]), ch)

    # Build extended time vector similar to MATLAB code
    step_t = t[1] - t[0] if len(t) > 1 else 1.0
    needed_points = int(np.ceil((np.nanmax(ch) - np.nanmin(ch)) / step_t)) if np.isfinite(ch).all() else 0
    needed_points = max(0, needed_points)

    # Original shape
    old_dim = old_pp_avg.shape  # (n_wl, n_t)
    new_n_t = n_t + needed_points

    # extended arrays: fill leading columns with NaN
    extendend_old_pp_avg = np.full((old_dim[0], new_n_t), np.nan, dtype=float)
    extendend_old_pp_avg[:, needed_points:needed_points + n_t] = old_pp_avg

    # extended time
    extendend_t = np.zeros(new_n_t, dtype=float)
    # fictitious times before t[0]
    ficticius_t_vect = ( -np.arange(needed_points, 0, -1) ) * step_t + t[0]
    extendend_t[:needed_points] = ficticius_t_vect
    extendend_t[needed_points:] = t

    # Now dechirp: for each wavelength row k, sample extendend_old_pp_avg[k] at times extendend_t + ch[k]
    dechirped_pp_avg = np.full_like(extendend_old_pp_avg, np.nan, dtype=float)
    for k in range(old_dim[0]):
        yrow = extendend_old_pp_avg[k, :]
        # use only finite samples as nodes for interpolation
        finite_mask = np.isfinite(yrow)
        if np.sum(finite_mask) < 2:
            # cannot interpolate, keep NaNs
            continue
        x_valid = extendend_t[finite_mask]
        y_valid = yrow[finite_mask]
        f = interp1d(x_valid, y_valid, kind='linear', bounds_error=False, fill_value=np.nan)
        target_times = extendend_t + ch[k]
        dechirped_pp_avg[k, :] = f(target_times)

    # Plot dechirped average for inspection
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    im3 = ax3.pcolormesh(t[index_t_min:index_t_max+1],
                         wl,
                         dechirped_pp_avg[:, index_t_min:index_t_max+1],
                         shading='auto', cmap=cmap)
    plt.colorbar(im3, ax=ax3)
    ax3.set_title("Corrected chirp (dechirped)")
    ax3.set_xlabel("Delay")
    ax3.set_ylabel("Wavelength")
    if show_plots:
        plt.show()

    # pack outputs
    dechirped_dataFrame_avg = (extendend_t, wl, dechirped_pp_avg)

    # chirp_data matrix in MATLAB like format
    chirp_data = np.zeros((len(wl)+1, len(t)+1), dtype=float)
    chirp_data[0, 1:] = t
    chirp_data[1:, 0] = wl
    chirp_data[1:, 1:] = old_pp_avg

    return dechirped_dataFrame_avg, chirp_data

        
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

        
def cut_spectra_stacked(wl, stacked, wl_lims):
    """
    TODO: fix this description
    """
    wl_cut, map_cut_TEMP = cut_spectra(wl, stacked[0], wl_lims)
    stacked_cut = np.zeros((stacked.shape[0], map_cut_TEMP.shape[0], map_cut_TEMP.shape[1]), dtype=np.float64)
    
    for i in range(stacked.shape[0]):
        _, map_cut = cut_spectra(wl, stacked[i], wl_lims)
        stacked_cut[i] = map_cut
    
    return wl_cut, stacked_cut

        
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

        
def track_maxima_fulltimeline(wl, t, map_data, wl_search, t_start, t_stop, maxSteps=np.inf):
    """
    Track maxima across the entire time vector using a seed at t_start.

    Behaviour:
      - Seed: find the absolute-maximum within wl_search at the time nearest t_start.
      - Propagate forward from t_start to the end (increasing indices).
      - Propagate backward from t_start down to t_stop (decreasing indices).
      - For times before t_stop, values are fixed to the value at t_stop.
      - maxSteps limits how many wavelength indices the chosen maximum may move
        between consecutive time steps. Default np.inf (no limit).

    Parameters
    ----------
    wl : 1D array (m,)
        Wavelength axis.
    t : 1D array (n,)
        Time axis.
    map_data : 2D array (m, n)
        Data matrix (wavelength × time).
    wl_search : sequence of two floats
        [wl_min, wl_max] search window (values in same units as wl).
    t_start : float
        Time value used to pick the seed maximum.
    t_stop : float
        Time value; for t < t_stop returned values are fixed to those at t_stop.
        Must satisfy t_stop <= t_start (they may be equal).
    maxSteps : int or float (optional)
        Max allowed jump in wavelength index between adjacent times. Default np.inf.

    Returns
    -------
    max_dynamics : ndarray (n,)
        Data values at the tracked maxima (signed).
    wl_maximum : ndarray (n,)
        Wavelength values at the tracked maxima.
    max_index : ndarray (n,)
        Integer indices into wl for the tracked maxima.
    t_out : ndarray (n,)
        Same as input t (returned for convenience).
    """
    wl = np.asarray(wl)
    t = np.asarray(t)
    map_data = np.asarray(map_data)

    if map_data.shape != (wl.size, t.size):
        raise ValueError(f"map_data shape {map_data.shape} must be (len(wl), len(t)) = {(wl.size, t.size)}")

    # resolve wl search window to indices (inclusive)
    wl0, wl1 = sorted(wl_search)
    idx_w0 = _find_nearest_index(wl, wl0)
    idx_w1 = _find_nearest_index(wl, wl1)
    wmin_idx, wmax_idx = min(idx_w0, idx_w1), max(idx_w0, idx_w1)

    # time indices
    idx_t_start = _find_nearest_index(t, t_start)
    idx_t_stop = _find_nearest_index(t, t_stop)
    if idx_t_stop > idx_t_start:
        raise ValueError("t_stop must be earlier (smaller) or equal to t_start")

    n_t = t.size
    max_index = np.empty(n_t, dtype=int)
    max_vals = np.empty(n_t, dtype=float)

    # 1) Seed at t_start (raw argmax inside wl window)
    rel = int(np.argmax(np.abs(map_data[wmin_idx:wmax_idx+1, idx_t_start])))
    seed_idx = wmin_idx + rel
    max_index[idx_t_start] = seed_idx
    max_vals[idx_t_start] = map_data[seed_idx, idx_t_start]

    # Helper to choose next index given prev_idx and raw argmax idx (raw within full wl window)
    def _choose_index_with_constraint(prev_idx, raw_idx, allowed_min, allowed_max):
        """Return chosen index respecting allowed range around prev_idx."""
        if np.isfinite(maxSteps):
            allowed_min2 = max(allowed_min, prev_idx - int(maxSteps))
            allowed_max2 = min(allowed_max, prev_idx + int(maxSteps))
            if raw_idx < allowed_min2 or raw_idx > allowed_max2:
                # restrict search to [allowed_min2, allowed_max2]
                sub = np.abs(map_data[allowed_min2:allowed_max2+1, current_t])
                if sub.size == 0:
                    return prev_idx
                rel_sub = int(np.argmax(sub))
                return allowed_min2 + rel_sub
            else:
                return raw_idx
        else:
            return raw_idx

    # 2) Propagate forward from t_start+1 to end
    prev_idx = seed_idx
    for current_t in range(idx_t_start + 1, n_t):
        window = np.abs(map_data[wmin_idx:wmax_idx+1, current_t])
        raw_rel = int(np.argmax(window))
        raw_idx = wmin_idx + raw_rel
        chosen = _choose_index_with_constraint(prev_idx, raw_idx, wmin_idx, wmax_idx)
        max_index[current_t] = chosen
        max_vals[current_t] = map_data[chosen, current_t]
        prev_idx = chosen

    # 3) Propagate backward from t_start-1 down to t_stop
    prev_idx = seed_idx
    for current_t in range(idx_t_start - 1, idx_t_stop - 1, -1):
        window = np.abs(map_data[wmin_idx:wmax_idx+1, current_t])
        raw_rel = int(np.argmax(window))
        raw_idx = wmin_idx + raw_rel
        chosen = _choose_index_with_constraint(prev_idx, raw_idx, wmin_idx, wmax_idx)
        max_index[current_t] = chosen
        max_vals[current_t] = map_data[chosen, current_t]
        prev_idx = chosen

    # 4) For times earlier than t_stop (indices 0 .. idx_t_stop-1) fix to value at idx_t_stop
    if idx_t_stop > 0:
        # ensure idx_t_stop entry is filled (it is, from step 3 if idx_t_stop <= idx_t_start,
        # but if idx_t_stop == idx_t_start it was already set)
        fixed_idx = max_index[idx_t_stop]
        for current_t in range(0, idx_t_stop, +1):
            #fixed_val = max_vals[idx_t_stop]
            max_index[current_t] = fixed_idx
            #max_vals[:idx_t_stop] = fixed_val
            max_vals[current_t] = map_data[fixed_idx, current_t]

    # Return arrays (same time order as t)
    wl_maximum = wl[max_index]
    return max_vals, wl_maximum, max_index, t

# Plotting functions 

def plot_spectra(t, wl, map_mat, ts,
                 ax=None,
                 clear=False,
                 show=True,
                 cmap_name='plasma',
                 linewidth=1.5,
                 alpha=1.0,
                 legend=True,
                 title=None):
    """
    Plot spectra extracted at times `ts` from a map (wl x t).

    Parameters
    ----------
    t : 1D array (n_t,)
        Time/delay axis.
    wl : 1D array (n_wl,)
        Wavelength axis.
    map_mat : 2D array (n_wl, n_t)
        Map containing spectra (rows = wavelength, columns = time).
    ts : scalar or sequence
        Times at which to extract spectra; passed to `extract_spectra`.
    ax : matplotlib.axes.Axes or None
        If provided, plotting will be done on this axis (overlay). If None a new figure/axis is created.
    clear : bool
        If True and `ax` provided, clear the axis before plotting.
    show : bool
        If True and a new figure is created, call plt.show(). If ax was provided and show=True,
        plt.draw() will be called.
    cmap_name : str
        Name of colormap for generating distinct colors.
    linewidth : float
        Line width for plotted spectra.
    alpha : float
        Line transparency.
    legend : bool
        Whether to show legend.
    title : str or None
        Optional title for the axis.

    Returns
    -------
    fig, ax
        Matplotlib figure and axis handles used.
    """

    # ---- get spectra and indices ----
    spectra, i_taken = extract_spectra(t, map_mat, ts)
    # spectra shape expected: (n_wl, n_spectra)
    n_spectra = spectra.shape[1]

    # ---- prepare axis ----
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 3))
        created_fig = True
    else:
        fig = ax.figure

    if clear:
        ax.cla()

    # ---- colors ----
    try:
        colors = create_diverging_colormap(n_spectra, cmap_name)
    except Exception:
        # fallback: use matplotlib colormap
        from matplotlib.cm import get_cmap
        cmap = get_cmap(cmap_name)
        colors = [cmap(i / max(1, n_spectra - 1)) for i in range(n_spectra)]

    # ---- plot each spectrum ----
    for i in range(n_spectra):
        spectrum = spectra[:, i]
        # label using the matching time from t via i_taken
        t_c = float(t[i_taken[i]])
        ax.plot(wl, spectrum,
                label=f'{t_c:.2f} fs',
                color=colors[i],
                linewidth=linewidth,
                alpha=alpha)

    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("ΔT/T (%)")
    ax.set_xlim([np.min(wl), np.max(wl)])
    if title is not None:
        ax.set_title(title)

    if legend:
        ax.legend(fontsize='small')

    # draw / show logic
    if created_fig and show:
        plt.tight_layout()
        plt.show()
    elif (not created_fig) and show:
        # if user provided ax and asked to show, update canvas
        fig.canvas.draw_idle()

    return fig, ax


def plot_dynamics(t, wl, map_mat, wls,
                  ax=None,
                  clear=False,
                  show=True,
                  cmap_name='plasma',
                  linewidth=1.5,
                  alpha=1.0,
                  legend=True,
                  title=None,
                  normalize = False,
                  absoluteValues = False):
    """
    Plot dynamics (signal vs time) extracted at specified wavelengths.

    Parameters
    ----------
    t : 1D array (n_t,)
        Time or delay axis.
    wl : 1D array (n_wl,)
        Wavelength axis.
    map_mat : 2D array (n_wl, n_t)
        Map of signal values (rows = wl, columns = time).
    wls : scalar or sequence
        Wavelength(s) at which to extract dynamics.
    ax : matplotlib.axes.Axes or None
        Axis on which to plot. If None, a new figure and axis are created.
    clear : bool
        Whether to clear the provided axis before plotting.
    show : bool
        Whether to immediately display the figure (plt.show()).
    cmap_name : str
        Colormap name for color generation.
    linewidth : float
        Line width.
    alpha : float
        Line transparency.
    legend : bool
        Whether to show a legend.
    title : str or None
        Optional title for the plot.

    Returns
    -------
    fig, ax
        The Matplotlib figure and axis handles.
    """

    # --- Extract dynamics ---
    dynamics, i_taken = extract_dyns(wl, map_mat, wls)
    n_dyns = dynamics.shape[0]

    # --- Handle axis creation ---
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 3))
        created_fig = True
    else:
        fig = ax.figure
        if clear:
            ax.cla()

    # --- Handle colormap ---
    try:
        colors = create_diverging_colormap(n_dyns, cmap_name)
    except Exception:
        from matplotlib.cm import get_cmap
        cmap = get_cmap(cmap_name)
        colors = [cmap(i / max(1, n_dyns - 1)) for i in range(n_dyns)]

    # --- Plot each dynamic ---
    for i in range(n_dyns):
        dynamic = dynamics[i, :]
        wl_c = wl[i_taken[i]]
        
        if absoluteValues:
            dynamic = np.abs(dynamic)
        if normalize:
            dynamic = dynamic / np.max(dynamic)
        
        ax.plot(t, dynamic,
                label=f'{wl_c:.2f} nm',
                color=colors[i],
                linewidth=linewidth,
                alpha=alpha)

    # --- Labels and formatting ---
    ax.set_xlabel("Delay (fs)")
    
    if normalize:
        ax.set_ylabel("ΔT/T (norm.)")
    else:
        ax.set_ylabel("ΔT/T (%)")
        
    ax.set_xlim([np.min(t), np.max(t)])
    if title:
        ax.set_title(title)
    if legend:
        ax.legend(fontsize='small')

    # --- Show logic ---
    if created_fig and show:
        plt.tight_layout()
        plt.show()
    elif (not created_fig) and show:
        fig.canvas.draw_idle()

    return fig, ax


def compute_clims_auto(matrix):
    """
    TODO: fix this description
    """
    vmax = np.nanmax(np.abs(matrix)) * 0.6
    vmin = -vmax
    return vmin, vmax

        
def plot_map(t, wl, map_mat, cmap_use = "PuOr_r", clims = "auto", show_colorbar=True):
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
    
    ax.set_xlabel("Delay (fs)")
    ax.set_ylabel("Wavelength (nm)")
    
    if show_colorbar:
        colorbar_pad = 0.02
        colorbar_size = 0.1
        # place a colorbar to the right of this axis without overlapping the next subplot
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size=colorbar_size, pad=colorbar_pad)
        cb = fig.colorbar(c, cax=cax)
        cb.set_label("ΔT/T")
    
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

        
def plot_tracked_wavelength_vs_time(
    wl, t, map_data,
    wl_search, t_start, t_stop, maxSteps=np.inf,
    show_map=True, cmap='PuOr_r', vmin=None, vmax=None,
    show_colorbar=True, figsize=(8,6), title=None
):
    """
    Plot the 2D map (wavelength×time) and overlay tracked wavelength vs time.

    Parameters
    ----------
    wl : 1D array (m,)
        Wavelength vector (y axis).
    t : 1D array (n,)
        Time vector (x axis).
    map_data : 2D array (m, n)
        Data matrix (rows = wl, cols = t).
    wl_search, t_start, t_stop, maxSteps :
        See track_maxima_fulltimeline signature.
    show_map : bool
        If True draw the pcolormesh map behind the tracked line.
    cmap : str
        Colormap for the map.
    vmin, vmax : float or None
        Color limits (if None they are auto-chosen from data).
    show_colorbar : bool
        Whether to display colorbar.
    figsize : tuple
        Figure size.
    title : str or None
        Plot title.

    Returns
    -------
    fig, ax, tracked_line, tracked_scatter, cbar
        Matplotlib handles (cbar may be None if show_colorbar=False).
    """

    # ensure arrays
    wl = np.asarray(wl)
    t = np.asarray(t)
    map_data = np.asarray(map_data)

    # call the tracker (must be defined/imported already)
    max_vals, wl_maximum, max_index, t_out = track_maxima_fulltimeline(
        wl, t, map_data, wl_search, t_start, t_stop, maxSteps=maxSteps
    )

    # plotting
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    cbar = None
    if show_map:
        if vmin is None or vmax is None:
            # symmetric around zero if data contains negative/positive:
            absmax = np.nanmax(np.abs(map_data))
            if vmin is None and vmax is None:
                vmin = -absmax
                vmax = absmax
            elif vmin is None:
                vmin = -max(absmax, abs(vmax))
            elif vmax is None:
                vmax = max(absmax, abs(vmin))

        pcm = ax.pcolormesh(t, wl, map_data, shading='auto', cmap=cmap, vmin=vmin, vmax=vmax)
        if show_colorbar:
            cbar = plt.colorbar(pcm, ax=ax)
            cbar.set_label("map value")

    # overlay the tracked wavelength vs time
    # make sure t_out is same ordering as t (track function returns t array same as input)
    tracked_line, = ax.plot(t_out, wl_maximum, '-', color='red', lw=1.6, label='tracked λ(t)')
    tracked_scatter = ax.scatter(t_out, wl_maximum, color='red', s=20)

    # add optional vertical lines for t_start and t_stop
    ax.axvline(x=t_start, color='green', linestyle='--', lw=1.0, label='t_start')
    ax.axvline(x=t_stop, color='blue', linestyle=':', lw=1.0, label='t_stop')

    ax.set_xlabel("Delay (fs)")
    ax.set_ylabel("Wavelength (nm)")
    if title is not None:
        ax.set_title(title)

    # legend (avoid duplicate labels)
    handles, labels = ax.get_legend_handles_labels()
    # remove duplicates in order
    seen = set()
    uniq = []
    for h, lab in zip(handles, labels):
        if lab not in seen:
            uniq.append(h)
            seen.add(lab)
    if uniq:
        ax.legend(uniq, [h.get_label() for h in uniq])

    #plt.tight_layout()
    return fig, ax, tracked_line, tracked_scatter, cbar

            
def plot_map_linear_log(t, wl, map_mat, t_split, cmap_use="PuOr_r", clims="auto"):
    """
    Plot a pump-probe 2D map with a linear x-axis on the left
    and a logarithmic x-axis on the right of a chosen split time.

    Parameters
    ----------
    t : np.ndarray
        1D array of time/delay values (must be strictly increasing, positive on the right).
    wl : np.ndarray
        1D array of wavelength values.
    map_mat : np.ndarray
        2D data array of shape (len(wl), len(t)).
    t_split : float
        Time value at which to split the x-axis between linear and logarithmic scales.
    cmap_use : str
        Colormap for the pcolormesh.
    clims : "auto" or [vmin, vmax]
        Color limits configuration.

    Returns
    -------
    fig, (ax_lin, ax_log), (c_lin, c_log)
        Figure, axes, and pcolormesh handles.
    """

    # Handle color limits
    if isinstance(clims, str) and clims.lower() == "auto":
        vmin_use, vmax_use = compute_clims_auto(map_mat)
    else:
        clims = np.asarray(clims).flatten()
        if clims.size != 2:
            raise ValueError("clims must be 'auto' or a sequence of two numbers [vmin, vmax].")
        vmin_use, vmax_use = np.sort(clims)

    # Split data
    idx_split = np.searchsorted(t, t_split)
    t_lin = t[:idx_split]
    t_log = t[idx_split:]

    map_lin = map_mat[:, :idx_split]
    map_log = map_mat[:, idx_split:]

    # Create two subplots sharing y-axis
    fig, (ax_lin, ax_log) = plt.subplots(
        1, 2, figsize=(10, 5), sharey=True, gridspec_kw={'width_ratios': [2, 2]}
    )

    # Plot linear region
    c_lin = ax_lin.pcolormesh(
        t_lin, wl, map_lin, shading="auto", cmap=cmap_use,
        vmin=vmin_use, vmax=vmax_use
    )
    ax_lin.set_xscale("linear")
    ax_lin.set_xlabel("Delay (fs)")
    ax_lin.set_ylabel("Wavelength (nm)")
    ax_lin.set_title("Linear region")

    # Plot logarithmic region
    c_log = ax_log.pcolormesh(
        t_log, wl, map_log, shading="auto", cmap=cmap_use,
        vmin=vmin_use, vmax=vmax_use
    )
    ax_log.set_xscale("log")
    ax_log.set_xlabel("Delay (fs)")
    ax_log.set_title("Logarithmic region")

    # Add colorbar shared between both
    cbar = fig.colorbar(c_lin, ax=[ax_lin, ax_log], orientation="vertical", fraction=0.02, pad=0.04)
    cbar.set_label("ΔT/T (%)")

    # Visual tweaks
    ax_lin.spines["right"].set_visible(False)
    ax_log.spines["left"].set_visible(False)
    ax_log.yaxis.tick_right()
    ax_log.yaxis.set_label_position("right")

    # Add diagonal "break" lines between the two axes
    d = .015
    kwargs = dict(transform=ax_lin.transAxes, color='k', clip_on=False)
    ax_lin.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    ax_lin.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
    kwargs.update(transform=ax_log.transAxes)
    ax_log.plot((-d, +d), (-d, +d), **kwargs)
    ax_log.plot((-d, +d), (1 - d, 1 + d), **kwargs)

    #plt.tight_layout()
    return fig, (ax_lin, ax_log), (c_lin, c_log)

        
def plot_dynamics_stack(t, wl, stacked, wl_choice, meas_indices=None, figsize=(8,3), cmap_name='plasma', show_mean_std=True):
    """
    Plot dynamics (t vs value) for ALL measurements at a chosen wavelength (or list of wavelengths).

    Parameters
    ----------
    t : 1D array (n_delays,)
        Delay/time axis.
    wl : 1D array (n_wl,)
        Wavelength axis.
    stacked : ndarray (n_meas, n_wl, n_delays)
        Stacked dataset.
    wl_choice : float or sequence of floats
        Wavelength(s) to plot. If a single float, nearest wl is used. If multiple, plots one panel per chosen wl.
    meas_indices : sequence or None
        If provided, plot only this subset of measurement indices (otherwise plot all).
    figsize : tuple
        Figure size.
    cmap_name : str
        Colormap name to color lines across measurements.
    show_mean_std : bool, default True
        If True, overlay the mean dynamic ± std deviation across measurements.

    Returns
    -------
    fig, ax or axs
    """
    stacked = np.asarray(stacked)
    if stacked.ndim != 3:
        raise ValueError("stacked must be 3D (n_meas, n_wl, n_delays)")

    n_meas, n_wl, n_delays = stacked.shape
    t = np.asarray(t)
    wl = np.asarray(wl)

    # choose measurements
    if meas_indices is None:
        meas_indices = np.arange(n_meas)
    else:
        meas_indices = np.asarray(meas_indices, dtype=int)

    # handle multiple wl choices
    if np.isscalar(wl_choice):
        wl_choice = [wl_choice]

    n_plots = len(wl_choice)
    fig, axs = plt.subplots(1, n_plots, figsize=(figsize[0]*n_plots, figsize[1]), squeeze=False)
    axs = axs.ravel()

    # colormap
    colors = create_diverging_colormap(n_meas, cmap_name)

    for ip, wl_val in enumerate(wl_choice):
        ax = axs[ip]
        idx = find_in_vector(wl, wl_val)
        dynamics = stacked[:, idx, :]  # shape (n_meas, n_delays)
        for k, mi in enumerate(meas_indices):
            ax.plot(t, dynamics[mi], color=colors[k], label=f"meas {mi}" if n_plots==1 else f"{wl[idx]:.2f} nm - meas {mi}")
        
        if show_mean_std:
            mean_dyn = np.mean(dynamics[meas_indices], axis=0)
            std_dyn = np.std(dynamics[meas_indices], axis=0)
            ax.plot(t, mean_dyn, color='red', lw=2.5, label='Mean')
            ax.fill_between(t, mean_dyn - std_dyn, mean_dyn + std_dyn,
                            color='gray', alpha=0.3, label='±1σ')
            
        ax.set_xlabel("Delay (fs)")
        ax.set_ylabel("ΔT/T (%)")
        ax.set_title(f"Wavelength {wl[idx]:.2f} nm")
        ax.set_xlim([t.min(), t.max()])
        if n_plots == 1:
            ax.legend(loc='upper right', fontsize='small')
    plt.tight_layout()
    return fig, axs if n_plots>1 else (fig, axs[0])        
        
# Spike detection

try:
    from numpy.lib.stride_tricks import sliding_window_view
    _has_swv = True
except Exception:
    _has_swv = False

def _rolling_median_1D(x, window):
    """
    Centered rolling median with odd window length.
    Returns an array same length as x (edges padded by reflecting).
    """
    if window <= 1:
        return x.copy()

    if window % 2 == 0:
        raise ValueError("window must be odd")

    half = window // 2
    # reflect-pad the array so center-window works at edges
    xp = np.pad(x, pad_width=half, mode='reflect')

    if _has_swv:
        # fast vectorized sliding windows
        views = sliding_window_view(xp, window_shape=window)
        med = np.median(views, axis=1)
        return med
    else:
        # safe fallback (still O(n*window) but fine for moderate sizes)
        n = len(x)
        med = np.empty(n, dtype=float)
        for i in range(n):
            seg = xp[i : i + window]
            med[i] = np.median(seg)
        return med

            
def detect_spikes(signal, window=11, thresh=6.0, min_distance=1):
    """
    Detect spike-like outliers in a 1D signal.

    Method:
      - compute centered rolling median (window must be odd)
      - residual = signal - median
      - robust scale = MAD = median(|residual|)
      - mark points where |residual| > thresh * MAD
      - optionally enforce minimum distance between detected spikes (min_distance samples)

    Parameters
    ----------
    signal : 1D array-like
        Input noisy signal.
    window : int, odd (default 11)
        Window length for centered median filter (should be odd, >1).
    thresh : float (default 6.0)
        Threshold multiplier on MAD to call a point a spike.
    min_distance : int (default 1)
        Minimum number of samples between reported spikes (keeps the largest residual in a cluster).

    Returns
    -------
    spike_idx : ndarray of ints
        Indices of detected spike points in the input signal.
    """
    s = np.asarray(signal, dtype=float)
    if s.ndim != 1:
        raise ValueError("signal must be 1D")

    if window < 3:
        # fallback: use global median
        med = np.median(s) * np.ones_like(s)
    else:
        med = _rolling_median_1D(s, window)

    resid = s - med
    mad = np.median(np.abs(resid))
    # avoid mad == 0
    if mad <= 0 or not np.isfinite(mad):
        mad = np.std(resid) if np.std(resid) > 0 else 1e-12

    mask = np.abs(resid) > (thresh * mad)

    # enforce min_distance: keep local maxima of |resid| within runs
    if min_distance > 1:
        # find indices where mask True and cluster them
        idx_true = np.nonzero(mask)[0]
        if idx_true.size == 0:
            return np.array([], dtype=int)
        clusters = []
        cur = [idx_true[0]]
        for k in idx_true[1:]:
            if k - cur[-1] <= min_distance:
                cur.append(k)
            else:
                clusters.append(cur)
                cur = [k]
        clusters.append(cur)
        chosen = [int(cluster[np.argmax(np.abs(resid[cluster]))]) for cluster in clusters]
        return np.array(chosen, dtype=int)
    else:
        return np.nonzero(mask)[0].astype(int)

            
def replace_spikes_with_interp(signal, spike_idx, extend=0):
    """
    Replace spike samples with linear interpolation from nearest good neighbors.

    Parameters
    ----------
    signal : 1D array-like
        Original signal.
    spike_idx : array-like of ints
        Indices of spike samples to replace.
    extend : int (default 0)
        Optionally expand each spike index to a symmetric window of +/- extend samples
        (useful if spikes are multi-sample).

    Returns
    -------
    cleaned : ndarray (float)
        New signal with spikes replaced by linear interpolation.
    """
    s = np.asarray(signal, dtype=float).copy()
    n = s.size
    if n == 0:
        return s

    if len(spike_idx) == 0:
        return s

    # build boolean mask of bad samples
    bad = np.zeros(n, dtype=bool)
    for idx in np.atleast_1d(spike_idx):
        if idx < 0 or idx >= n:
            continue
        lo = max(0, idx - extend)
        hi = min(n-1, idx + extend)
        bad[lo:hi+1] = True

    good_idx = np.nonzero(~bad)[0]
    bad_idx = np.nonzero(bad)[0]

    if good_idx.size == 0:
        # nothing to interpolate from — return constant or original
        return s

    # linear interpolation: for bad indices, use np.interp over good points
    # np.interp will extrapolate at ends using first/last good values.
    x = np.arange(n)
    s_clean = s.copy()
    s_clean[bad_idx] = np.interp(bad_idx, good_idx, s[good_idx])

    return s_clean

        
def plot_spikes(signal, spike_idx, cleaned=None, ax=None, marker_kw=None, line_kw=None):
    """
    Plot original signal and mark detected spikes. Optionally overlay cleaned signal.

    Parameters
    ----------
    signal : 1D array-like
        Original signal.
    spike_idx : array-like of ints
        Indices of detected spikes.
    cleaned : 1D array-like or None
        Cleaned signal to overlay.
    ax : matplotlib axis or None
        Axis to plot on; if None a new figure is created.
    marker_kw : dict or None
        kwargs for spike markers (default red 'x', s=60).
    line_kw : dict or None
        kwargs for cleaned signal plot (default blue line).
    """
    s = np.asarray(signal, dtype=float)
    n = s.size
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 3))
    else:
        fig = ax.figure

    if marker_kw is None:
        marker_kw = dict(color='red', marker='x', linestyle='None', markersize=8, label='spikes')
    if line_kw is None:
        line_kw = dict(color='blue', linewidth=1.5, label='cleaned')

    x = np.arange(n)
    ax.plot(x, s, color='gray', alpha=0.7, label='original')
    if len(spike_idx) > 0:
        ax.plot(spike_idx, s[spike_idx], **marker_kw)

    if cleaned is not None:
        ax.plot(x, cleaned, **line_kw)
    ax.set_xlabel('sample')
    ax.set_ylabel('value')
    ax.legend()
    return ax

        
def detect_spikes_stack_at_wl(stacked, wl, wl_choice, window=11, thresh=6.0, min_distance=1):
    """
    Detect spikes in a stacked dataset (n_meas, n_wl, n_delays) by applying the 1D
    detect_spikes function to each measurement's time trace at the wavelength nearest to wl_choice.

    Parameters
    ----------
    stacked : ndarray, shape (n_meas, n_wl, n_delays)
    wl : 1D array of length n_wl (wavelengths)
    wl_choice : float (wavelength to use as detector) or integer index
    window, thresh, min_distance : passed to detect_spikes

    Returns
    -------
    spike_mask : ndarray(bool), shape (n_meas, n_delays)
        True where that measurement and delay is flagged as a spike.
    detected_indices : list of ndarray
        For each measurement, the array of spike indices (may be empty).
    wl_index : int
        Wavelength index used for detection.
    """
    stacked = np.asarray(stacked)
    if stacked.ndim != 3:
        raise ValueError("stacked must be 3D (n_meas, n_wl, n_delays)")
    n_meas, n_wl, n_delays = stacked.shape
    wl = np.asarray(wl)

    wl_idx = int(np.argmin(np.abs(wl - float(wl_choice))))

    spike_mask = np.zeros((n_meas, n_delays), dtype=bool)
    detected_indices = [None] * n_meas

    # apply detect_spikes to each measurement's trace at wl_idx
    for mi in range(n_meas):
        sig = stacked[mi, wl_idx, :]
        idxs = detect_spikes(sig, window=window, thresh=thresh, min_distance=min_distance)
        detected_indices[mi] = idxs
        if idxs.size > 0:
            spike_mask[mi, idxs] = True

    return spike_mask, detected_indices, wl_idx
    
        
def detect_spikes_stack(stacked, wl_thresh=0.5, point_thresh=0.1, mode='absolute', min_count_same_time=1):
    """
    Detect spikes in stacked dataset using the rule:
      - compute median_map = median over measurements -> shape (n_wl, n_delays)
      - compute difference per measurement: D = |stacked - median_map|
      - (mode=='relative'): normalize D by max(|median_map|, eps)
      - for each (measurement, delay) count how many wavelengths have D > point_thresh
      - if count >= threshold_count -> (measurement, delay) is a spike

    Parameters
    ----------
    stacked : ndarray (n_meas, n_wl, n_delays)
    wl_thresh : float
        If 0 < wl_thresh < 1, treated as fraction of wavelengths (e.g. 0.2 means 20%).
        If wl_thresh >= 1, treated as integer number of wavelengths.
    point_thresh : float
        Threshold for per-wavelength comparison. In 'relative' mode this is a fraction
        (e.g. 0.1 = 10% difference). In 'absolute' mode it's in the same units as data.
    mode : {'relative', 'absolute'}
        Comparison mode.
    min_count_same_time : int
        (Optional) require that at least this many measurements share spike at same delay
        before flagging? (not used now; placeholder)

    Returns
    -------
    spike_mask : ndarray(bool) shape (n_meas, n_delays)
        True where that measurement/time is considered spike.
    info : dict
        Additional diagnostic arrays: 'median_map', 'diff' (abs or rel), 'counts' (n_wl exceeding point_thresh)
    """
    stacked = np.asarray(stacked)
    if stacked.ndim != 3:
        raise ValueError("stacked must be shape (n_meas, n_wl, n_delays)")

    n_meas, n_wl, n_delays = stacked.shape

    # median across measurements
    median_map = np.median(stacked, axis=0)   # shape (n_wl, n_delays)

    # compute absolute diff or relative diff
    if mode == 'relative':
        eps = 1e-12
        denom = np.maximum(np.abs(median_map), eps)
        diff = np.abs(stacked - median_map[None, ...]) / denom[None, ...]
    elif mode == 'absolute':
        diff = np.abs(stacked - median_map[None, ...])
    else:
        raise ValueError("mode must be 'relative' or 'absolute'")

    # boolean per-wavelength exceed
    exceed = diff > point_thresh   # shape (n_meas, n_wl, n_delays)

    # count across wavelengths for each (meas, delay)
    counts = np.sum(exceed, axis=1)  # shape (n_meas, n_delays)

    # interpret wl_thresh: fraction -> integer count
    if 0 < wl_thresh < 1:
        threshold_count = int(np.ceil(wl_thresh * n_wl))
    else:
        threshold_count = int(wl_thresh)

    spike_mask = counts >= threshold_count  # shape (n_meas, n_delays)

    info = {
        'median_map': median_map,
        'diff': diff,
        'exceed': exceed,
        'counts': counts,
        'threshold_count': threshold_count
    }
    return spike_mask, info

        
def replace_spikes_stack_with_median_spectrum(stacked, spike_mask):
    """
    Replace full spectra at flagged (measurement, delay) positions by the median spectrum
    computed across measurements at that delay.

    Parameters
    ----------
    stacked : ndarray (n_meas, n_wl, n_delays)
    spike_mask : ndarray(bool) shape (n_meas, n_delays)

    Returns
    -------
    cleaned : ndarray same shape as stacked
    """
    stacked = np.asarray(stacked).copy()
    if stacked.ndim != 3:
        raise ValueError("stacked must be 3D")
    n_meas, n_wl, n_delays = stacked.shape
    spike_mask = np.asarray(spike_mask, dtype=bool)
    if spike_mask.shape != (n_meas, n_delays):
        raise ValueError("spike_mask must have shape (n_meas, n_delays)")

    # median spectrum across measurements at each delay: shape (n_wl, n_delays)
    median_map = np.median(stacked, axis=0)

    # indices to replace
    meas_idx, delay_idx = np.nonzero(spike_mask)
    if meas_idx.size > 0:
        # vectorized assignment: for each pair (mi, d) assign median_map[:, d]
        for mi, d in zip(meas_idx, delay_idx):
            stacked[mi, :, d] = median_map[:, d]

    return stacked

        
def plot_spike_mask_overlay(t, wl, stacked, spike_mask, wl_choice=None, meas_indices=None,
                            figsize=(8,4), alpha=0.5, cmap_name='viridis'):
    """
    Plot dynamics of all measurements (overlapped) at chosen wl (or averaged across wl if wl_choice is None),
    and overlay markers where spike_mask is True.

    Parameters
    ----------
    t : 1D delays
    stacked : (n_meas, n_wl, n_delays)
    spike_mask : (n_meas, n_delays) boolean
    wl_choice : float or None
       If float -> nearest wavelength plotted; if None -> average across wavelengths.
    meas_indices : sequence or None -> which measurements to plot.
    """
    stacked = np.asarray(stacked)
    n_meas, n_wl, n_delays = stacked.shape
    t = np.asarray(t)

    if meas_indices is None:
        meas_indices = np.arange(n_meas)
    else:
        meas_indices = np.asarray(meas_indices, dtype=int)

    if wl_choice is None:
        series = stacked.mean(axis=1)  # (n_meas, n_delays)
    else:
        idx = find_in_vector(np.asarray(wl), wl_choice)
        series = stacked[:, idx, :]

    fig, ax = plt.subplots(figsize=figsize)

    colors = create_diverging_colormap(n_meas, cmap_name)

    for k, mi in enumerate(meas_indices):
        ax.plot(t, series[mi], color=colors[k], alpha=alpha)
        spikes_t = t[spike_mask[mi]]
        spikes_y = series[mi, spike_mask[mi]]
        if spikes_t.size>0:
            ax.scatter(spikes_t, spikes_y, color='red', marker='x', s=30)

    ax.set_xlabel("Delay")
    ax.set_ylabel("Signal")
    ax.set_title("Overlaid dynamics with spikes marked (red x)")
    plt.tight_layout()
    return fig, ax

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