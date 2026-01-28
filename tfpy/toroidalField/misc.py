#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# misc.py


import numpy as np
from numba import jit 
from functools import lru_cache
from ..config import tfParams


def resize_center_pad_zeros(arr: np.ndarray, p: int, q: int) -> np.ndarray:
    """
    Resizes an array with odd dimensions to p*q dimensions. Centers crop when shrinking 
    and centers pad with zeros when expanding.
    
    Args:
        arr (np.ndarray): Input array (m*n where m and n are odd numbers)
        p (int): Target number of rows (must be odd)
        q (int): Target number of columns (must be odd)
    
    Returns:
        np.ndarray: Resized array of shape p*q
    """
    # Get original dimensions
    m, n = arr.shape
    # Row processing
    if p <= m:
        # Take center p rows by slicing
        start_row = (m - p) // 2
        adjusted_rows = arr[start_row:start_row + p, :]
    else:
        # Create zero-padded rows and place original array in center
        adjusted_rows = np.zeros((p, n), dtype=arr.dtype)
        start_row = (p - m) // 2
        adjusted_rows[start_row:start_row + m, :] = arr
    # Column processing
    if q <= n:
        # Take center q columns by slicing
        start_col = (n - q) // 2
        adjusted = adjusted_rows[:, start_col:start_col + q]
    else:
        # Create zero-padded columns and place intermediate array in center
        adjusted = np.zeros((p, q), dtype=arr.dtype)
        start_col = (q - n) // 2
        adjusted[:, start_col:start_col + n] = adjusted_rows
    return adjusted


def numba_convolve2d(mat: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    return numba_convolve2d_impl(mat, kernel)


@jit(nopython=True, cache=True)
def numba_convolve2d_impl(mat: np.ndarray, kernel: np.ndarray) -> np.ndarray:

    h_mat, w_mat = mat.shape
    h_kernel, w_kernel = kernel.shape
    output_h = h_mat + h_kernel - 1
    output_w = w_mat + w_kernel - 1
    output = np.zeros((output_h, output_w), dtype=mat.dtype)

    for y_out in range(output_h):
        for x_out in range(output_w):
            val = 0.0

            # Calculate the range of indices in 'mat' that overlap with 'kernel'
            # The convolution sum is over k_y, k_x: mat[mat_y, mat_x] * kernel[k_y, k_x]
            # where the indices satisfy the convolution definition (flipped kernel).
            # The kernel indices corresponding to mat[mat_y, mat_x] and output[y_out, x_out] are:
            # k_y = y_out - mat_y
            # k_x = x_out - mat_x

            # Determine valid range for mat_y
            # Condition 1: 0 <= mat_y < h_mat
            # Condition 2: 0 <= k_y < h_kernel  =>  0 <= y_out - mat_y < h_kernel  =>  y_out - h_kernel + 1 <= mat_y <= y_out

            y_mat_start = max(0, y_out - h_kernel + 1)
            y_mat_end = min(h_mat, y_out + 1)
            
            # Determine valid range for mat_x
            x_mat_start = max(0, x_out - w_kernel + 1)
            x_mat_end = min(w_mat, x_out + 1)

            for y_mat in range(y_mat_start, y_mat_end):
                k_y = y_out - y_mat
                for x_mat in range(x_mat_start, x_mat_end):
                    k_x = x_out - x_mat
                    val += mat[y_mat, x_mat] * kernel[k_y, k_x]
            
            output[y_out, x_out] = val
    
    return output

