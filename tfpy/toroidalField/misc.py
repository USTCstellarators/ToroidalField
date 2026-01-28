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

    if not tfParams.cache:
        return numba_convolve2d_impl(mat, kernel)
    else:
        mat_tuple = tuple(tuple(row) for row in mat)
        kernel_tuple = tuple(tuple(row) for row in kernel)
        result_tuple = numba_convolve2d_cached(mat_tuple, kernel_tuple)
        return np.array(result_tuple)


@jit(nopython=True)
def numba_convolve2d_impl(mat: np.ndarray, kernel: np.ndarray) -> np.ndarray:

    h_mat, w_mat = mat.shape
    h_kernel, w_kernel = kernel.shape
    output_h = h_mat + h_kernel - 1
    output_w = w_mat + w_kernel - 1
    output = np.zeros((output_h, output_w))
    
    kernel_flipped = np.zeros_like(kernel)
    for y in range(h_kernel):
        for x in range(w_kernel):
            kernel_flipped[y, x] = kernel[h_kernel - 1 - y, w_kernel - 1 - x]
    
    for y_out in range(output_h):
        for x_out in range(output_w):
            val = 0.0

            y_mat_start = max(0, y_out - h_kernel + 1)
            y_mat_end = min(h_mat, y_out + 1)
            x_mat_start = max(0, x_out - w_kernel + 1)
            x_mat_end = min(w_mat, x_out + 1)
            
            y_kernel_start = max(0, h_kernel - y_out - 1)
            y_kernel_end = min(h_kernel, h_mat + h_kernel - y_out - 1)
            x_kernel_start = max(0, w_kernel - x_out - 1)
            x_kernel_end = min(w_kernel, w_mat + w_kernel - x_out - 1)
            
            sub_mat = mat[y_mat_start:y_mat_end, x_mat_start:x_mat_end]
            sub_kernel = kernel_flipped[y_kernel_start:y_kernel_end, x_kernel_start:x_kernel_end]
            val = np.sum(sub_mat * sub_kernel)
            output[y_out, x_out] = val
    
    return output


@lru_cache(maxsize=16384)
def numba_convolve2d_cached(mat_tuple: tuple, kernel_tuple: tuple) -> tuple:
    mat = np.array(mat_tuple, dtype=np.float64)
    kernel = np.array(kernel_tuple, dtype=np.float64)
    result = numba_convolve2d_impl(mat, kernel)
    return tuple(tuple(row) for row in result)

