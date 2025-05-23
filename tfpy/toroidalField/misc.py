#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# misc.py


import numpy as np 


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
