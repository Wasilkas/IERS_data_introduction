import numpy as np
import pandas as pd
from astropy.time import Time
import math


def remove_leap_seconds(data: np.ndarray, leap_sec: int) -> (np.ndarray, np.ndarray):
    a = data.flatten()[::-1]
    removed_secs = np.zeros(len(data.flatten()))
    prev = 0
    for i in range(len(data.flatten()) - 1):
        if a[i] - a[i + 1] > 0.9:
            a[prev:i+1] -= leap_sec
            removed_secs[prev:i+1] += leap_sec
            prev = i + 1
            leap_sec -= 1
    a[prev:] -= leap_sec
    return a[::-1], removed_secs[::-1]


def prepare_d_ut_sample_iers(data: np.ndarray, mjds: np.ndarray, mode: bool) -> tuple[np.ndarray, np.ndarray]:
    tidal = pd.read_csv('tidal.csv')
    n = len(data)
    removal = np.zeros(n)
    for i in range(n):
        alpha = evaluate_tidal_params_of_d_ut(mjds[i])
        for j in range(62):
            argument = (tidal['l'][j] * alpha[0] + tidal['l_marked'][j] * alpha[1] + tidal['F'][j] * alpha[2] +
                        tidal['D'][j] * alpha[3] + tidal['Omega'][j] * alpha[4]) / 180 * math.pi
            removal[i] += tidal['B'][j] * 10 ** (-4) * math.sin(argument) + tidal['C'][j] * 10 ** (-4) * math.cos(argument)

    data = data - removal if mode else data + removal

    return data, removal


def evaluate_tidal_params_of_d_ut(mjd: int) -> list[float]:
    t = (Time(mjd, format='mjd').jd1 - 2451545.0) / 36525
    l = 134.96340251 + (1717915923.2178 * t + 31.8792 * t ** 2 + 0.051635 * t ** 3 - 0.00024470 * t ** 4) / 3600
    l_marked = 357.52910918 + (
                129596581.0481 * t - 0.5532 * t ** 2 + 0.000136 * t ** 3 - 0.00001149 * t ** 4) / 3600
    f = 93.27209062 + (1739527262.8478 * t - 12.7512 * t ** 2 - 0.001037 * t ** 3 - 0.00000417 * t ** 4) / 3600
    d = 297.85019547 + (1602961601.2090 * t - 6.3706 * t ** 2 + 0.006593 * t ** 3 - 0.00003169 * t ** 4) / 3600
    omega = 125.04455501 + (-6962890.5431 * t + 7.4722 * t ** 2 + 0.007702 * t ** 3 - 0.00005939 * t ** 4) / 3600
    return [l, l_marked, f, d, omega]


def add_leap_seconds_to_fcast(data: np.ndarray, leap_sec: int):
    for i in range(len(data)):
        data[i] += leap_sec
    return data