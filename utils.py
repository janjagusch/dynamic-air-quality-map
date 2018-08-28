import io
import numpy as np
import pandas as pd
import folium
import scipy.interpolate
import cv2 as cv
import PIL as pil

# Coordinates for removing zoom buttons and "leaflet" annotation.
_Y_START = 72
_Y_END = 16
_X_START = 42
_X_END = 43


def read_air_quality(station_id):
    # Reads mardid.h5 file with station_id as key.
    air_quality = pd.read_hdf("input/madrid.h5", key=str(station_id))
    air_quality["station_id"] = station_id
    return air_quality


def _min_max_scale(in_val, in_min, in_max, out_min, out_max):
    # Standardizes in_val.
    return (in_val - in_min) / (in_max - in_min) * (out_max - out_min) + out_min


def min_max_scale_series(series, out_min=0, out_max=1):
    # Standardizes series.
    in_min = series.min()
    in_max = series.max()
    return series.apply(lambda x: _min_max_scale(x, 0, in_max, out_min, out_max))


def make_map(location=(40.4168, -3.7038), width=1280, height=1280, zoom_start=12, tiles="CartoDB dark_matter", exclude_buttons=True):
    # Makes folium map.    
    height_buttons = _Y_START + _Y_END
    width_buttons = _X_START + _X_END
    if exclude_buttons:
        width += width_buttons
        height += height_buttons
    m = folium.Map(location=location,
                   width=width,
                   height=height,
                   zoom_start=zoom_start,
                   tiles=tiles)
    return m


def raster_map(m):
    # Converts m in html format to rastered m.    
    height = int(m.height[0])
    width = int(m.width[0])
    m_png = m._to_png()
    m_rastered = pil.Image.open(io.BytesIO(m_png))
    m_rastered = np.array(m_rastered)
    m_rastered = m_rastered[:height, :width, :]
    return m_rastered


def remove_buttons_map(m):
    # Removes zoom buttons and label from m.
    y, x = m.shape[:2]
    y_min = _Y_START
    y_max = y - _Y_END
    x_min = _X_START
    x_max = x - _X_END
    return m[y_min:y_max, x_min:x_max]


def process_map(m):
    # Rasters m, removes buttons and changes color channels.
    m = raster_map(m)
    m = remove_buttons_map(m)
    m = cv.cvtColor(m, cv.COLOR_BGRA2RGB)
    return m


def _interpolate(x, y, step=1./12):
    # Interpolate intra-hour values.
    tck = scipy.interpolate.splrep(x, y, s=0)
    xnew = np.arange(start=0, stop=24, step=1./12)
    ynew = scipy.interpolate.splev(xnew, tck, der=0)
    return xnew, ynew


def interpolate(df, x_col, y_col, step=1./12):
    # Interpolate intra-hour values for data frame.
    x_new, y_new = _interpolate(df[x_col], df[y_col], step)
    
    return pd.DataFrame({
        x_col: x_new,
        y_col: y_new
    })


def time_decimal_to_string(t):
    # Converts time decimal (8.5) to string ("08:30 AM").    
    h = int(t)
    m = int(round(t % 1 * 60))
    if h >= 12:
        period = "PM"
    else:
        period = "AM"
    if h > 12:
        h = h - 12
    if h < 10:
        h = str(0) + str(h)
    else:
        h = str(h)
    if m < 10:
        m = str(0) + str(m)
    else:
        m = str(m)
    return h + ":" + m + " " + period