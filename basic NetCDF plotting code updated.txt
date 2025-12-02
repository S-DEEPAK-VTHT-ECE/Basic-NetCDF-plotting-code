"""
Advanced Radar PPI Batch Plotting
- Matplotlib polar PPI with Py-ART colormaps, range rings & km labels.
- Interactive Plotly PPI (HTML) with hover (azimuth, range_km, value).
- Batch for all variables with Azimuth_Count and Number_of_Range_Bins (or Number_of_Tx_Range_Bins).
- Saves PNG/JPG/PDF + GIF/MP4 animations + HTML.
"""

import os
import math
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib import ticker
import imageio
from netCDF4 import Dataset

# Plotly
import plotly.express as px
import plotly.io as pio

# Try import pyart colormap helpers
try:
    import pyart
    from pyart import graph as pyart_graph
    has_pyart = True
except Exception:
    has_pyart = False
    # we'll fall back to matplotlib colormaps

# ---------------- User settings ----------------
file_path = r"C:\Users\sdeep\OneDrive\Desktop\NetCDF files extraction using py code\DWRIGCAR-Level1A-IGCAR_OPERATIONAL-26-IGCAR_OPERATIONAL_250KM-2025-07-21-044856-ELE-21.0.nc"
output_root_name = "Radar_Plots_Advanced"
range_ring_step_km = 10        # rings every 10 km (changeable)
range_label_km = [10, 50, 100, 150, 200, 250]  # extra labels (customize)
save_png = True
save_jpg = True
save_pdf = True
create_gif_mp4 = True
save_plotly_html = True
dpi = 200

# ------------------------------------------------

base_folder = os.path.dirname(file_path)
output_root = os.path.join(base_folder, output_root_name)
os.makedirs(output_root, exist_ok=True)

ds = xr.open_dataset(file_path)
nc = Dataset(file_path, mode="r")

# Read global attributes for axis ranges
g = ds.attrs
Zmin = float(g.get("Zmin", -20.0)); Zmax = float(g.get("Zmax", 60.0))
Vmin = float(g.get("Vmin", -60.0)); Vmax = float(g.get("Vmax", 60.0))
Swmin = float(g.get("Swmin", 0.0)); Swmax = float(g.get("Swmax", 10.0))

# Range axis (km): use Distance_Info if available (likely in km or meters)
if "Distance_Info" in ds.variables:
    ranges = np.array(ds["Distance_Info"])
    # convert to km if values appear large (>1000)
    ranges_km = ranges / 1000.0 if ranges.max() > 1000 else ranges.copy()
    print("Using Distance_Info from file for ranges (km).")
else:
    # fallback to Range_Bin_Resolution or Netcdf_Resolution attr (meters)
    res_m = float(g.get("Range_Bin_Resolution", g.get("Netcdf_Resolution", 150.0)))
    n_bins = ds.dims.get("Number_of_Range_Bins", ds.dims.get("Number_of_Tx_Range_Bins", 0))
    ranges_km = np.arange(n_bins) * (res_m / 1000.0)
    print(f"Using resolution {res_m} m -> {res_m/1000.0} km per bin.")

# Azimuths (degrees)
if "Azimuth_Info" in ds.variables:
    azim = np.array(ds["Azimuth_Info"].isel(Elevation_Count=0)).flatten()
    if azim.max() <= 2 * math.pi:  # might be in radians
        azim = np.rad2deg(azim)
    print("Using Azimuth_Info from file.")
else:
    azim = np.arange(ds.dims["Azimuth_Count"])
    print("Using default azimuths 0..359.")

n_elev = int(ds.dims.get("Elevation_Count", 1))

# Map variables -> preferred pyart colormap names or matplotlib fallbacks
VAR_CMAP_PREF = {
    "reflectivity": ["pyart_NWSRef", "pyart_NWS_Ref", "viridis"],
    "reflect": ["pyart_NWSRef", "viridis"],
    "velocity": ["RdBu_r", "pyart_NWSVel", "seismic"],
    "vel": ["RdBu_r", "pyart_NWSVel", "seismic"],
    "power": ["inferno"],
    "txpower": ["inferno"],
    "rxpower": ["inferno"],
    "zdr": ["pyart_NWS_ZDR", "PiYG"],
    "phidp": ["pyart_NWS_PHIDP", "twilight"],
    "rho": ["viridis"],
    "sw": ["magma"],
    "sqI": ["viridis"],
    "snr": ["viridis"]
}

import matplotlib as mpl

def choose_cmap(var_name, da_array):
    """Attempt to get pyart colormap, otherwise use best Matplotlib fallback."""
    name = var_name.lower()
    # auto-detect type
    for key in VAR_CMAP_PREF:
        if key in name:
            candidates = VAR_CMAP_PREF[key]
            break
    else:
        candidates = ["viridis"]

    # Try pyart variants if pyart available
    if has_pyart:
        for c in candidates:
            # many pyart colormaps are registered with prefix 'pyart_' in matplotlib
            pyart_name_variants = [c, c.replace("pyart_", ""), "pyart_" + c, c.replace("pyart_", "").upper()]
            for v in pyart_name_variants:
                try:
                    # matplotlib will find registered pyart colormaps if available
                    return mpl.cm.get_cmap(v)
                except Exception:
                    continue

    # fallback to matplotlib get_cmap by candidate names
    for c in candidates:
        try:
            return mpl.cm.get_cmap(c)
        except Exception:
            continue

    # default fallback
    return mpl.cm.get_cmap("viridis")

def get_vmin_vmax(var_name, da):
    """Return sensible vmin/vmax for variable using global attrs or percentiles."""
    name = var_name.lower()
    if "reflect" in name or "dbz" in da.attrs.get("units", "").lower():
        return Zmin, Zmax
    if "vel" in name or "m/s" in da.attrs.get("units", "").lower():
        return Vmin, Vmax
    if "sw" in name:
        return Swmin, Swmax
    # for power fields, use percentiles for robust range
    data_flat = np.array(da).flatten()
    data_flat = data_flat[~np.isnan(data_flat)]
    if data_flat.size:
        p1, p99 = np.percentile(data_flat, [1, 99])
        return float(p1), float(p99)
    return None, None

# Plotting helpers: range rings and labels
def add_range_rings(ax, max_range_km, step_km=10, labels=None):
    # ax is polar axis
    rings = np.arange(step_km, max_range_km + step_km, step_km)
    theta = np.linspace(0, 2 * np.pi, 360)
    for r in rings:
        ax.plot(theta, np.full_like(theta, r), linestyle='--', linewidth=0.6, alpha=0.7)
    # add labels at top (north)
    if labels is None:
        labels = rings
    for r in labels:
        ax.text(0, r, f"{r} km", ha='center', va='bottom', fontsize=8, color='white',
                bbox=dict(facecolor='black', alpha=0.4, pad=1))

# Core function to plot one variable (matplotlib & plotly)
def process_and_plot_variable(var_name):
    if var_name not in ds.variables:
        print(f"Variable {var_name} not present.")
        return

    da = ds[var_name]
    dims = da.dims
    units = da.attrs.get("units", "")
    # Determine range dimension name: use Number_of_Range_Bins or Number_of_Tx_Range_Bins
    if "Number_of_Range_Bins" in dims:
        n_range = ds.dims["Number_of_Range_Bins"]
        ranges = ranges_km[:n_range]
    elif "Number_of_Tx_Range_Bins" in dims:
        n_range = ds.dims["Number_of_Tx_Range_Bins"]
        ranges = ranges_km[:n_range] if len(ranges_km) >= n_range else np.arange(n_range) * (ranges_km[1]-ranges_km[0])
    else:
        # fallback to full ranges_km
        ranges = ranges_km.copy()
        n_range = ranges.size

    # Build variable-specific output folder
    var_folder = os.path.join(output_root, var_name)
    os.makedirs(var_folder, exist_ok=True)

    cmap = choose_cmap(var_name, da)
    vmin, vmax = get_vmin_vmax(var_name, da)

    # Collect frames for animation
    frames = []

    # Process each elevation (if present)
    for elev in range(n_elev):
        # Extract data slice and coerce to (azimuth, range) array
        if da.ndim == 3:
            # typical order is (Elevation_Count, Azimuth_Count, Range)
            slice_ = np.array(da.isel(Elevation_Count=elev))
            # Ensure shape is (Azimuth_Count, Range)
            if slice_.shape[0] != ds.dims["Azimuth_Count"]:
                slice_ = slice_.reshape((ds.dims["Azimuth_Count"], -1))
        elif da.ndim == 2:
            # maybe (Elevation_Count, Azimuth_Count) or (Azimuth_Count, Range)
            slice_ = np.array(da)
            if slice_.shape[0] == 1:
                slice_ = slice_.reshape((ds.dims["Azimuth_Count"], -1))
        else:
            slice_ = np.array(da)

        az_count, rng_count = slice_.shape[0], slice_.shape[1]
        # Build mesh for plotting (theta radians, r in km)
        theta_mid = np.deg2rad(azim[:az_count])
        r_mid = ranges[:rng_count]

        theta_grid, r_grid = np.meshgrid(theta_mid, r_mid, indexing='ij')  # shape (az, rng)

        # ---------- Matplotlib polar PPI ----------
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='polar')
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)

        # pcolormesh wants 2D arrays of theta,r matching data shape
        # Use shading='auto' and pass theta_grid, r_grid
        pcm = ax.pcolormesh(theta_grid, r_grid, slice_, shading='auto', cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_ylim(0, r_mid.max())

        # add range rings & labels
        max_r = float(r_mid.max())
        add_range_rings(ax, max_r, step_km=range_ring_step_km, labels=range_label_km)

        # Add title and colorbar
        ax.set_title(f"{var_name} | Elevation {elev} | Units: {units}", pad=20)
        cb = fig.colorbar(pcm, ax=ax, pad=0.08)
        cb.set_label(units)

        # Save static images
        fname_base = f"{var_name}_E{elev}"
        if save_png:
            fig.savefig(os.path.join(var_folder, fname_base + ".png"), dpi=dpi, bbox_inches="tight")
        if save_jpg:
            fig.savefig(os.path.join(var_folder, fname_base + ".jpg"), dpi=150, bbox_inches="tight")
        if save_pdf:
            fig.savefig(os.path.join(var_folder, fname_base + ".pdf"), bbox_inches="tight")

        # store frame for animation
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(image)

        plt.close(fig)

    # ---------- Save animation if multiple elevation frames or single frame (still saved) ----------
    if create_gif_mp4 and frames:
        try:
            gif_path = os.path.join(var_folder, f"{var_name}_elev_animation.gif")
            imageio.mimsave(gif_path, frames, fps=1)
            print("Saved GIF:", gif_path)
        except Exception as e:
            print("GIF creation failed:", e)

        try:
            mp4_path = os.path.join(var_folder, f"{var_name}_elev_animation.mp4")
            imageio.mimsave(mp4_path, frames, fps=1, macro_block_size=None)  # mp4 codec may require ffmpeg
            print("Saved MP4 (if ffmpeg available):", mp4_path)
        except Exception as e:
            print("MP4 creation failed (ffmpeg may be missing):", e)

    # ---------- Interactive Plotly export ----------
    if save_plotly_html:
        # We'll create an image-like interactive heatmap using azimuth (y) x range (x) grid for simplicity.
        # y axis = azimuth degrees, x axis = range_km
        # For multi-elevation, create tabs (here we save separate HTML files per elevation)
        for elev in range(n_elev):
            if da.ndim == 3:
                img = np.array(da.isel(Elevation_Count=elev))
                if img.shape[0] != ds.dims["Azimuth_Count"]:
                    img = img.reshape((ds.dims["Azimuth_Count"], -1))
            elif da.ndim == 2:
                img = np.array(da)
                if img.shape[0] == 1:
                    img = img.reshape((ds.dims["Azimuth_Count"], -1))
            else:
                img = np.array(da)

            # restrict to correct range length
            nr = img.shape[1]
            xranges = ranges[:nr]
            yaz = azim[:img.shape[0]]

            # Plotly express imshow — axes correspond to x (ranges) and y (azimuths)
            fig = px.imshow(img,
                            x=xranges,
                            y=yaz,
                            origin='lower',
                            aspect='auto',
                            labels={'x': 'Range (km)', 'y': 'Azimuth (deg)', 'color': units},
                            title=f"{var_name} | Elevation {elev}")

            # Choose a plotly color_scale roughly matching matplotlib colormap choice
            # Use simple mapping for better compatibility with Plotly standard palettes
            lname = var_name.lower()
            if "reflect" in lname:
                color_scale = "turbo"   # reflectivity-like (user can change)
            elif "vel" in lname:
                color_scale = "RdBu"
            elif "zdr" in lname:
                color_scale = "RdYlBu"
            elif "power" in lname or "txpower" in lname or "rxpower" in lname:
                color_scale = "inferno"
            else:
                color_scale = "Viridis"

            fig.update_traces(coloraxis=None)
            fig.update_layout(coloraxis_showscale=True)
            fig.update_traces(colorscale=color_scale, zmin=vmin, zmax=vmax)

            # Add hovertemplate to show Azimuth, Range and Value properly
            fig.update_traces(hovertemplate="Azimuth: %{y}°<br>Range: %{x} km<br>Value: %{z}<extra></extra>")

            html_path = os.path.join(var_folder, f"{var_name}_E{elev}_interactive.html")
            try:
                fig.write_html(html_path)
                print("Saved interactive HTML:", html_path)
            except Exception as e:
                print("Failed to save interactive HTML:", e)

    print(f"Completed variable: {var_name}. Outputs at: {var_folder}")

# -------------------------
# Which variables to batch-plot?
# Pick variables that have Azimuth_Count and Number_of_Range_Bins or Number_of_Tx_Range_Bins
# -------------------------
plot_vars = []
for vname, var in ds.variables.items():
    dims = var.dims
    if ("Azimuth_Count" in dims) and (("Number_of_Range_Bins" in dims) or ("Number_of_Tx_Range_Bins" in dims)):
        plot_vars.append(vname)

print("Variables to be plotted (detected):", plot_vars)

# Run processing on each detected variable
for v in plot_vars:
    try:
        process_and_plot_variable(v)
    except Exception as e:
        print(f"Error processing {v}: {e}")

# Close datasets
ds.close()
nc.close()
print("All done. Output root:", output_root)
