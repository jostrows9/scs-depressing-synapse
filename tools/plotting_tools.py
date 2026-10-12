import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

fontsize = 8

def plot_raster_plot(ax, 
                    spike_data, 
                    ylabel: str = '', 
                    marker_color: str = '#4c4d4f', 
                    marker_size: float = 0.2): 
    """
    Plot timeseries spike data as a raster plot. 

    Args: 
        - ax: plot axes to plot on 
        - spike_data: spiking data, neurons x time
        - ylabel: y-axis label 
        - marker_color: marker color
    """
    num_neurons = len(spike_data)
    for i in range(num_neurons): ax.plot(spike_data[i],i*np.ones(len(spike_data[i])), ".",  c=marker_color, markersize=marker_size)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.yaxis.set_tick_params(labelsize=fontsize)
    ax.xaxis.set_tick_params(labelsize=fontsize)

def plot_time_series_data(ax, 
                    time,
                    time_series_data, 
                    ylabel: str = '', 
                    line_color: str = '#4c4d4f', 
                    line_width: float = 1): 
    """
    Plot data over time. 

    Args: 
        - ax: plot axes to plot on 
        - time: simulation time
        - time_series_data: timeseries data (membrane potential, fr, etc.)
        - ylabel: y-axis label 
        - line_color: line color
    """
    ax.plot(time, time_series_data, color=line_color, linewidth=line_width)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.yaxis.set_tick_params(labelsize=fontsize)
    ax.xaxis.set_tick_params(labelsize=fontsize)
    