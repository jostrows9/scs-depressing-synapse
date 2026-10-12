from matplotlib import pyplot as plt
from neuron import h
import numpy as np
import pickle

import sys
sys.path.append('../scs-depressing-synapse/')

import cells as cll
from tools.general_tools import ensure_dir
import tools.neuron_functions as nf
import tools.plotting_tools as pt
import tools.analysis_tools as at

def run_mn_pool_supraspinal_scs_simulation(scs_amp: float,
                                   scs_freq: int, 
                                   perc_supra_intact: float = 1, 
                                   supra_freq: int = 60, 
                                   scs_syn_type: str = 'depressing',
                                   syn_params: dict = None,
                                   simulation_duration: int = 4000,
                                   record_membrane: bool = False,
                                   save_data_folder: str = '', 
                                   plot_sim: bool = False, 
                                   seed: int = 672945):
    """ 
    Run simulation of a motoneuron pool receiving modulation from
    SCS and residual supraspinal inputs. 

    Args: 
        - scs_amp: float, amplitude of stimulation (% of max # of SCS inputs)
        - scs_freq: int, frequency of stimulation (Hz)
        - perc_supra_intact: float, remaining supraspinal inputs (% of intact model = 300)
        - supra_freq: int, frequency of supraspinal inputs (Hz)
        - scs_syn_type: str, type of synapse to use for SCS inputs (options: 'simpleDepressing', 'AMPADepressing', 'static')
        - syn_params: dict, parameters for any depressing synapse models (more details in `tools/neuron_functions.py`), if not provided, default parameters will be used
        - simulation_duration: int, length of simulation (ms)
        - record_membrane: bool, records membrane potential of model motoneurons
        - save_data_folder: str, name of folder to save data (if not provided, data will not be saved)
        - plot_sim: bool, plot simulation results
    
    Out: N/A, see `save_data_folder` and `plot_sim` for simulation output options
    """

    h.load_file('stdrun.hoc')
    np.random.seed(seed) # set the seed so the network is always the same

    # Set SCS parameters
    num_scs_total = 60
    num_scs_effective = int(scs_amp*num_scs_total)

    # Set supraspinal parameters
    num_supraspinal_total = 300
    num_supraspinal = int(num_supraspinal_total*perc_supra_intact)
    rate_supraspinal = supra_freq # firing rate (in Hz)
    
    # Set MN parameters
    num_mn = 20
    mn_drug = True
    mn_avg_diameter = 36
    
    # Set synaptic parameters
    synaptic_weight = 0.000148
    shape = 1.2
    tau = 5 #ms

    # Create lists to hold the neurons and spike recorders
    supraspinal_neurons = []
    scs_neurons = []
    supraspinal_spike_times = []
    scs_pulse_times = []
    
    # Create a population of SCS pulses and record their spikes
    if num_scs_effective > 0:
        scs_neurons = nf.create_input_neurons(num_scs_effective, scs_freq, noise=0, first_spike=0)
        scs_pulse_times = nf.create_spike_recorder_input_neurons(scs_neurons)

    # Create a population of supraspinal neurons and record their spikes
    if num_supraspinal > 0: 
        supraspinal_neurons = nf.create_input_neurons(num_supraspinal, rate_supraspinal, noise=1, first_spike=0)
        supraspinal_spike_times = nf.create_spike_recorder_input_neurons(supraspinal_neurons)

    # Create a population of MNs and record their spikes
    mn_L = mn_avg_diameter + np.random.randn(num_mn)*0.1*mn_avg_diameter
    mns = [cll.MotoneuronNoDendrites("WT", drug=mn_drug, L=mn_L[imn]) for imn in range(num_mn)]
    mn_spike_times = nf.create_spike_recorder_mns(mns)

    # Connect a population of supraspinal fibers to MNs
    W_supraspinal = 0
    if num_supraspinal > 0:
        W_supraspinal = np.random.gamma(shape, scale=synaptic_weight/shape, size=[num_supraspinal,num_mn])
        syn_supraspinal, nc_supraspinal = nf.create_synapses(supraspinal_neurons, mns, W_supraspinal, tau)

    # Connect a population of scs pulses to MNs
    W_scs = 0
    if num_scs_effective>0:
        W_scs = np.random.gamma(shape, scale=synaptic_weight/shape, size=[num_scs_effective, num_mn])
        delay = np.random.lognormal(1, 0.5, size=[num_scs_effective, num_mn]) # from Greiner 2021
        syn_scs, nc_scs = nf.create_synapses(scs_neurons, mns, W_scs, tau, delay, syn_type=scs_syn_type, syn_params=syn_params)

    # Record motoneuron membranes
    membrane_potentials = [[]]
    simulation_time_vector = []
    if record_membrane: 
        for i in range(num_mn):
            membrane_potentials.append(h.Vector().record(mns[i].soma(0.5)._ref_v))
        simulation_time_vector = h.Vector().record(h._ref_t)

    # Run simulation
    print(f"Running simulation for {scs_syn_type} SCS synapses, {scs_amp} SCS amplitude, {scs_freq}Hz SCS frequency, {num_supraspinal} supraspinal inputs, {rate_supraspinal}Hz supraspinal frequency!") 
    h.finitialize()
    h.tstop = simulation_duration
    h.run()

    # Convert outputs to numpy
    supraspinal_spike_times = [np.array(supraspinal_spike_times[i]) if len(supraspinal_spike_times[i]) > 0 else [] for i in range(num_supraspinal)]
    scs_pulse_times = [np.array(scs_pulse_times[i]) if len(scs_pulse_times[i]) > 0 else [] for i in range(num_scs_effective)]
    mn_spike_times = [np.array(mn_spike_times[i]) if len(mn_spike_times[i]) > 0 else [] for i in range(num_mn)]
    membrane_potentials = [np.array(membrane_potentials[i]) if len(membrane_potentials[i]) > 0 else [] for i in range(len(membrane_potentials))]
    simulation_time_vector = np.array(simulation_time_vector)

    # Estimate EMG 
    firings_int = [[int(spike) for spike in mn_firings] for mn_firings in mn_spike_times]
    firings_mat = np.array([[1 if i in mn_firings else 0 for i in range(0, simulation_duration)] for mn_firings in firings_int])
    emg_signal = at.estimate_emg(firings_mat)

    # Save simulation data
    if save_data_folder != '': 
        ensure_dir(save_data_folder)
        data_filename = f"mnNum_{num_mn}_supraspinalNum_{num_supraspinal}_supraspinalFR_{rate_supraspinal}_SCSFreq_{scs_freq}_SCSTotal_{num_scs_total}_SCSAmp_{scs_amp}_SynW_{synaptic_weight}_seed_{seed}_scsSynType_{scs_syn_type}.pickle"

        data={}
        data["mn_spikes"] = mn_spike_times
        data["supraspinal_spike_times"] = supraspinal_spike_times
        data["scs_pulse_times"] = scs_pulse_times
        data["scs_frequency"] = scs_freq
        data["scs_amp"] = scs_amp
        data["num_scs_total"] = num_scs_total
        data["W_scs"] = W_scs
        data["supraspinal_rate"] = rate_supraspinal
        data["num_supraspinal"] = num_supraspinal
        data["simulation_duration"] = simulation_duration
        data["num_mn"] = num_mn
        data["synaptic_weight"] = synaptic_weight
        data["W_supraspinal"] = W_supraspinal
        data["mn_L"] = mn_L
        data["simulation_time_vector"] = simulation_time_vector
        data["membrane_potentials"] = membrane_potentials
        data["emg_signal"] = emg_signal
        data["scs_syn_type"] = scs_syn_type
        data["syn_params"] = syn_params


        f=open(save_data_folder+data_filename,"wb")
        pickle.dump(data,f)
        f.close()

    # Visualize simulation data
    if plot_sim: 
        plt.subplots(4, 1, sharex='col')
        plt.suptitle(f'MN Pool Firing:\n {scs_amp} SCS Amp, {scs_freq} Hz SCS Freq,\n {num_supraspinal} Supraspinal Inputs', fontsize=8)

        # Plot supraspinal firing
        ax = plt.subplot(4, 1, 1)
        pt.plot_raster_plot(ax, supraspinal_spike_times, ylabel='Supraspinal\n fibers')
        ax.set_ylim([0, 300])

        # Plot SCS pulses
        ax = plt.subplot(4, 1, 2)
        pt.plot_raster_plot(ax, scs_pulse_times, ylabel='Ia-afferent\n input')
        ax.set_ylim([0, 60])

        # Plot MN firing
        ax = plt.subplot(4, 1, 3)
        pt.plot_raster_plot(ax, mn_spike_times, ylabel='MN spikes')
        ax.set_xlabel('Time (ms)', fontsize=8)

        # Plot MN firing
        ax = plt.subplot(4, 1, 4)
        pt.plot_time_series_data(ax, np.linspace(0, simulation_duration, len(emg_signal)), emg_signal, ylabel='EMG signal')
        ax.set_xlabel('Time (ms)', fontsize=8)

        plt.tight_layout()

        if record_membrane: 
            plt.subplots(3, 1, sharex='col', figsize=(5,6))
            plt.suptitle(f'Motoneuron Membrane Potential:\n {scs_amp} SCS Amp, {scs_freq} SCS Freq,\n {num_supraspinal} Supraspinal Inputs', fontsize=8)

            # Plot motoneuron membrane potential
            ax = plt.subplot(3, 1, 1)
            pt.plot_time_series_data(ax, simulation_time_vector, membrane_potentials[1], ylabel='Motoneuron membrane\n potential (mV)')
            ax.set_ylim(-74, -60)

            # Plot supraspinal firing
            ax = plt.subplot(3, 1, 2)
            pt.plot_raster_plot(ax, supraspinal_spike_times, ylabel='Supraspinal\n fibers')
            ax.set_ylim([0, 300])

            # Plot SCS pulses
            ax = plt.subplot(3, 1, 3)
            pt.plot_raster_plot(ax, scs_pulse_times, ylabel='Ia-afferent\n input')
            ax.set_ylim([0, 60])
            ax.set_xlabel('Time (ms)', fontsize=8)

            plt.tight_layout()

        plt.show()

    
if __name__ == '__main__': 
    scs_amp = 0.8
    scs_freq = 40
    perc_supra_intact = 0.3

    syn_params = {'tau_rec': 800, 'tau_facil': 0, 'U': 0.5, 'e': 0, 'u0': 0}

    run_mn_pool_supraspinal_scs_simulation(scs_amp=scs_amp, 
        scs_freq=scs_freq, 
        perc_supra_intact=perc_supra_intact,
        record_membrane=True,
        simulation_duration=1000,
        scs_syn_type='static',
        syn_params=syn_params,
        plot_sim=True)


        
