from neuron import h
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import choice



def create_input_neurons(N, rate, noise, first_spike=0):
    supraspinal_neurons = []
    if type(rate) == np.ndarray:
        for i in range(N):
            if rate[i]==0:
                cell = h.NetStim()
                cell.interval = 1000.0    # Inter-spike interval in ms
                cell.noise = noise
                cell.number = 1e999
                cell.start = 1e999
                supraspinal_neurons.append(cell)
            else:
                cell = h.NetStim()
                cell.interval = 1000.0/ rate[i]   # Inter-spike interval in ms
                cell.noise = noise
                cell.number = 1e999
                cell.start = first_spike
                supraspinal_neurons.append(cell)

    else:
        for _ in range(N):
            cell = h.NetStim()
            cell.interval = 1000.0 / rate  # Inter-spike interval in ms
            cell.noise = noise
            cell.number = 1e999
            cell.start = first_spike
            supraspinal_neurons.append(cell)
    return supraspinal_neurons

def create_spike_recorder_input_neurons(neurons):
    num_neurons = len(neurons)
    spike_times = [h.Vector() for _ in range(num_neurons)]
    spike_detector = [h.NetCon(neurons[i], None) for i in range(num_neurons)]
    for i in range(num_neurons): spike_detector[i].record(spike_times[i])
    return spike_times


def create_spike_recorder_mns(neurons):
    MN_spike_times = [h.Vector() for i in range(len(neurons))]
    MN_spike_detector = []
    for i in range(len(neurons)):
        sp_detector = h.NetCon(neurons[i].soma(0.5)._ref_v, None, sec=neurons[i].soma)
        sp_detector.threshold = -5
        MN_spike_detector.append(sp_detector)
        MN_spike_detector[i].record(MN_spike_times[i])
    return MN_spike_times

def create_synapses(source,
                    target,
                    W,
                    tau,
                    delay=0,
                    cell_source=False, 
                    syn_type='static',
                    syn_params=None):
    syn_list=[]
    nc_list=[]

    for itarget in range(len(target)):
        syn_list.append([])
        nc_list.append([])

        for isource in range(len(source)):
            if target[itarget].is_artificial(): 
                syn_ = target[itarget].cell
            else: 
                if syn_type == 'inhibitory': 
                    syn_ = h.Exp2Syn(target[itarget].soma(0.5))
                    syn_.tau1 = 5 # 1.5
                    syn_.tau2 = 5 # 2
                    syn_.e = -80 # -75
                elif syn_type == 'static':
                    syn_ = h.ExpSyn(target[itarget].soma(0.5))
                    syn_.tau = tau
                elif syn_type == 'depressing':
                    syn_ = h.tmgsyn(target[itarget].soma(0.5))
                    syn_.tau_1 = tau
                    syn_.tau_rec = syn_params.get('tau_rec', 800)
                    syn_.tau_facil = syn_params.get('tau_facil', 800)
                    syn_.U = syn_params.get('U1', 0.5)
                    syn_.e = syn_params.get('e', 0)
                    syn_.u0 = syn_params.get('u0', 0)

            if cell_source: 
                nc = h.NetCon(source[isource].cell, syn_)
            else: 
                nc = h.NetCon(source[isource], syn_)
                
            nc.weight[0] = W[isource,itarget]
            nc_list[-1].append(nc)
            syn_list[-1].append(syn_)
            if type(delay) == np.ndarray: 
                nc.delay = delay[isource,itarget]
            else: 
                nc.delay = delay

    return syn_list,nc_list


def create_inhomongenous_poisson_process(max_fr, tStop, integration_step, offset_x, frequency, offset_y, limit_fr=np.inf): 
    """ Simulate an inhomongenous Poisson process. TODO: full documentation here"""

    T = np.arange(0, tStop, integration_step, dtype=float)
    Y = np.maximum(max_fr * np.sin(2 * np.pi * frequency * T / 1000 - offset_x) + offset_y,0.001)

    Y = np.minimum(Y, limit_fr)
    max_lambda = np.max(Y)
    samples = tStop/integration_step

    total_spikes = np.random.poisson(lam=max_lambda*tStop/1000)
    # create homogenous process
    homogeneous_spikes = np.sort([np.random.uniform(0, samples) for _ in range(total_spikes)])
    # thin process based on inhomogenous rate
    inhomogeneous_spikes = [homogeneous_spikes[i]*integration_step for i in range(total_spikes) if np.random.uniform(0, 1) <= Y[int(homogeneous_spikes[i])]/max_lambda]
    
    return [int(spike) for spike in inhomogeneous_spikes]

def create_inhomogeneous_input_neurons(N, max_rate, tStop=6000, offset_x=0, frequency=0.45, offset_y=0, limit_fr=np.inf): 
    # NOTE freq is in Hz (cycles per s)
    
    input_neurons = []
    
    for _ in range(N):
        cell = h.VecStim()
        firings = create_inhomongenous_poisson_process(max_rate, tStop=tStop, integration_step=1, offset_x=offset_x, frequency=frequency, offset_y=offset_y, limit_fr=limit_fr)
        vec = h.Vector(firings)
        cell.play(vec)
        input_neurons.append(cell)
    return input_neurons

def create_depressing_scs_train(frequency, tau_reuptake, simulation_duration, dt=0.025, prob_syn_release=1, n_vesicle=10): 
    # set up variables 
    t_vec_len = int(simulation_duration/dt)
    n_all = n_vesicle*np.ones(t_vec_len) # docked vesicles in each synapse
    scs_pulses = np.array([1 if np.mod(x, int(1/frequency*1000/dt)) == 0 else 0 for x in range(t_vec_len)]) # pulses from SCS
    release = []

    # run simulation of synaptic failure, predict spike train
    for t in range(t_vec_len - 1): 
        if (scs_pulses[t] == 1): # if SCS pulse succeeds  
            p = 1-(1-prob_syn_release)**n_all[t]
            if np.random.rand() <= p: # if synaptic success 
                redock_time = np.random.exponential(tau_reuptake)
                redock_time = int(np.round(np.min([t_vec_len-t, redock_time])))
                n_all[t:] = np.max([0, n_all[t]-1])
                n_all[t+redock_time:] += 1 
                release.append(t*dt)

    return release

def create_depressing_scs_neurons(N, frequency, syn_params, simulation_duration, dt=0.025): 
    scs_neurons = []

    tau_reuptake = syn_params.get('tau_reuptake', 700)
    prob_syn_release = syn_params.get('prob_syn_release', 1)
    n_vesicle = syn_params.get('n_vesicle', 10)

    for _ in range(N): 
        cell = h.VecStim()
        firings = create_depressing_scs_train(frequency, tau_reuptake, simulation_duration, dt, prob_syn_release=prob_syn_release, n_vesicle=n_vesicle)
        vec = h.Vector(firings)
        cell.play(vec)
        scs_neurons.append(cell)

    return scs_neurons
