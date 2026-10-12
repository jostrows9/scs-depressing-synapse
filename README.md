# scs-depressing-synapse

Repo for running a simulation of SCS + supraspinal drive. Can be run with either a static or dynamic (TsodyksMarkram) model of the SCS-motoneuron synapse. 

## Build NEURON mod files 

Run `nrnivmodl mod_files` to load repo-specific NEURON files. 

## Run simulation

The basic simulation can be run using: `python scripts/run_rdd_mn_pool_supraspinal_scs_simulation.py`. 

Parameters are as follows: 

```
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
```