# Sensor-corruption

## Corruption Framework
A Corruption framework was built to corrupt the raw data. There are three components to the framework: *Type*, *Granularity*, and *Severity*.

### Corruption Type
There are various different corruption types we applied.

- **Stochastic**: Adds random noise to the signal window, simulating stochastic sensor noise.
- **Dropout**: Randomly zeros out a fraction of timesteps in the signal window, simulating intermittent sensor failure or signal loss.
- **Bias**: Adds a constant offset to the signal window, simulating miscalibration or aging.
- **Gain**: Scales the signal by a constant multiplier, simulating sensor sensitivity degradation or amplification.
- **Drift**: Adds a linearly growing offset over the window, simulating gradual sensor drift over time.
- **Resolution**: Reduces the number of decimal places in the signal, simulating loss of sensor precision.

### Granularity
Corruption can be applied at different levels of granularity by specifying which channels are affected. The dataset contains 6 channels across 2 sensors. Body accelerometer (x, y, z) and gyroscope (x, y, z). This allows corruption to be applied to an entire sensor modality, a single axis within a modality, or any combination, enabling targeted investigation.

### Severity
Each corruption type has an associated severity parameter that controls the degree of corruption applied. The meaning of severity is type-sepcific.

- **Stochastic**: Corresponds to the fraction of the channel's standard deviation used as the noise scale.  
Typical values: (0.25, 0.5, 1.0, 1.25)
- **Dropout**: Corresponds to the fraction of timesteps zeroed out in the window.
Typical value: (0.1, 0.25, 0.5, 0.75)
- **Bias**: Corresponds to a fraction of the channel's standard deviation added as a constant offset.  
Typical Values: (0.25, 0.5, 1.0, 1.5)
- **Gain**: Corresponds to a constant multiplier applied to the signal.   
Typical values: (0.5, 0.75, 1.25, 2.0)
- **Drift**: Corresponds to a multiplier of the channel's standard deviation by the end of the signal window.  
Typical values: (1, 2, 3, 4)
- **Resolution**: Corresponds to number of decimal places retained.  
Typical values: (3, 4, 5, 6)

## Creating a python environment
First create a virtual python environment
```
$ python3 -m venv .venv
```

Activate environment
```
$ source .venv/bin/activate
```

Install required libraries
```
(venv) $ pip install -r requirements.txt
```

Run files
```
(venv) $ python src/data.py
```

Deactivate
```
(venv) $ deactivate
```
