# Research data

The simulations in the vine modulation project define their probability laws, effect sizes and random seeds in code. Small numerical tables and selected recording examples are included so readers can reproduce the figures without downloading the raw recordings.

## Hippocampal recordings

| Dataset | Public source | Analysis inputs |
| --- | --- | --- |
| Grosmark and Buzsáki recordings | [DANDI 000044, version 0.261106.0236](https://doi.org/10.48324/dandi.000044/0.261106.0236) | NWB session files with spikes, field potentials, position and sleep intervals |
| Tirole and colleagues' track recordings | [Dryad dataset](https://doi.org/10.5061/dryad.ksn02v76h) | Session files with spike times, position and field potentials during running and rest |

The analysis detects sharp-wave ripples and constructs cell-by-ripple participation matrices. Each dataset supplies its own definitions of sleep or rest. Obtain the original files, attribution and terms from the respective archives. [Preprocessing and figure commands](user-guide/vine-modulation.md)

## Other DVC examples

The Allen Visual Behavior Neuropixels scripts work with the session NWB files. The finance benchmark selects its series and date ranges in YAML. Both accept local input or output locations. See the [runner documentation](howto/runner.md) for the configured experiments.
