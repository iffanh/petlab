Create ensemble module should be able to mutate the Eclipse-compatible deck based on variables that are uncertain. Uncertain parameters should be assigned a particular $KEYWORD that will be later be replaced by some values sampled from a particular distribution.

Perhaps we need to specify it in a .json format, e.g. 

What's needed for the first version: 

1. Number of ensemble, N_e
2. Parameters to change, and the associated distribution. e.g. PORO from N(0.2, 0.01)