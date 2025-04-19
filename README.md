# GDL-Mini-project
 ## Code for the Geometric Deep Learning mini-project. 
In this project, we address the prohibitive computational cost of $O(N^6)$ present in Epping et al (2024) when finding a critical weight variance
for initialization in GCNs through the use of Bayesian Optimization.

More importantly, we study the effect of dynamical isometry and orthogonal weights in GCNs and prove they can prevent oversmoothing 
in deeper setting without resorting to normalization or residual techniques.

## Executing the code.

### It is recommended to follow the notebook from Google Colab. However, the code is also directly executable in python scripts.

Some of the experiments take a long time (running a GCN 5 times for each combination of (n_layers,activation) or running BO for the same combinations).
To run the code, download the repository and from the root directory run python -m experiments.[NAME_OF_EXPERIMENT] (no .py).
