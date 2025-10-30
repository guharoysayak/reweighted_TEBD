# reweighted_TEBD (rTEBD)

Reweighted TEBD (rTEBD) is a new Matrix Product Density Operator based tensor network simulation technique for improved quantum dynamics simulations. In this repository, there are rTEBD codes for a fermionic and a spin model that are shown in https://arxiv.org/abs/2412.08730. Additionally, there are MPS based TEBD code for both the fermionic and the spin model that is used for the comparison of rTEBD with MPS. According to the paper, the fermionic rTEBD code uses the fermionic scheme and the spin model rTEBD code uses the bosonic scheme. 

The relevant parameters in the rTEBD code are the bond dimension ($\chi), the system size ($L$), total simulation time in units of $J$ ($T$), total number of time steps ($N$) which gives the Trotter step ($\delta t$), and the parameter $\gamma$ (g) as defined in the paper. For g = 1, this code is usual MPDO based TEBD code and for any other value of g > 1, this is an rTEBD code.

DOI for this code release: 10.5281/zenodo.17479681
