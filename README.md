# Parallalel
Parallel programs that use MPI C/C++

Nbody.cpp does an N-body simulation and divides the task among multiple processes using MPI. 
The simulation uses the leapfrog method to move bodies.
Initial and final positions will be written to a separate .txt file.
Number of bodies, timesteps and lenght of timestep can be altered in the code.
