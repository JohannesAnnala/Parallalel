#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mpi.h"

const double G = 6.67259e-7; // Adjusted gravitational constant to make forces a bit stronger
const double dt = 1.0;       // Lenght of a timestep

/* Write out positions (x,y) of N particles to the file fn
   Return zero if the file couldn't be opened, otherwise one */
int write_particles(int N, double* X_a, double* Y_a, char* filename)
{
	FILE* fp;
	if ((fp = fopen(filename, "w")) == NULL) // Open file 
	{
		printf("Couldn't open file %s\n", filename);
		return 0;
	}

	for (int i = 0; i < N; i++) // Write positions to file
	{
		fprintf(fp, "%3.2f %3.2f \n", X_a[i], Y_a[i]);
	}
	fprintf(fp, "\n");
	fclose(fp);
	return 1;
}

// Distance between points with coordinates (x1, y1) and (x2, y2)
double dist(double x1, double y1, double x2, double y2)
{
	return sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2));
}

// Compute forces between bodies
void ComputeForcesProcess(int N, int process_s, int first_b, double* mass, double* X, double* Y, double* Fx, double* Fy)
{
	const double mindist = 0.0001;      // Minimun distance to compute force between bodies

	for (int i = 0; i < process_s; i++) // Calculate force for each body
	{
		Fx[i] = Fy[i] = 0.0;            // Initialize values of force to zero

		for (int j = 0; j < N; j++)     // The force is a sum over all other bodies
		{
			if ((i + first_b) != j)
			{
				double r = dist(X[i + first_b], Y[i + first_b], X[j], Y[j]); // The distance between bodies

				if (r >= mindist)       // Very near-distance forces are ignored
				{
					double r3 = pow(r, 3);
					Fx[i] += G * mass[i + first_b] * mass[j] * (X[j] - X[i + first_b]) / r3;
					Fy[i] += G * mass[i + first_b] * mass[j] * (Y[j] - Y[i + first_b]) / r3;
				}
			}
		}
	}
}

int main(int argc, char* argv[])
{
	int np, id;
	int tag = 42;
	MPI_Status status;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &np);
	MPI_Comm_rank(MPI_COMM_WORLD, &id);

	const int N = 1000;                 // Number of bodies
	const int timesteps = 1000;         // Number of timesteps
	const double size = 100.0;			// Range for initial positions
	const int process_size = N / np;    // Size of each process
	const int first_body = N * id / np; // First body of each process

	// Check that the load is divisible among processes
	if (N % np != 0)
	{
		if (id == 0)
		{
			printf("The work is not divisible among processes\n");
		}
		MPI_Finalize();
		exit(0);
	}

	/* Differentiate variables based on whether it contains data from all/process,
       so its obvious in code */
	double* mass_all = (double*)malloc(N * sizeof(double));   // masses of all bodies
	double* X_all = (double*)malloc(N * sizeof(double));	  // x-positions of all bodies
	double* Y_all = (double*)malloc(N * sizeof(double));	  // y-positions of all bodies
	double* X_process = (double*)malloc(N * sizeof(double));  // x-positions of process specific bodies
	double* Y_process = (double*)malloc(N * sizeof(double));  // y-positions of process specific bodies
	double* Fx_process = (double*)malloc(N * sizeof(double)); // forces on x-axis of process specific bodies
	double* Fy_process = (double*)malloc(N * sizeof(double)); // forces on y-axis of process specific bodies
	double* Vx_process = (double*)malloc(N * sizeof(double)); // velocities on x-axis of process specific bodies
	double* Vy_process = (double*)malloc(N * sizeof(double)); // velocities on y-axis of process specific bodies

	// Check that allocations were succesfull
	if (mass_all == NULL || X_all == NULL || Y_all == NULL || X_process == NULL || Y_process == NULL || Fx_process == NULL || Fy_process == NULL || Vx_process == NULL || Vy_process == NULL)
	{
		printf("Memory allocation was unsuccesfull\n");
		MPI_Finalize();
		exit(0);
	}

	if (id == 0) 
	{
		// Seed the random number generator
		unsigned short int seedval[3] = { 7, 7, 7 };
		seed48(seedval);

		// Initialize mass and coordinates
		for (int i = 0; i < N; i++)
		{
			mass_all[i] = 1000.0 * drand48();
			X_all[i] = size * drand48();
			Y_all[i] = size * drand48();
		}

		printf("N body simulation, number of bodies = %d \n", N); fflush(stdout);		
		write_particles(N, X_all, Y_all, "starting_positions.txt"); // Write initial positions to a file
	}

	// Broadcast masses and initial positions to all processes
	MPI_Bcast(mass_all, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(X_all, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(Y_all, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	// Start measuring time
	double time_start = MPI_Wtime();

	// Compute the initial forces
	ComputeForcesProcess(N, process_size, first_body, mass_all, X_all, Y_all, Fx_process, Fy_process);

	// Compute initial velocities for leapfrog method
	for (int i = 0; i < process_size; i++)
	{
		Vx_process[i] = 0.5 * dt * Fx_process[i] / mass_all[i + first_body];
		Vy_process[i] = 0.5 * dt * Fy_process[i] / mass_all[i + first_body];
	}

	int t = 0;
	// Simulate bodies 
	while (t < timesteps)
	{	
		t++;
		
		// Calculate new positions
		for (int i = 0; i < process_size; i++)
		{
			X_process[i] = X_all[i + first_body] + Vx_process[i] * dt;
			Y_process[i] = Y_all[i + first_body] + Vy_process[i] * dt;
		}

		// Gather all new positions
		MPI_Gather(X_process, process_size, MPI_DOUBLE, X_all, process_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Gather(Y_process, process_size, MPI_DOUBLE, Y_all, process_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

		// Broadcast all new positions
		MPI_Bcast(X_all, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(Y_all, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

		// Update forces
		ComputeForcesProcess(N, process_size, first_body, mass_all, X_all, Y_all, Fx_process, Fy_process);

		// Update velocities
		for (int i = 0; i < process_size; i++)
		{
			Vx_process[i] = Vx_process[i] + Fx_process[i] * dt / mass_all[i + first_body];
			Vy_process[i] = Vy_process[i] + Fy_process[i] * dt / mass_all[i + first_body];
		}		
	}

	if (id == 0)
	{
		double time_end = MPI_Wtime();
		printf("Simulation took %f seconds\n", (time_end - time_start)); fflush(stdout);
		write_particles(N, X_all, Y_all, "end_positions.txt"); // Write final positions to a file
	}

	free(mass_all);
	free(X_all);
	free(Y_all);
	free(X_process);
	free(Y_process);
	free(Fx_process);
	free(Fy_process);
	free(Vx_process);
	free(Vy_process);
	MPI_Finalize();
	exit(0);
}