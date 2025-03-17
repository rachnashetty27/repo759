#include <cstddef>
#include <iostream>
#include <fstream>  
#include <cmath>
#include <cstdlib>  
#include <ctime>   
#include <chrono>
#include <random>
#include <omp.h>  // OpenMP library

using std::chrono::high_resolution_clock;
using std::chrono::duration;

// Constants
const double G = 1.0;          // Gravitational constant
const double softening = 0.1;  // Softening length
const double dt = 0.01;        // Time step
const double board_size = 4.0; // Simulation boundary

// Function to calculate acceleration using OpenMP parallelization
void getAcc(const double pos[][3], const double mass[], double acc[][3], int N, int num_threads) {
    // Reset accelerations
    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < N; i++) {
        acc[i][0] = acc[i][1] = acc[i][2] = 0.0;
    }

    // Compute pairwise gravitational forces using OpenMP
    #pragma omp parallel for num_threads(num_threads) collapse(2)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (i != j) {
                double dx = pos[j][0] - pos[i][0];
                double dy = pos[j][1] - pos[i][1];
                double dz = pos[j][2] - pos[i][2];

                double distSqr = dx * dx + dy * dy + dz * dz + softening * softening;
                double invDist = 1.0 / sqrt(distSqr);
                double invDistCube = invDist * invDist * invDist;

                acc[i][0] += G * mass[j] * dx * invDistCube;
                acc[i][1] += G * mass[j] * dy * invDistCube;
                acc[i][2] += G * mass[j] * dz * invDistCube;
            }
        }
    }
}

// Main simulation function
int main(int argc, char *argv[]) {
    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_sec;
    start = high_resolution_clock::now();

    // Check if correct number of arguments are provided
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <number_of_particles> <simulation_end_time> <num_threads>" << std::endl;
        return 1;
    }

    int N = std::stoi(argv[1]);       // Number of particles
    double tEnd = std::stod(argv[2]); // Simulation time
    int num_threads = std::stoi(argv[3]); // Number of threads

    // Allocate dynamic arrays
    double* mass = new double[N];
    double(*pos)[3] = new double[N][3];
    double(*vel)[3] = new double[N][3];
    double(*acc)[3] = new double[N][3];

    // Random number generators
    std::mt19937 generator(std::random_device{}());
    std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);
    std::normal_distribution<double> normal_dist(0.0, 1.0);

    // Initialize particles with random values
    for (int i = 0; i < N; i++) {
        mass[i] = uniform_dist(generator);
        pos[i][0] = normal_dist(generator);
        pos[i][1] = normal_dist(generator);
        pos[i][2] = normal_dist(generator);
        vel[i][0] = normal_dist(generator);
        vel[i][1] = normal_dist(generator);
        vel[i][2] = normal_dist(generator);
    }

    // Initial acceleration calculation
    getAcc(pos, mass, acc, N, num_threads);

    // Simulation loop
    int Nt = int(tEnd / dt);
    for (int step = 0; step < Nt; step++) {
        
        // (1/2) Kick - Update velocity using current acceleration (Parallelized)
        #pragma omp parallel for num_threads(num_threads)
        for (int i = 0; i < N; i++) {
            vel[i][0] += 0.5 * dt * acc[i][0];
            vel[i][1] += 0.5 * dt * acc[i][1];
            vel[i][2] += 0.5 * dt * acc[i][2];
        }

        // Drift - Update positions using velocity (Parallelized)
        #pragma omp parallel for num_threads(num_threads)
        for (int i = 0; i < N; i++) {
            pos[i][0] += dt * vel[i][0];
            pos[i][1] += dt * vel[i][1];
            pos[i][2] += dt * vel[i][2];
        }

        // Update accelerations
        getAcc(pos, mass, acc, N, num_threads);

        // (1/2) Kick - Update velocity again with new acceleration (Parallelized)
        #pragma omp parallel for num_threads(num_threads)
        for (int i = 0; i < N; i++) {
            vel[i][0] += 0.5 * dt * acc[i][0];
            vel[i][1] += 0.5 * dt * acc[i][1];
            vel[i][2] += 0.5 * dt * acc[i][2];
        }
    }

    // Free memory
    delete[] mass;
    delete[] pos;
    delete[] vel;
    delete[] acc;

    end = high_resolution_clock::now();
    duration_sec = std::chrono::duration_cast<duration<double, std::milli> >(end - start);
    std::cout << "Parallel Simulation time: " << duration_sec.count() << "ms\n";

    return 0;
}
