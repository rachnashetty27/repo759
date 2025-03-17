#include <iostream>
#include <cmath>
#include <chrono>
#include <omp.h>
#include <vector>

using namespace std;
using namespace std::chrono;

// Constants
const double G = 1.0;          // Gravitational constant
const double softening = 0.1;  // Softening length
const double dt = 0.01;        // Time step
const double board_size = 4.0; // Simulation boundary

// Function to calculate acceleration using OpenMP with Dynamic Scheduling
void getAcc(const vector<vector<double>>& pos, const vector<double>& mass, vector<vector<double>>& acc, int N, int num_threads) {
    #pragma omp parallel for num_threads(num_threads) schedule(dynamic)
    for (int i = 0; i < N; i++) {
        acc[i][0] = acc[i][1] = acc[i][2] = 0.0;
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

int main(int argc, char* argv[]) {
    if (argc != 4) {
        cerr << "Usage: " << argv[0] << " <num_particles> <simulation_time> <num_threads>\n";
        return 1;
    }

    int N = stoi(argv[1]);         // Number of particles
    double tEnd = stod(argv[2]);   // Simulation end time
    int num_threads = stoi(argv[3]);  // Number of OpenMP threads

    // Start timing
    auto start = high_resolution_clock::now();

    // Initialize positions, velocities, accelerations, and masses
    vector<vector<double>> pos(N, vector<double>(3));
    vector<vector<double>> vel(N, vector<double>(3, 0.0));
    vector<vector<double>> acc(N, vector<double>(3, 0.0));
    vector<double> mass(N, 1.0);

    // Random initial positions
    srand(42);
    for (int i = 0; i < N; i++) {
        pos[i][0] = (double(rand()) / RAND_MAX) * board_size - (board_size / 2);
        pos[i][1] = (double(rand()) / RAND_MAX) * board_size - (board_size / 2);
        pos[i][2] = (double(rand()) / RAND_MAX) * board_size - (board_size / 2);
    }

    // Compute initial accelerations
    getAcc(pos, mass, acc, N, num_threads);

    // Simulation loop
    int Nt = int(tEnd / dt);
    for (int step = 0; step < Nt; step++) {
        // (1/2) Kick - Update velocity using current acceleration
        #pragma omp parallel for num_threads(num_threads)
        for (int i = 0; i < N; i++) {
            vel[i][0] += 0.5 * dt * acc[i][0];
            vel[i][1] += 0.5 * dt * acc[i][1];
            vel[i][2] += 0.5 * dt * acc[i][2];
        }

        // Drift - Update positions using velocity
        #pragma omp parallel for num_threads(num_threads)
        for (int i = 0; i < N; i++) {
            pos[i][0] += dt * vel[i][0];
            pos[i][1] += dt * vel[i][1];
            pos[i][2] += dt * vel[i][2];
        }

        // Debugging: Print the position of the first particle every 10 steps
        if (step % 10 == 0) {
            cout << "Step " << step << " first particle pos: (" 
                 << pos[0][0] << ", " << pos[0][1] << ", " << pos[0][2] << ")\n";
        }

        // Update accelerations
        getAcc(pos, mass, acc, N, num_threads);

        // (1/2) Kick - Update velocity again with new acceleration
        #pragma omp parallel for num_threads(num_threads)
        for (int i = 0; i < N; i++) {
            vel[i][0] += 0.5 * dt * acc[i][0];
            vel[i][1] += 0.5 * dt * acc[i][1];
            vel[i][2] += 0.5 * dt * acc[i][2];
        }
    }

    // Stop timing
    auto end = high_resolution_clock::now();
    duration<double, std::milli> duration_sec = end - start;

    // Print execution time
    cout << "Dynamic Scheduling Time: " << duration_sec.count() << " ms\n";

    return 0;
}
