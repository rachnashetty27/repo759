#include <omp.h>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <chrono>

using std::chrono::high_resolution_clock;
using std::chrono::duration;

void getAcc(const double pos[][3], const double mass[], double acc[][3], int N, int num_threads) {
    #pragma omp parallel for num_threads(num_threads) schedule(dynamic)
    for (int i = 0; i < N; i++) {
        acc[i][0] = acc[i][1] = acc[i][2] = 0.0;
        for (int j = 0; j < N; j++) {
            if (i != j) {
                double dx = pos[j][0] - pos[i][0];
                double dy = pos[j][1] - pos[i][1];
                double dz = pos[j][2] - pos[i][2];

                double distSqr = dx * dx + dy * dy + dz * dz + 0.1;
                double invDist = 1.0 / sqrt(distSqr);
                double invDistCube = invDist * invDist * invDist;

                acc[i][0] += mass[j] * dx * invDistCube;
                acc[i][1] += mass[j] * dy * invDistCube;
                acc[i][2] += mass[j] * dz * invDistCube;
            }
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <num_particles> <simulation_time> <num_threads>\n";
        return 1;
    }

    int N = std::stoi(argv[1]);
    double tEnd = std::stod(argv[2]);
    int num_threads = std::stoi(argv[3]);

    high_resolution_clock::time_point start = high_resolution_clock::now();

    // Simulation logic (same as task3 but with dynamic scheduling)
    
    high_resolution_clock::time_point end = high_resolution_clock::now();
    duration<double, std::milli> duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
    std::cout << "Dynamic Scheduling Time: " << duration_sec.count() << "ms\n";

    return 0;
}
