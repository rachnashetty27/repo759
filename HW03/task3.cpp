#include <iostream>
#include <cstdlib>
#include <chrono>
#include "msort.h"

using namespace std;

// Function to generate random numbers in an array
void fill_array(int* arr, int n) {
    for (int i = 0; i < n; i++) {
        arr[i] = rand() % 2001 - 1000;  // Random integers in range [-1000, 1000]
    }
}

int main(int argc, char* argv[]) {
    // Ensure correct number of arguments
    if (argc != 4) {
        cerr << "Usage: ./task3 <array_size> <num_threads> <threshold>\n";
        return 1;
    }

    // Parse command-line arguments
    int n = atoi(argv[1]);       // Array size
    int threads = atoi(argv[2]); // Number of OpenMP threads
    int threshold = atoi(argv[3]); // Threshold for recursion

    // Allocate memory for the array
    int* arr = new int[n];

    // Fill array with random values
    fill_array(arr, n);

    // Measure execution time
    auto start = chrono::high_resolution_clock::now();

    // Run parallel merge sort
    msort(arr, 0, n - 1, threshold, threads);

    // Stop measuring execution time
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> elapsed = end - start;

    // Print required outputs
    cout << arr[0] << endl;      // First element of sorted array
    cout << arr[n - 1] << endl;  // Last element of sorted array
    cout << elapsed.count() << endl; // Execution time in milliseconds

    // Free allocated memory
    delete[] arr;

    return 0;
}
