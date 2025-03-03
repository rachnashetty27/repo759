#include <iostream>
#include <omp.h>
#include "msort.h"

using namespace std;

// Function to merge two halves of an array
void merge(int* arr, int left, int mid, int right) {
    int n1 = mid - left + 1;
    int n2 = right - mid;

    // Create temporary arrays
    int* L = new int[n1];
    int* R = new int[n2];

    // Copy data into temp arrays
    for (int i = 0; i < n1; i++) L[i] = arr[left + i];
    for (int i = 0; i < n2; i++) R[i] = arr[mid + 1 + i];

    int i = 0, j = 0, k = left;

    // Merge temp arrays back into arr[left..right]
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k] = L[i];
            i++;
        } else {
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    // Copy any remaining elements of L[]
    while (i < n1) {
        arr[k] = L[i];
        i++;
        k++;
    }

    // Copy any remaining elements of R[]
    while (j < n2) {
        arr[k] = R[j];
        j++;
        k++;
    }

    delete[] L;
    delete[] R;
}

// Parallel merge sort function using OpenMP tasks
void msort(int* arr, int left, int right, int threshold, int threads) {
    if (left < right) {
        if ((right - left) <= threshold) {
            // Use serial merge sort when size is below threshold
            for (int i = left + 1; i <= right; i++) {
                int key = arr[i];
                int j = i - 1;
                while (j >= left && arr[j] > key) {
                    arr[j + 1] = arr[j];
                    j--;
                }
                arr[j + 1] = key;
            }
        } else {
            int mid = left + (right - left) / 2;

            // Create OpenMP tasks for parallel execution
            #pragma omp parallel num_threads(threads)
            {
                #pragma omp single
                {
                    #pragma omp task shared(arr) if ((right - left) > threshold)
                    msort(arr, left, mid, threshold, threads);

                    #pragma omp task shared(arr) if ((right - left) > threshold)
                    msort(arr, mid + 1, right, threshold, threads);

                    #pragma omp taskwait  // Ensure both halves are sorted before merging
                    merge(arr, left, mid, right);
                }
            }
        }
    }
}
