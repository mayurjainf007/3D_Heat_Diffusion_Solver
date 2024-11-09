#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

#define SIZE 256         // Size of the 3D grid
#define ITERATIONS 1000   // Number of iterations
#define SIZE3 (SIZE * SIZE * SIZE)

// Function to allocate a 1D grid array for better memory locality
double *allocate_grid(int size) {
    return malloc(size * sizeof(double));
}

void free_grid(double *grid) {
    free(grid);
}

// Optimized heat distribution function using OpenMP
void heat_distribution_openmp(double *grid, double *new_grid) {
    for (int iter = 0; iter < ITERATIONS; iter++) {
        #pragma omp parallel for collapse(2) schedule(static)
        for (int i = 1; i < SIZE - 1; i++) {
            for (int j = 1; j < SIZE - 1; j++) {
                for (int k = 1; k < SIZE - 1; k++) {
                    int idx = i * SIZE * SIZE + j * SIZE + k;
                    new_grid[idx] = 0.25 * (grid[(i+1) * SIZE * SIZE + j * SIZE + k] +
                                            grid[(i-1) * SIZE * SIZE + j * SIZE + k] +
                                            grid[i * SIZE * SIZE + (j+1) * SIZE + k] +
                                            grid[i * SIZE * SIZE + (j-1) * SIZE + k] +
                                            grid[i * SIZE * SIZE + j * SIZE + (k+1)] +
                                            grid[i * SIZE * SIZE + j * SIZE + (k-1)]);
                }
            }
        }
        
        // Swap grids for the next iteration
        double *temp = grid;
        grid = new_grid;
        new_grid = temp;
    }
}

int main() {
    double *grid = allocate_grid(SIZE3);
    double *new_grid = allocate_grid(SIZE3);

    // Initialize the grid with random values
    for (int i = 0; i < SIZE3; i++) {
        grid[i] = (double)rand() / RAND_MAX;
    }

    // Measure the time taken by the optimized method
    clock_t start_optimized = clock();
    heat_distribution_openmp(grid, new_grid);
    clock_t end_optimized = clock();
    double time_taken_optimized = (double)(end_optimized - start_optimized) / CLOCKS_PER_SEC;
    printf(" Grid Size: %d\n Iterartions: %d\n Time taken by optimized method: %.2f "
         "seconds\n",
         SIZE, ITERATIONS, time_taken_optimized);

    // Free allocated memory
    free_grid(grid);
    free_grid(new_grid);

    return 0;
}
