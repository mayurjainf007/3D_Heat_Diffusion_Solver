
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#include <immintrin.h>
#include <math.h>

#define SIZE 256         // Size of the 3D grid
#define ITERATIONS 10000  // Number of iterations

// Allocate a 1D grid array for optimized OpenMP approach
double *allocate_grid_1D(int size) {
    return malloc(size * sizeof(double));
}

void free_grid_1D(double *grid) {
    free(grid);
}

// Allocate a 3D grid array for the naive method
double ***allocate_grid_3D(int size) {
    double ***grid = malloc(size * sizeof(double **));
    for (int i = 0; i < size; i++) {
        grid[i] = malloc(size * sizeof(double *));
        for (int j = 0; j < size; j++) {
            grid[i][j] = malloc(size * sizeof(double));
        }
    }
    return grid;
}

void free_grid_3D(double ***grid, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            free(grid[i][j]);
        }
        free(grid[i]);
    }
    free(grid);
}

// Naive heat distribution function without optimization
void heat_distribution_naive_method(double ***grid, double ***new_grid) {
    for (int iter = 0; iter < ITERATIONS; iter++) {
        for (int i = 1; i < SIZE - 1; i++) {
            for (int j = 1; j < SIZE - 1; j++) {
                for (int k = 1; k < SIZE - 1; k++) {
                    new_grid[i][j][k] = 0.125 * (grid[i - 1][j][k] + grid[i + 1][j][k] +
                                                 grid[i][j - 1][k] + grid[i][j + 1][k] +
                                                 grid[i][j][k - 1] + grid[i][j][k + 1] +
                                                 2.0 * grid[i][j][k]);
                }
            }
        }
        double ***temp = grid;
        grid = new_grid;
        new_grid = temp;
    }
}

// Optimized heat distribution function using OpenMP
void heat_distribution_openmp(double *grid, double *new_grid) {
    int size3 = SIZE * SIZE * SIZE;
    for (int iter = 0; iter < ITERATIONS/25; iter++) {
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
        double *temp = grid;
        grid = new_grid;
        new_grid = temp;
    }
}

int main() {

    // Naive method
    double ***grid_3D = allocate_grid_3D(SIZE);
    double ***new_grid_3D = allocate_grid_3D(SIZE);

    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            for (int k = 0; k < SIZE; k++) {
                grid_3D[i][j][k] = (double)rand() / RAND_MAX;
                new_grid_3D[i][j][k] = 0.0;
            }
        }
    }

    clock_t start_naive = clock();
    heat_distribution_naive_method(grid_3D, new_grid_3D);
    clock_t end_naive = clock();

    double time_taken_naive = (double)(end_naive - start_naive) / CLOCKS_PER_SEC;
    printf("Naive method - Grid Size: %d, Iterations: %d, Time taken: %.2f seconds\n", SIZE, ITERATIONS, time_taken_naive);

    free_grid_3D(grid_3D, SIZE);
    free_grid_3D(new_grid_3D, SIZE);

    // OpenMP optimized version
    int size3 = SIZE * SIZE * SIZE;
    double *grid_1D = allocate_grid_1D(size3);
    double *new_grid_1D = allocate_grid_1D(size3);

    for (int i = 0; i < size3; i++) {
        grid_1D[i] = (double)rand() / RAND_MAX;
    }

    clock_t start_optimized = clock();
    heat_distribution_openmp(grid_1D, new_grid_1D);
    clock_t end_optimized = clock();

    double time_taken_optimized = (double)(end_optimized - start_optimized) / CLOCKS_PER_SEC;
    printf("Optimized method - Grid Size: %d, Iterations: %d, Time taken: %.2f seconds\n", SIZE, ITERATIONS, time_taken_optimized);

    free_grid_1D(grid_1D);
    free_grid_1D(new_grid_1D);

    printf(" Speedup: %.2f\n", time_taken_naive / time_taken_optimized);

    return 0;
}
