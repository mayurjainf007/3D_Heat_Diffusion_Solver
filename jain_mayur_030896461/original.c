#include <immintrin.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define SIZE 256         // Size of the 3D grid
#define ITERATIONS 10000 // Number of iterations

double ***allocate_grid(int size) {
  double ***grid = malloc(size * sizeof(double **));
  for (int i = 0; i < size; i++) {
    grid[i] = malloc(size * sizeof(double *));
    for (int j = 0; j < size; j++) {
      grid[i][j] = malloc(size * sizeof(double));
    }
  }
  return grid;
}

// Free the allocated 3D array
void free_grid(double ***grid, int size) {
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      free(grid[i][j]);
    }
    free(grid[i]);
  }
  free(grid);
}

void heat_distribution_naive_method(double ***grid, double ***new_grid) {
  for (int iter = 0; iter < ITERATIONS; iter++) {
    for (int i = 1; i < SIZE - 1; i++) {
      for (int j = 1; j < SIZE - 1; j++) {
        for (int k = 1; k < SIZE - 1; k++) {
          // Update the grid based on neighbors (Jacobi method)
          new_grid[i][j][k] = 0.125 * (grid[i - 1][j][k] + grid[i + 1][j][k] +
                                       grid[i][j - 1][k] + grid[i][j + 1][k] +
                                       grid[i][j][k - 1] + grid[i][j][k + 1] +
                                       2.0 * grid[i][j][k]);
        }
      }
    }
    // Swap the grids
    double ***temp = grid;
    grid = new_grid;
    new_grid = temp;
  }
}

int main() {
  // Allocate memory for the grids
  double ***grid = allocate_grid(SIZE);
  double ***new_grid = allocate_grid(SIZE);
  // Initialize the grid
  for (int i = 0; i < SIZE; i++) {
    for (int j = 0; j < SIZE; j++) {
      for (int k = 0; k < SIZE; k++) {
        grid[i][j][k] = rand() / (double)RAND_MAX;
        new_grid[i][j][k] = 0.0;
      }
    }
  }
  // Performing heat distribution using Naive method
  clock_t start_naive = clock();
  heat_distribution_naive_method(grid, new_grid);
  clock_t end_naive = clock();

  double time_taken_naive = (double)(end_naive - start_naive) / CLOCKS_PER_SEC;
  printf(" Grid Size: %d\n Iterartions: %d\n Time taken by naive method: %.2f "
         "seconds\n",
         SIZE, ITERATIONS, time_taken_naive);

  return 0;
}