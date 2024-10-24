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

void heat_distribution_simd_optimised(double ***grid, double ***new_grid) {
  for (int iter = 0; iter < ITERATIONS; iter++) {
    for (int i = 1; i < SIZE - 1; i++) {
      for (int j = 1; j < SIZE - 1; j++) {
        for (int k = 1; k < SIZE - 1; k += 4) { // Process 4 elements at a time
          // Load neighboring values using AVX2 intrinsics
          __m256d left = _mm256_loadu_pd(&grid[i - 1][j][k]);
          __m256d right = _mm256_loadu_pd(&grid[i + 1][j][k]);
          __m256d front = _mm256_loadu_pd(&grid[i][j - 1][k]);
          __m256d back = _mm256_loadu_pd(&grid[i][j + 1][k]);
          __m256d below = _mm256_loadu_pd(&grid[i][j][k - 1]);
          __m256d above = _mm256_loadu_pd(&grid[i][j][k + 1]);
          __m256d center = _mm256_loadu_pd(&grid[i][j][k]);

          // Compute the new values using Jacobi formula
          __m256d sum = _mm256_add_pd(left, right);
          sum = _mm256_add_pd(sum, front);
          sum = _mm256_add_pd(sum, back);
          sum = _mm256_add_pd(sum, below);
          sum = _mm256_add_pd(sum, above);

          __m256d factor = _mm256_set1_pd(0.125); // 1/8th for Jacobi method
          __m256d center_factor = _mm256_set1_pd(2.0);
          __m256d center_mult = _mm256_mul_pd(center, center_factor);
          sum = _mm256_add_pd(sum, center_mult);
          __m256d new_value = _mm256_mul_pd(sum, factor);

          // Store the results back to the new_grid
          _mm256_storeu_pd(&new_grid[i][j][k], new_value);
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

  // Performing heat distribution using SIMD method
  clock_t start_simd = clock();
  heat_distribution_simd_optimised(grid, new_grid);
  clock_t end_simd = clock();

  double time_taken_simd = (double)(end_simd - start_simd) / CLOCKS_PER_SEC;
  printf(" Grid Size: %d\n Iterartions: %d\n Time taken by SIMD method: %.2f "
         "seconds\n",
         SIZE, ITERATIONS, time_taken_simd);

  printf(" Speedup: %.2f\n", time_taken_naive / time_taken_simd);

  return 0;
}