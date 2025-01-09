// #include <stdio.h>
// #include <stdlib.h>
// #include <string.h>
// #include <omp.h>

// // Function to allocate memory for a matrix
// /**
//  * @brief Allocates memory for a 2D matrix dynamically.
//  *
//  * @param rows Number of rows in the matrix.
//  * @param cols Number of columns in the matrix.
//  * @return Pointer to the allocated matrix.
//  */
// double **allocate_matrix(int rows, int cols)
// {
// 	double **matrix = (double **)malloc(rows * sizeof(double *));
// 	for (int i = 0; i < rows; i++)
// 	{
// 		matrix[i] = (double *)malloc(cols * sizeof(double));
// 	}
// 	return matrix;
// }

// // Function to free allocated memory for a matrix
// /**
//  * @brief Frees memory allocated for a 2D matrix.
//  *
//  * @param matrix Pointer to the matrix.
//  * @param rows Number of rows in the matrix.
//  */
// void free_matrix(double **matrix, int rows)
// {
// 	for (int i = 0; i < rows; i++)
// 	{
// 		free(matrix[i]);
// 	}
// 	free(matrix);
// }

// // Function to read a matrix from a file
// /**
//  * @brief Reads a matrix from the given file.
//  *
//  * @param fp Pointer to the input file.
//  * @param rows Pointer to store the number of rows in the matrix.
//  * @param cols Pointer to store the number of columns in the matrix.
//  * @return Pointer to the allocated and filled matrix.
//  */
// double **read_matrix(FILE *fp, int *rows, int *cols)
// {
// 	if (fscanf(fp, "%d,%d", rows, cols) != 2)
// 	{
// 		return NULL; // End of file or invalid format
// 	}
// 	double **matrix = allocate_matrix(*rows, *cols);
// 	for (int i = 0; i < *rows; i++)
// 	{
// 		for (int j = 0; j < *cols; j++)
// 		{
// 			fscanf(fp, "%lf,", &matrix[i][j]);
// 		}
// 	}
// 	return matrix;
// }

// // Function to write a matrix to a file
// /**
//  * @brief Writes a matrix to the output file in a structured format.
//  *
//  * @param fp Pointer to the output file.
//  * @param matrix Pointer to the matrix.
//  * @param rows Number of rows in the matrix.
//  * @param cols Number of columns in the matrix.
//  * @param operation_details Description of the operation.
//  */
// void write_matrix(FILE *fp, double **matrix, int rows, int cols, const char *operation_details)
// {
// 	fprintf(fp, "Operation: %s\n", operation_details);
// 	fprintf(fp, "Dimensions: %dx%d\n", rows, cols);
// 	fprintf(fp, "Result:\n");
// 	for (int i = 0; i < rows; i++)
// 	{
// 		for (int j = 0; j < cols; j++)
// 		{
// 			fprintf(fp, "%10.2lf \t", matrix[i][j]);
// 		}
// 		fprintf(fp, "\n");
// 	}
// 	fprintf(fp, "-----------------------------------\n");
// }

// // Function to multiply two matrices using OpenMP
// /**
//  * @brief Multiplies two matrices using OpenMP for parallelism.
//  *
//  * @param A Pointer to matrix A.
//  * @param rows_A Number of rows in matrix A.
//  * @param cols_A Number of columns in matrix A.
//  * @param B Pointer to matrix B.
//  * @param rows_B Number of rows in matrix B.
//  * @param cols_B Number of columns in matrix B.
//  * @param C Pointer to the result matrix.
//  * @param num_threads Number of threads to use for computation.
//  */
// void multiply_matrices(double **A, int rows_A, int cols_A, double **B, int rows_B, int cols_B, double **C, int num_threads)
// {
// 	omp_set_num_threads(num_threads);

// #pragma omp parallel for collapse(2)
// 	for (int i = 0; i < rows_A; i++)
// 	{
// 		for (int j = 0; j < cols_B; j++)
// 		{
// 			C[i][j] = 0.0;
// 			for (int k = 0; k < cols_A; k++)
// 			{
// 				C[i][j] += A[i][k] * B[k][j];
// 			}
// 		}
// 	}
// }

// // Main function to execute matrix multiplication
// int main(int argc, char *argv[])
// {
// 	if (argc != 3)
// 	{
// 		fprintf(stderr, "Usage: %s <input_file> <num_threads>\n", argv[0]);
// 		return 1;
// 	}

// 	char *input_file = argv[1];
// 	int num_threads = atoi(argv[2]);

// 	FILE *fp = fopen(input_file, "r");
// 	if (!fp)
// 	{
// 		perror("Error opening input file");
// 		return 1;
// 	}

// 	FILE *output_fp = fopen("output.txt", "w");
// 	if (!output_fp)
// 	{
// 		perror("Error opening output file");
// 		fclose(fp);
// 		return 1;
// 	}

// 	printf("\nStarting Matrix Multiplication...\n");

// 	while (!feof(fp))
// 	{
// 		int rows_A, cols_A, rows_B, cols_B;

// 		double **A = read_matrix(fp, &rows_A, &cols_A);
// 		if (!A)
// 			break; // End of file or invalid format

// 		double **B = read_matrix(fp, &rows_B, &cols_B);
// 		if (!B)
// 		{
// 			free_matrix(A, rows_A);
// 			break;
// 		}

// 		if (cols_A != rows_B)
// 		{
// 			printf("\n[ERROR] Cannot multiply matrices of dimensions (%d,%d) and (%d,%d)\n", rows_A, cols_A, rows_B, cols_B);
// 			free_matrix(A, rows_A);
// 			free_matrix(B, rows_B);
// 			continue;
// 		}

// 		double **C = allocate_matrix(rows_A, cols_B);
// 		int max_threads = rows_A > cols_B ? rows_A : cols_B;
// 		if (num_threads > max_threads)
// 		{
// 			num_threads = max_threads;
// 		}

// 		double start_time = omp_get_wtime();
// 		multiply_matrices(A, rows_A, cols_A, B, rows_B, cols_B, C, num_threads);
// 		double end_time = omp_get_wtime();

// 		char operation_details[100];
// 		snprintf(operation_details, sizeof(operation_details), "Multiplication of A (%dx%d) and B (%dx%d)", rows_A, cols_A, rows_B, cols_B);
// 		write_matrix(output_fp, C, rows_A, cols_B, operation_details);

// 		printf("\n[INFO] Multiplied matrices of dimensions (%d,%d) and (%d,%d)\n", rows_A, cols_A, rows_B, cols_B);
// 		printf("Result saved to 'result.txt'\n");
// 		printf("Time Taken: %.2f seconds using %d threads\n", end_time - start_time, num_threads);

// 		free_matrix(A, rows_A);
// 		free_matrix(B, rows_B);
// 		free_matrix(C, rows_A);
// 	}

// 	fclose(fp);
// 	fclose(output_fp);

// 	printf("\nMatrix Multiplication Completed. Results saved in 'output.txt'\n");
// 	return 0;
// }

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

// Function to allocate memory for a matrix
/**
 * @brief Allocates memory for a 2D matrix dynamically.
 *
 * @param rows Number of rows in the matrix.
 * @param cols Number of columns in the matrix.
 * @return Pointer to the allocated matrix.
 */
double **allocate_matrix(int rows, int cols)
{
	double **matrix = (double **)malloc(rows * sizeof(double *));
	for (int i = 0; i < rows; i++)
	{
		matrix[i] = (double *)malloc(cols * sizeof(double));
	}
	return matrix;
}

// Function to free allocated memory for a matrix
/**
 * @brief Frees memory allocated for a 2D matrix.
 *
 * @param matrix Pointer to the matrix.
 * @param rows Number of rows in the matrix.
 */
void free_matrix(double **matrix, int rows)
{
	for (int i = 0; i < rows; i++)
	{
		free(matrix[i]);
	}
	free(matrix);
}

// Function to read a matrix from a file
/**
 * @brief Reads a matrix from the given file.
 *
 * @param fp Pointer to the input file.
 * @param rows Pointer to store the number of rows in the matrix.
 * @param cols Pointer to store the number of columns in the matrix.
 * @return Pointer to the allocated and filled matrix.
 */
double **read_matrix(FILE *fp, int *rows, int *cols)
{
	if (fscanf(fp, "%d,%d", rows, cols) != 2)
	{
		return NULL; // End of file or invalid format
	}
	double **matrix = allocate_matrix(*rows, *cols);
	for (int i = 0; i < *rows; i++)
	{
		for (int j = 0; j < *cols; j++)
		{
			if (fscanf(fp, "%lf,", &matrix[i][j]) != 1)
			{
				free_matrix(matrix, *rows);
				return NULL; // Invalid matrix data
			}
		}
	}
	return matrix;
}

// Function to write a matrix to a file
/**
 * @brief Writes a matrix to the output file in a structured format.
 *
 * @param fp Pointer to the output file.
 * @param matrix Pointer to the matrix.
 * @param rows Number of rows in the matrix.
 * @param cols Number of columns in the matrix.
 * @param operation_details Description of the operation.
 */
void write_matrix(FILE *fp, double **matrix, int rows, int cols, const char *operation_details)
{
	fprintf(fp, "Operation: %s\n", operation_details);
	fprintf(fp, "Dimensions: %dx%d\n", rows, cols);
	fprintf(fp, "Result:\n");
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			fprintf(fp, "%10.2lf\t", matrix[i][j]);
		}
		fprintf(fp, "\n");
	}
	fprintf(fp, "-----------------------------------\n");
}

// Function to multiply two matrices using OpenMP
/**
 * @brief Multiplies two matrices using OpenMP for parallelism.
 *
 * @param A Pointer to matrix A.
 * @param rows_A Number of rows in matrix A.
 * @param cols_A Number of columns in matrix A.
 * @param B Pointer to matrix B.
 * @param rows_B Number of rows in matrix B.
 * @param cols_B Number of columns in matrix B.
 * @param C Pointer to the result matrix.
 * @param num_threads Number of threads to use for computation.
 */
void multiply_matrices(double **A, int rows_A, int cols_A, double **B, int rows_B, int cols_B, double **C, int num_threads)
{
	omp_set_num_threads(num_threads);

#pragma omp parallel for collapse(2)
	for (int i = 0; i < rows_A; i++)
	{
		for (int j = 0; j < cols_B; j++)
		{
			C[i][j] = 0.0;
			for (int k = 0; k < cols_A; k++)
			{
				C[i][j] += A[i][k] * B[k][j];
			}
		}
	}
}

// Main function to execute matrix multiplication
int main(int argc, char *argv[])
{
	if (argc != 4)
	{
		fprintf(stderr, "Usage: %s <input_file> <output_file> <num_threads>\n", argv[0]);
		return 1;
	}

	char *input_file = argv[1];
	char *output_file = argv[2];
	int num_threads = atoi(argv[3]);

	FILE *fp = fopen(input_file, "r");
	if (!fp)
	{
		perror("Error opening input file");
		return 1;
	}

	FILE *output_fp = fopen(output_file, "w");
	if (!output_fp)
	{
		perror("Error opening output file");
		fclose(fp);
		return 1;
	}

	printf("\nStarting Matrix Multiplication...\n");

	while (!feof(fp))
	{
		int rows_A, cols_A, rows_B, cols_B;

		double **A = read_matrix(fp, &rows_A, &cols_A);
		if (!A)
			break; // End of file or invalid format

		double **B = read_matrix(fp, &rows_B, &cols_B);
		if (!B)
		{
			free_matrix(A, rows_A);
			break;
		}

		if (cols_A != rows_B)
		{
			fprintf(stderr, "\n[ERROR] Cannot multiply matrices of dimensions (%d,%d) and (%d,%d)\n", rows_A, cols_A, rows_B, cols_B);
			free_matrix(A, rows_A);
			free_matrix(B, rows_B);
			continue;
		}

		double **C = allocate_matrix(rows_A, cols_B);
		int max_threads = rows_A > cols_B ? rows_A : cols_B;
		if (num_threads > max_threads)
		{
			num_threads = max_threads;
		}

		double start_time = omp_get_wtime();
		multiply_matrices(A, rows_A, cols_A, B, rows_B, cols_B, C, num_threads);
		double end_time = omp_get_wtime();

		char operation_details[100];
		snprintf(operation_details, sizeof(operation_details), "Multiplication of A (%dx%d) and B (%dx%d)", rows_A, cols_A, rows_B, cols_B);
		write_matrix(output_fp, C, rows_A, cols_B, operation_details);

		printf("\n[INFO] Multiplied matrices of dimensions (%d,%d) and (%d,%d)\n", rows_A, cols_A, rows_B, cols_B);
		printf("Time Taken: %.2f seconds using %d threads\n", end_time - start_time, num_threads);

		free_matrix(A, rows_A);
		free_matrix(B, rows_B);
		free_matrix(C, rows_A);
	}

	fclose(fp);
	fclose(output_fp);

	printf("\nMatrix Multiplication Completed. Results saved in '%s'\n", output_file);
	return 0;
}