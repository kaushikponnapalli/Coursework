#include <mpi.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "mm.h"
#include "stats.h"
#include "support.h"

/* lower triangular matrix multiplication */
void mm3(double *restrict A, const double *restrict B,
           const double *restrict C, size_t N)
{
    int rank;
    int size;
    size_t istart = 0;       /* start row for rank */
    size_t iend = 0;         /* end row for rank */
    size_t num_elements = 0; /* number of elements for rank */
    double *restrict Alocal = NULL;
    double *restrict Blocal = NULL;
    double *restrict Clocal = NULL;
    int *counts = NULL;
    int *displs = NULL;

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* compute start row, end row, and number of elements for rank */
    istart = f2(rank, size, N);
    iend = f2(rank + 1, size, N);
    num_elements = 0.5*iend*(iend + 1) - 0.5*istart*(istart + 1);

    /* allocate space for local data */
    Alocal = malloc(sizeof(double) * num_elements);
    Blocal = malloc(sizeof(double) * num_elements);
    Clocal = malloc(sizeof(double) * triangular(N));

    /* only on root */
    if (rank == 0) {
        /* copy C to Clocal */
        memcpy(Clocal, C, sizeof(double) * triangular(N));

        /* allocate counts and displacements arrays */
        counts = malloc(sizeof(int) * size);
        displs = malloc(sizeof(int) * size);

        /* for each process */
        for (int i = 0; i < size; i++) {
            /* compute displacement from start of A and B */
            displs[i] = 0.5*f2(i, size, N)*(f2(i, size, N) + 1);

            /* compute number of elements (count) */
            counts[i] = 0.5*(f2(i + 1, size, N)*(f2(i + 1, size, N) + 1) - f2(i, size, N)*(f2(i, size, N) + 1));

        }
    }

    /* synchronize and start stats */
    MPI_Barrier(MPI_COMM_WORLD);
    start_stats();

    /* scatter B from root to Blocal */
    MPI_Scatterv(B, counts, displs, MPI_DOUBLE, Blocal, num_elements, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* broadcast Clocal from root */
    MPI_Bcast(Clocal, triangular(N), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* first column is 0; last column is same as last row */
    for (size_t j = 0; j < iend; j++) {
        /* start at the right row for the column, end at last row */
        for (size_t i = max(istart, j); i < iend; i++) {
            /* compute offset from start of Alocal and Blocal */
            size_t ii = triangular(i) - triangular(istart);
            double sum = 0.0;
            for (size_t k = j; k <= i; k++) {
                /* compute offset from start of Clocal */
                size_t kk = triangular(k);
                sum += Blocal[ii + k] * Clocal[kk + j];
            }
            Alocal[ii + j] = sum;
        }
    }

    /* gather A to root from Alocal */
    MPI_Gatherv(Alocal, num_elements, MPI_DOUBLE, A, counts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* stop stats and synchronize */
    stop_stats();
    MPI_Barrier(MPI_COMM_WORLD);

    /* free allocated data */
    free(Alocal);
    free(Blocal);
    free(Clocal);
    free(counts);
    free(displs);
}
