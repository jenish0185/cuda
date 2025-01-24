// b.   Modify the program so that it doesn't use locks.

#include <stdio.h>
#include <unistd.h>
#include <omp.h>

int booksAvailable = 3;

void borrower(int id)
{
    for (int i = 0; i < 2; i++) // Each borrower tries twice
    {
// Critical section for borrowing a book
#pragma omp critical
        {
            if (booksAvailable > 0)
            {
                printf("[Borrower %d] wants to borrow a book. Books available = %d\n", id, booksAvailable);
                booksAvailable--;
                printf("[Borrower %d] successfully borrowed a book. Books now available = %d\n", id, booksAvailable);
            }
            else
            {
                printf("[Borrower %d] wants to borrow a book, but none are available.\n", id);
            }
        }

        printf("[Borrower %d] is reading a book.\n", id);
        usleep(10000); // Simulate reading

// Critical section for returning a book
#pragma omp critical
        {
            printf("[Borrower %d] is returning a book.\n", id);
            booksAvailable++;
            printf("[Borrower %d] returned a book. Books now available = %d\n", id, booksAvailable);
        }

        usleep(10000); // Pause before the next iteration
    }
}

int main()
{
#pragma omp parallel num_threads(5) // 5 borrowers
    {
        int id = omp_get_thread_num();
        borrower(id);
    }

    return 0;
}