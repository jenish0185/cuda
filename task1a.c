#include <stdio.h>
#include <unistd.h>
#include <omp.h>

omp_lock_t lock;
int booksAvailable = 3;

void borrower(int id) {
    for (int i = 0; i < 2; i++) { // Reduce iterations for demonstration
        // Check and borrow a book
        omp_set_lock(&lock);
        if (booksAvailable > 0) {
            printf("Borrower %d wants to borrow a book. \n", id);
            printf("Books available = %d\n", booksAvailable);
            booksAvailable--;
            printf("Borrower %d gets a book.\n", id);
            printf("Books available = %d\n", booksAvailable);
            omp_unset_lock(&lock);

            // Simulate reading the book
            printf("Borrower %d is reading a book. \n", id);
            usleep(10000);

            // Return the book
            omp_set_lock(&lock);
            printf("Borrower %d is returning a book. \n", id);
            booksAvailable++;
            printf("Books available = %d\n", booksAvailable);
        } else {
            printf("Borrower %d wants to borrow a book but none are available.\n", id);
        }
        omp_unset_lock(&lock); // Release lock in all cases

        // Simulate some wait time before retrying
        usleep(10000);
    }
}

int main() {
    omp_init_lock(&lock);

#pragma omp parallel num_threads(5) // Reduce to 5 borrowers for clarity
    {
        int id = omp_get_thread_num();
        borrower(id);
    }

    omp_destroy_lock(&lock);
    return 0;
}
