#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <crypt.h>
#include <unistd.h>
#include <time.h>
#include <omp.h>

/******************************************************************************
  Demonstrates how to crack an encrypted password using a simple
  "brute force" algorithm. Works on passwords that consist only of 2 uppercase
  letters and a 2 digit integer.

  Compile with:
    cc -o CrackAZ99 CrackAZ99.c -lcrypt

  If you want to analyse the output then use the redirection operator to send
  output to a file that you can view using an editor or the less utility:
    ./CrackAZ99 > output.txt

  Dr Kevan Buckley, University of Wolverhampton, 2018 Modified by Dr. Ali Safaa 2019
******************************************************************************/

int count=0;     // A counter used to track the number of combinations explored so far
int password_found = 0; // Flag to indicate the password is found

/**
 Required by lack of standard function in C.   
*/

void substr(char *dest, char *src, int start, int length){
  memcpy(dest, src + start, length);
  *(dest + length) = '\0';
}

/**
 This function can crack the kind of password explained above. All combinations
 that are tried are displayed and when the password is found, #, is put at the 
 start of the line. Note that one of the most time consuming operations that 
 it performs is the output of intermediate results, so performance experiments 
 for this kind of program should not include this. i.e. comment out the printfs.
*/

void crack(char *salt_and_encrypted) {
    int x, y, z;          // Loop counters
    char salt[7];         // Salt used for hashing
    char plain[7];        // The combination of letters and numbers being checked
    struct crypt_data cdata; // Thread-safe structure for crypt_r
    cdata.initialized = 0;

    substr(salt, salt_and_encrypted, 0, 6);

    // Parallelize the outermost loop
    #pragma omp parallel for private(y, z, plain, cdata) shared(password_found, count)
    for (x = 'A'; x <= 'Z'; x++) {
        if (password_found) continue; // Exit early if another thread found the password

        for (y = 'A'; y <= 'Z'; y++) {
            for (z = 0; z <= 99; z++) {
                if (password_found) break;

                sprintf(plain, "%c%c%02d", x, y, z); // Generate a candidate password
                char *enc = crypt_r(plain, salt, &cdata);

                #pragma omp atomic
                count++;

                if (strcmp(salt_and_encrypted, enc) == 0) {
                    #pragma omp critical
                    {
                        if (!password_found) {
                            password_found = 1;
                            printf("#%-8d%s %s\n", count, plain, enc);
                        }
                    }
                }
            }
        }
    }
}

int main()
{
  char encrypted_password[100]; 

  printf("Enter the encrypted password (SHA-512 hash with salt): ");
  scanf("%99s", encrypted_password); 

  printf("\nCracking process started !\n");
  printf("Encrypted Password: %s\n", encrypted_password);

  double start_time = omp_get_wtime(); 

  crack(encrypted_password); 

  double end_time = omp_get_wtime();            
  double cpu_time_used = end_time - start_time; 

  if (!password_found)
  {
    printf("PASSWORD COULD NOT BE FOUND\n");
  }

  printf("Time Taken: %.2f seconds\n", cpu_time_used);

  return 0;
}
