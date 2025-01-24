#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

long encode(char *s)
{
    long a, b, c, x;
    a = s[0];
    b = s[1];
    c = s[2];
    x = ((((a * 69) + c) * 137) + b) * 39;
    x = x % 54321;
    return x;
}

int main()
{
    char s[4];
    long x, y;
    int i, j, k;
    int found = 0; // Flag to indicate if the result is found
    printf("Enter the code: ");
    scanf("%ld", &x);
    s[3] = '\0';

    // Parallelize nested loops with OpenMP collapse
    #pragma omp parallel for collapse(3) private(i, j, k, s, y) num_threads(16)
    for (i = 0; i < 26; i++)
    {
        for (j = 0; j < 26; j++)
        {
            for (k = 0; k < 26; k++)
            {
                if (found) continue; // Skip work if the result is found

                // Generate the string for the current iteration
                s[0] = i + 'a';
                s[1] = j + 'a';
                s[2] = k + 'a';

                // Calculate the encoded value
                y = encode(s);

                // Check if the encoded value matches the input
                if (x == y)
                {
                    #pragma omp critical
                    {
                        if (!found)
                        {
                            printf("The letters for code %ld are %s\n", y, s);
                            found = 1; // Set the flag to stop further computation
                        }
                    }
                }
            }
        }
    }

    if (!found)
    {
        printf("No matching letters found for code %ld.\n", x);
    }

    return 0;
}
