#include <stdio.h>

long encode(char *s)
{
  long a,b,c,x;
  a = s[0];
  b = s[1];
  c = s[2];
  x = ((((a*69)+c)*137)+b)*39;
  x = x % 54321;
  return x;
}

void main()
{
  char s[100];
  long x;
  printf("Enter 3 lowercase letters: ");
  scanf("%s",s);
  s[3]='\0';
  x=encode(s);
  printf("Code for %s is %ld\n",s,x);
}