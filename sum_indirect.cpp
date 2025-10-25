#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>
#include <string.h>

#include "sums.h"

void 
setup(int64_t N, int64_t A[])
{
   printf(" inside sum_indirect problem_setup, N=%lld \n", N);
   for(int64_t i =0; i< N; i++){
      A[i]=static_cast<int64_t>(lrand48()%N);
   }
}

int64_t
sum(int64_t N, int64_t A[])
{
   printf(" inside sum_indirect perform_sum, N=%lld \n", N);
   int64_t s = 0;
   int64_t idx = A[0];

   for(int64_t i = 0; i < N; i++){
      s += A[idx];
      idx = A[idx];
   }

   return s;
}

