#include "utils.h"
#include "helper_math.h"

// template <typename scalar_t>
// __device__ uint32_t fast_hash(const uint32_t pos_grid[D]){

//     constexpr uint32_t primes[7] = { 1, 2654435761, 805459861, 3674653429, 2097192037, 1434869437, 2165219737 };

//     uint32_t result = 0;
//     #pragma unroll
//     for (uint32_t i = 0; i < D; ++i) {
//         result ^= pos_grid[i] * primes[i];
//     }

//     return result;
// }



























