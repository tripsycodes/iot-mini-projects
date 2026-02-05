#ifndef _UTIL_MILLIS_H_
#define _UTIL_MILLIS_H_

#include <stdint.h>
#include <time.h>

static inline uint64_t millis(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);

    return (uint64_t)(ts.tv_sec * 1000ULL + ts.tv_nsec / 1000000ULL);
}

#endif // _UTIL_MILLIS_H_
