#ifndef _UTIL_BYTE_H_
#define _UTIL_BYTE_H_

#include <stdint.h>

static inline uint8_t lowByte(uint16_t w) { return (uint8_t)((w) & 0xFF); }
static inline uint8_t highByte(uint16_t w) { return (uint8_t)((w) >> 8); }

#endif // _UTIL_BYTE_H_
