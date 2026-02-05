#ifndef BITWRITE_H_
#define BITWRITE_H_

#define bitSet(value, bit) ((value) |= (1UL << (bit)))
#define bitClear(value, bit) ((value) &= ~(1UL << (bit)))
#define bitRead(value, bit) (((value) >> (bit)) & 0x01)

#define bitWrite(value, bit, bitvalue)                                         \
    (bitvalue ? bitSet(value, bit) : bitClear(value, bit))

#endif // BITWRITE_H_
