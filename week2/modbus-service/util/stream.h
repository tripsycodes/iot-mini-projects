#ifndef _UTIL_STREAM_H_
#define _UTIL_STREAM_H_

#include <stddef.h>
#include <stdint.h>
#include <termios.h>

typedef struct ring_buffer RingBuffer;

#define DEFAULT_READ_TIMEOUT_MS 100 // wait 100 ms

// max buffer sizes
#define MAX_TX_BUFFER_SIZE 1024
#define MAX_RX_BUFFER_SIZE 1024

class Stream {
private:
  char *path;    // path of the serial file descriptor [linux filesystem]
  uint16_t baud; // baud
  int fd;
  RingBuffer *rxbuffer;

public:
  Stream(const char *path, uint16_t baud);
  ~Stream();
  int32_t read();
  int write(uint8_t b);
  int write(uint8_t *buffer, size_t size);
  int flush();
  bool available();
};

#endif // _UTIL_STREAM_H_
