#include "util/stream.h"
#define RBTYPE uint32_t
#include "util/ring-buffer.h"
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <sys/ioctl.h>
#include <sys/select.h>
#include <unistd.h>

static void setup_serial_configuration(int fd, uint16_t baud) {
    termios tty{};
    tcgetattr(fd, &tty);
    cfsetispeed(&tty, baud);
    cfsetospeed(&tty, baud);
    tty.c_cflag |= (CLOCAL | CREAD); // enable receiver
    tty.c_cflag &= ~CSIZE;
    tty.c_cflag |= CS8;      // 8 data bits
    tty.c_cflag &= ~PARENB;  // no parity
    tty.c_cflag &= ~CSTOPB;  // 1 stop bit
    tty.c_cflag &= ~CRTSCTS; // no flow control
    tty.c_lflag = 0;         // raw input
    tty.c_iflag = 0;
    tty.c_oflag = 0;
    tty.c_cc[VMIN] = 0; // non-blocking
    tty.c_cc[VTIME] = 0;
    tcsetattr(fd, TCSANOW, &tty);
}

Stream::Stream(const char *path, uint16_t baud) {
    path = path;
    baud = baud;

    rxbuffer = ringbuffer_init(MAX_RX_BUFFER_SIZE);
    if (rxbuffer == NULL) {
        perror("RingBuffer malloc");
    }

    // try to open the serial device
    fd = open(path, O_RDWR | O_NOCTTY | O_NONBLOCK);
    if (fd < 0) {
        perror("open");
    }

    // NOTE: setting up the serial configutations
    setup_serial_configuration(fd, baud);
}

Stream::~Stream() {
    //  close the opened file descriptor
    ringbuffer_deinit(rxbuffer); // safely deallocate the ring buffer
    if (close(fd) == -1) {
        // handle the error
    }
}

static inline int serial_read(int fd, uint8_t *buffer, size_t size,
                              int timeout_ms) {
    fd_set set;
    FD_ZERO(&set);
    FD_SET(fd, &set);

    timeval tv{};
    tv.tv_sec = timeout_ms / 1000;
    tv.tv_usec = timeout_ms % 1000;

    int rv = select(fd + 1, &set, nullptr, nullptr, &tv);
    if (rv > 0) {
        ssize_t n;
        do {
            n = read(fd, buffer, size);
        } while (n == -1 && errno == EINTR);
        return n;
    }

    if (rv == 0) {
        return 0;
    }

    return -1; // NOTE: what cases does it capture
}

// BUG: potential
// TODO: test this (most probably wrong or buggy)
int32_t Stream::read() {
    uint8_t buffer;
    int32_t ret =
        serial_read(fd, &buffer, sizeof(buffer), DEFAULT_READ_TIMEOUT_MS);
    if (ret <= 0) {
        return -1;
    }
    return buffer;
}

static inline size_t serial_write(int fd, uint8_t b) {
    size_t len = write(fd, &b, sizeof(b));
    return len;
}

static inline int serial_write(int fd, uint8_t *buffer, size_t size) {
    size_t len = 0;
    while (len < size) {
        int bytes_sent = write(fd, buffer + len, size - len);
        if (bytes_sent == 0) {
            return -1; // return falure if send failed even for one byte
        } else if (bytes_sent == -1 && errno == EINTR) {
            // if the write was blocked by interrupt, continue to try writing
            continue;
        } else {
            len += bytes_sent;
        }
    }
    return len; // return the number of bytes sent
}

// TODO: test this
int Stream::write(uint8_t b) { return serial_write(fd, b); }

// TODO: test this
int Stream::write(uint8_t *buffer, size_t size) {
    return serial_write(fd, buffer, size);
}

// TODO: test this
int Stream::flush() { return tcdrain(fd); }

// TODO: test this
bool Stream::available() {
    int bytes = 0;
    if (ioctl(fd, FIONREAD, &bytes) == -1) {
        return -1; // error
    }
    return bytes != 0;
}
