/**
 * MIT License
 *
 * Copyright (c) 2025 Sahil
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef RING_BUFFER_H_
#define RING_BUFFER_H_

#include <stdint.h>

/**
 * @brief Type of data stored in the ring buffer.
 */
#ifndef RBTYPE
#define RBTYPE int
#endif

#ifndef MIN
#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#endif

#ifndef MAX
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#endif

#if defined(RB_DEBUG)
#include <assert.h>
#include <stdio.h>
#define RB_LOG(a, ...) fprintf(stderr, "[%s] " __VA_ARGS__, a)
#else
#define RB_LOG(a, ...)
#endif // RB_DEBUG

#ifndef NO_STRING
#include <string.h>
#endif

#include <stddef.h>

/**
 * @brief Structure representing a ring (circular) buffer.
 */
typedef struct ring_buffer {
    RBTYPE *buffer;  /**< Pointer to buffer memory */
    size_t capacity; /**< Maximum number of elements (excluding extra slot) */
    size_t head;     /**< Index to write to */
    size_t tail;     /**< Index to read from */
} RingBuffer;

#ifndef RB_NO_MALLOC
#include <stdlib.h>

/**
 * @brief Dynamically allocates and initializes a ring buffer.
 * @param size Number of elements the buffer should hold.
 * @return Pointer to initialized ring buffer or NULL on failure.
 */
RingBuffer *ringbuffer_init(size_t size);

/**
 * @brief Frees memory allocated for the ring buffer.
 * @param prb Pointer to ring buffer to be deallocated.
 */
void ringbuffer_deinit(RingBuffer *prb);
#endif // RB_NO_MALLOC

/**
 * @brief Initializes a statically allocated ring buffer.
 * @param prb Pointer to ring buffer structure.
 * @param buffer Pointer to preallocated memory (size + 1).
 * @param size Maximum number of elements.
 * @return 0 on success, -1 on error.
 */
int ringbuffer_static_init(RingBuffer *prb, RBTYPE *buffer, size_t size);

/**
 * @brief Returns number of elements currently in the buffer.
 * @param prb Pointer to ring buffer.
 * @return Number of elements in buffer.
 */
size_t ringbuffer_size(RingBuffer *prb);

/**
 * @brief Checks if the ring buffer is empty.
 * @param prb Pointer to ring buffer.
 * @return 1 if empty, 0 otherwise.
 */
int ringbuffer_empty(RingBuffer *prb);

/**
 * @brief Checks if the ring buffer is full.
 * @param prb Pointer to ring buffer.
 * @return 1 if full, 0 otherwise.
 */
int ringbuffer_full(RingBuffer *prb);

/**
 * @brief Resets the ring buffer (clears contents).
 * @param prb Pointer to ring buffer.
 */
void ringbuffer_reset(RingBuffer *prb);

/**
 * @brief Writes data to the ring buffer.
 * @param prb Pointer to ring buffer.
 * @param pIn Input array.
 * @param size Number of elements to write.
 * @param nwrite Optional output: number of elements actually written.
 * @return 0 on success, -1 on error.
 */
int ringbuffer_write(RingBuffer *prb, const RBTYPE *pIn, const size_t size,
                     size_t *nwrite);

/**
 * @brief Reads data from the ring buffer.
 * @param prb Pointer to ring buffer.
 * @param pOut Output array.
 * @param size Maximum number of elements to read.
 * @param nread Optional output: number of elements actually read.
 * @return 0 on success, -1 on error.
 */
int ringbuffer_read(RingBuffer *prb, RBTYPE *pOut, const size_t size,
                    size_t *nread);

/**
 * @brief Peeks into the ring buffer without modifying it.
 * @param prb Pointer to ring buffer.
 * @param pOut Output array.
 * @param size Maximum number of elements to peek.
 * @param nread Optional output: number of elements peeked.
 * @return 0 on success, -1 on error.
 */
int ringbuffer_peek(RingBuffer *prb, RBTYPE *pOut, const size_t size,
                    size_t *nread);

// =============== IMPLEMENTATIONS ==============

#ifndef RB_NO_MALLOC
RingBuffer *ringbuffer_init(size_t size) {
    RingBuffer *prb = (RingBuffer *)malloc(sizeof(RingBuffer));
    if (prb == NULL) {
        RB_LOG("ERROR", "failed to allocate memory for ringbuffer\n");
        return NULL;
    }

    prb->capacity = size;
    prb->head = 0;
    prb->tail = 0;

    prb->buffer = (RBTYPE *)malloc(sizeof(RBTYPE) * (size + 1));
    if (prb->buffer == NULL) {
        free(prb);
        return NULL;
    }

    return prb;
}

void ringbuffer_deinit(RingBuffer *prb) {
    if (prb) {
        if (prb->buffer)
            free(prb->buffer);
        free(prb);
    }
}
#endif // RB_NO_MALLOC

// note: sizeof(buffer[]) > size; eg: size + 1
int ringbuffer_static_init(RingBuffer *prb, RBTYPE *buffer, size_t size) {
    if (prb == NULL) {
        RB_LOG("ERROR", "null pointer provided for ring buffer.\n");
        return -1;
    }
    if (buffer == NULL) {
        RB_LOG("ERROR", "null pointer provided for buffer.\n");
        return -1;
    }

    prb->capacity = size;
    prb->head = 0;
    prb->tail = 0;
    prb->buffer = buffer;

    return 0; // SUCCESS
}

size_t ringbuffer_size(RingBuffer *prb) {
    if (prb == NULL) {
        RB_LOG("ERROR", "null buffer provided for prb in size.\n");
        return 0;
    }

    if (prb->head < prb->tail) {
        return prb->capacity - prb->tail + prb->head + 1;
    } else {
        return prb->head - prb->tail;
    }
}

int ringbuffer_empty(RingBuffer *prb) {
    if (prb == NULL) {
        RB_LOG("ERROR", "null pointer provided in ringbuffer_empty.\n");
        return 1; // default on error: empty
    }

    return prb->head == prb->tail;
}

int ringbuffer_full(RingBuffer *prb) {
    if (prb == NULL) {
        RB_LOG("ERROR", "null pointer provided in ringbuffer_full.\n");
        return 0; // default on error: not full
    }

    return prb->capacity == ringbuffer_size(prb);
}

void ringbuffer_reset(RingBuffer *prb) {
    if (prb == NULL) {
        RB_LOG("ERROR", "null pointer provided in ringbuffer_reset.\n");
        return;
    }

    prb->head = 0;
    prb->tail = 0;
}

int ringbuffer_write(RingBuffer *prb, const RBTYPE *pIn, const size_t size,
                     size_t *nwrite) {
    if (nwrite != NULL)
        *nwrite = 0;

    if (prb == NULL) {
        RB_LOG("ERROR", "null pointer `prb` provided in ringbuffer_write.\n");
        return -1;
    }

    if (pIn == NULL) {
        RB_LOG("ERROR", "null pointer `pIn` provided in ringbuffer_write.\n");
        return -1;
    }

    if (prb->buffer == NULL) {
        RB_LOG("ERROR", "prb has invalid buffer, error in ringbuffer_write.\n");
        return -1;
    }

    const size_t msize = prb->capacity;
    const size_t head = prb->head;
    const size_t filled = ringbuffer_size(prb);
    const size_t towrite = MIN(msize - filled, size);

    if (towrite + head > msize) {
        size_t lend = msize - head + 1;
        size_t lbeg = towrite - lend;
#if defined(NO_STRING)
        for (size_t i = 0; i < lend; i++) {
            prb->buffer[head + i] = pIn[i];
        }
        for (size_t i = 0; i < lbeg; i++) {
            prb->buffer[i] = pIn[lend + i];
        }
#else
        memcpy(prb->buffer + head, pIn, lend * sizeof(RBTYPE));
        if (lbeg > 0)
            memcpy(prb->buffer, pIn + lend, lbeg * sizeof(RBTYPE));
#endif // NO_STRING
        prb->head = (head + towrite) % (msize + 1);
    } else {
#if defined(NO_STRING)
        for (size_t i = 0; i < towrite; i++) {
            prb->buffer[head + i] = pIn[i];
        }
#else
        memcpy(prb->buffer + head, pIn, towrite * sizeof(RBTYPE));
#endif // NO_STRING
        prb->head += towrite;
    }

    if (nwrite != NULL)
        *nwrite = towrite;

    return 0; // SUCCESS
}

int ringbuffer_read(RingBuffer *prb, RBTYPE *pOut, const size_t size,
                    size_t *nread) {
    if (nread != NULL)
        *nread = 0;

    if (prb == NULL) {
        RB_LOG("ERROR", "null pointer prb provided in ringbuffer_read.\n");
        return -1;
    }

    if (pOut == NULL) {
        RB_LOG("ERROR", "null pointer pOut provided in ringbuffer_read.\n");
        return -1;
    }

    if (prb->buffer == NULL) {
        RB_LOG("ERROR", "invalid buffer pointer for prb in ringbuffer_read.\n");
        return -1;
    }

    const size_t msize = prb->capacity;
    const size_t tail = prb->tail;
    const size_t avail = ringbuffer_size(prb);
    const size_t toread = MIN(avail, size);

    if (tail + toread > msize) {
        size_t lend = msize - tail + 1;
        size_t lbeg = toread - lend;

#if defined(NO_STRING)
        for (size_t i = 0; i < lend; i++) {
            pOut[i] = prb->buffer[tail + i];
        }
        for (size_t i = 0; i < lbeg; i++) {
            pOut[lend + i] = prb->buffer[i];
        }
#else
        memcpy(pOut, prb->buffer + tail, lend * sizeof(RBTYPE));
        if (lbeg >= 0)
            memcpy(pOut + lend, prb->buffer, lbeg * sizeof(RBTYPE));
#endif // NO_STRING
        prb->tail = (tail + toread) % (msize + 1);
    } else {
#if defined(NO_STRING)
        for (size_t i = 0; i < toread; i++) {
            pOut[i] = prb->buffer[tail + i];
        }
#else
        memcpy(pOut, prb->buffer + tail, toread * sizeof(RBTYPE));
#endif // NO_STRING
        prb->tail += toread;
    }

    if (nread != NULL)
        *nread = toread;

    return 0; // SUCCESS
}

int ringbuffer_peek(RingBuffer *prb, RBTYPE *pOut, const size_t size,
                    size_t *nread) {
    if (nread != NULL)
        *nread = 0;

    if (prb == NULL) {
        RB_LOG("ERROR", "null pointer prb provided in ringbuffer_read.\n");
        return -1;
    }

    if (pOut == NULL) {
        RB_LOG("ERROR", "null pointer pOut provided in ringbuffer_read.\n");
        return -1;
    }

    if (prb->buffer == NULL) {
        RB_LOG("ERROR", "invalid buffer pointer for prb in ringbuffer_read.\n");
        return -1;
    }

    const size_t msize = prb->capacity;
    const size_t tail = prb->tail;
    const size_t avail = ringbuffer_size(prb);
    const size_t toread = MIN(avail, size);

    if (tail + toread > msize) {
        size_t lend = msize - tail + 1;
        size_t lbeg = toread - lend;

#if defined(NO_STRING)
        for (size_t i = 0; i < lend; i++) {
            pOut[i] = prb->buffer[tail + i];
        }
        for (size_t i = 0; i < lbeg; i++) {
            pOut[lend + i] = prb->buffer[i];
        }
#else
        memcpy(pOut, prb->buffer + tail, lend * sizeof(RBTYPE));
        if (lbeg >= 0)
            memcpy(pOut + lend, prb->buffer, lbeg * sizeof(RBTYPE));
#endif // NO_STRING
    } else {
#if defined(NO_STRING)
        for (size_t i = 0; i < toread; i++) {
            pOut[i] = prb->buffer[tail + i];
        }
#else
        memcpy(pOut, prb->buffer + tail, toread * sizeof(RBTYPE));
#endif // NO_STRING
    }

    if (nread != NULL)
        *nread = toread;

    return 0; // SUCCESS
}

#endif // RING_BUFFER_H_
