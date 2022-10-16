/**
 * @file memlib-passthrough.c
 * @brief A module that passes through memlib calls to libc.
 *
 * This file allows compiling student malloc implementations so that they can
 * be used as an interpositioning library, and thereby run actual programs.
 */
#include <assert.h>
#include <stdint.h>
#include <unistd.h>

#include "config.h"
#include "memlib.h"

/* private global variables */
static bool init = false;
static unsigned char *heap;         /* Starting address of heap */
static unsigned char *mem_brk;      /* Current position of break */

static void ensure_init(void) {
    if (!init) {
        mem_brk = heap = sbrk(0);
        assert(mem_brk != (void *)-1);
        init = true;
    }
}

void *mem_sbrk(intptr_t incr) {
    ensure_init();

    unsigned char *res = sbrk(incr);
    if (res == (void *)-1) {
        return res;
    }

    assert(res == mem_brk);
    mem_brk += incr;
    return (void *) res;
}

void *mem_heap_lo(void) {
    ensure_init();
    return (void *)heap;
}

void *mem_heap_hi(void) {
    ensure_init();
    return (void *)(mem_brk - 1);
}

size_t mem_heapsize(void) {
    ensure_init();
    return (size_t)(mem_brk - heap);
}

size_t mem_pagesize(void) {
    return (size_t)getpagesize();
}
