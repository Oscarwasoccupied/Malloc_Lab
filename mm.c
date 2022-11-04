/**
 * @file mm.c
 * @brief A 64-bit struct-based explicit free list memory allocator with 15
 * buckets. Optimization includes removing footer and mini blocks.
 *
 * 15-213: Introduction to Computer Systems
 * Version Final: Segregated Free List
 *************************************************************************
 * 1. Structure for blocks
 * For Segregated free list, without any optimization, a allocated block
 * consists of a header, a payload, a footer; a free block wiil have a header
 * a pointer to previous block, a pointer to next block, and a footer.
 * But with optimization, miniblock of size 16 is introduced. An allocated
 * block wiil only have a header and a payload. And, a free block can have
 * only a header with one pointer to the next block.
 * For the header/footer of each block, without optimization, there is an
 * allocated bit that indicate whether current block is allocated or not.
 * After optimization, two more bits are added to the block. The first one
 * is a pre-mini bit that indicates whether previous block is a mini block
 * or not. The second bit is a pre-allocated bit that indicates whether the
 * previous block is allocated or not.
 *
 * 2. Minimum block size
 * Base on the 16 bytes alignments requirements, for allocated block,
 * the minimum size is 16 bytes (8 bytes header + 8 bytes payload);
 * for free block, the minimum size is also 16 bytes (8 bytes header +
 * 8 bytes pointer to the next free block).
 *
 * 3. Management of free blocks
 * There are total 15 buckets free block list, includes a mini-block bucket
 * that consists of block is only 16 bytes which is connect together using
 * a singly linked list. And for others 14 free list, each of them connect
 * blocks using a doubly linked list.
 *
 *************************************************************************
 * Segregate list. Pass Checkpoint. Utilization=59.3, Throughput=10692,
 * Performance= 53.6
 * @author Xianwei Zou <xianweiz@andrew.cmu.edu>
 */

#include <assert.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "memlib.h"
#include "mm.h"

/* Do not change the following! */

#ifdef DRIVER
/* create aliases for driver tests */
#define malloc mm_malloc
#define free mm_free
#define realloc mm_realloc
#define calloc mm_calloc
#define memset mem_memset
#define memcpy mem_memcpy
#endif /* def DRIVER */

/* You can change anything from here onward */

/*
 *****************************************************************************
 * If DEBUG is defined (such as when running mdriver-dbg), these macros      *
 * are enabled. You can use them to print debugging output and to check      *
 * contracts only in debug mode.                                             *
 *                                                                           *
 * Only debugging macros with names beginning "dbg_" are allowed.            *
 * You may not define any other macros having arguments.                     *
 *****************************************************************************
 */
#ifdef DEBUG
/* When DEBUG is defined, these form aliases to useful functions */
#define dbg_printf(...) printf(__VA_ARGS__)
#define dbg_requires(expr) assert(expr)
#define dbg_assert(expr) assert(expr)
#define dbg_ensures(expr) assert(expr)
#define dbg_printheap(...) print_heap(__VA_ARGS__)
#else
/* When DEBUG is not defined, no code gets generated for these */
/* The sizeof() hack is used to avoid "unused variable" warnings */
#define dbg_printf(...) (sizeof(__VA_ARGS__), -1)
#define dbg_requires(expr) (sizeof(expr), 1)
#define dbg_assert(expr) (sizeof(expr), 1)
#define dbg_ensures(expr) (sizeof(expr), 1)
#define dbg_printheap(...) ((void)sizeof(__VA_ARGS__))
#endif

/* Basic constants */

typedef uint64_t word_t; // word_t is just unsigned long: 8 bytes

/** @brief Word and header size (bytes) */
static const size_t wsize = sizeof(word_t);

/** @brief Double word size (bytes) */
static const size_t dsize = 2 * wsize;

/** @brief Minimum block size (bytes) */
static const size_t min_block_size = dsize;

/** @brief number of seg list buckets */
static const size_t num_buckets = 15;

/** @brief better fit find limits
 *      larger value means finding a more suitable block size, with larger
 * utility, but smaller throughput, since more times are needed for searching.
 * Experimentally find the best value is around 6 to find a trade off between
 * utility and throughput
 */
static const size_t find_fit_time_limits = 3;

/** @brief 2^min_bucket_pow is the upper bound for size
 *      of the first bucket.
 * range = (2^(index+min_bucket_pow-1)-1, 2^(index+min_bucket_pow))
 */
static const size_t min_bucket_pow = 4;

/**
 * Everytime extends heap by this amount
 * (Must be divisible by dsize)
 */
static const size_t chunksize = (1 << 12);

/**
 * Get the allocted flag byte from the header/footer
 *      (unsigned long) 0x1 = 0x 0000 0000 0000 0001
 *          alloc = header & alloc_mask
 */
static const word_t alloc_mask = 0x1;

/**
 * @brief to extract the alloc bit for previous block
 */
static const word_t pre_alloc_mask = 0x2;

/**
 * @brief to extract the mini bit to tell if the previous
 * block is a free block of mini size
 */
static const word_t pre_mini_mask = 0x4;

/**
 * (unsigned long) ~0xF = 0x FFFF FFFF FFFF FFF0
 *      size = header & size_mask
 */
static const word_t size_mask = ~(word_t)0xF;

/** @brief Represents the header and payload of one block in the heap */
typedef struct block {
    /** @brief Header contains size + allocation flag */
    word_t header;

    union {
        struct {
            struct block *next;
            struct block *prev;
        };
        char payload[0];
    };

} block_t;

/* Global variables */

/** @brief Pointer to first block in the heap */
static block_t *heap_start = NULL;

/** @brief Point to the free block list **/
static block_t *bucket_list[num_buckets];

/*
 *****************************************************************************
 * The functions below are short wrapper functions to perform                *
 * bit manipulation, pointer arithmetic, and other helper operations.        *
 *                                                                           *
 * We've given you the function header comments for the functions below      *
 * to help you understand how this baseline code works.                      *
 *                                                                           *
 * Note that these function header comments are short since the functions    *
 * they are describing are short as well; you will need to provide           *
 * adequate details for the functions that you write yourself!               *
 *****************************************************************************
 */

/*
 * ---------------------------------------------------------------------------
 *                        BEGIN SHORT HELPER FUNCTIONS
 * ---------------------------------------------------------------------------
 */

/**
 * @brief Returns the maximum of two integers.
 * @param[in] x
 * @param[in] y
 * @return `x` if `x > y`, and `y` otherwise.
 */
static size_t max(size_t x, size_t y) {
    return (x > y) ? x : y;
}

/**
 * @brief Rounds `size` up to next multiple of n
 * @param[in] size
 * @param[in] n
 * @return The size after rounding up
 */
static size_t round_up(size_t size, size_t n) {
    return n * ((size + (n - 1)) / n);
}

/**
 * @brief Packs the `size` and `alloc` of a block into a word suitable for
 *        use as a packed value.
 *
 * Packed values are used for both headers and footers.
 *
 * The allocation status is packed into the lowest bit of the word.
 *
 * @param[in] size The size of the block being represented
 * @param[in] alloc True if the block is allocated
 * @return The packed value
 */
static word_t pack(size_t size, bool alloc, bool pre_alloc, bool pre_mini) {
    word_t word = size;
    if (alloc) {
        word |= alloc_mask;
    }
    if (pre_alloc) {
        word |= pre_alloc_mask;
    }
    if (pre_mini) {
        word |= pre_mini_mask;
    }
    return word;
}

/**
 * @brief Extracts the size represented in a packed word.
 *
 * This function simply clears the lowest 4 bits of the word, as the heap
 * is 16-byte aligned.
 *
 * @param[in] word
 * @return The size of the block represented by the word
 */
static size_t extract_size(word_t word) {
    return (word & size_mask);
}

/**
 * @brief Extracts the size of a block from its header.
 * @param[in] block
 * @return The size of the block
 */
static size_t get_size(block_t *block) {
    return extract_size(block->header);
}

/**
 * @brief Given a payload pointer, returns a pointer to the corresponding
 *        block.
 * @param[in] bp A pointer to a block's payload
 * @return The corresponding block
 */
static block_t *payload_to_header(void *bp) {
    return (block_t *)((char *)bp - offsetof(block_t, payload));
}

/**
 * @brief Given a block pointer, returns a pointer to the corresponding
 *        payload.
 * @param[in] block
 * @return A pointer to the block's payload
 * @pre The block must be a valid block, not a boundary tag.
 */
static void *header_to_payload(block_t *block) {
    dbg_requires(get_size(block) != 0);
    return (void *)(block->payload);
}

/**
 * @brief Given a block pointer, returns a pointer to the corresponding
 *        footer.
 * @param[in] block
 * @return A pointer to the block's footer
 * @pre The block must be a valid block, not a boundary tag.
 */
static word_t *header_to_footer(block_t *block) {
    dbg_requires(get_size(block) != 0 &&
                 "Called header_to_footer on the epilogue block");
    return (word_t *)(block->payload + get_size(block) - dsize);
}

/**
 * @brief Given a block footer, returns a pointer to the corresponding
 *        header.
 * @param[in] footer A pointer to the block's footer
 * @return A pointer to the start of the block
 * @pre The footer must be the footer of a valid block, not a boundary tag.
 */
static block_t *footer_to_header(word_t *footer) {
    size_t size = extract_size(*footer);
    dbg_assert(size != 0 && "Called footer_to_header on the prologue block");
    return (block_t *)((char *)footer + wsize - size);
}

/**
 * @brief Returns the payload size of a given block.
 *
 * The payload size is equal to the entire block size minus the sizes of the
 * block's header and footer.
 *
 * @param[in] block
 * @return The size of the block's payload
 */
static size_t get_payload_size(block_t *block) {
    size_t asize = get_size(block);
    return asize - wsize;
}

/**
 * @brief Returns the allocation status of a given header value.
 *
 * This is based on the lowest bit of the header value.
 *
 * @param[in] word
 * @return The allocation status correpsonding to the word
 */
static bool extract_alloc(word_t word) {
    return (bool)(word & alloc_mask);
}

/**
 * @brief Returns the allocation status of a block, based on its header.
 * @param[in] block
 * @return The allocation status of the block
 */
static bool get_alloc(block_t *block) {
    return extract_alloc(block->header);
}

/**
 * @brief Returns the allocation status of a previous block of a given header
 * value
 *
 * @param word
 * @return The previous block allocation status correpsonding to the word
 */
static bool extract_prev_alloc(word_t word) {
    return (bool)((word & pre_alloc_mask) >> 1);
}

/**
 * @brief Returns the allocation status of a previous block, based on header of
 * the current block
 *
 * @param block
 * @return The allocation status of the previous sblock
 */
static bool get_prev_alloc(block_t *block) {
    return extract_prev_alloc(block->header);
}

/**
 * @brief Check if the previous block is a mini-block of a given header value
 * @param word
 * @return The previous block is mini block or not correpsonding to the word
 */
static bool extract_prev_mini(word_t word) {
    return (bool)((word & pre_mini_mask) >> 2);
}

/**
 * @brief Check if the previous block is a mini-block
 * @param block
 * @return The previous block is mini block or not
 */
static bool get_prev_mini(block_t *block) {
    return extract_prev_mini(block->header);
}

/**
 * @brief Writes an epilogue header at the given address.
 *
 * The epilogue header has size 0, and is marked as allocated.
 *
 * @param[out] block The location to write the epilogue header
 */
static void write_epilogue(block_t *block) {
    dbg_requires(block != NULL);
    dbg_requires((char *)block == mem_heap_hi() - 7);
    block->header = pack(0, true, false, false);
}

/**
 * @brief Finds the next consecutive block on the heap.
 *
 * This function accesses the next block in the "implicit list" of the heap
 * by adding the size of the block.
 *
 * @param[in] block A block in the heap
 * @return The next consecutive block on the heap
 * @pre The block is not the epilogue
 */
static block_t *find_next(block_t *block) {
    dbg_requires(block != NULL);
    dbg_requires(get_size(block) != 0 &&
                 "Called find_next on the last block in the heap");
    return (block_t *)((char *)block + get_size(block));
}

/**
 * @brief write footer for a non-min free block
 *
 * @param block block pointer
 * @param size size of the current block
 * @param alloc indicate if this is a allocated block
 * @param pre_alloc indicate if previous is a allocated block
 * @param pre_mini indicate if previous is a mini block
 */
static void write_footer(block_t *block, size_t size, bool alloc,
                         bool pre_alloc, bool pre_mini) {
    word_t *footerp = header_to_footer(block);
    *footerp = pack(size, alloc, pre_alloc, pre_mini);
}

/**
 * @brief write header of a block based on previous block condition
 *  (allocated, mini-block), write footer if it is a non-mini free block
 */
static void write_header_footer(block_t *block, bool pre_alloc, bool pre_mini) {
    size_t size = get_size(block);
    bool alloc = get_alloc(block);
    block->header = pack(size, alloc, pre_alloc, pre_mini);
    // need to write a footer for non-mini free block
    if ((!alloc) && (size > min_block_size)) {
        write_footer(block, size, alloc, pre_alloc, pre_mini);
    }
}

/**
 * @brief Writes a block starting at the given address.
 *
 * This function writes both a header and footer, where the location of the
 * footer is computed in relation to the header.
 *
 * @param[out] block The location to begin writing the block header
 * @param[in] size The size of the new block
 * @param[in] alloc The allocation status of the new block
 */
static void write_block(block_t *block, size_t size, bool alloc, bool pre_alloc,
                        bool pre_mini) {
    dbg_requires(block != NULL);
    dbg_requires(size > 0);
    bool mini = (size == min_block_size);
    block->header = pack(size, alloc, pre_alloc, pre_mini);
    // need to write a footer for non-mini free block
    if (!alloc && (size > min_block_size)) {
        write_footer(block, size, alloc, pre_alloc, pre_mini);
    }

    // update the next block when you change the current block
    block_t *block_next = find_next(block);
    write_header_footer(block_next, alloc, mini);
}

/**
 * @brief Finds the footer of the previous block on the heap.
 * @param[in] block A block in the heap
 * @return The location of the previous block's footer
 */
static word_t *find_prev_footer(block_t *block) {
    // Compute previous footer position as one word before the header
    return &(block->header) - 1;
}

/**
 * @brief Finds the previous consecutive block on the heap.
 *
 * This is the previous block in the "implicit list" of the heap.
 * If the function is called on the first block in the heap, NULL will be
 * returned, since the first block in the heap has no previous block!
 * The position of the previous block is found by reading the previous
 * block's footer to determine its size, then calculating the start of the
 * previous block based on its size.
 *
 * @param[in] block A block in the heap
 * @return The previous consecutive block in the heap.
 */
static block_t *find_prev(block_t *block) {
    dbg_requires(block != NULL);
    // when previous block is a mini block
    if (get_prev_mini(block)) {
        return (block_t *)((char *)block - min_block_size);
    }

    word_t *footerp = find_prev_footer(block);
    // When the previous block is a prologue
    if (extract_size(*footerp) == 0) {
        return NULL;
    }

    // when previous block is not a mini block
    return footer_to_header(footerp);
}

/**
 * @brief find the suitable segregated free list according to the size
 * of the free block
 *  range = 2^(index+4-1)+1 ~ 2^(index+4)
 *  seg index        list block size range
 *      0                       16
 *      1                  17 ~ 32
 *      2                  33 ~ 64
 *      3                  65 ~ 128
 *      4                 129 ~ 256
 *      5                 257 ~ 512
 *      6                 513 ~ 1024
 *      7                1025 ~ 2048
 *      8                2049 ~ 4096
 *      9                4097 ~ 8192
 *      10               8193 ~ 16,384
 *      11             16,385 ~ 32,768
 *      12             32,769 ~ 65,536
 *      13             65,537 ~ 131,072
 *      14                   >= 131,073
 * @pre size of the free block must be larger than the mini_block_size
 */
static size_t find_seg_list(size_t asize) {
    dbg_requires(asize >= min_block_size); // check size

    for (size_t i = 0; i < num_buckets - 1; i++) {
        if (asize <= (1 << (i + min_bucket_pow))) {
            return i;
        }
    }
    // when size is even larger than 2^18 = 131,073
    // put them all into bucket 14
    return num_buckets - 1;
}

/*
 * ---------------------------------------------------------------------------
 *                        END SHORT HELPER FUNCTIONS
 * ---------------------------------------------------------------------------
 */

/******** The remaining content below are helper and debug routines ********/

/**
 * @brief print address and size for each bloch in the heap
 */
static void print_heap() {
    dbg_printf("Heap Start here\n");
    for (block_t *block = heap_start; block < (heap_start + mem_heapsize());
         block = find_next(block)) {
        dbg_printf("block start at %p, size %ld \n", block, get_size(block));
    }
    dbg_printf("Heap End here\n");
}

/**
 * @brief Insert a free block into corresponding free bucket list according to
 * the free block size using LIFO strategy. Notice that the miniblock is in a
 * singly linked list, and other 14 free lists are doubly linked list.
 * @return void
 * @pre block is not NULL, block is marked as free
 */
static void free_list_insert(block_t *block) {
    dbg_requires(block != NULL);     // check NULL
    dbg_requires(!get_alloc(block)); // check free

    // Find a suitable bucket list for this free block
    size_t index = find_seg_list(get_size(block));

    // When this free block should be insert into the index 0 mini-block free
    // list note the mini-block free list is a singly linked list
    if (index == 0) {
        // when the mini-block list is empty
        if (bucket_list[index] == NULL) {
            block->next = NULL;
            bucket_list[index] = block;
            return;
        }
        // when the mini-block list is not empty
        block->next = bucket_list[index];
        bucket_list[index] = block;
    }
    // When this free block should be insert into other free lists except
    // mini-block free list
    else {
        // When the bucket find is empty, initialize the bucket with current
        // block
        if (bucket_list[index] == NULL) {
            block->next = NULL;
            block->prev = NULL;
            bucket_list[index] = block;
            return;
        }

        // When the bucket is not empty
        bucket_list[index]->prev = block;
        block->next = bucket_list[index];
        block->prev = NULL;
        bucket_list[index] = block;
    }
}

/**
 * @brief Delete a free block from its bucket list
 * @pre block is not NULL, block is free
 */
static void free_list_remove(block_t *block) {
    dbg_requires(block != NULL); // check NULL

    // Find this free block is in which bucket free list
    size_t index = find_seg_list(get_size(block));
    // This bucket free list cannot be an empty list
    dbg_assert(bucket_list[index] != NULL);

    // when this free block should be insert into the index 0 mini-block free
    // list note the mini-block free list is a singly linked list
    if (index == 0) {
        // When this block is the first in the list
        if (bucket_list[index] == block) {
            bucket_list[index] = block->next;
        }
        // When this block is not the first in the list
        else {
            block_t *node = bucket_list[index];
            block_t *tmp;
            // search through the whole list
            while ((node != NULL) && (node != block)) {
                tmp = node;
                node = node->next;
            }
            tmp->next = node->next;
        }
    }
    // When this free block should be insert into other free lists except
    // mini-block free list
    else {
        // When this bucket list only has one block
        if ((block->next == NULL) && (block->prev == NULL)) {
            bucket_list[index] = NULL;
        }
        // When this bucket list has many blocks
        else {
            // when this block is the first of the list
            if (block->prev == NULL) {
                block->next->prev = NULL;
                bucket_list[index] = block->next;
                block->next = NULL;
            }
            // when the block is the end of the list
            else if (block->next == NULL) {
                block->prev->next = NULL;
                block->next = NULL;
            }
            // when this block is in the middle
            else {
                block->prev->next = block->next;
                block->next->prev = block->prev;
                block->next = NULL;
                block->prev = NULL;
            }
        }
    }
}

/**
 * @brief Merge two consecutive free space in the heap into one free block to
 * decrease fragmentation.
 *
 * @param[in] block
 * @return
 */
static block_t *coalesce_block(block_t *block) {

    block_t *prev_block = find_prev(block);
    bool pre_alloc = get_prev_alloc(block);  // get block previous alloc bit
    block_t *next_block = find_next(block);  // get next block address
    bool next_alloc = get_alloc(next_block); // get next block alloc bit

    bool pre_pre_alloc; // get the previous of the previous block alloc bit
    bool pre_mini = get_prev_mini(block);
    bool pre_pre_mini; // get the previous of the previous block mini bit

    size_t cur_size = get_size(block); // get the size of the current block

    /* Case 1: allocated + free + allocated */
    if (pre_alloc && next_alloc) {
        // nothing to coalsece, just return the current block
        write_block(block, cur_size, false, pre_alloc, pre_mini);
        return block;
    }
    /* Case 2: allocated + free + free */
    else if (pre_alloc && (!next_alloc)) {
        // To merge current free block with next free block
        // Change the header and footer (size) of current block
        cur_size = cur_size + get_size(next_block); // get the new size
        free_list_remove(next_block);
        write_block(block, cur_size, false, pre_alloc, pre_mini);
    }
    /* Case 3: free + free + allocated */
    else if ((!pre_alloc) && (next_alloc)) {
        // To merge the current free block with the pre free block
        cur_size = cur_size + get_size(prev_block); // get the new size
        pre_pre_alloc = get_prev_alloc(prev_block);
        pre_pre_mini = get_prev_mini(prev_block);
        free_list_remove(prev_block);
        // Change the pointer to previous block
        block = prev_block;
        // Change the header and footer (size) of the coalesed block
        write_block(block, cur_size, false, pre_pre_alloc, pre_pre_mini);
    }
    /* Case 4: free + free + free */
    else {
        // To merge the current free block with the pre and nextfree block
        cur_size = cur_size + get_size(prev_block) + get_size(next_block);
        pre_pre_alloc = get_prev_alloc(prev_block);
        pre_pre_mini = get_prev_mini(prev_block);
        free_list_remove(prev_block);
        free_list_remove(next_block);
        // Change the pointer to the previous block
        block = prev_block;
        // Change the header and footer of the coalesed block
        write_block(block, cur_size, false, pre_pre_alloc, pre_pre_mini);
    }

    return block;
}

/**
 * @brief extend the heap when initialize the heap or when there is not enough
 * space in the heap
 *
 * @param[in] size
 * @return
 */
static block_t *extend_heap(size_t size) {
    void *bp;
    bool prev_alloc = get_prev_alloc(payload_to_header(mem_sbrk(0)));
    bool prev_mini = get_prev_mini(payload_to_header(mem_sbrk(0)));

    // Allocate an even number of words to maintain alignment
    size = round_up(size, dsize);
    if ((bp = mem_sbrk(size)) == (void *)-1) {
        return NULL;
    }

    // Initialize free block header/footer
    block_t *block = payload_to_header(bp); // to cover the old epilogue
    block_t *next_block = (block_t *)((char *)block + size);
    write_epilogue(next_block);
    write_block(block, size, false, prev_alloc, prev_mini);

    // Coalesce in case the previous block was free
    block = coalesce_block(block);
    free_list_insert(block);
    return block;
}

/**
 * @brief Try to split a free block out, each time after a free block is
 * allocted, to decrease fragmentation.
 *
 * @param[in] block
 * @param[in] asize
 */
static block_t *split_block(block_t *block, size_t asize) {
    dbg_requires(get_alloc(block));

    size_t block_size = get_size(block);

    // when allocated block has space larger than min_block_size, it can split a
    // free block out
    if ((block_size - asize) >= min_block_size) {
        block_t *block_next = (block_t *)((char *)block + asize);
        bool pre_alloc = get_prev_alloc(block);
        bool pre_mini = get_prev_mini(block);
        bool mini = (asize == min_block_size);

        // write the seperated free block first
        write_block(block_next, block_size - asize, false, true, mini);
        // update the allocated block
        write_block(block, asize, true, pre_alloc, pre_mini);
        block_next = find_next(block);
        return block_next;
    }

    dbg_ensures(get_alloc(block));
    return NULL;
}

/**
 * @brief Using the better fit strategy that try to approac the best fit
 * strategy. Set with a timer, that gives you 5 times to find a smaller and
 * suitable free block after you alreay find you first fit
 * @param[in] asize
 * @return
 */
static block_t *find_fit(size_t asize) {
    size_t index = find_seg_list(asize);

    for (size_t i = index; i < num_buckets; i++) {
        block_t *better_fit = NULL;
        // gives 6 chances to search for a better fit block
        size_t timer = find_fit_time_limits;
        block_t *block;
        for (block = bucket_list[i]; block != NULL; block = block->next) {
            if (asize <= get_size(block)) {
                // initial the better_fit
                if (better_fit == NULL) {
                    better_fit = block;
                }
                // have a smaller block size choice
                if (get_size(block) < get_size(better_fit)) {
                    better_fit = block;
                }
                // use one search time
                timer = timer - 1;
            }
            // time used up, not allowed for further search
            if (timer == 0) {
                break;
            }
        }
        // if found suitable block in this bucket list
        if (better_fit != NULL) {
            return better_fit;
        }
    }
    return NULL; // not found
}

/**
 * @brief Check Prologue and Epilogue. Called by mm_checkheap
 *
 * @param line
 * @return true prologue and epilogue is correct
 * @return false prologue and epilogue is correct
 */
static bool check_prologue_epilogue(int line) {
    block_t *prologue = (block_t *)mem_heap_lo();
    block_t *epilogue = (block_t *)(mem_heap_hi() - 7);
    // check prologue
    if (get_size(prologue) != 0 || get_alloc(prologue) == false) {
        dbg_printf("[line %d]: Prologue is incorrrect.\n", line);
        return false;
    }
    // check epilogue
    if (get_size(epilogue) != 0 || get_alloc(epilogue) == false) {
        dbg_printf("[line %d]: Epilogue is incorrrect.\n", line);
        return false;
    }
    return true;
}
/**
 * @brief Check each block's address alignment
 * @param line
 * @param block
 * @return true
 * @return false
 */
static bool check_align(int line, block_t *block) {
    if ((!get_prev_mini(block)) && (get_size(block) % dsize != 0)) {
        dbg_printf("[line %d]: Block size %lu not aligned.\n", line,
                   get_size(block));
        return false;
    }
    return true;
}

/**
 * @brief Check blocks lie within heap boundaries
 * @param line
 * @param block
 * @return true
 * @return false
 */
static bool check_boundary(int line, block_t *block) {
    if (block < heap_start || block > (block_t *)(mem_heap_hi() - 7L)) {
        dbg_printf("[line: %d]: Block %p is out of the bound\n", line,
                   (void *)block);
        return false;
    }
    return true;
}

/**
 * @brief Check each block's header and footer matching, check block and
 *      next block matching
 * @param line
 * @param block
 * @return true
 * @return false
 */
static bool check_head_foot_matching(int line, block_t *block) {
    if (!get_alloc(block) && (get_size(block) != min_block_size)) {
        word_t *footer = (word_t *)header_to_footer(block);
        block_t *block_next = find_next(block);

        // check size bit
        if (get_size(block) != extract_size(*footer)) {
            dbg_printf("[line %d]: Block %p header and footer displays "
                       "different alloc flag.\n",
                       line, block);
            return false;
        }
        // check alloc bit
        if (get_alloc(block) != extract_alloc(*footer)) {
            dbg_printf("[line %d]: Block %p header and footer displays "
                       "different block size.\n",
                       line, block);
            return false;
        }

        // check pre alloc bit
        if (get_alloc(block) != get_prev_alloc(block_next)) {
            dbg_printf("[line %d]: Block %p header and footer displays "
                       "different block size.\n",
                       line, block);
            return false;
        }
    }
    return true;
}

/**
 * @brief Check block size
 *
 * @param line
 * @param block
 * @return true
 * @return false
 */
static bool check_block_size(int line, block_t *block) {
    if (get_size(block) < min_block_size) {
        dbg_printf(
            "[line: %d]: Block %p is smaller than the min_block_size %lu\n",
            line, (void *)block, min_block_size);
        return false;
    }
    return true;
}

/**
 * @brief Check coalescing: no consecutive free blocks in the heap
 *
 * @param line
 * @param block
 * @return true
 * @return false
 */
static bool check_coalesce(int line, block_t *block) {
    if (!get_alloc(block) && !get_alloc(find_next(block))) {
        dbg_printf("[line: %d]: Two consecutive free blocks: %p and %p\n ",
                   line, (void *)block, (void *)find_next(block));
        return false;
    }
    return true;
}
/**
 * @brief Check All next/previous pointers are consistent (if A next pointer
 * points to B, B previous pointer should point to A).
 * @param line
 * @param block
 * @return true
 * @return false
 */
static bool check_pointer_matching(int line, block_t *block) {
    if (!get_alloc(block)) {
        if (get_size(block) != min_block_size) {
            // Check A prev pointer points to B, B next pointer points to A
            if (block->prev != NULL) {
                if (block != block->prev->next) {
                    dbg_printf("[line %d]: Pointer %p is incorrrect.\n", line,
                               (void *)block);
                    return false;
                }
            }
            // Check A next pointer points to B, B prev pointer points to A
            if (block->next != NULL) {
                if (block != block->next->prev) {
                    dbg_printf("[line %d]: Pointer %p is incorrrect.\n", line,
                               (void *)block);
                    return false;
                }
            }
        }
    }
    return true;
}

/**
 * @brief Check if free list pointers are between mem heap lo() and mem heap
 * high().
 *
 * @param line
 * @param i
 * @param block
 */
static bool check_pointer_range(int line, size_t i, block_t *block) {
    if (i == 0) {
        if (((void *)block < mem_heap_lo()) ||
            ((void *)block > mem_heap_hi())) {
            dbg_printf("[line %d]: Miniblock pointer %p is incorrrect.\n", line,
                       (void *)block);
            return false;
        }
    } else {
        if (((void *)block < mem_heap_lo()) ||
            ((void *)block > mem_heap_hi())) {
            if (bucket_list[i] != block) {
                if (((void *)block->prev < mem_heap_lo()) ||
                    ((void *)block->prev > mem_heap_hi())) {
                    dbg_printf(
                        "[line %d]: Block prev pointer %p is incorrect.\n",
                        line, (void *)block->prev);
                    return false;
                }
            }
            dbg_printf("[line %d]: Block next pointer %p is incorrect.\n", line,
                       (void *)block->prev);
            return false;
        }
    }
    return true;
}
/**
 * @brief Check important information about blocks and pointers in the heap when
 * in the bdg mode
 * @param[in] line line that call this function
 * @return
 */
bool mm_checkheap(int line) {
    dbg_printf("\n[line %d] Checking...\n", line);

    // 1. Check Prologue and Epilogue
    if (!check_prologue_epilogue(line)) {
        return false;
    }

    int heap_free_count = 0;   // counter for free blocks in heap
    int bucket_free_count = 0; // counter for free blocks in bucket

    for (block_t *block = heap_start; get_size(block) != 0;
         block = find_next(block)) {
        // 2. Check each block's address alignment
        if (!check_align(line, block)) {
            return false;
        }

        // 3. Check blocks lie within heap boundaries
        if (!check_boundary(line, block)) {
            return false;
        }

        // 4. Check each block's header and footer matching, check block and
        // next block matching
        if (!check_head_foot_matching(line, block)) {
            return false;
        }

        // 5. Check block size
        if (!check_block_size(line, block)) {
            return false;
        }

        // 6. Check coalescing: no consecutive free blocks in the heap
        if (!check_coalesce(line, block)) {
            return false;
        }

        // 7. Check next/previous pointer matching
        if (!check_pointer_matching(line, block)) {
            return false;
        }

        // heap free count
        if (!get_alloc(block)) {
            heap_free_count = heap_free_count + 1;
        }
    }

    for (size_t i = 0; i < num_buckets; i++) {
        for (block_t *block = bucket_list[i]; block != NULL;
             block = block->next) {
            bucket_free_count = bucket_free_count + 1;

            // 8. Check if free list pointers are between mem heap lo() and mem
            // heap high().
            if (!check_pointer_range(line, i, block)) {
                return false;
            }

            // 9. Check blocks in each list bucket fall within bucket size
            // range.
            if (find_seg_list(get_size(block)) != i) {
                dbg_printf(
                    "[line %d]: Free block %p. of size %lu should not be "
                    "in %luth bucket.\n",
                    line, (void *)block, get_size(block), i);
                return false;
            }
        }
    }

    // 10. Count free blocks by iterating through every block and traversing
    // free list by pointers and see if they match.
    if (bucket_free_count != heap_free_count) {
        dbg_printf("[line %d]: Free lists in heap are %d, Free lists in "
                   "buckets are %d, does not match.\n",
                   line, heap_free_count, bucket_free_count);
        return false;
    }
    return true;
}

/**
 * @brief Inialize the heap when malloc is called the first time.
 *
 * @return
 */
bool mm_init(void) {
    // Create the initial empty heap
    word_t *start = (word_t *)(mem_sbrk(2 * wsize));

    if (start == (void *)-1) {
        return false;
    }

    start[0] = pack(0, true, false, false); // Heap prologue (block footer)
    start[1] = pack(0, true, true, false);  // Heap epilogue (block header)

    // Heap starts with first "block header", currently the epilogue
    heap_start = (block_t *)&(start[1]);
    for (int i = 0; i < 15; i++) {
        bucket_list[i] = NULL;
    }

    // Extend the empty heap with a free block of chunksize bytes
    if (extend_heap(chunksize) == NULL) {
        return false;
    }
    return true;
}

/**
 * @brief calling malloc with size will return a block in the heap.
 *
 * @param[in] size
 * @return
 */
void *malloc(size_t size) {
    dbg_requires(mm_checkheap(__LINE__));

    size_t asize;      // Adjusted block size
    size_t extendsize; // Amount to extend heap if no fit is found
    block_t *block;
    void *bp = NULL;

    // Initialize heap if it isn't initialized
    if (heap_start == NULL) {
        mm_init();
    }

    // Ignore spurious request
    if (size == 0) {
        dbg_ensures(mm_checkheap(__LINE__));
        return bp;
    }

    // Adjust block size to include overhead and to meet alignment requirements
    if (size > wsize) {
        asize = round_up(size + wsize, dsize);
    } else {
        asize = min_block_size;
    }

    // Search the free list for a fit
    block = find_fit(asize);

    // If no fit is found, request more memory, and then and place the block
    if (block == NULL) {
        // Always request at least chunksize
        extendsize = max(asize, chunksize);
        block = extend_heap(extendsize);
        // extend_heap returns an error
        if (block == NULL) {
            return bp;
        }
    }

    // The block should be marked as free
    dbg_assert(!get_alloc(block));
    // Mark the found free block as allocated
    write_block(block, get_size(block), true, get_prev_alloc(block),
                get_prev_mini(block));
    // Always remove block from free list when allocated
    free_list_remove(block);
    // Check if the allocated block has some free space left
    block_t *left_block = split_block(block, asize);
    if (left_block != NULL) {
        // Insert this seperated free block back the free list
        free_list_insert(left_block);
    }

    bp = header_to_payload(block);
    dbg_ensures(mm_checkheap(__LINE__));
    return bp;
}

/**
 * @brief freeing space in heap that created by malloc
 *
 * @param[in] bp
 */
void free(void *bp) {
    dbg_requires(mm_checkheap(__LINE__));
    dbg_printf("CALLING FREE\n");
    if (bp == NULL) {
        return;
    }

    block_t *block = payload_to_header(bp);
    size_t size = get_size(block);

    // The block should be marked as allocated
    dbg_assert(get_alloc(block));

    // Mark the block as free
    write_block(block, size, false, get_prev_alloc(block),
                get_prev_mini(block));

    // Try to coalesce the block with its neighbors
    block = coalesce_block(block);
    // Insert the freed block back to free list
    free_list_insert(block);

    dbg_ensures(mm_checkheap(__LINE__));
}

/**
 * @brief resize allocated memory
 *
 * This function is used to resize the memory block
 * which is allocated bny malloc or calloc before,
 * and returns a pointer to the new, resized memory.
 *
 * @param[in] ptr
 * @param[in] size
 * @return a pointer to the resized memory
 */
void *realloc(void *ptr, size_t size) {
    block_t *block = payload_to_header(ptr);
    size_t copysize;
    void *newptr;

    // If size == 0, then free block and return NULL
    if (size == 0) {
        free(ptr);
        return NULL;
    }

    // If ptr is NULL, then equivalent to malloc
    if (ptr == NULL) {
        return malloc(size);
    }

    // Otherwise, proceed with reallocation
    newptr = malloc(size);

    // If malloc fails, the original block is left untouched
    if (newptr == NULL) {
        return NULL;
    }

    // Copy the old data
    copysize = get_payload_size(block); // gets size of old payload
    if (size < copysize) {
        copysize = size;
    }
    memcpy(newptr, ptr, copysize);

    // Free the old block
    free(ptr);

    return newptr;
}

/**
 * @brief allocate memory and set its content to 0
 *
 * This function allocate the requested memory, initialize its content to 0,
 * and returns a pointer to the block payload.
 *
 * @param[in] elements
 * @param[in] size
 * @return pointer to the payload of the block allocated
 */
void *calloc(size_t elements, size_t size) {
    void *bp;
    size_t asize = elements * size;

    if (elements == 0) {
        return NULL;
    }
    if (asize / elements != size) {
        // Multiplication overflowed
        return NULL;
    }

    bp = malloc(asize);
    if (bp == NULL) {
        return NULL;
    }

    // Initialize all bits to 0
    memset(bp, 0, asize);

    return bp;
}

/*
 *****************************************************************************
 * Do not delete the following super-secret(tm) lines!                       *
 *                                                                           *
 * 53 6f 20 79 6f 75 27 72 65 20 74 72 79 69 6e 67 20 74 6f 20               *
 *                                                                           *
 * 66 69 67 75 72 65 20 6f 75 74 20 77 68 61 74 20 74 68 65 20               *
 * 68 65 78 61 64 65 63 69 6d 61 6c 20 64 69 67 69 74 73 20 64               *
 * 6f 2e 2e 2e 20 68 61 68 61 68 61 21 20 41 53 43 49 49 20 69               *
 *                                                                           *
 * 73 6e 27 74 20 74 68 65 20 72 69 67 68 74 20 65 6e 63 6f 64               *
 * 69 6e 67 21 20 4e 69 63 65 20 74 72 79 2c 20 74 68 6f 75 67               *
 * 68 21 20 2d 44 72 2e 20 45 76 69 6c 0a c5 7c fc 80 6e 57 0a               *
 *                                                                           *
 *****************************************************************************
 */
