#ifndef HASHMAP_H
#define HASHMAP_H

#include <cuda_runtime.h>
#include <cstring>  // for strcmp, strcpy, strlen
#include <cmath>    // for log10f

const int MAX_NAME_LEN = 64;
const int TABLE_SIZE = 1048576;  // 1M, adjust as needed

struct Entry {
    char username[MAX_NAME_LEN];
    float score;
    int lock;
    bool occupied;
};

__device__ Entry* table;
__device__ int global_lock = 0;

__device__ unsigned int hash(const char* str) {
    unsigned int h = 0;
    while (*str) {
        h = h * 31 + *str++;
    }
    return h % TABLE_SIZE;
}

__device__ void acquire(int* l) {
    while (atomicCAS(l, 0, 1) != 0);
    // memory fence if needed, but for simplicity omit
}

__device__ void release(int* l) {
    atomicExch(l, 0);
}

__device__ int find(const char* username) {
    unsigned int idx = hash(username);
    for (int i = 0; i < TABLE_SIZE; ++i) {
        int probe = (idx + i) % TABLE_SIZE;
        if (!table[probe].occupied) {
            return -1;  // not found
        }
        if (strcmp(table[probe].username, username) == 0) {
            return probe;
        }
    }
    return -1;  // table full, error
}

__device__ int find_empty(unsigned int start) {
    for (int i = 0; i < TABLE_SIZE; ++i) {
        int probe = (start + i) % TABLE_SIZE;
        if (!table[probe].occupied) {
            return probe;
        }
    }
    return -1;  // full
}

__device__ void add_score(const char* username, float delta) {
    int idx = find(username);
    if (idx != -1) {
        acquire(&table[idx].lock);
        table[idx].score += delta;
        release(&table[idx].lock);
        return;
    }

    // Not found, acquire global lock
    acquire(&global_lock);

    // Double-check
    idx = find(username);
    if (idx != -1) {
        release(&global_lock);
        acquire(&table[idx].lock);
        table[idx].score += delta;
        release(&table[idx].lock);
    } else {
        // Insert
        unsigned int h = hash(username);
        int empty_idx = find_empty(h);
        if (empty_idx == -1) {
            // Error: table full
            release(&global_lock);
            return;
        }
        strcpy(table[empty_idx].username, username);
        table[empty_idx].score = delta;
        table[empty_idx].lock = 0;
        table[empty_idx].occupied = true;
        release(&global_lock);
    }
}

#endif