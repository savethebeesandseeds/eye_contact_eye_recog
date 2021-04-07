#include <stdio.h>
#include <string.h>
#include <stdint.h>

#define CHUNK_SIZE 64
#define TOTAL_LEN_LEN 8

/*
 * Comments from pseudo-code at https://en.wikipedia.org/wiki/SHA-2 are reproduced here.
 * When useful for clarification, portions of the pseudo-code are reproduced here too.
 */

/*
 * Initialize array of round constants:
 * (first 32 bits of the fractional parts of the cube roots of the first 64 primes 2..311):
 */
static const uint32_t k[] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

struct buffer_state {
    const uint8_t * p;
    size_t len;
    size_t total_len;
    int single_one_delivered;
    int total_len_delivered;
};

static inline uint32_t right_rot(uint32_t value, unsigned int count)
{
    /*
     * Defined behaviour in standard C for all count where 0 < count < 32,
     * which is what we need here.
     */
    return value >> count | value << (32 - count);
}

static void init_buf_state(struct buffer_state * state, const void * input, size_t len)
{
    state->p = input;
    state->len = len;
    state->total_len = len;
    state->single_one_delivered = 0;
    state->total_len_delivered = 0;
}

static int calc_chunk(uint8_t chunk[CHUNK_SIZE], struct buffer_state * state)
{
    size_t space_in_chunk;

    if (state->total_len_delivered) {
        return 0;
    }

    if (state->len >= CHUNK_SIZE) {
        memcpy(chunk, state->p, CHUNK_SIZE);
        state->p += CHUNK_SIZE;
        state->len -= CHUNK_SIZE;
        return 1;

    }

    memcpy(chunk, state->p, state->len);
    chunk += state->len;
    space_in_chunk = CHUNK_SIZE - state->len;
    state->p += state->len;
    state->len = 0;

    /* If we are here, space_in_chunk is one at minimum. */
    if (!state->single_one_delivered) {
        *chunk++ = 0x80;
        space_in_chunk -= 1;
        state->single_one_delivered = 1;
    }

    /*
     * Now:
     * - either there is enough space left for the total length, and we can conclude,
     * - or there is too little space left, and we have to pad the rest of this chunk with zeroes.
     * In the latter case, we will conclude at the next invokation of this function.
     */
    if (space_in_chunk >= TOTAL_LEN_LEN) {
        const size_t left = space_in_chunk - TOTAL_LEN_LEN;
        const size_t len = state->total_len * 8;
        memset(chunk, 0x00, left);
        chunk += left;

#if SIZE_MAX > UINT32_MAX
        chunk[0] = (uint8_t) (len >> 56);
        chunk[1] = (uint8_t) (len >> 48);
        chunk[2] = (uint8_t) (len >> 40);
        chunk[3] = (uint8_t) (len >> 32);
#else
        chunk[0] = 0;
        chunk[1] = 0;
        chunk[2] = 0;
        chunk[3] = 0;
#endif

#if SIZE_MAX > UINT16_MAX
        chunk[4] = (uint8_t) (len >> 24);
        chunk[5] = (uint8_t) (len >> 16);
#else
        chunk[4] = 0;
        chunk[5] = 0;
#endif

#if SIZE_MAX > UINT8_MAX
        chunk[6] = (uint8_t) (len >> 8);
#else
        chunk[6] = 0;
#endif

        chunk[7] = (uint8_t) len;

        state->total_len_delivered = 1;
    } else {
        memset(chunk, 0x00, space_in_chunk);
    }

    return 1;
}

/*
 * Limitations:
 * - len must be small enough for (8 * len) to fit in len. Otherwise, the results are unpredictable.
 * - sizeof size_t is assumed to be either 8, 16, 32 or 64. Otherwise, the results are unpredictable.
 * - Since input is a pointer in RAM, the data to hash should be in RAM, which could be a problem
 *   for large data sizes.
 * - SHA algorithms theoretically operate on bit strings. However, this implementation has no support
 *   for bit string lengths that are not multiples of eight, and it really operates on arrays of bytes.
 *   the len parameter is a number of bytes.
 */
void calc_sha_256(unsigned char hash[32], const void * input, size_t len)
{
    /*
     * Note 1: All variables are 32 bit unsigned integers and addition is calculated modulo 232
     * Note 2: For each round, there is one round constant k[i] and one entry in the message schedule array w[i], 0 = i = 63
     * Note 3: The compression function uses 8 working variables, a through h
     * Note 4: Big-endian convention is used when expressing the constants in this pseudocode,
     *     and when parsing message block data from bytes to words, for example,
     *     the first word of the input message "abc" after padding is 0x61626380
     */

    /*
     * Initialize hash values:
     * (first 32 bits of the fractional parts of the square roots of the first 8 primes 2..19):
     */

    uint32_t h0 = 0x6a09e667;
    uint32_t h1 = 0xbb67ae85;
    uint32_t h2 = 0x3c6ef372;
    uint32_t h3 = 0xa54ff53a;
    uint32_t h4 = 0x510e527f;
    uint32_t h5 = 0x9b05688c;
    uint32_t h6 = 0x1f83d9ab;
    uint32_t h7 = 0x5be0cd19;

    /* 512-bit chunks is what we will operate on. */

    uint8_t chunk[64];

    struct buffer_state state;

    init_buf_state(&state, input, len);

    while (calc_chunk(chunk, &state)) {
        uint32_t a, b, c, d, e, f, g, h;

        /*
         * create a 64-entry message schedule array w[0..63] of 32-bit words
         * (The initial values in w[0..63] don't matter, so many implementations zero them here)
         * copy chunk into first 16 words w[0..15] of the message schedule array
         */
        uint32_t w[64];
        const uint8_t *p = chunk;
        int i;

        memset(w, 0x00, sizeof w);
        for (i = 0; i < 16; i++) {
            w[i] = (uint32_t) p[0] << 24 | (uint32_t) p[1] << 16 |
                (uint32_t) p[2] << 8 | (uint32_t) p[3];
            p += 4;
        }

        /* Extend the first 16 words into the remaining 48 words w[16..63] of the message schedule array: */
        for (i = 16; i < 64; i++) {
            const uint32_t s0 = right_rot(w[i - 15], 7) ^ right_rot(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const uint32_t s1 = right_rot(w[i - 2], 17) ^ right_rot(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }

        /* Initialize working variables to current hash value: */
        a = h0;
        b = h1;
        c = h2;
        d = h3;
        e = h4;
        f = h5;
        g = h6;
        h = h7;

        /* Compression function main loop: */
        for (i = 0; i < 64; i++) {
            const uint32_t s1 = right_rot(e, 6) ^ right_rot(e, 11) ^ right_rot(e, 25);
            const uint32_t ch = (e & f) ^ (~e & g);
            const uint32_t temp1 = h + s1 + ch + k[i] + w[i];
            const uint32_t s0 = right_rot(a, 2) ^ right_rot(a, 13) ^ right_rot(a, 22);
            const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t temp2 = s0 + maj;

            h = g;
            g = f;
            f = e;
            e = d + temp1;
            d = c;
            c = b;
            b = a;
            a = temp1 + temp2;
        }

        /* Add the compressed chunk to the current hash value: */
        h0 = h0 + a;
        h1 = h1 + b;
        h2 = h2 + c;
        h3 = h3 + d;
        h4 = h4 + e;
        h5 = h5 + f;
        h6 = h6 + g;
        h7 = h7 + h;
    }

    /* Produce the final hash value (big-endian): */
    hash[0] = (uint8_t) (h0 >> 24);
    hash[1] = (uint8_t) (h0 >> 16);
    hash[2] = (uint8_t) (h0 >> 8);
    hash[3] = (uint8_t) h0;
    hash[4] = (uint8_t) (h1 >> 24);
    hash[5] = (uint8_t) (h1 >> 16);
    hash[6] = (uint8_t) (h1 >> 8);
    hash[7] = (uint8_t) h1;
    hash[8] = (uint8_t) (h2 >> 24);
    hash[9] = (uint8_t) (h2 >> 16);
    hash[10] = (uint8_t) (h2 >> 8);
    hash[11] = (uint8_t) h2;
    hash[12] = (uint8_t) (h3 >> 24);
    hash[13] = (uint8_t) (h3 >> 16);
    hash[14] = (uint8_t) (h3 >> 8);
    hash[15] = (uint8_t) h3;
    hash[16] = (uint8_t) (h4 >> 24);
    hash[17] = (uint8_t) (h4 >> 16);
    hash[18] = (uint8_t) (h4 >> 8);
    hash[19] = (uint8_t) h4;
    hash[20] = (uint8_t) (h5 >> 24);
    hash[21] = (uint8_t) (h5 >> 16);
    hash[22] = (uint8_t) (h5 >> 8);
    hash[23] = (uint8_t) h5;
    hash[24] = (uint8_t) (h6 >> 24);
    hash[25] = (uint8_t) (h6 >> 16);
    hash[26] = (uint8_t) (h6 >> 8);
    hash[27] = (uint8_t) h6;
    hash[28] = (uint8_t) (h7 >> 24);
    hash[29] = (uint8_t) (h7 >> 16);
    hash[30] = (uint8_t) (h7 >> 8);
    hash[31] = (uint8_t) h7;
}



/*
 * Simple MD5 implementation
 *
 * Compile with: gcc -o md5 -O3 -lm md5.c
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

// leftrotate function definition
#define LEFTROTATE(x, c) (((x) << (c)) | ((x) >> (32 - (c))))

// These vars will contain the hash
uint32_t h0, h1, h2, h3;

void md5(uint8_t *initial_msg, size_t initial_len) {

    // Message (to prepare)
    uint8_t *msg = NULL;

    // Note: All variables are unsigned 32 bit and wrap modulo 2^32 when calculating

    // r specifies the per-round shift amounts

    uint32_t r[] = {7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
                    5,  9, 14, 20, 5,  9, 14, 20, 5,  9, 14, 20, 5,  9, 14, 20,
                    4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
                    6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21};

    // Use binary integer part of the sines of integers (in radians) as constants// Initialize variables:
    uint32_t k[] = {
        0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee,
        0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501,
        0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be,
        0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821,
        0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa,
        0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
        0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed,
        0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a,
        0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c,
        0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70,
        0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05,
        0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
        0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039,
        0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
        0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1,
        0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391};

    h0 = 0x67452301;
    h1 = 0xefcdab89;
    h2 = 0x98badcfe;
    h3 = 0x10325476;

    // Pre-processing: adding a single 1 bit
    //append "1" bit to message
    /* Notice: the input bytes are considered as bits strings,
       where the first bit is the most significant bit of the byte.[37] */

    // Pre-processing: padding with zeros
    //append "0" bit until message length in bit ≡ 448 (mod 512)
    //append length mod (2 pow 64) to message

    int new_len = ((((initial_len + 8) / 64) + 1) * 64) - 8;

    msg = calloc(new_len + 64, 1); // also appends "0" bits
                                   // (we alloc also 64 extra bytes...)
    memcpy(msg, initial_msg, initial_len);
    msg[initial_len] = 128; // write the "1" bit

    uint32_t bits_len = 8*initial_len; // note, we append the len
    memcpy(msg + new_len, &bits_len, 4);           // in bits at the end of the buffer

    // Process the message in successive 512-bit chunks:
    //for each 512-bit chunk of message:
    int offset;
    for(offset=0; offset<new_len; offset += (512/8)) {

        // break chunk into sixteen 32-bit words w[j], 0 ≤ j ≤ 15
        uint32_t *w = (uint32_t *) (msg + offset);

#ifdef DEBUG
        printf("offset: %d %x\n", offset, offset);

        int j;
        for(j =0; j < 64; j++) printf("%x ", ((uint8_t *) w)[j]);
        puts("");
#endif

        // Initialize hash value for this chunk:
        uint32_t a = h0;
        uint32_t b = h1;
        uint32_t c = h2;
        uint32_t d = h3;

        // Main loop:
        uint32_t i;
        for(i = 0; i<64; i++) {

#ifdef ROUNDS
            uint8_t *p;
            printf("%i: ", i);
            p=(uint8_t *)&a;
            printf("%2.2x%2.2x%2.2x%2.2x ", p[0], p[1], p[2], p[3], a);

            p=(uint8_t *)&b;
            printf("%2.2x%2.2x%2.2x%2.2x ", p[0], p[1], p[2], p[3], b);

            p=(uint8_t *)&c;
            printf("%2.2x%2.2x%2.2x%2.2x ", p[0], p[1], p[2], p[3], c);

            p=(uint8_t *)&d;
            printf("%2.2x%2.2x%2.2x%2.2x", p[0], p[1], p[2], p[3], d);
            puts("");
#endif


            uint32_t f, g;

             if (i < 16) {
                f = (b & c) | ((~b) & d);
                g = i;
            } else if (i < 32) {
                f = (d & b) | ((~d) & c);
                g = (5*i + 1) % 16;
            } else if (i < 48) {
                f = b ^ c ^ d;
                g = (3*i + 5) % 16;
            } else {
                f = c ^ (b | (~d));
                g = (7*i) % 16;
            }

#ifdef ROUNDS
            printf("f=%x g=%d w[g]=%x\n", f, g, w[g]);
#endif
            uint32_t temp = d;
            d = c;
            c = b;
            //printf("rotateLeft(%x + %x + %x + %x, %d)\n", a, f, k[i], w[g], r[i]);
            b = b + LEFTROTATE((a + f + k[i] + w[g]), r[i]);
            a = temp;



        }

        // Add this chunk's hash to result so far:

        h0 += a;
        h1 += b;
        h2 += c;
        h3 += d;

    }

    // cleanup
    free(msg);

}

int main (void) {
	const char *s = "Rosetta code";
    const int SHA256_DIGEST_LENGTH = 32;
    const int msg_len = strlen(s);
    unsigned char hash[SHA256_DIGEST_LENGTH];
    calc_sha_256(hash, &s, msg_len);
	for (int i = 0; i < SHA256_DIGEST_LENGTH; i++){
        printf("%p", hash[i]);
        //printf("Hello, World!\n");
	}
	putchar('\n');

    // benchmark
    // int i;
    // for (i = 0; i < 1000000; i++) {
        md5((uint8_t *)s, msg_len);
    // }

    //var char digest[16] := h0 append h1 append h2 append h3 //(Output is in little-endian)
    uint8_t *p;

    // display result

    p=(uint8_t *)&h0;
    printf("%2.2x%2.2x%2.2x%2.2x", p[0], p[1], p[2], p[3], h0);

    p=(uint8_t *)&h1;
    printf("%2.2x%2.2x%2.2x%2.2x", p[0], p[1], p[2], p[3], h1);

    p=(uint8_t *)&h2;
    printf("%2.2x%2.2x%2.2x%2.2x", p[0], p[1], p[2], p[3], h2);

    p=(uint8_t *)&h3;
    printf("%2.2x%2.2x%2.2x%2.2x", p[0], p[1], p[2], p[3], h3);
    puts("");


	return 0;
}
