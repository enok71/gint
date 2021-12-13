/* -*- mode: c; c-basic-offset: 4; -*- */
/*******************************************************************************
 *
 * Copyright (c) 2020 Oskar Enoksson. All rights reserved.
 * Licensed under the MIT license. See LICENSE file in the project root for details.
 *
 * Description:
 * Implementation of arithmetic on unlimited polynomials over GF(2)
 *
 *******************************************************************************/

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <string.h>
#include <stdio.h>
#include <stdbool.h>

// Max value size
static Py_ssize_t PYGF2X_MAX_DIGITS = (9000000/PyLong_SHIFT);

// This number controls when malloc is used instead of automatic arrays
#define STATIC_LIMIT 32

// Define to enable DBG_ASSERT checks
//#define DEBUG_PYGF2X

// Define to enable DBG_PRINTF and DBG_PRINTF_DIGITS printouts
//#define DEBUG_PYGF2X_VERBOSE

// Define to restrict DBG_PRINTF and DBG_PRINTF_DIGITS printouts to a specific function
//#define DEBUG_PYGF2X_VERBOSE_FUNCTION "mul_nl_nr"
//#define DEBUG_PYGF2X_VERBOSE_FUNCTION "inverse"

#ifdef DEBUG_PYGF2X_VERBOSE_FUNCTION
#define DEBUG_PYGF2X_COND if(strcmp(__func__,DEBUG_PYGF2X_VERBOSE_FUNCTION)==0)
#else
#define DEBUG_PYGF2X_COND
#endif

#ifdef DEBUG_PYGF2X
#define DBG_ASSERT(x) { if (! (x) ) { fprintf(stderr,"Assertion failed on line %d: %s\n",__LINE__,#x); my_abort(); } }
#ifdef __GNUC__
#include <execinfo.h>
#endif
#include <stdlib.h>
static void my_abort() {
#ifdef __GNUC__
    void *array[3];
    size_t size;

    // get void*'s for all entries on the stack
    size = backtrace(array, 3);

    // print out all the frames to stderr
    backtrace_symbols_fd(array, size, STDERR_FILENO);
#endif
    exit(1);
}
#else
#define DBG_ASSERT(x)
#endif

#ifdef DEBUG_PYGF2X_VERBOSE
#define DBG_PRINTF(...) { DEBUG_PYGF2X_COND printf(__VA_ARGS__); }
#if (PyLong_SHIFT == 30)
#define DBG_PRINTF_DIGITS(msg,digits,ndigs) { DEBUG_PYGF2X_COND { DBG_PRINTF(msg); for(int i=(ndigs)-1; i>=0; i--) DBG_PRINTF("%08x'", (digits)[i]); DBG_PRINTF("\n"); }; }
#elif (PyLong_SHIFT == 15)
#define DBG_PRINTF_DIGITS(msg,digits,ndigs) { DEBUG_PYGF2X_COND { DBG_PRINTF(msg); for(int i=(ndigs)-1; i>=0; i--) DBG_PRINTF("%04x'", (digits)[i]); DBG_PRINTF("\n"); }; }
#else
#error
#endif
#else
#define DBG_PRINTF(...)
#define DBG_PRINTF_DIGITS(...)
#endif

#define GF2X_MAX(a,b) (((a)>(b)) ? (a) : (b))
#define GF2X_MIN(a,b) (((a)<(b)) ? (a) : (b))

#define LIMIT_DIV_BITWISE 0

static const uint16_t sqr_8[256];
static const uint16_t mul_5_5[32][32];

#if PyLong_SHIFT==15
#define mul_digit_digit(l, r) mul_15_15(l, r)
#define mul_digit_nr(p, l, r, nr) mul_15_nr(p, l, r, nr)
#elif PyLong_SHIFT==30
#define mul_digit_digit(l, r) mul_30_30(l, r)
#define mul_digit_nr(p, l, r, nr) mul_30_nr(p, l, r, nr)
#else
#error
#endif

#if defined(__GNUC__) && defined(__PCLMUL__)
#include "intel_clmul.h"
#elif defined(__GNUC__) && (defined(__ARM_NEON) || defined(__ARM_NEON__)) && defined(__ARM_FEATURE_CRYPTO)
#include "armv8_crypto.h"
#elif defined(__GNUC__) && (defined(__ARM_NEON) || defined(__ARM_NEON__))
#include "armv7_neon.h"
#else
#include "generic.h"
#endif

// Squares up to 255 (8-bit chunk size)
static const uint16_t sqr_8[256] = {
    0x0000,0x0001,0x0004,0x0005,0x0010,0x0011,0x0014,0x0015,0x0040,0x0041,0x0044,0x0045,0x0050,0x0051,0x0054,0x0055,
    0x0100,0x0101,0x0104,0x0105,0x0110,0x0111,0x0114,0x0115,0x0140,0x0141,0x0144,0x0145,0x0150,0x0151,0x0154,0x0155,
    0x0400,0x0401,0x0404,0x0405,0x0410,0x0411,0x0414,0x0415,0x0440,0x0441,0x0444,0x0445,0x0450,0x0451,0x0454,0x0455,
    0x0500,0x0501,0x0504,0x0505,0x0510,0x0511,0x0514,0x0515,0x0540,0x0541,0x0544,0x0545,0x0550,0x0551,0x0554,0x0555,
    0x1000,0x1001,0x1004,0x1005,0x1010,0x1011,0x1014,0x1015,0x1040,0x1041,0x1044,0x1045,0x1050,0x1051,0x1054,0x1055,
    0x1100,0x1101,0x1104,0x1105,0x1110,0x1111,0x1114,0x1115,0x1140,0x1141,0x1144,0x1145,0x1150,0x1151,0x1154,0x1155,
    0x1400,0x1401,0x1404,0x1405,0x1410,0x1411,0x1414,0x1415,0x1440,0x1441,0x1444,0x1445,0x1450,0x1451,0x1454,0x1455,
    0x1500,0x1501,0x1504,0x1505,0x1510,0x1511,0x1514,0x1515,0x1540,0x1541,0x1544,0x1545,0x1550,0x1551,0x1554,0x1555,
    0x4000,0x4001,0x4004,0x4005,0x4010,0x4011,0x4014,0x4015,0x4040,0x4041,0x4044,0x4045,0x4050,0x4051,0x4054,0x4055,
    0x4100,0x4101,0x4104,0x4105,0x4110,0x4111,0x4114,0x4115,0x4140,0x4141,0x4144,0x4145,0x4150,0x4151,0x4154,0x4155,
    0x4400,0x4401,0x4404,0x4405,0x4410,0x4411,0x4414,0x4415,0x4440,0x4441,0x4444,0x4445,0x4450,0x4451,0x4454,0x4455,
    0x4500,0x4501,0x4504,0x4505,0x4510,0x4511,0x4514,0x4515,0x4540,0x4541,0x4544,0x4545,0x4550,0x4551,0x4554,0x4555,
    0x5000,0x5001,0x5004,0x5005,0x5010,0x5011,0x5014,0x5015,0x5040,0x5041,0x5044,0x5045,0x5050,0x5051,0x5054,0x5055,
    0x5100,0x5101,0x5104,0x5105,0x5110,0x5111,0x5114,0x5115,0x5140,0x5141,0x5144,0x5145,0x5150,0x5151,0x5154,0x5155,
    0x5400,0x5401,0x5404,0x5405,0x5410,0x5411,0x5414,0x5415,0x5440,0x5441,0x5444,0x5445,0x5450,0x5451,0x5454,0x5455,
    0x5500,0x5501,0x5504,0x5505,0x5510,0x5511,0x5514,0x5515,0x5540,0x5541,0x5544,0x5545,0x5550,0x5551,0x5554,0x5555,
};


// Multiplication table up to 31x31, that is, 5-bit chunks.
static const uint16_t mul_5_5[32][32] = {
    {0x000,0x000,0x000,0x000,0x000,0x000,0x000,0x000,0x000,0x000,0x000,0x000,0x000,0x000,0x000,0x000,
     0x000,0x000,0x000,0x000,0x000,0x000,0x000,0x000,0x000,0x000,0x000,0x000,0x000,0x000,0x000,0x000},
    {0x000,0x001,0x002,0x003,0x004,0x005,0x006,0x007,0x008,0x009,0x00a,0x00b,0x00c,0x00d,0x00e,0x00f,
     0x010,0x011,0x012,0x013,0x014,0x015,0x016,0x017,0x018,0x019,0x01a,0x01b,0x01c,0x01d,0x01e,0x01f},
    {0x000,0x002,0x004,0x006,0x008,0x00a,0x00c,0x00e,0x010,0x012,0x014,0x016,0x018,0x01a,0x01c,0x01e,
     0x020,0x022,0x024,0x026,0x028,0x02a,0x02c,0x02e,0x030,0x032,0x034,0x036,0x038,0x03a,0x03c,0x03e},
    {0x000,0x003,0x006,0x005,0x00c,0x00f,0x00a,0x009,0x018,0x01b,0x01e,0x01d,0x014,0x017,0x012,0x011,
     0x030,0x033,0x036,0x035,0x03c,0x03f,0x03a,0x039,0x028,0x02b,0x02e,0x02d,0x024,0x027,0x022,0x021},
    {0x000,0x004,0x008,0x00c,0x010,0x014,0x018,0x01c,0x020,0x024,0x028,0x02c,0x030,0x034,0x038,0x03c,
     0x040,0x044,0x048,0x04c,0x050,0x054,0x058,0x05c,0x060,0x064,0x068,0x06c,0x070,0x074,0x078,0x07c},
    {0x000,0x005,0x00a,0x00f,0x014,0x011,0x01e,0x01b,0x028,0x02d,0x022,0x027,0x03c,0x039,0x036,0x033,
     0x050,0x055,0x05a,0x05f,0x044,0x041,0x04e,0x04b,0x078,0x07d,0x072,0x077,0x06c,0x069,0x066,0x063},
    {0x000,0x006,0x00c,0x00a,0x018,0x01e,0x014,0x012,0x030,0x036,0x03c,0x03a,0x028,0x02e,0x024,0x022,
     0x060,0x066,0x06c,0x06a,0x078,0x07e,0x074,0x072,0x050,0x056,0x05c,0x05a,0x048,0x04e,0x044,0x042},
    {0x000,0x007,0x00e,0x009,0x01c,0x01b,0x012,0x015,0x038,0x03f,0x036,0x031,0x024,0x023,0x02a,0x02d,
     0x070,0x077,0x07e,0x079,0x06c,0x06b,0x062,0x065,0x048,0x04f,0x046,0x041,0x054,0x053,0x05a,0x05d},
    {0x000,0x008,0x010,0x018,0x020,0x028,0x030,0x038,0x040,0x048,0x050,0x058,0x060,0x068,0x070,0x078,
     0x080,0x088,0x090,0x098,0x0a0,0x0a8,0x0b0,0x0b8,0x0c0,0x0c8,0x0d0,0x0d8,0x0e0,0x0e8,0x0f0,0x0f8},
    {0x000,0x009,0x012,0x01b,0x024,0x02d,0x036,0x03f,0x048,0x041,0x05a,0x053,0x06c,0x065,0x07e,0x077,
     0x090,0x099,0x082,0x08b,0x0b4,0x0bd,0x0a6,0x0af,0x0d8,0x0d1,0x0ca,0x0c3,0x0fc,0x0f5,0x0ee,0x0e7},
    {0x000,0x00a,0x014,0x01e,0x028,0x022,0x03c,0x036,0x050,0x05a,0x044,0x04e,0x078,0x072,0x06c,0x066,
     0x0a0,0x0aa,0x0b4,0x0be,0x088,0x082,0x09c,0x096,0x0f0,0x0fa,0x0e4,0x0ee,0x0d8,0x0d2,0x0cc,0x0c6},
    {0x000,0x00b,0x016,0x01d,0x02c,0x027,0x03a,0x031,0x058,0x053,0x04e,0x045,0x074,0x07f,0x062,0x069,
     0x0b0,0x0bb,0x0a6,0x0ad,0x09c,0x097,0x08a,0x081,0x0e8,0x0e3,0x0fe,0x0f5,0x0c4,0x0cf,0x0d2,0x0d9},
    {0x000,0x00c,0x018,0x014,0x030,0x03c,0x028,0x024,0x060,0x06c,0x078,0x074,0x050,0x05c,0x048,0x044,
     0x0c0,0x0cc,0x0d8,0x0d4,0x0f0,0x0fc,0x0e8,0x0e4,0x0a0,0x0ac,0x0b8,0x0b4,0x090,0x09c,0x088,0x084},
    {0x000,0x00d,0x01a,0x017,0x034,0x039,0x02e,0x023,0x068,0x065,0x072,0x07f,0x05c,0x051,0x046,0x04b,
     0x0d0,0x0dd,0x0ca,0x0c7,0x0e4,0x0e9,0x0fe,0x0f3,0x0b8,0x0b5,0x0a2,0x0af,0x08c,0x081,0x096,0x09b},
    {0x000,0x00e,0x01c,0x012,0x038,0x036,0x024,0x02a,0x070,0x07e,0x06c,0x062,0x048,0x046,0x054,0x05a,
     0x0e0,0x0ee,0x0fc,0x0f2,0x0d8,0x0d6,0x0c4,0x0ca,0x090,0x09e,0x08c,0x082,0x0a8,0x0a6,0x0b4,0x0ba},
    {0x000,0x00f,0x01e,0x011,0x03c,0x033,0x022,0x02d,0x078,0x077,0x066,0x069,0x044,0x04b,0x05a,0x055,
     0x0f0,0x0ff,0x0ee,0x0e1,0x0cc,0x0c3,0x0d2,0x0dd,0x088,0x087,0x096,0x099,0x0b4,0x0bb,0x0aa,0x0a5},
    {0x000,0x010,0x020,0x030,0x040,0x050,0x060,0x070,0x080,0x090,0x0a0,0x0b0,0x0c0,0x0d0,0x0e0,0x0f0,
     0x100,0x110,0x120,0x130,0x140,0x150,0x160,0x170,0x180,0x190,0x1a0,0x1b0,0x1c0,0x1d0,0x1e0,0x1f0},
    {0x000,0x011,0x022,0x033,0x044,0x055,0x066,0x077,0x088,0x099,0x0aa,0x0bb,0x0cc,0x0dd,0x0ee,0x0ff,
     0x110,0x101,0x132,0x123,0x154,0x145,0x176,0x167,0x198,0x189,0x1ba,0x1ab,0x1dc,0x1cd,0x1fe,0x1ef},
    {0x000,0x012,0x024,0x036,0x048,0x05a,0x06c,0x07e,0x090,0x082,0x0b4,0x0a6,0x0d8,0x0ca,0x0fc,0x0ee,
     0x120,0x132,0x104,0x116,0x168,0x17a,0x14c,0x15e,0x1b0,0x1a2,0x194,0x186,0x1f8,0x1ea,0x1dc,0x1ce},
    {0x000,0x013,0x026,0x035,0x04c,0x05f,0x06a,0x079,0x098,0x08b,0x0be,0x0ad,0x0d4,0x0c7,0x0f2,0x0e1,
     0x130,0x123,0x116,0x105,0x17c,0x16f,0x15a,0x149,0x1a8,0x1bb,0x18e,0x19d,0x1e4,0x1f7,0x1c2,0x1d1},
    {0x000,0x014,0x028,0x03c,0x050,0x044,0x078,0x06c,0x0a0,0x0b4,0x088,0x09c,0x0f0,0x0e4,0x0d8,0x0cc,
     0x140,0x154,0x168,0x17c,0x110,0x104,0x138,0x12c,0x1e0,0x1f4,0x1c8,0x1dc,0x1b0,0x1a4,0x198,0x18c},
    {0x000,0x015,0x02a,0x03f,0x054,0x041,0x07e,0x06b,0x0a8,0x0bd,0x082,0x097,0x0fc,0x0e9,0x0d6,0x0c3,
     0x150,0x145,0x17a,0x16f,0x104,0x111,0x12e,0x13b,0x1f8,0x1ed,0x1d2,0x1c7,0x1ac,0x1b9,0x186,0x193},
    {0x000,0x016,0x02c,0x03a,0x058,0x04e,0x074,0x062,0x0b0,0x0a6,0x09c,0x08a,0x0e8,0x0fe,0x0c4,0x0d2,
     0x160,0x176,0x14c,0x15a,0x138,0x12e,0x114,0x102,0x1d0,0x1c6,0x1fc,0x1ea,0x188,0x19e,0x1a4,0x1b2},
    {0x000,0x017,0x02e,0x039,0x05c,0x04b,0x072,0x065,0x0b8,0x0af,0x096,0x081,0x0e4,0x0f3,0x0ca,0x0dd,
     0x170,0x167,0x15e,0x149,0x12c,0x13b,0x102,0x115,0x1c8,0x1df,0x1e6,0x1f1,0x194,0x183,0x1ba,0x1ad},
    {0x000,0x018,0x030,0x028,0x060,0x078,0x050,0x048,0x0c0,0x0d8,0x0f0,0x0e8,0x0a0,0x0b8,0x090,0x088,
     0x180,0x198,0x1b0,0x1a8,0x1e0,0x1f8,0x1d0,0x1c8,0x140,0x158,0x170,0x168,0x120,0x138,0x110,0x108},
    {0x000,0x019,0x032,0x02b,0x064,0x07d,0x056,0x04f,0x0c8,0x0d1,0x0fa,0x0e3,0x0ac,0x0b5,0x09e,0x087,
     0x190,0x189,0x1a2,0x1bb,0x1f4,0x1ed,0x1c6,0x1df,0x158,0x141,0x16a,0x173,0x13c,0x125,0x10e,0x117},
    {0x000,0x01a,0x034,0x02e,0x068,0x072,0x05c,0x046,0x0d0,0x0ca,0x0e4,0x0fe,0x0b8,0x0a2,0x08c,0x096,
     0x1a0,0x1ba,0x194,0x18e,0x1c8,0x1d2,0x1fc,0x1e6,0x170,0x16a,0x144,0x15e,0x118,0x102,0x12c,0x136},
    {0x000,0x01b,0x036,0x02d,0x06c,0x077,0x05a,0x041,0x0d8,0x0c3,0x0ee,0x0f5,0x0b4,0x0af,0x082,0x099,
     0x1b0,0x1ab,0x186,0x19d,0x1dc,0x1c7,0x1ea,0x1f1,0x168,0x173,0x15e,0x145,0x104,0x11f,0x132,0x129},
    {0x000,0x01c,0x038,0x024,0x070,0x06c,0x048,0x054,0x0e0,0x0fc,0x0d8,0x0c4,0x090,0x08c,0x0a8,0x0b4,
     0x1c0,0x1dc,0x1f8,0x1e4,0x1b0,0x1ac,0x188,0x194,0x120,0x13c,0x118,0x104,0x150,0x14c,0x168,0x174},
    {0x000,0x01d,0x03a,0x027,0x074,0x069,0x04e,0x053,0x0e8,0x0f5,0x0d2,0x0cf,0x09c,0x081,0x0a6,0x0bb,
     0x1d0,0x1cd,0x1ea,0x1f7,0x1a4,0x1b9,0x19e,0x183,0x138,0x125,0x102,0x11f,0x14c,0x151,0x176,0x16b},
    {0x000,0x01e,0x03c,0x022,0x078,0x066,0x044,0x05a,0x0f0,0x0ee,0x0cc,0x0d2,0x088,0x096,0x0b4,0x0aa,
     0x1e0,0x1fe,0x1dc,0x1c2,0x198,0x186,0x1a4,0x1ba,0x110,0x10e,0x12c,0x132,0x168,0x176,0x154,0x14a},
    {0x000,0x01f,0x03e,0x021,0x07c,0x063,0x042,0x05d,0x0f8,0x0e7,0x0c6,0x0d9,0x084,0x09b,0x0ba,0x0a5,
     0x1f0,0x1ef,0x1ce,0x1d1,0x18c,0x193,0x1b2,0x1ad,0x108,0x117,0x136,0x129,0x174,0x16b,0x14a,0x155},
};


static inline int nbits(PyLongObject *integer)
// return 1-based index of the most significant non-zero bit, or 0 if all bits are zero
{
    return _PyLong_NumBits((PyObject *)integer);
}

static void mul_ATOM_nr(digit * restrict const p,
                        const uint8_t l,
                        const digit * restrict const r0, int nr)
//
// Multiply a bignum polynomial by a ATOM-bit polynomial
//
{
    DBG_ASSERT(l<(1<<ATOM));
    twodigits pi = 0;
    for(int id_r=0; id_r<nr; id_r++) {
#if (PyLong_SHIFT == 15)
        pi ^= mul_ATOM_15(l, r0[id_r]);
#elif (PyLong_SHIFT == 30)
        pi ^= mul_ATOM_30(l, r0[id_r]);
#else
#error
#endif
        p[id_r] ^= pi & PyLong_MASK;
        pi >>= PyLong_SHIFT;
    }
    if(pi)
        p[nr] ^= pi;
}

static void mul_15_nr(digit * restrict const p,
                      const uint16_t l,
                      const digit * restrict const r0, int nr)
//
// Multiply a bignum polynomial by a 15-bit polynomial
//
{
    DBG_ASSERT(l<(1<<15));
    twodigits pi = 0;
    for(int id_r=0; id_r<nr; id_r++) {
#if (PyLong_SHIFT == 15)
        pi ^= mul_15_15(l, r0[id_r]);
#elif (PyLong_SHIFT == 30)
        pi ^= mul_15_30(l, r0[id_r]);
#else
#error
#endif
        p[id_r] ^= pi & PyLong_MASK;
        pi >>= PyLong_SHIFT;
    }
    if(pi)
        p[nr] ^= pi;
}

#if (PyLong_SHIFT == 30)
static void mul_30_nr(digit * restrict const p,
                      const uint32_t l,
                      const digit * restrict const r0, int nr)
//
// Multiply a bignum polynomial by a 30-bit polynomial
//
{
    DBG_ASSERT(l<(1<<30));
    twodigits pi = 0;
    for(int id_r=0; id_r<nr; id_r++) {
#if (PyLong_SHIFT == 15)
        pi ^= mul_15_30(r0[id_r], l);
#elif (PyLong_SHIFT == 30)
        pi ^= mul_30_30(r0[id_r], l);
#else
#error
#endif
        p[id_r] ^= pi & PyLong_MASK;
        pi >>= PyLong_SHIFT;
    }
    if(pi)
        p[nr] ^= pi;
}
#endif

static PyObject *
pygf2x_sqr(PyObject *self, PyObject *args)
//
// Square one Python integer, interpreted as polynomial over GF(2)
//
{
    (void)self;

    PyLongObject *f;
    if (!PyArg_ParseTuple(args, "O", &f)) {
        PyErr_SetString(PyExc_TypeError, "Failed to parse arguments");
        return NULL;
    }

    if( ! PyLong_Check(f) ) {
        PyErr_SetString(PyExc_TypeError, "Arguments must be integer");
        return NULL;
    }
    if(((PyVarObject *)f)->ob_size < 0) {
        PyErr_SetString(PyExc_ValueError, "Argument must be non-negative");
        return NULL;
    }
    if(((PyVarObject *)f)->ob_size > PYGF2X_MAX_DIGITS) {
        PyErr_SetString(PyExc_ValueError, "Factor out of range");
        return NULL;
    }

    int nbits_f = nbits(f);
    int nbits_p = 2*nbits_f -1;
    int ndigs_p = (nbits_p + (PyLong_SHIFT-1))/PyLong_SHIFT;
    int ndigs_f = ((PyVarObject *)f)->ob_size;

    if(nbits_p > (PYGF2X_MAX_DIGITS * PyLong_SHIFT)) {
        PyErr_SetString(PyExc_OverflowError, "Result of square is out of range");
        return NULL;
    }
    
    PyLongObject *p = _PyLong_New(2*ndigs_f); // This may be 1 digit more than needed (ndigs_p);
    ((PyVarObject *)p)->ob_size = ndigs_p;
    
    DBG_PRINTF("Bits per digit   = %-4d\n",PyLong_SHIFT);
    DBG_PRINTF("factor bits      = %-4d\n",nbits_f);
    DBG_PRINTF("Square digits    = %-4d\n",ndigs_p);

    square_n(p->ob_digit, f->ob_digit, ndigs_f);

    DBG_PRINTF_DIGITS("Square:", p->ob_digit, ndigs_p);

    return (PyObject *)p;
}

#include "mul_nl_nr.h"

static PyObject *
pygf2x_mul(PyObject *self, PyObject *args)
//
// Multiply two Python integers, interpreted as polynomials over GF(2)
//
{
    (void)self;

    PyLongObject *fl, *fr;
    if (!PyArg_ParseTuple(args, "OO", &fl, &fr)) {
        PyErr_SetString(PyExc_TypeError, "Failed to parse arguments");
        return NULL;
    }

    if( ! PyLong_Check(fl) ||
        ! PyLong_Check(fr) ) {
        PyErr_SetString(PyExc_TypeError, "Both arguments must be integers");
        return NULL;
    }
    if(((PyVarObject *)fl)->ob_size < 0 ||
       ((PyVarObject *)fr)->ob_size < 0) {
        PyErr_SetString(PyExc_ValueError, "Both arguments must be non-negative");
        return NULL;
    }

    if(((PyVarObject *)fl)->ob_size == 0 ||
       ((PyVarObject *)fr)->ob_size == 0) {
        PyLongObject *p = _PyLong_New(0);
        return (PyObject *)p;
    }

    if(((PyVarObject *)fl)->ob_size > PYGF2X_MAX_DIGITS
       ||((PyVarObject *)fr)->ob_size > PYGF2X_MAX_DIGITS
       ) {
        PyErr_SetString(PyExc_ValueError, "Factor is out of range");
        return NULL;
    }

    int nbits_l = nbits(fl);
    int nbits_r = nbits(fr);
    int nbits_p = nbits_l + nbits_r -1;
    int ndigs_l = (nbits_l + (PyLong_SHIFT-1))/PyLong_SHIFT;
    int ndigs_r = (nbits_r + (PyLong_SHIFT-1))/PyLong_SHIFT;
    int ndigs_p = (nbits_p + (PyLong_SHIFT-1))/PyLong_SHIFT;

    if(ndigs_p > PYGF2X_MAX_DIGITS) {
        PyErr_SetString(PyExc_OverflowError, "Result of multiplication is out of range");
        return NULL;
    }

    const bool use_heap = (ndigs_l + ndigs_r > STATIC_LIMIT);
    digit result_static[STATIC_LIMIT];
    digit * restrict const result = use_heap ? malloc((ndigs_l + ndigs_r)*sizeof(digit)) : result_static;
    memset(result,0,(ndigs_l + ndigs_r)*sizeof(digit));
    
    DBG_PRINTF("Bits per digit   = %-4d\n",PyLong_SHIFT);
    DBG_PRINTF("Left factor bits = %-4d\n",nbits_l);
    DBG_PRINTF("Right factor bits= %-4d\n",nbits_r);
    DBG_PRINTF("Product digits   = %-4d\n",ndigs_p);

    mul_nl_nr(result, fl->ob_digit, ndigs_l, fr->ob_digit, ndigs_r);

    DBG_PRINTF_DIGITS("Product          :",result,ndigs_p);

    PyLongObject *p = _PyLong_New(ndigs_p);
    memcpy(p->ob_digit, result, sizeof(digit)*ndigs_p);
    if(use_heap)
        free(result);

    return (PyObject *)p;
}

#include "div_bitwise.h"
#include "inverse.h"

static PyObject *
pygf2x_inv(PyObject *self, PyObject *args)
//
// Multiplicative inverse of one Python integer, interpreted as polynomial over GF(2)
//
{
    (void)self;

    int nbits_e;
    PyLongObject *d;
    if (!PyArg_ParseTuple(args, "Oi", &d, &nbits_e)) {
        PyErr_SetString(PyExc_TypeError, "Failed to parse arguments");
        return NULL;
    }

    if( ! PyLong_Check(d) ) {
        PyErr_SetString(PyExc_TypeError, "Argument must be integer");
        return NULL;
    }
    if(((PyVarObject *)d)->ob_size == 0) {
        PyErr_SetString(PyExc_ZeroDivisionError, "Inverse of zero is undefined");
        return NULL;
    }
    if(((PyVarObject *)d)->ob_size < 0) {
        PyErr_SetString(PyExc_ValueError, "Argument must be positive");
        return NULL;
    }
    if(((PyVarObject *)d)->ob_size > PYGF2X_MAX_DIGITS) {
        PyErr_SetString(PyExc_ValueError, "Inverse operand is out of range");
        return NULL;
    }
    if(nbits_e <= 0) {
        PyErr_SetString(PyExc_ValueError, "Inverse bit_length must be positive");
        return NULL;
    }
    if(nbits_e <= 0 || nbits_e > (PYGF2X_MAX_DIGITS*PyLong_SHIFT)) {
        PyErr_SetString(PyExc_OverflowError, "Requested bit_length of inverse is out of range");
        return NULL;
    }
    
    int nbits_d = nbits(d);
    int ndigs_d = (nbits_d + (PyLong_SHIFT-1))/PyLong_SHIFT;

    int ndigs_e = (nbits_e + (PyLong_SHIFT-1))/PyLong_SHIFT;
    DBG_PRINTF("ndigs_e          = %-4d\n",ndigs_e);
    
    PyLongObject *e = _PyLong_New(ndigs_e);
    memset(e->ob_digit, 0, ndigs_e*sizeof(digit));
    
    DBG_PRINTF("Bits per digit   = %-4d\n",PyLong_SHIFT);
    DBG_PRINTF("Denominator bits = %-4d\n",nbits_d);
    DBG_PRINTF("Requested bits   = %-4d\n",nbits_e);

    inverse(e->ob_digit, ndigs_e, nbits_e,
            d->ob_digit, ndigs_d, nbits_d);

    DBG_PRINTF_DIGITS("Inverse:", e->ob_digit, ndigs_e);

    return (PyObject *)e;
}

static void rshift(digit digits[], int ndigs, int nb_shift)
// Shift in-place nb_shift bits to the right
// nb_shift must be >=0
{
    DBG_ASSERT(nb_shift>=0);
    DBG_ASSERT(ndigs>=0);
    if(ndigs==0)
        return;
    
    int nd_shift = nb_shift/PyLong_SHIFT;
    nb_shift     = nb_shift%PyLong_SHIFT;
    
    for(int i=0; nd_shift+1+i < ndigs; i++)
        digits[i] = (digits[nd_shift+i] >> nb_shift) |
            ((digits[nd_shift+1+i] << (PyLong_SHIFT - nb_shift)) & PyLong_MASK);
    if(ndigs > nd_shift)
        digits[ndigs-1-nd_shift] = (digits[ndigs-1] >> nb_shift);

    // Fill evacuated digits with zero
    for(int i=GF2X_MAX(ndigs, nd_shift) - nd_shift; i < ndigs; i++)
        digits[i] = 0;
}

static PyObject *
pygf2x_divmod(PyObject *self, PyObject *args)
//
// Divide two Python integers, interpreted as polynomials over GF(2)
// Return quotient and remainder
//
{
    (void)self;

    PyLongObject *numerator, *denominator;
    if (!PyArg_ParseTuple(args, "OO", &numerator, &denominator)) {
        PyErr_SetString(PyExc_TypeError, "Failed to parse arguments");
        return NULL;
    }
    
    if( ! PyLong_Check(numerator) ||
        ! PyLong_Check(denominator) ) {
        PyErr_SetString(PyExc_TypeError, "Both arguments must be integers");
        return NULL;
    }
    if(((PyVarObject *)numerator)->ob_size < 0 ||
       ((PyVarObject *)denominator)->ob_size < 0) {
        PyErr_SetString(PyExc_ValueError, "Both arguments must be non-negative");
        return NULL;
    }
    if(((PyVarObject *)numerator)->ob_size > PYGF2X_MAX_DIGITS ||
       ((PyVarObject *)denominator)->ob_size > PYGF2X_MAX_DIGITS) {
        PyErr_SetString(PyExc_ValueError, "Numerator or denominator out of range");
        return NULL;
    }

    int nbits_d = nbits(denominator);
    int ndigs_d = (nbits_d + (PyLong_SHIFT-1))/PyLong_SHIFT;
    if(nbits_d == 0) {
        PyErr_SetString(PyExc_ZeroDivisionError, "Denominator is zero");
        return NULL;
    }
    int nbits_u = nbits(numerator);
    int ndigs_u = (nbits_u + (PyLong_SHIFT-1))/PyLong_SHIFT;
    
    int nbits_q = nbits_u > nbits_d-1 ? nbits_u - (nbits_d-1) : 0;
    int nbits_r = nbits_u > nbits_d-1 ? nbits_u : nbits_d-1; // Enough room to store u initially, and r finally
    int ndigs_q = (nbits_q + (PyLong_SHIFT-1))/PyLong_SHIFT;
    int ndigs_r = (nbits_r + (PyLong_SHIFT-1))/PyLong_SHIFT;

    PyLongObject *q = _PyLong_New(ndigs_q);
    digit *restrict q_digits = q->ob_digit;
    memset(q_digits,0,ndigs_q*sizeof(digit));
    
    digit * restrict const r_digits = malloc(ndigs_r*sizeof(digit)); // Initialize to numerator
    memset(r_digits+ndigs_u,0,(ndigs_r-ndigs_u)*sizeof(digit));
    memcpy(r_digits, numerator->ob_digit, ndigs_u*sizeof(digit));
    
    DBG_PRINTF("Bits per digit   = %-4d\n",PyLong_SHIFT);
    DBG_PRINTF("Numerator bits   = %-4d\n",nbits_u);
    DBG_PRINTF("Denominator bits = %-4d\n",nbits_d);
    DBG_PRINTF("Quotient bits    = %-4d\n",nbits_q);
    DBG_PRINTF("Remainder bits  <= %-4d\n",nbits_d-1);

    DBG_PRINTF_DIGITS("Numerator        :",numerator->ob_digit,ndigs_u);
    DBG_PRINTF_DIGITS("Denominator      :",denominator->ob_digit,ndigs_d);

    if(nbits_u==nbits_d) {
        // The special case of quotient==1
        q_digits[0] = 1;
        for(int i=0; i<ndigs_d; i++)
            r_digits[i] ^= denominator->ob_digit[i];
    } else if(nbits_d==1) {
        // The special case of denominator==1
        for(int i=0; i<ndigs_u; i++)
            q_digits[i] = r_digits[i];
        for(int i=0; i<ndigs_r; i++)
            r_digits[i] = 0;
    } else if(nbits_u>=nbits_d) {
        if(nbits_d < LIMIT_DIV_BITWISE) {
            // Use bitwise Euclidean division for small denominators because it is possibly more efficient
            div_bitwise(q_digits, r_digits, denominator->ob_digit, nbits_u, nbits_d);
        } else {
            /*
             *   u = q*d + r
             * Let |x| denote nbits(x) (the one-based index of the highest set bit), then
             *   |u| = |q|+|d|-1
             *   |r| <= |d|-1
             * assuming |u| >= |d|
             *
             * Let e be an approximate inverse inv(d) with |e| correct binary digits, i.e.:
             *   d*e = (1<<(|e|+|d|-2)) + f
             * where |f| <= |d|-1. Then:
             *   u*e = q*d*e + r*e
             *       = q*( (1<<(|e|+|d|-2)) + f ) + r*e
             * Right shift (|e|+|d|-2) to recover q:
             *   (u*e)>>(|e|+|d|-2) = q + (q*f)>>(|e|+|d|-2) + (r*e)>>(|e|+|d|-2)
             * The rightmost term vanishes because |r*e|= |r|+|e|-1 <= |d|-1+|e|-1 = |e|+|d|-2
             * The bitlength of the term (q*f)>>(|e|+|d|-2) is
             *   |q|+|f|-1 - (|e|+|d|-2) <= |q| + (|d|-1) -1 - (|e|+|d|-2) = |q| - |e|
             * Thus if |e| >= |q| then we have a correct q
             * So if the inverse is computed with this accuracy then q can computed in one step with
             *   q = (u*e) >> (|e|+|d|-2)
             * If a less accurate e is computed then the n.o. correct digits of q is |e|
             * If |e|<=|d| then q can then be computed in a finite number of steps, using Newton iteration:
             *   r_0 = u
             *   e = inv(d,|e|), where |e|<=|d|
             *   while |r_i| >= |d|
             *     dq_i = ((r_i >> (|r_i|-|e|)) * e) >> |e|-1
             *     shft = |r_i| - |d| - (|e|-1))
             *     q_{i+1} = q_i + (dq_i << shft)
             *     r_{i+1} = r_i - ((dq_i * e) << shft)
             * Note that only the |e| most significant bits of r_i needs to be multiplied with e in
             * each step above.
             *
             * In the implementation below we use |e| which is an entire Python Digit, to make implementation
             * simpler and faster for large polynomials.
             *
             */
            
            // Choose accuracy of inverse to compute:
            // If nbits_d >= nbits_q just compute q with one single step in the Euclidean division loop below
            // Otherwise take multiple steps, each of size nbits_d or less
            int nbits_e = GF2X_MIN(nbits_q, nbits_d);
            // Round up to nearest digit size
            int ndigs_e = (nbits_e + (PyLong_SHIFT-1))/PyLong_SHIFT;
            nbits_e = PyLong_SHIFT*ndigs_e;

            // Compute the inverse e = (d)^-1
            digit * restrict const e = malloc(ndigs_e*sizeof(digit));
            memset(e, 0, ndigs_e*sizeof(digit));
            inverse(e, ndigs_e, nbits_e,
                    denominator->ob_digit, ndigs_d, nbits_d);
            DBG_PRINTF_DIGITS("inverse          :",e,ndigs_e);

            DBG_PRINTF("ndigs_e=%d, ndigs_u=%d, ndigs_q=%d\n", ndigs_e, ndigs_u, ndigs_q);

            digit * restrict const dr = malloc((ndigs_e + ndigs_d)*sizeof(digit));
            
            // Start with computing the most significant, incomplete digit of q, if it exists.
            if(nbits_q%PyLong_SHIFT != 0)
                {
                    int nbits_ei = nbits_q%PyLong_SHIFT;
                    int nbits_ri = (nbits_r-1)%PyLong_SHIFT+1;
                    DBG_PRINTF("nbits_ei=%d, nbits_ri=%d\n",nbits_ei,nbits_ri);
                    digit ri = (nbits_ri > nbits_ei) ? r_digits[ndigs_r-1] >> (nbits_ri - nbits_ei) :
                        (r_digits[ndigs_r-2]>>(PyLong_SHIFT - (nbits_ei - nbits_ri))) |
                        (r_digits[ndigs_r-1]<<(nbits_ei - nbits_ri));
                    digit ei = e[ndigs_e-1] >> (PyLong_SHIFT - nbits_ei);

                    // dq = ((r >> (nr-ne)) *e) >> (ne-1)
                    digit dq = mul_digit_digit(ei, ri) >> (nbits_ei -1);
                    DBG_PRINTF("ei=%x, ri=%x, dq=%x\n",ei,ri,dq);
                    DBG_ASSERT(dq >= (1u<<(nbits_ei-1)) && dq<(1u<<nbits_ei));
                    // |dq| = nbits_ei
                    q_digits[ndigs_q-1] = dq;

                    int nbits_qi = nbits_r - nbits_d - (nbits_ei -1);
                    DBG_ASSERT(nbits_qi%PyLong_SHIFT == 0);
                    int ndigs_qi = nbits_qi/PyLong_SHIFT;

                    // dr = (dq*d) << nqi
                    memset(dr, 0, (ndigs_d+1)*sizeof(digit));
                    mul_nl_nr(dr, &dq, 1, denominator->ob_digit, ndigs_d);
                    DBG_PRINTF_DIGITS("dr  :",dr,ndigs_d+1);
                    DBG_PRINTF("dq=%x\n",dq);
                    DBG_ASSERT(ndigs_r -1 - ndigs_qi < ndigs_d +1);
                    for(int i=ndigs_qi; i<ndigs_r; i++) {
                        r_digits[i] ^= dr[i - ndigs_qi];
                    }
                    DBG_PRINTF_DIGITS("r_0              :",r_digits,ndigs_r);           
                
                    nbits_r -= nbits_ei;
                }
            // Loop over whole digits
            DBG_ASSERT(nbits_e%PyLong_SHIFT == 0);
            DBG_ASSERT((nbits_r - nbits_d +1)%PyLong_SHIFT == 0);
            digit * restrict const dq = malloc((ndigs_e<<1)*sizeof(digit));
            for(; nbits_r >= nbits_d; nbits_r -= nbits_e) {
                int ndigs_ei = GF2X_MIN(ndigs_e, (nbits_r - nbits_d +1)/PyLong_SHIFT);
                int nbits_ei = ndigs_ei*PyLong_SHIFT;
                DBG_PRINTF("nbits_ei=%d, ndigs_ei=%d\n",nbits_ei,ndigs_ei);

                // dq = ((r >> (nr-ne)) *e) >> (ne-1)
                memset(dq, 0, (2*ndigs_ei)*sizeof(digit));
                int ndigs_ri = (nbits_r + (PyLong_SHIFT-1))/PyLong_SHIFT;
                {
                    int nbits_ri = (nbits_r-1)%PyLong_SHIFT+1;
                    memset(dr, 0, ndigs_ei*sizeof(digit));
                    for(int i=0; i<ndigs_ei; i++)
                        dr[i] = (r_digits[ndigs_ri - ndigs_ei +i] << (PyLong_SHIFT - nbits_ri) & PyLong_MASK) |
                            r_digits[ndigs_ri - ndigs_ei +i -1] >> nbits_ri;
                    DBG_PRINTF_DIGITS("r>>(nr-ne)       :",dr,ndigs_ei);
                    mul_nl_nr(dq, &e[ndigs_e - ndigs_ei], ndigs_ei, dr, ndigs_ei);
                }
                rshift(dq, 2*ndigs_ei, nbits_ei-1);
                // |dq| is now = nbits_ei (the uppermost ndigs_ei digits is 0)
                DBG_PRINTF_DIGITS("dq               :",dq,2*ndigs_ei);
                
                int ndigs_qi = nbits_r - nbits_d - (nbits_ei -1);
                DBG_ASSERT(ndigs_qi%PyLong_SHIFT == 0);
                ndigs_qi /= PyLong_SHIFT;
                DBG_PRINTF("nbits_r=%d, nbits_d=%d, nbits_e=%d, ndigs_qi=%d\n",nbits_r,nbits_d, nbits_e, ndigs_qi);
                
                // dr = (dq*d) << nqi
                memset(dr, 0, (ndigs_ei + ndigs_d)*sizeof(digit));
                mul_nl_nr(dr, dq, ndigs_ei, denominator->ob_digit, ndigs_d);
                DBG_PRINTF_DIGITS("dr               :",dr,ndigs_ei+ndigs_d);

                for(int i=ndigs_qi; i < ndigs_ri; i++)
                    if(dr[i-ndigs_qi])
                        r_digits[i] ^= dr[i-ndigs_qi];
                for(int i=0; i<ndigs_ei; i++)
                    if(dq[i])
                        q_digits[ndigs_qi+i] ^= dq[i];
                DBG_PRINTF_DIGITS("r                :",r_digits,ndigs_r);
            }
            free(e);
            free(dq);
            free(dr);
        }
    }

    DBG_PRINTF_DIGITS("Quotient         :",q_digits,ndigs_q);

    // Remove leading zero digits from remainder
    while(ndigs_r > 0 && r_digits[ndigs_r-1] == 0)
        ndigs_r -= 1;
    DBG_PRINTF_DIGITS("Remainder        :",r_digits,ndigs_r);
    
    PyLongObject *r = _PyLong_New(ndigs_r);
    memcpy(r->ob_digit, r_digits, sizeof(digit)*ndigs_r);
    free(r_digits);

    return Py_BuildValue("OO", q, r);
}

PyObject *pygf2x_get_MAX_BITS(PyObject *self,
                              PyObject *nbits_obj)
{
    // Return maximum allowed n.o. bits
    return PyLong_FromSsize_t(PYGF2X_MAX_DIGITS*PyLong_SHIFT);
}

PyObject *pygf2x_set_MAX_BITS(PyObject *self,
                              PyObject *nbits_obj)
{
    // Set maximum allowed n.o. bits
    // Argument must be a multiple of built-in integer digit size (15 or 30)
    if (! PyLong_CheckExact(nbits_obj)) {
        PyErr_SetString(PyExc_ValueError, "Argument to set_max_bits must be an integer");
        return NULL;
    }
    Py_ssize_t nbits = PyLong_AsSsize_t(nbits_obj);
    if(nbits%PyLong_SHIFT) {
        PyErr_SetString(PyExc_ValueError, "Argument is not a multiple of digit size (sys.int_info.bits_per_digit)");
        return NULL;
    }
    PYGF2X_MAX_DIGITS = nbits/PyLong_SHIFT;

    Py_INCREF(Py_None);
    return Py_None;
}

PyMethodDef pygf2x_functions[] =
    {
        {
            "divmod",
            pygf2x_divmod,
            METH_VARARGS,
            "Divide two integers as polynomials over GF(2) (returns quotient and remainder)"
        },
        {
            "mul",
            pygf2x_mul,
            METH_VARARGS,
            "Multiply two integers as polynomials over GF(2)"
        },
        {
            "sqr",
            pygf2x_sqr,
            METH_VARARGS,
            "Square one integer as polynomial over GF(2)"
        },
        {
            "inv",
            pygf2x_inv,
            METH_VARARGS,
            "Multiplicative inverse of integer as polynomial over GF(2), with given precision"
        },
        {
            "get_MAX_BITS",
            pygf2x_get_MAX_BITS,
            METH_NOARGS,
            "Get maximum allowed gint bit_length"
        },
        {
            "set_MAX_BITS",
            pygf2x_set_MAX_BITS,
            METH_O,
            "Set maximum allowed gint bit_length"
        },
        {
            NULL,                   // const char  *ml_name;  /* The name of the built-in function/method   */
            NULL,                   // PyCFunction ml_meth;   /* The C function that implements it          */
            0,                      // int         ml_flags;  /* Combination of METH_xxx flags, which mostly*/
            //                        /* describe the args expected by the C func   */
            NULL                    // const char  *ml_doc;   /* The __doc__ attribute, or NULL             */
        }
    };


struct PyModuleDef pygf2x_module =
    {
        // Python module definition
        .m_base = PyModuleDef_HEAD_INIT,
        .m_name = "pygf2x",       // Name of the module.
        .m_doc  = NULL,                   // Docstring for the module - in this case empty.
        .m_size = -1,                     // Used by sub-interpreters, if you do not know what
        // it is then you do not need it, keep -1 .
        .m_methods = pygf2x_functions,  // Structures of type `PyMethodDef` with functions
        // (or "methods") provided by the module.
        .m_slots = NULL,
        .m_traverse = NULL,
        .m_clear = NULL,
        .m_free = NULL
    };


PyMODINIT_FUNC PyInit_pygf2x(void)
{
    // Python module initialization
    PyObject *pygf2x = PyModule_Create(&pygf2x_module);

    return pygf2x;
}
