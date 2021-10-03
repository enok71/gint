/* -*- mode: c; c-basic-offset: 4; -*- */
/*******************************************************************************
 *
 * Copyright (c) 2020 Oskar Enoksson. All rights reserved.
 * Licensed under the MIT license. See LICENSE file in the project root for details.
 *
 * Description:
 * Base (generic) implementation of arithmetic on infinite polynomials over GF(2)
 *
 *******************************************************************************/

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <string.h>
#include <stdio.h>

#if defined(__SSE__)
#include <immintrin.h>
#elif defined(__ARM_NEON)
#include <mmintrin.h>
#endif

// Define to enable DBG_ASSERT checks
#define DEBUG_PYGF2X

// Define to enable DBG_PRINTF and DBG_PRINTF_DIGITS printouts
//#define DEBUG_PYGF2X_VERBOSE

// Define to restrict DBG_PRINTF and DBG_PRINTF_DIGITS printouts to a specific function
//#define DEBUG_PYGF2X_VERBOSE_FUNCTION "rinverse"

#ifdef DEBUG_PYGF2X_VERBOSE_FUNCTION
#define DEBUG_PYGF2X_COND if(strcmp(__func__,DEBUG_PYGF2X_VERBOSE_FUNCTION)==0)
#else
#define DEBUG_PYGF2X_COND
#endif

#ifdef DEBUG_PYGF2X
#include <execinfo.h>
#define DBG_ASSERT(x) { if (! (x) ) { fprintf(stderr,"Assertion failed on line %d: %s\n",__LINE__,#x); my_abort(); } }
static void my_abort() {
  void *array[3];
  size_t size;

  // get void*'s for all entries on the stack
  size = backtrace(array, 3);

  // print out all the frames to stderr
  backtrace_symbols_fd(array, size, STDERR_FILENO);
  exit(1);
}
#else
#define DBG_ASSERT(x)
#endif

#ifdef DEBUG_PYGF2X_VERBOSE
#define DBG_PRINTF(...) { DEBUG_PYGF2X_COND printf(__VA_ARGS__); }
#define DBG_PRINTF_DIGITS(msg,digits,ndigs) { DEBUG_PYGF2X_COND { DBG_PRINTF(msg); for(int i=(ndigs)-1; i>=0; i--) DBG_PRINTF("%08x'", (digits)[i]); DBG_PRINTF("\n"); }; }
#else
#define DBG_PRINTF(...)
#define DBG_PRINTF_DIGITS(...)
#endif

#define GF2X_MAX(a,b) ((a>b) ? (a) : (b))
#define GF2X_MIN(a,b) ((a<b) ? (a) : (b))

#define LIMIT_DIV_BITWISE 0
#define LIMIT_DIV_EUCLID 0
#define NDIV 10
#define inv_NDIV inv_10

// Squares up to 255x255 (8-bit chunk size)
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


//
// Inverse table for (1<<(N-1))..(1<<(2^N)-1), N-bit denominators with msb=1.
// Table lists inv(d) where
// inv(d)*d = ( 1<<(2N-2) ) + r
// inv(d) is N-bits (msb always=1) and rem(d) is N-1 bits
//

static const uint8_t inv_8[1<<7] = {
    0x80,0x81,0x82,0x83,0x84,0x85,0x86,0x87,0x88,0x89,0x8a,0x8b,0x8c,0x8d,0x8e,0x8f,
    0x92,0x93,0x90,0x91,0x96,0x97,0x94,0x95,0x9a,0x9b,0x98,0x99,0x9e,0x9f,0x9c,0x9d,
    0xaa,0xab,0xa8,0xa9,0xae,0xaf,0xac,0xad,0xa2,0xa3,0xa0,0xa1,0xa6,0xa7,0xa4,0xa5,
    0xb9,0xb8,0xbb,0xba,0xbd,0xbc,0xbf,0xbe,0xb1,0xb0,0xb3,0xb2,0xb5,0xb4,0xb7,0xb6,
    0xff,0xfe,0xfd,0xfc,0xfa,0xfb,0xf8,0xf9,0xf5,0xf4,0xf7,0xf6,0xf0,0xf1,0xf2,0xf3,
    0xe9,0xe8,0xeb,0xea,0xec,0xed,0xee,0xef,0xe3,0xe2,0xe1,0xe0,0xe6,0xe7,0xe4,0xe5,
    0xdb,0xda,0xd9,0xd8,0xde,0xdf,0xdc,0xdd,0xd1,0xd0,0xd3,0xd2,0xd4,0xd5,0xd6,0xd7,
    0xcc,0xcd,0xce,0xcf,0xc9,0xc8,0xcb,0xca,0xc6,0xc7,0xc4,0xc5,0xc3,0xc2,0xc1,0xc0
};
/*
static const uint16_t inv_9[1<<8] = {
    0x100,0x101,0x102,0x103,0x104,0x105,0x106,0x107,0x108,0x109,0x10a,0x10b,0x10c,0x10d,0x10e,0x10f,
    0x110,0x111,0x112,0x113,0x114,0x115,0x116,0x117,0x118,0x119,0x11a,0x11b,0x11c,0x11d,0x11e,0x11f,
    0x124,0x125,0x126,0x127,0x120,0x121,0x122,0x123,0x12c,0x12d,0x12e,0x12f,0x128,0x129,0x12a,0x12b,
    0x134,0x135,0x136,0x137,0x130,0x131,0x132,0x133,0x13c,0x13d,0x13e,0x13f,0x138,0x139,0x13a,0x13b,
    0x154,0x155,0x156,0x157,0x150,0x151,0x152,0x153,0x15c,0x15d,0x15e,0x15f,0x158,0x159,0x15a,0x15b,
    0x145,0x144,0x147,0x146,0x141,0x140,0x143,0x142,0x14d,0x14c,0x14f,0x14e,0x149,0x148,0x14b,0x14a,
    0x173,0x172,0x171,0x170,0x177,0x176,0x175,0x174,0x17b,0x17a,0x179,0x178,0x17f,0x17e,0x17d,0x17c,
    0x162,0x163,0x160,0x161,0x166,0x167,0x164,0x165,0x16a,0x16b,0x168,0x169,0x16e,0x16f,0x16c,0x16d,
    0x1fe,0x1ff,0x1fc,0x1fd,0x1fb,0x1fa,0x1f9,0x1f8,0x1f4,0x1f5,0x1f6,0x1f7,0x1f1,0x1f0,0x1f3,0x1f2,
    0x1eb,0x1ea,0x1e9,0x1e8,0x1ee,0x1ef,0x1ec,0x1ed,0x1e1,0x1e0,0x1e3,0x1e2,0x1e4,0x1e5,0x1e6,0x1e7,
    0x1d2,0x1d3,0x1d0,0x1d1,0x1d7,0x1d6,0x1d5,0x1d4,0x1d8,0x1d9,0x1da,0x1db,0x1dd,0x1dc,0x1df,0x1de,
    0x1c7,0x1c6,0x1c5,0x1c4,0x1c2,0x1c3,0x1c0,0x1c1,0x1cd,0x1cc,0x1cf,0x1ce,0x1c8,0x1c9,0x1ca,0x1cb,
    0x1b7,0x1b6,0x1b5,0x1b4,0x1b2,0x1b3,0x1b0,0x1b1,0x1bd,0x1bc,0x1bf,0x1be,0x1b8,0x1b9,0x1ba,0x1bb,
    0x1a3,0x1a2,0x1a1,0x1a0,0x1a6,0x1a7,0x1a4,0x1a5,0x1a9,0x1a8,0x1ab,0x1aa,0x1ac,0x1ad,0x1ae,0x1af,
    0x198,0x199,0x19a,0x19b,0x19d,0x19c,0x19f,0x19e,0x192,0x193,0x190,0x191,0x197,0x196,0x195,0x194,
    0x18c,0x18d,0x18e,0x18f,0x189,0x188,0x18b,0x18a,0x186,0x187,0x184,0x185,0x183,0x182,0x181,0x180
};
*/
static const uint16_t inv_10[1<<9] = {
    0x200,0x201,0x202,0x203,0x204,0x205,0x206,0x207,0x208,0x209,0x20a,0x20b,0x20c,0x20d,0x20e,0x20f,
    0x210,0x211,0x212,0x213,0x214,0x215,0x216,0x217,0x218,0x219,0x21a,0x21b,0x21c,0x21d,0x21e,0x21f,
    0x222,0x223,0x220,0x221,0x226,0x227,0x224,0x225,0x22a,0x22b,0x228,0x229,0x22e,0x22f,0x22c,0x22d,
    0x232,0x233,0x230,0x231,0x236,0x237,0x234,0x235,0x23a,0x23b,0x238,0x239,0x23e,0x23f,0x23c,0x23d,
    0x249,0x248,0x24b,0x24a,0x24d,0x24c,0x24f,0x24e,0x241,0x240,0x243,0x242,0x245,0x244,0x247,0x246,
    0x259,0x258,0x25b,0x25a,0x25d,0x25c,0x25f,0x25e,0x251,0x250,0x253,0x252,0x255,0x254,0x257,0x256,
    0x26b,0x26a,0x269,0x268,0x26f,0x26e,0x26d,0x26c,0x263,0x262,0x261,0x260,0x267,0x266,0x265,0x264,
    0x27b,0x27a,0x279,0x278,0x27f,0x27e,0x27d,0x27c,0x273,0x272,0x271,0x270,0x277,0x276,0x275,0x274,
    0x2aa,0x2ab,0x2a8,0x2a9,0x2ae,0x2af,0x2ac,0x2ad,0x2a2,0x2a3,0x2a0,0x2a1,0x2a6,0x2a7,0x2a4,0x2a5,
    0x2bb,0x2ba,0x2b9,0x2b8,0x2bf,0x2be,0x2bd,0x2bc,0x2b3,0x2b2,0x2b1,0x2b0,0x2b7,0x2b6,0x2b5,0x2b4,
    0x28a,0x28b,0x288,0x289,0x28e,0x28f,0x28c,0x28d,0x282,0x283,0x280,0x281,0x286,0x287,0x284,0x285,
    0x29b,0x29a,0x299,0x298,0x29f,0x29e,0x29d,0x29c,0x293,0x292,0x291,0x290,0x297,0x296,0x295,0x294,
    0x2e5,0x2e4,0x2e7,0x2e6,0x2e1,0x2e0,0x2e3,0x2e2,0x2ed,0x2ec,0x2ef,0x2ee,0x2e9,0x2e8,0x2eb,0x2ea,
    0x2f4,0x2f5,0x2f6,0x2f7,0x2f0,0x2f1,0x2f2,0x2f3,0x2fc,0x2fd,0x2fe,0x2ff,0x2f8,0x2f9,0x2fa,0x2fb,
    0x2c5,0x2c4,0x2c7,0x2c6,0x2c1,0x2c0,0x2c3,0x2c2,0x2cd,0x2cc,0x2cf,0x2ce,0x2c9,0x2c8,0x2cb,0x2ca,
    0x2d4,0x2d5,0x2d6,0x2d7,0x2d0,0x2d1,0x2d2,0x2d3,0x2dc,0x2dd,0x2de,0x2df,0x2d8,0x2d9,0x2da,0x2db,
    0x3ff,0x3fe,0x3fd,0x3fc,0x3fa,0x3fb,0x3f8,0x3f9,0x3f5,0x3f4,0x3f7,0x3f6,0x3f0,0x3f1,0x3f2,0x3f3,
    0x3ea,0x3eb,0x3e8,0x3e9,0x3ef,0x3ee,0x3ed,0x3ec,0x3e0,0x3e1,0x3e2,0x3e3,0x3e5,0x3e4,0x3e7,0x3e6,
    0x3d6,0x3d7,0x3d4,0x3d5,0x3d3,0x3d2,0x3d1,0x3d0,0x3dc,0x3dd,0x3de,0x3df,0x3d9,0x3d8,0x3db,0x3da,
    0x3c3,0x3c2,0x3c1,0x3c0,0x3c6,0x3c7,0x3c4,0x3c5,0x3c9,0x3c8,0x3cb,0x3ca,0x3cc,0x3cd,0x3ce,0x3cf,
    0x3a7,0x3a6,0x3a5,0x3a4,0x3a2,0x3a3,0x3a0,0x3a1,0x3ad,0x3ac,0x3af,0x3ae,0x3a8,0x3a9,0x3aa,0x3ab,
    0x3b2,0x3b3,0x3b0,0x3b1,0x3b7,0x3b6,0x3b5,0x3b4,0x3b8,0x3b9,0x3ba,0x3bb,0x3bd,0x3bc,0x3bf,0x3be,
    0x38e,0x38f,0x38c,0x38d,0x38b,0x38a,0x389,0x388,0x384,0x385,0x386,0x387,0x381,0x380,0x383,0x382,
    0x39b,0x39a,0x399,0x398,0x39e,0x39f,0x39c,0x39d,0x391,0x390,0x393,0x392,0x394,0x395,0x396,0x397,
    0x36d,0x36c,0x36f,0x36e,0x368,0x369,0x36a,0x36b,0x367,0x366,0x365,0x364,0x362,0x363,0x360,0x361,
    0x379,0x378,0x37b,0x37a,0x37c,0x37d,0x37e,0x37f,0x373,0x372,0x371,0x370,0x376,0x377,0x374,0x375,
    0x346,0x347,0x344,0x345,0x343,0x342,0x341,0x340,0x34c,0x34d,0x34e,0x34f,0x349,0x348,0x34b,0x34a,
    0x352,0x353,0x350,0x351,0x357,0x356,0x355,0x354,0x358,0x359,0x35a,0x35b,0x35d,0x35c,0x35f,0x35e,
    0x333,0x332,0x331,0x330,0x336,0x337,0x334,0x335,0x339,0x338,0x33b,0x33a,0x33c,0x33d,0x33e,0x33f,
    0x327,0x326,0x325,0x324,0x322,0x323,0x320,0x321,0x32d,0x32c,0x32f,0x32e,0x328,0x329,0x32a,0x32b,
    0x318,0x319,0x31a,0x31b,0x31d,0x31c,0x31f,0x31e,0x312,0x313,0x310,0x311,0x317,0x316,0x315,0x314,
    0x30c,0x30d,0x30e,0x30f,0x309,0x308,0x30b,0x30a,0x306,0x307,0x304,0x305,0x303,0x302,0x301,0x300,
};

//
// Inverse table for (1..(1<<(2^N)-1), N-bit denominators with msb=1.
// Only odd inputs are listed (because even numbers don't have this kind of inverse)
// Table lists inv(d>>1)>>1 where
// inv(d)*d = ( r<<(N-1) ) + 1
// inv(d) is N-bits (lsb always=1) and rem(d) is N-1 bits
//
static const uint8_t rinv_8[1<<7] = {
    0x00,0x7f,0x2a,0xed,0x24,0x4b,0xce,0x19,0x08,0xd7,0xa2,0xc5,0x2c,0xe3,0x46,0x31,
    0x10,0x2f,0x3a,0xbd,0x34,0x1b,0xde,0x49,0x18,0x87,0xb2,0x95,0x3c,0xb3,0x56,0x61,
    0x20,0xdf,0x0a,0x4d,0x04,0xeb,0xee,0xb9,0x28,0x77,0x82,0x65,0x0c,0x43,0x66,0x91,
    0x30,0x8f,0x1a,0x1d,0x14,0xbb,0xfe,0xe9,0x38,0x27,0x92,0x35,0x1c,0x13,0x76,0xc1,
    0x40,0x3f,0x6a,0xad,0x64,0x0b,0x8e,0x59,0x48,0x97,0xe2,0x85,0x6c,0xa3,0x06,0x71,
    0x50,0x6f,0x7a,0xfd,0x74,0x5b,0x9e,0x09,0x58,0xc7,0xf2,0xd5,0x7c,0xf3,0x16,0x21,
    0x60,0x9f,0x4a,0x0d,0x44,0xab,0xae,0xf9,0x68,0x37,0xc2,0x25,0x4c,0x03,0x26,0xd1,
    0x70,0xcf,0x5a,0x5d,0x54,0xfb,0xbe,0xa9,0x78,0x67,0xd2,0x75,0x5c,0x53,0x36,0x81
};

static inline int nbits(PyLongObject *integer) {
    // return 1-based index of the most significant non-zero bit, or 0 if all bits are zero
    return _PyLong_NumBits((PyObject *)integer);
}

static uint32_t
mul_15_15(uint16_t l, uint16_t r)
// Multiply two unsigned 15-bit polynomials over GF(2) (stored in uint16_t)
// Return as a 29-bit polynomial (stored in uint32_t)
{
    uint8_t l0 = l & 0x1f;
    l>>=5;
    uint8_t l1 = l&0x1f;
    l>>=5;
    uint8_t l2 = l&0x1f;
    uint8_t r0 = r&0x1f;
    r>>=5;
    uint8_t r1 = r&0x1f;
    r>>=5;
    uint8_t r2 = r&0x1f;

    uint32_t p = mul_5_5[l2][r2];
    p <<= 5;
    p ^= mul_5_5[l1][r2] ^ mul_5_5[l2][r1];
    p <<= 5;
    p ^= mul_5_5[l0][r2] ^ mul_5_5[l1][r1] ^ mul_5_5[l2][r0];
    p <<= 5;
    p ^= mul_5_5[l0][r1] ^ mul_5_5[l1][r0];
    p <<= 5;
    p ^= mul_5_5[l0][r0];

    return p;
}

static uint64_t
mul_30_30(uint32_t l, uint32_t r) 
// Multiply two unsigned 30-bit polynomials over GF(2) (stored in uint32_t)
// Return as a 59-bit polynomial (stored in uint64_t)
{
#if defined(__PCLMUL__)
    __m128i li = {l,0};
    __m128i ri = {r,0};
    __m128i pi = _mm_clmulepi64_si128(li,ri,0);
    return pi[0];
#elif defined(__ARM_NEON)
    poly8x8_t li1 = {l>>16, l>>16, l>>16, l>>16, l>>24, l>>24, l>>24, l>>24};
    poly8x8_t ri  = {r    , r>> 8, r>>16, r>>24, r    , r>> 8, r>>16, r>>24};
    poly16x8  pi1 = vmull_p8(li1,ri);
    poly8x8_t li0 = {l    , l    , l    , l    , l>> 8, l>> 8, l>> 8, l>> 8};
    poly16x8  pi0 = vmull_p8(li0,ri);
    uint64_t  p0  = pi1[7];
    p <<= 8;
    p ^= pi1[6] ^ pi1[3];
    p <<= 8;
    p ^= pi1[5] ^ pi1[2] ^ pi0[7];
    p <<= 8;
    p ^= pi1[4] ^ pi1[1] ^ pi0[6] ^ pi0[3];
    p <<= 8;
    p ^= pi1[0] ^ pi0[5] ^ pi0[2];
    p <<= 8;
    p ^= pi0[4] ^ pi0[1];
    p <<= 8;
    p ^= pi0[0];

    return p;
#else
    // Use Karatsubas formula and mul_15_15
    uint16_t ll = l &0x7fff;
    uint16_t lh = (l >> 15) &0x7fff;
    uint16_t rl = r &0x7fff;
    uint16_t rh = (r >> 15) &0x7fff;

    uint32_t z0 = mul_15_15(ll,rl);
    uint32_t z2 = mul_15_15(lh,rh);
    uint32_t z1 = mul_15_15(ll ^ lh, rl ^ rh) ^z2 ^z0;

    return ((((uint64_t)z2 << 15) ^ (uint64_t)z1 ) << 15) ^ (uint64_t)z0;
#endif
}

static void mul_5_nr(digit * restrict const p,
		     uint8_t l,
		     const digit * restrict const r0, int nr)
{
#if (PyLong_SHIFT == 15)
    for(int id_r=0; id_r<nr; id_r++) {
	uint16_t ri = r0[id_r];
	uint16_t pi_0 = mul_5_5[l][ri&0x1f];
	ri >>= 5;
	uint16_t pi_1 = mul_5_5[l][ri&0x1f];
	ri >>= 5;
	uint16_t pi_2 = mul_5_5[l][ri&0x1f];
	p[id_r] ^= ((((pi_2 << 5) ^ pi_1) << 5) ^ pi_0) & PyLong_MASK;
	p[id_r+1] ^= pi_2) >> 5;
    }
#else // PyLong_SHIFT == 30
    for(int id_r=0; id_r<nr; id_r++) {
	uint32_t ri = r0[id_r];
	uint16_t pi_0 = mul_5_5[l][ri&0x1f];
	ri >>= 5;
	uint16_t pi_1 = mul_5_5[l][ri&0x1f];
	ri >>= 5;
	uint16_t pi_2 = mul_5_5[l][ri&0x1f];
	ri >>= 5;
	uint16_t pi_3 = mul_5_5[l][ri&0x1f];
	ri >>= 5;
	uint16_t pi_4 = mul_5_5[l][ri&0x1f];
	ri >>= 5;
	uint16_t pi_5 = mul_5_5[l][ri&0x1f];
	p[id_r] ^= ((((((((((pi_5 << 5) ^ pi_4) << 5) ^ pi_3) << 5) ^ pi_2) << 5) ^ pi_1) << 5) ^ pi_0) & PyLong_MASK;
	p[id_r+1] ^= pi_5 >> 5;
    }
#endif
}

static void mul_15_nr(digit * restrict const p,
		      uint16_t l,
		      const digit * restrict const r0, int nr)
{
    for(int id_r=0; id_r<nr; id_r++) {
#if (PyLong_SHIFT == 15)
	uint32_t pi = mul_15_15(l, r0[id_r]);
	p[id_r] ^= (pi & PyLong_MASK);
	p[id_r+1] ^= (pi >> PyLong_SHIFT);
#elif PyLong_SHIFT == 30
	uint16_t ri_0 = r0[id_r]&0x7fff;
	uint16_t ri_1 = r0[id_r]>>15;
	uint32_t pi_0 = mul_15_15(l, ri_0);
	uint32_t pi_1 = mul_15_15(l, ri_1);
	p[id_r] ^= pi_0 ^ ((pi_1 & ((1<<15)-1))<<15);
	p[id_r+1] ^= (pi_1 >> 15);
#else
#error
#endif
    }
}

static void mul_digit_nr(digit * restrict const p,
			 digit l,
		      const digit * restrict const r0, int nr)
{
    for(int id_r=0; id_r<nr; id_r++) {
#if (PyLong_SHIFT == 15)
	twodigits pi = mul_15_15(l, r0[id_r]);
#elif PyLong_SHIFT == 30
	twodigits pi = mul_30_30(l, r0[id_r]);
#else
#error
#endif
	p[id_r] ^= (pi & PyLong_MASK);
	p[id_r+1] ^= (pi >> PyLong_SHIFT);
    }
}

static void
mul_60_60(uint64_t p[2], uint64_t l, uint64_t r)
{
#if defined(__PCLMUL__)
    __m128i li = {l,0};
    __m128i ri = {r,0};
    __m128i pi = _mm_clmulepi64_si128(li,ri,0);
    p[0] = pi[0];
    p[1] = pi[1];
#else
    // Use Karatsubas formula and mul_30_30
    uint32_t ll = l &0x3fffffff;
    uint32_t lh = (l >> 30) &0x3fffffff;
    uint32_t rl = r &0x3fffffff;
    uint32_t rh = (r >> 30) &0x3fffffff;

    uint64_t z0 = mul_30_30(ll,rl);
    uint64_t z2 = mul_30_30(lh,rh);
    uint64_t z1 = mul_30_30(ll ^ lh, rl ^ rh) ^z2 ^z0;

    p[0] = (((z2 << 30) ^ z1) << 30) ^ z0;
    p[1] = ((z1 >> 30) ^ z2) >> 4;
#endif    
}

static void
square_n(digit * restrict result, int ndigs_p, const digit *fdigits, int ndigs_f)
{
    int idp=0;
    for(int id=0; id<ndigs_f; id++) {
	digit ic = fdigits[id];
#if (PyLong_SHIFT == 15)
	digit pc0 = sqr_8[ic&0xff];
	result[idp++] ^= pc0;
	ic>>=8;
	digit pc1 = sqr_8[ic&0x7f];
	if(idp < ndigs_p)
	    result[idp++] ^= pc1;
#elif (PyLong_SHIFT == 30)		
	uint16_t pc0 = sqr_8[ic&0xff];
	ic>>=8;
	uint16_t pc1 = sqr_8[ic&0x7f];
	result[idp++] ^= (pc1 << 16) ^ pc0;
	ic>>=7;
	uint16_t pc2 = sqr_8[ic&0xff];
	ic>>=8;
	uint16_t pc3 = sqr_8[ic&0x7f];
	if(idp < ndigs_p)
	    result[idp++] ^= (pc3 << 16) ^ pc2;
#else
#error
#endif
    }
}

static PyObject *
pygf2x_sqr(PyObject *self, PyObject *args) {
    // Square one Python integer, interpreted as polynomial over GF(2)
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

    int nbits_f = nbits(f);
    int nbits_p = 2*nbits_f -1;
    int ndigs_p = (nbits_p + (PyLong_SHIFT-1))/PyLong_SHIFT;
    int ndigs_f = ((PyVarObject *)f)->ob_size;

    PyLongObject *p = _PyLong_New(ndigs_p);
    digit *restrict result = p->ob_digit;
    memset(result,0,ndigs_p*sizeof(digit));
    
    DBG_PRINTF("Bits per digit   = %-4d\n",PyLong_SHIFT);
    DBG_PRINTF("factor bits      = %-4d\n",nbits_f);
    DBG_PRINTF("Square digits    = %-4d\n",ndigs_p);

    square_n(result, ndigs_p, f->ob_digit, ndigs_f);

    DBG_PRINTF_DIGITS("Square:", result, ndigs_p);

    return (PyObject *)p;
}

static void mul_nl_nr(digit * restrict const p,
		      const digit * restrict const l0, int nl,
		      const digit * restrict const r0, int nr)
//
// Recursive function for Karatsuba multiplication
//
{
#ifdef DEBUG_PYGF2X
    static int depth = 0;
#endif
    if(nl == 1) {
	// Stop recursing
	DBG_PRINTF("%-2d:[0,%d],[0,%d]\n",depth,nl,nr);
	if(l0[0]<(1<<5))
	    mul_5_nr(p, l0[0], r0, nr);
#if (PyLong_SHIFT == 30)
	else if(l0[0]<(1<<15))
	    mul_15_nr(p, l0[0], r0, nr);
#endif
	else
	    mul_digit_nr(p, l0[0], r0, nr);
    } else if(nr == 1) {
	// Stop recursing
	DBG_PRINTF("%-2d:[0,%d],[0,%d]\n",depth,nl,nr);
	if(r0[0]<(1<<5))
	    mul_5_nr(p, r0[0], l0, nl);
#if (PyLong_SHIFT == 30)
	else if(r0[0]<(1<<15))
	    mul_15_nr(p, r0[0], l0, nl);
#endif
	else
	    mul_digit_nr(p, r0[0], l0, nl);
    } else if(nl > 2*nr) {
    // Divide l to form more equal sized pieces
	int nc = nl/nr; // Number of chunks
	for(int ic=0; ic<nc; ic++) {
	    int icu = (ic+1)*nl/nc;
	    int icl = ic*nl/nc;
	    mul_nl_nr(p+icl, l0+icl, icu-icl, r0, nr);
	}
    } else if(nr > 2*nl) {
    // Divide l to form more equal sized pieces
	int nc = nr/nl; // Number of chunks
	for(int ic=0; ic<nc; ic++) {
	    int icu = (ic+1)*nr/nc;
	    int icl = ic*nr/nc;
	    mul_nl_nr(p+icl, l0, nl, r0+icl, icu-icl);
	}
    } else if(nl>1 && nr>1) {
    // Use Karatsuba
	int m = GF2X_MIN(nl/2,nr/2);
	const digit * restrict l1 = l0+m;
	const digit * restrict r1 = r0+m;
	DBG_PRINTF("%-2d:[0,%d,%d],[0,%d,%d]\n",depth,m,nl,m,nr);

	digit r01[nr-m];               // r01 = r0^r1
	for(int i=0; i<m; i++)
	    r01[i] = r0[i] ^ r1[i];
	for(int i=m; i<nr-m; i++)
	    r01[i] = r1[i];
	
	digit l01[nl-m];               // l01 = l0^l1
	for(int i=0; i<m; i++)
	    l01[i] = l0[i] ^ l1[i];
	for(int i=m; i<nl-m; i++)
	    l01[i] = l1[i];

#ifdef DEBUG_PYGF2X
	depth += 1;
#endif
	digit z0[2*m];                 // z0 = l0*r0
	memset(z0,0,sizeof(z0));
	mul_nl_nr(z0, l0, m, r0, m);

	digit z2[(nl-m)+(nr-m)];       // z2 = l1*r1
	memset(z2,0,sizeof(z2));
	mul_nl_nr(z2, l1, nl-m, r1, nr-m);
	
	digit z1[(nl-m)+(nr-m)];       // z1 = l01*r01
	memset(z1,0,sizeof(z1));
	mul_nl_nr(z1, l01, nl-m, r01, nr-m);

#ifdef DEBUG_PYGF2X
	depth -= 1;
#endif
	for(int id_p=0; id_p<2*m; id_p++)
	    p[id_p] ^= z0[id_p];

	for(int id_p=0; id_p<2*m; id_p++)
	    (p+m)[id_p] ^= z0[id_p] ^ z1[id_p] ^ z2[id_p];
	for(int id_p=2*m; id_p<(nl-m)+(nr-m); id_p++)
	    (p+m)[id_p] ^= z1[id_p] ^ z2[id_p];

	for(int id_p=0; id_p<(nl-m)+(nr-m); id_p++)
	    (p+2*m)[id_p] ^= z2[id_p];
    }
}

static PyObject *
pygf2x_mul(PyObject *self, PyObject *args) {
    // Multiply two Python integers, interpreted as polynomials over GF(2)
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

    int nbits_l = nbits(fl);
    int nbits_r = nbits(fr);
    int nbits_p = nbits_l + nbits_r -1;
    int ndigs_l = (nbits_l + (PyLong_SHIFT-1))/PyLong_SHIFT;
    int ndigs_r = (nbits_r + (PyLong_SHIFT-1))/PyLong_SHIFT;
    int ndigs_p = (nbits_p + (PyLong_SHIFT-1))/PyLong_SHIFT;

    digit result[ndigs_l + ndigs_r];
    memset(result,0,sizeof(result));
    
    DBG_PRINTF("Bits per digit   = %-4d\n",PyLong_SHIFT);
    DBG_PRINTF("Left factor bits = %-4d\n",nbits_l);
    DBG_PRINTF("Right factor bits= %-4d\n",nbits_r);
    DBG_PRINTF("Product digits   = %-4d\n",ndigs_p);

    mul_nl_nr(result, fl->ob_digit, ndigs_l, fr->ob_digit, ndigs_r);

    DBG_PRINTF_DIGITS("Product          :",result,ndigs_p);

    PyLongObject *p = _PyLong_New(ndigs_p);
    memcpy(p->ob_digit, result, sizeof(digit)*ndigs_p);

    return (PyObject *)p;
}

static inline uint16_t extract_chunk(const digit *digits, int ib, int nch, const int ndigs)
// return nch-bit chunk of data extracted from digits, at msbit position ib.
// Max value for nch is 15 because PyLong_SHIFT can be 15
{
    DBG_ASSERT(nch <= 15);
    DBG_ASSERT(nch <= PyLong_SHIFT);
    DBG_ASSERT(nch-1 <= ib);
    DBG_ASSERT(ib < PyLong_SHIFT*ndigs);
    int id  = ib/PyLong_SHIFT;  // Digit position of most significant bit
    int ibd = ib%PyLong_SHIFT;  // Bit position in digit
    uint16_t ch;
    if(ibd >= nch-1) {
	// The chunk is contained in one single digit
	DBG_ASSERT(id >=0 && id < ndigs);
	ch = digits[id] >> (ibd - (nch-1));
    } else {
	// The chunk is split between two digits
	DBG_ASSERT(id >=1 && id < ndigs);
	ch = (digits[id  ] << ((nch-1) - ibd))
	    |(digits[id-1] >> (PyLong_SHIFT - ((nch-1) - ibd)));
    }
    return ch;
}

static inline void add_chunk(digit *digits, int ib, int nch, uint16_t value, const int ndigs)
// Add (xor) nch-bit chunk of data to digits, at msbit position ib.
// Max value for nch is 15 because PyLong_SHIFT can be 15
{
    DBG_ASSERT(nch <= 15);
    DBG_ASSERT(nch <= PyLong_SHIFT);
    DBG_ASSERT(nch-1 <= ib);
    DBG_ASSERT(ib < PyLong_SHIFT*ndigs);
    DBG_ASSERT(value < (1<<nch));
    int id  = ib/PyLong_SHIFT;  // Digit position of most significant bit
    int ibd = ib%PyLong_SHIFT;  // Bit position in digit
    if(ibd >= nch-1) {
	// The chunk is contained in one single digit
	DBG_ASSERT(id >=0 && id < ndigs);
	digits[id] ^= value << (ibd - (nch-1));
    } else {
	// The chunk is split between two digits
	DBG_ASSERT(id >=1 && id < ndigs);
	digits[id  ] ^= value >> ((nch-1) - ibd);
	digits[id-1] ^= (value & ((1 << ((nch-1) - ibd)) - 1 )) << (PyLong_SHIFT - ((nch-1) - ibd));
    }
}

static void
div_bitwise(digit * restrict q_digits,
	    digit * restrict r_digits,
	    const digit * restrict d_digits,
	    int nbits_n, int nbits_d)
{
    for(int ib_r = nbits_n-1; ib_r >= nbits_d-1; ib_r--) {
	int id_r = (ib_r/PyLong_SHIFT);    // Digit position
	int ibd_r = ib_r-id_r*PyLong_SHIFT; // Bit position in digit
	if(r_digits[id_r] & (1<<ibd_r)) {
	    // Numerator bit is set. Set quotient bit and subtract denominator
	    int ib_q  = ib_r - nbits_d +1;
	    int id_q  = ib_q/PyLong_SHIFT;       // Digit position
	    int ibd_q = ib_q%PyLong_SHIFT;       // Bit position in digit
	    q_digits[id_q] |= (1<<ibd_q);
	    for(int ib_d  = nbits_d-1; ib_d >= 0 ; ib_d--) {
		int id_d  = ib_d/PyLong_SHIFT;   // Digit position
		int ibd_d = ib_d%PyLong_SHIFT;   // Bit position in digit
		int ib_dr  = ib_r - ((nbits_d-1) - ib_d);
		int id_dr  = ib_dr/PyLong_SHIFT; // Digit position
		int ibd_dr = ib_dr%PyLong_SHIFT; // Bit position in digit
		r_digits[id_dr] ^= ((d_digits[id_d] >> ibd_d) & 1) << ibd_dr;
	    }
	}
    }
}

static void
div_NDIVwise(digit * restrict q_digits,
	     digit * restrict r_digits,
	     const digit * restrict d_digits,
	     int nbits_n, int nbits_d,
	     int ndigs_n, int ndigs_d, int ndigs_q, int ndigs_r)
{
    uint16_t inv_dh;
    // Extract dh = most significant "chunk" of denominator, and invert it.
    // bitsize is MIN(NDIV,nbits_d)
    const int nbits_dh = GF2X_MIN(NDIV,nbits_d);
    {
	// Extract the highest chunk of denominator
	int ib_d = nbits_d-1; // Highest bit position in highest digit of denominator
	uint16_t dh = extract_chunk(d_digits, ib_d, nbits_dh, ndigs_d);

	// Invert dh using tablulated inverse
	dh <<= (NDIV - nbits_dh);
	DBG_ASSERT(dh>=(1 << (NDIV-1)) && dh < (1 << NDIV));
	inv_dh = inv_NDIV[dh - (1 << (NDIV-1))];
	DBG_ASSERT(inv_dh < (1 << NDIV));
	inv_dh >>= (NDIV - nbits_dh);
    }
    // inv_dh now contains inverse of dh (nbits_dh bits)

    // Update q and r, processing ndiv bits in each step.
    // ndiv = nbits_dh = min(NDIV,nbits_d), except in the last step where ndiv is smaller to match nbits_d
    // Loop over most significant bit-position in remainder.
    int ib_r_;
    for(int ib_r  = nbits_n - 1; ib_r >= nbits_d - 1; ib_r = ib_r_) {
	// Compute ib_r_ = next remainder bit position ib_r, after the current iteration
	if (ib_r > nbits_dh + (nbits_d - 2))
	    ib_r_ = ib_r - nbits_dh;
	else
	    ib_r_ = nbits_d - 2;
	// ndiv is the number of remainder bits processed in this step.
	const int ndiv = ib_r - ib_r_;

	// Extract rh = the remainder chunk to be eliminated. Bitsize is ndiv.
	uint16_t rh = extract_chunk(r_digits, ib_r, ndiv, ndigs_r);
	DBG_ASSERT(rh < (1<<10));
	DBG_ASSERT(inv_dh < (1<<10));

	// Compute qh = the quotient chunk. Bitsize is ndiv.
	uint16_t qh;
	{
	    //  qh = (rh*inv(dh))>>(nbits_d-1)
	    uint32_t qh_tmp = 
		(( mul_5_5[rh >> 5  ][inv_dh & 0x1f] ^
		   mul_5_5[rh & 0x1f][inv_dh >> 5  ]) << 5)
		^( mul_5_5[rh & 0x1f][inv_dh & 0x1f] )
		^( mul_5_5[rh >> 5  ][inv_dh >> 5  ] << 10);
	    // qh_tmp is now nbits_dh + ndiv - 1 bits
	    qh_tmp >>= (nbits_dh -1);
	    // qh_tmp is now ndiv bits
	    DBG_ASSERT((int)qh_tmp < (1 << ndiv));
	    qh = qh_tmp;
	}

	// Update q by inserting qh in the right position
	int ib_q  = ib_r - (nbits_d - 1);   // Bit position
	add_chunk(q_digits, ib_q, ndiv, qh, ndigs_q);

	// Update remainder r -= (qh*d) << (ib_r_ - (nbits_d - 1))
	// Loop over 5-bit chunks in denominator, starting from least significant end
	// (the last, most significant chunk of the denominator may be smaller than 5 bits)
	for(int jb_d = 0; jb_d < nbits_d; jb_d += 5 ) {
	    // Note: jb_d is the least significant bit position of the denominator chunk processed in this step
	    int ib_d = GF2X_MIN(jb_d + 4, nbits_d - 1);
	    int nbits_dc = ib_d - jb_d + 1;
	    // Extract the denominator chunk
	    int jd_d  = jb_d/PyLong_SHIFT;
	    int jbd_d = jb_d%PyLong_SHIFT;
	    DBG_ASSERT(jd_d < ndigs_d);
	    uint16_t dc = (d_digits[jd_d] >> jbd_d) & 0x1f;

	    // Multiply qh with denominator chunk
	    // The result qhdc is at most ndiv+nbits_dc-1 bits
	    DBG_ASSERT(dc<(1<<nbits_dc));
	    DBG_ASSERT(qh<(1<<ndiv));
	    uint16_t qhdc = (mul_5_5[dc][qh>>5] << 5) ^ mul_5_5[dc][qh&0x1f];
	    int nbits_qhdc = ndiv + nbits_dc -1;
	    DBG_ASSERT(qhdc < (1 << nbits_qhdc));

	    // Add qhdc to the remainder at the appropriate position
	    int ib_qhdc = ib_r - ((nbits_d-1) - jb_d) + (nbits_dc-1);
	    add_chunk(r_digits, ib_qhdc, nbits_qhdc, qhdc, ndigs_r);
	}
	DBG_ASSERT(extract_chunk(r_digits, ib_r, ndiv, ndigs_r) == 0);
    }
}

static void
inverse(digit *restrict e_digits, int ndigs_e, int nbits_e,
	const digit * restrict d_digits, int ndigs_d, int nbits_d)
//
// Compute inverse e to d such that
// e*d == (1 << (nbits_e + nbits_d -2)) + r
// where nbits_r < nbits_d
// nbits_e can be equal, smaller or bigger than nbits_d, allowing
// an inverse with arbitrary accuracy
//
{
    DBG_ASSERT(nbits_d>0);
    DBG_ASSERT(ndigs_d==(nbits_d + (PyLong_SHIFT-1))/PyLong_SHIFT);
    DBG_ASSERT(nbits_e>0);
    DBG_ASSERT(ndigs_e==(nbits_e + (PyLong_SHIFT-1))/PyLong_SHIFT);
    DBG_PRINTF("inv: nbits_d=%d, nbits_e=%d\n", nbits_d, nbits_e);
    DBG_PRINTF("inv: ndigs_d=%d, ndigs_e=%d\n", ndigs_d, ndigs_e);
    DBG_PRINTF_DIGITS("d=", d_digits, ndigs_d);
    
    // Shift the entire d to the left so that the most significant digit has most significant bit =1
    // Truncate it, or fill it with zero, so that it has ndigs_e digits.
    digit d[ndigs_e];
    memset(d,0,sizeof(d));
    {
	const int shift = (PyLong_SHIFT-1) - (nbits_d + (PyLong_SHIFT-1))%PyLong_SHIFT;
	int n0 = GF2X_MAX(0,ndigs_d-ndigs_e);
	DBG_PRINTF("inv: shift=%d, n0=%d\n", shift, n0);
	// Copy and shift all digits from d_digits that can fit into d
	for(int n=ndigs_d-1; n>n0; n--) {
	    d[n-(ndigs_d-ndigs_e)] = ((d_digits[n]<<shift)&PyLong_MASK) | (d_digits[n-1]>>(PyLong_SHIFT-shift));
	}
	d[n0-(ndigs_d-ndigs_e)] = (d_digits[n0] << shift) &PyLong_MASK;
	if(n0 > 0)
	    d[n0-(ndigs_d-ndigs_e)] |= d_digits[n0-1] >> (PyLong_SHIFT-shift);
	DBG_PRINTF_DIGITS("d<<(ne-nd)=", d, ndigs_e);
    }
    if(nbits_e <= 8) {
	// Compute the whole inverse using table
	uint16_t dh = d[ndigs_e-1] >> (PyLong_SHIFT-nbits_e);
	// Invert dh using tablulated inverse
	e_digits[ndigs_e-1] = inv_8[(dh << (8-nbits_e)) - (1 << (8-1))] >> (8-nbits_e);
	DBG_ASSERT(e_digits[ndigs_e-1] < (1u<<nbits_e));
	return;
    }
    //
    // Find initial approximate inverse using table
    //
    {
	// Extract the highest chunk of denominator
	uint16_t dh = d[ndigs_e-1] >> (PyLong_SHIFT-8);
	e_digits[ndigs_e-1] = inv_8[dh - (1 << (8-1))];  // Invert dh using tablulated inverse
	DBG_ASSERT(e_digits[ndigs_e-1] < (1 << 8));
    }
    // e now contains 8 correct bits
    DBG_PRINTF_DIGITS("e=", e_digits, ndigs_e);
    //
    // Take the first Newton-step from 8 correct bits to 15
    //
    if(nbits_e <= 15) {
	// Compute the full inverse in this step
	uint16_t dh = d[ndigs_e-1] >> (PyLong_SHIFT-nbits_e);
	uint16_t x2 = sqr_8[e_digits[ndigs_e-1]];
	e_digits[ndigs_e-1] = mul_15_15(x2, dh) >> 14;         // nbits_e + 15 -1 - nbits_e = 14
	DBG_ASSERT(e_digits[ndigs_e-1] < (1u<<nbits_e));
	return;
    }
    {
	uint16_t dh = d[ndigs_e-1] >> (PyLong_SHIFT-15);
	uint16_t x2 = sqr_8[e_digits[ndigs_e-1]];
	e_digits[ndigs_e-1] = mul_15_15(x2, dh) >> 14; // 15 + 15 -1 - 15 = 14
	DBG_ASSERT(e_digits[ndigs_e-1]<(1<<15));
    }
    // e now contains 15 correct bits
    DBG_PRINTF_DIGITS("e=", e_digits, ndigs_e);
#if (PyLong_SHIFT == 30)
    //
    // Take next Newton-step from 15 correct bits to 30
    //
    if(nbits_e <= 30) {
	// Compute the full inverse in this step
	uint32_t dh = d[ndigs_e-1] >> (PyLong_SHIFT-nbits_e);
	uint32_t x2 = ((uint32_t)sqr_8[e_digits[ndigs_e-1] >> 8] << 16) ^ (uint32_t)sqr_8[e_digits[ndigs_e-1] &0xff];
	// x2 is 2*15-1
	e_digits[ndigs_e-1] = mul_30_30(x2, dh) >> 28; // nbits_e + 29 -1 - nbits_e = 28
	DBG_ASSERT(e_digits[ndigs_e-1] < (1u<<nbits_e));
	return;
    }
    {
	uint32_t dh = d[ndigs_e-1];
	uint32_t x2 = ((uint32_t)sqr_8[e_digits[ndigs_e-1] >> 8] << 16) ^ (uint32_t)sqr_8[e_digits[ndigs_e-1] &0xff];
	e_digits[ndigs_e-1] = mul_30_30(x2, dh) >> 28; // 30 + 29 -1 - 30 = 28
	DBG_ASSERT(e_digits[ndigs_e-1]<(1<<30));
    }
    // e now contains 30 correct bits
    DBG_PRINTF_DIGITS("e=", e_digits, ndigs_e);
#endif
    // e now contains one full correct digit

    //
    // Repeat Newton-steps.
    // In each step the number of correct digits is doubled
    //
    int ncorrect;
    for(ncorrect=1; (ncorrect<<1)<ndigs_e; ncorrect<<=1) {
	DBG_PRINTF("ncorrect=%d\n",ncorrect);
	int n = ncorrect<<1;
	digit x2[n];
	memset(x2, 0, sizeof(x2));
	square_n(x2, n, &e_digits[ndigs_e-ncorrect], ncorrect);
	DBG_PRINTF_DIGITS("x2=", x2, n);
    
	int nn = n<<1;
	digit etmp[nn];
	memset(etmp, 0, sizeof(etmp));
	mul_nl_nr(etmp, &d[ndigs_e-n], n, x2, n);
	DBG_PRINTF_DIGITS("etmp=", etmp, nn);

	// Discard lowest 2*(PyLong_SHIFT*n-1) bits of etmp
	for(int i=1; i<=n; i++)
	    e_digits[ndigs_e-i] = ((etmp[nn-i] << 2) &PyLong_MASK) | (etmp[nn-1-i] >> (PyLong_SHIFT-2));
	DBG_PRINTF_DIGITS("e=", e_digits, ndigs_e);
    }
    // e_digits now contains <ncorrect> correct digits
    
    //
    // Last Newton-step
    //
    {
	DBG_PRINTF("ncorrect=%d\n",ncorrect);
	int n = ncorrect<<1;
	digit x2[n];
	memset(x2, 0, sizeof(x2));
	square_n(x2, n, &e_digits[ndigs_e-ncorrect], ncorrect);
	DBG_PRINTF_DIGITS("x2=", x2, n);
	DBG_PRINTF_DIGITS("d0=", d, ndigs_e);
    
	int nn = ndigs_e<<1;
	digit etmp[nn];
	memset(etmp, 0, sizeof(etmp));
	mul_nl_nr(etmp, d, ndigs_e, &x2[n-ndigs_e], ndigs_e);
	DBG_PRINTF_DIGITS("etmp=", etmp, nn);

	// Discard all but the ndigs_e*PyLong_SHIFT-shift significant bits
	for(int i=1; i<=ndigs_e; i++)
	    e_digits[ndigs_e-i] = ((etmp[nn-i] << 2) &PyLong_MASK) | (etmp[nn-1-i] >> (PyLong_SHIFT-2));
	DBG_PRINTF_DIGITS("e=", e_digits, ndigs_e);

	const int shift = (PyLong_SHIFT-1) - (nbits_e -1)%PyLong_SHIFT;

	// Shift back to e_digits
	for(int i=0; i<ndigs_e-1; i++)
	    e_digits[i] = (e_digits[i]>>shift) | ((e_digits[i+1]<<(PyLong_SHIFT-shift)) &PyLong_MASK);
	e_digits[ndigs_e-1] = e_digits[ndigs_e-1]>>shift;
	DBG_PRINTF_DIGITS("e=", e_digits, ndigs_e);
    }
    return;
}


static PyObject *
pygf2x_inv(PyObject *self, PyObject *args) {
    // Multiplicative inverse of one Python integer, interpreted as polynomial over GF(2)
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
    if(nbits_e <= 0 || nbits_e > ((1<<16)-1)) {
        PyErr_SetString(PyExc_ValueError, "Requested bit_length of inverse is out of range");
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


static void
rinverse(digit *restrict e_digits, int ndigs_e, int nbits_e,
	const digit * restrict d_digits, int ndigs_d, int nbits_d)
//
// Compute inverse e to d such that
// e*d == (r << (nbits_e-1)) + 1
// where nbits_r < nbits_d
// nbits_e can be equal, smaller or bigger than nbits_d, allowing
// an inverse with arbitrary accuracy
//
{
    DBG_ASSERT(nbits_d>0);
    DBG_ASSERT(ndigs_d==(nbits_d + (PyLong_SHIFT-1))/PyLong_SHIFT);
    DBG_ASSERT((d_digits[0] & 1) != 0);
    DBG_ASSERT(nbits_e>0);
    DBG_ASSERT(ndigs_e==(nbits_e + (PyLong_SHIFT-1))/PyLong_SHIFT);
    DBG_PRINTF("inv: nbits_d=%d, nbits_e=%d\n", nbits_d, nbits_e);
    DBG_PRINTF("inv: ndigs_d=%d, ndigs_e=%d\n", ndigs_d, ndigs_e);
    DBG_PRINTF_DIGITS("d=", d_digits, ndigs_d);

    // Copy the lowest nbits_e bits from d_digits to d as starting value
    // Shift the entire d to the left so that the most significant digit has most significant bit =1
    // Truncate it, or fill it with zero, so that it has ndigs_e digits.
    digit d[ndigs_e];
    if(ndigs_e<=ndigs_d) {
	memcpy(d, d_digits, ndigs_e*sizeof(digit));
    } else {
	memcpy(d, d_digits, ndigs_d*sizeof(digit));
	memset(&d[ndigs_d],0,(ndigs_e-ndigs_d)*sizeof(digit));
    }
    if(nbits_e <= 8) {
	// Compute the whole inverse using table
	uint16_t dl = d[0];
	// Invert dl using tablulated inverse
	e_digits[0] = ((rinv_8[dl>>1] <<1) |1) & ((1 << nbits_e) -1);  // Invert dl using tablulated inverse
	DBG_ASSERT(e_digits[0] < (1u << nbits_e));
	return;
    }
    //
    // Find initial approximate inverse using table
    //
    {
	// Extract the lowest chunk of denominator
	uint16_t dl = d[0] & ((1 << 8) -1);
	e_digits[0] = ((rinv_8[dl>>1] <<1) |1) & ((1 << 8) -1);  // Invert dl using tablulated inverse
	DBG_ASSERT(e_digits[0] < (1u << 8));
    }
    // e now contains 8 correct bits
    DBG_PRINTF_DIGITS("e=", e_digits, ndigs_e);
    //
    // Take the first Newton-step from 8 correct bits to 15
    //
    if(nbits_e <= 15) {
	// Compute the full inverse in this step
	uint16_t dl = d[0] & ((1 << 15) -1);
	uint16_t x2 = sqr_8[e_digits[0]];
	e_digits[0] = mul_15_15(x2, dl) & ((1 << nbits_e) -1);
	DBG_ASSERT(e_digits[0] < (1u << nbits_e));
	return;
    }
    {
	uint16_t dl = d[0] & ((1 << 15) -1);
	uint16_t x2 = sqr_8[e_digits[0]];
	e_digits[0] = mul_15_15(x2, dl) & ((1 << 15) -1);
	DBG_ASSERT(e_digits[0] < (1 << 15));
    }
    // e now contains 15 correct bits
    DBG_PRINTF_DIGITS("e=", e_digits, ndigs_e);
#if (PyLong_SHIFT == 30)
    //
    // Take next Newton-step from 15 correct bits to 30
    //
    if(nbits_e <= 30) {
	// Compute the full inverse in this step
	uint32_t dl = d[0] & ((1 << 30) -1);
	uint32_t x2 = ((uint32_t)sqr_8[e_digits[0] >> 8] << 16) ^ (uint32_t)sqr_8[e_digits[0] &0xff];
	// x2 is 2*15-1
	e_digits[0] = mul_30_30(x2, dl) & ((1 << nbits_e) -1);
	DBG_ASSERT(e_digits[0] < (1u << nbits_e));
	return;
    }
    {
	uint32_t dl = d[0] & ((1 << 30) -1);
	uint32_t x2 = ((uint32_t)sqr_8[e_digits[0] >> 8] << 16) ^ (uint32_t)sqr_8[e_digits[0] &0xff];
	e_digits[0] = mul_30_30(x2, dl) & ((1 << 30) -1);
	DBG_ASSERT(e_digits[0] < (1 << 30));
    }
    // e now contains 30 correct bits
    DBG_PRINTF_DIGITS("e=", e_digits, ndigs_e);
#endif
    // e now contains one full correct digit

    //
    // Repeat Newton-steps.
    // In each step the number of correct digits is doubled
    //
    int ncorrect;
    for(ncorrect=1; (ncorrect<<1)<ndigs_e; ncorrect<<=1) {
	DBG_PRINTF("Newton Step: ncorrect=%d\n",ncorrect);
	int n = ncorrect<<1;
	digit x2[n];
	memset(x2, 0, sizeof(x2));
	square_n(x2, n, &e_digits[0], ncorrect);
	DBG_PRINTF_DIGITS("x2=", x2, n);
	DBG_PRINTF_DIGITS("d0=", d, ndigs_e);
    
	int nn = n<<1;
	digit etmp[nn];
	memset(etmp, 0, sizeof(etmp));
	mul_nl_nr(etmp, d, n, x2, n);
	DBG_PRINTF_DIGITS("etmp=", etmp, nn);

	// Discard everything except the n lowest digits of etmp
	for(int i=0; i<n; i++)
	    e_digits[i] = etmp[i];
	DBG_PRINTF_DIGITS("e=", e_digits, ndigs_e);
    }
    // e_digits now contains <ncorrect> correct digits
    
    //
    // Last Newton-step
    //
    {
	DBG_PRINTF("Newton Final: ncorrect=%d\n",ncorrect);
	int n = ncorrect<<1;
	digit x2[n];
	memset(x2, 0, sizeof(x2));
	square_n(x2, n, e_digits, ncorrect);
	DBG_PRINTF_DIGITS("x2=", x2, n);
	DBG_PRINTF_DIGITS("d0=", d, ndigs_e);
    
	int nn = ndigs_e<<1;
	digit etmp[nn];
	memset(etmp, 0, sizeof(etmp));
	DBG_ASSERT(ndigs_e<=n);
	mul_nl_nr(etmp, d, ndigs_e, x2, ndigs_e);
	DBG_PRINTF_DIGITS("etmp=", etmp, nn);

	// Discard everything except the nbits_e lowest bits of etmp
	DBG_ASSERT(ndigs_e<=nn);
	for(int i=0; i<ndigs_e; i++)
	    e_digits[i] = etmp[i];
	DBG_PRINTF_DIGITS("e=", e_digits, ndigs_e);
	e_digits[ndigs_e-1] &= ((1 << ((nbits_e-1) % PyLong_SHIFT +1)) -1);
	DBG_PRINTF_DIGITS("e=", e_digits, ndigs_e);
    }
    return;
}


static PyObject *
pygf2x_rinv(PyObject *self, PyObject *args) {
    // Multiplicative inverse of one Python integer, interpreted as polynomial over GF(2)
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
    if(((PyVarObject *)d)->ob_size <= 0) {
        PyErr_SetString(PyExc_ValueError, "Argument must be positive");
        return NULL;
    }
    if((d->ob_digit[0] & 1) == 0) {
        PyErr_SetString(PyExc_ValueError, "Argument must be non-even");
        return NULL;
    }
    if(nbits_e <= 0 || nbits_e > ((1<<16)-1)) {
        PyErr_SetString(PyExc_ValueError, "Requested bit_length of inverse is out of range");
        return NULL;
    }
    
    int nbits_d = nbits(d);
    int ndigs_d = (nbits_d + (PyLong_SHIFT-1))/PyLong_SHIFT;

    int ndigs_e = (nbits_e + (PyLong_SHIFT-1))/PyLong_SHIFT;
    DBG_PRINTF("ndigs_e          = %-4d\n",ndigs_e);
    
    DBG_PRINTF("Bits per digit   = %-4d\n",PyLong_SHIFT);
    DBG_PRINTF("Denominator bits = %-4d\n",nbits_d);
    DBG_PRINTF("Requested bits   = %-4d\n",nbits_e);

    digit e_digits[ndigs_e];
    memset(e_digits, 0, sizeof(e_digits));
    
    rinverse(e_digits, ndigs_e, nbits_e,
	     d->ob_digit, ndigs_d, nbits_d);

    // Remove leading zero digits from inverse
    while(ndigs_e > 0 && e_digits[ndigs_e-1] == 0)
      ndigs_e -= 1;
    
    PyLongObject *e = _PyLong_New(ndigs_e);
    memcpy(e->ob_digit, e_digits, ndigs_e*sizeof(digit));
    
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
    for(int i=GF2X_MAX(ndigs, nd_shift) - nd_shift; i < ndigs; i++)
	digits[i] = 0;
}

static PyObject *
pygf2x_divmod(PyObject *self, PyObject *args)
// Divide two Python integers, interpreted as polynomials over GF(2)
// Return quotient and remainder
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

    int nbits_d = nbits(denominator);
    int ndigs_d = (nbits_d + (PyLong_SHIFT-1))/PyLong_SHIFT;
    if(nbits_d == 0) {
        PyErr_SetString(PyExc_ZeroDivisionError, "Denominator is zero");
        return NULL;
    }
    int nbits_u = nbits(numerator);
    int ndigs_u = (nbits_u + (PyLong_SHIFT-1))/PyLong_SHIFT;
    
    int nbits_q = nbits_u > nbits_d-1 ? nbits_u - (nbits_d-1) : 0;
    int nbits_r = nbits_u > nbits_d-1 ? nbits_u : nbits_d-1; // The final n.o. remainder bits is <= this value.
    int ndigs_q = (nbits_q + (PyLong_SHIFT-1))/PyLong_SHIFT;
    int ndigs_r = (nbits_r + (PyLong_SHIFT-1))/PyLong_SHIFT;

    PyLongObject *q = _PyLong_New(ndigs_q);
    digit *restrict q_digits = q->ob_digit;
    memset(q_digits,0,ndigs_q*sizeof(digit));
    
    digit r_digits[ndigs_r]; // Initialize to numerator
    memset(r_digits+ndigs_u,0,(ndigs_r-ndigs_u)*sizeof(digit));
    memcpy(r_digits, numerator->ob_digit, ndigs_u*sizeof(digit));
    
    DBG_PRINTF("Bits per digit   = %-4d\n",PyLong_SHIFT);
    DBG_PRINTF("Numerator bits   = %-4d\n",nbits_u);
    DBG_PRINTF("Denominator bits = %-4d\n",nbits_d);
    DBG_PRINTF("Quotient bits    = %-4d\n",nbits_q);
    DBG_PRINTF("Remainder bits  <= %-4d\n",nbits_r);

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
	} else if(nbits_d < LIMIT_DIV_EUCLID) {
	    // Use Euclidean division based on tabled inverse of NDIV bit chunks
	    div_NDIVwise(q_digits, r_digits, denominator->ob_digit, nbits_u, nbits_d, ndigs_u, ndigs_d, ndigs_q, ndigs_r);
	} else {
	    /*
	     *   u = q*d + r
	     * Let nx denote nbits(x), then
	     *   nu = nq+nd-1
	     *   nr <= nd-1
	     * assuming nu >= nd
	     *
	     * Let e be an approximate inverse inv(d) with ne correct binary digits, i.e.:
	     *   d*e = (1<<(ne+nd-2)) + f
	     * where nf <= nd-1. Then:
	     *   u*e = q*d*e + r*e
	     *       = q*( (1<<(ne+nd-2)) + f ) + r*e
	     * Right shift (ne+nd-2) to recover q:
	     *   (u*e)>>(ne+nd-2) = q + (q*f)>>(ne+nd-2) + (r*e)>>(ne+nd-2)
	     * The rightmost term vanishes because nbits(r*e)= nr+ne-1 <= nd-1+ne-1 = ne+nd-2
	     * The bitlength of the term (q*f)>>(ne+nd-2) is
	     *   nq+nf-1 - (ne+nd-2) <= (nu-nd+1) + (nd-1) - (ne+nd-2) = nu - (ne+nd-2)
	     * Thus if nu - (ne+nd-2) <= 0 then we have a correct q, i.e. if
	     *   ne >= nu-nd +2
	     * otherwise the n.o. correct digits of q is
	     *   (nu-nd+1) - (nu - (ne+nd-2)) = 1+ne-2 = ne-1
	     *
	     */

	    // Just take one step in the Euclidean division loop below
	    int nbits_e = nbits_u - nbits_d +2;
	    int ndigs_e = (nbits_e + (PyLong_SHIFT-1))/PyLong_SHIFT;

	    digit e[ndigs_e];
	    memset(e, 0, sizeof(e));
	    inverse(e, ndigs_e, nbits_e,
		    denominator->ob_digit, ndigs_d, nbits_d);
	    DBG_PRINTF_DIGITS("inverse          :",e,ndigs_e);

	    DBG_PRINTF("ndigs_e=%d, ndigs_u=%d, ndigs_q=%d\n",ndigs_e,ndigs_u,ndigs_q);
	    for(int nbits_ri=nbits_r; nbits_ri>=nbits_d; nbits_ri -= nbits_e-1) {
		// Improve the n.o. correct bits by nbits_e-1 for each loop step
		int ndigs_ri = (nbits_ri + (PyLong_SHIFT-1))/PyLong_SHIFT;
		digit qtmp[ndigs_e + ndigs_ri];
		memset(qtmp, 0, sizeof(qtmp));
		mul_nl_nr(qtmp, e, ndigs_e, r_digits, ndigs_ri);
		// qtmp is now nbits_e + nbits_ri -1 bits. We just want the nbits_ri - nbits_d +1 most significant bits
		// So shift (nbits_e + nbits_d -2) bits right
		rshift(qtmp, ndigs_e + ndigs_ri, nbits_e + nbits_d -2);
		DBG_PRINTF_DIGITS("qtmp             :",qtmp,ndigs_e+ndigs_ri);
		
		int ndigs_qtmp = (nbits_ri + PyLong_SHIFT - nbits_d)/PyLong_SHIFT;
		for(int i=0; i<ndigs_qtmp; i++)
		    q_digits[i] ^= qtmp[i];
		DBG_PRINTF_DIGITS("q                :",q_digits,ndigs_q);
		int ndigs_rtmp = ndigs_qtmp + ndigs_d;
		digit rtmp[ndigs_rtmp];
		memset(rtmp, 0, sizeof(rtmp));
		mul_nl_nr(rtmp, qtmp, ndigs_qtmp, denominator->ob_digit, ndigs_d);
		for(int i=0; i<ndigs_rtmp; i++)
		    r_digits[i] ^= rtmp[i];
		DBG_PRINTF_DIGITS("r                :",r_digits,ndigs_r);
	    }
	}
    }

    DBG_PRINTF_DIGITS("Quotient         :",q_digits,ndigs_q);

    // Remove leading zero digits from remainder
    while(ndigs_r > 0 && r_digits[ndigs_r-1] == 0)
      ndigs_r -= 1;
    DBG_PRINTF_DIGITS("Remainder        :",r_digits,ndigs_r);
    
    PyLongObject *r = _PyLong_New(ndigs_r);
    memcpy(r->ob_digit, r_digits, sizeof(digit)*ndigs_r);

    return Py_BuildValue("OO", q, r);
}

PyMethodDef pygf2x_generic_functions[] =
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
        "rinv",
        pygf2x_rinv,
        METH_VARARGS,
        "Multiplicative inverse of integer as polynomial over GF(2), with given precision"
    },
    {
        NULL,                   // const char  *ml_name;  /* The name of the built-in function/method   */
        NULL,                   // PyCFunction ml_meth;   /* The C function that implements it          */
        0,                      // int         ml_flags;  /* Combination of METH_xxx flags, which mostly*/
                                //                        /* describe the args expected by the C func   */
        NULL                    // const char  *ml_doc;   /* The __doc__ attribute, or NULL             */
    }
};


struct PyModuleDef pygf2x_generic_module =
{
  // Python module definition
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "pygf2x_generic",       // Name of the module.
    .m_doc  = NULL,                   // Docstring for the module - in this case empty.
    .m_size = -1,                     // Used by sub-interpreters, if you do not know what
                                      // it is then you do not need it, keep -1 .
    .m_methods = pygf2x_generic_functions,  // Structures of type `PyMethodDef` with functions
                                      // (or "methods") provided by the module.
    .m_slots = NULL,
    .m_traverse = NULL,
    .m_clear = NULL,
    .m_free = NULL
};


PyMODINIT_FUNC PyInit_pygf2x_generic(void)
{
  // Python module initialization
    PyObject *pygf2x = PyModule_Create(&pygf2x_generic_module);

    //PyModule_AddIntConstant(pygf2x, "BITWISE_DIVISION_LIMIT", 5);
    //BITWISE_DIVISION_LIMIT = PyDict_GetItemString(PyModule_GetDict(pygf2x), "BITWISE_DIVISION_LIMIT");
    
    return pygf2x;
}
