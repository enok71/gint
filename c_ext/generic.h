/* -*- mode: c; c-basic-offset: 4; -*- */
/*******************************************************************************
 *
 * Copyright (c) 2020 Oskar Enoksson. All rights reserved.
 * Licensed under the MIT license. See LICENSE file in the project root for details.
 *
 * Description:
 * Generic functions for polynomials over GF(2)
 *
 *******************************************************************************/
#define ATOM 5
#define mul_ATOM_15 mul_5_15
#define mul_ATOM_30 mul_5_30
#define KARATSUBA_LIMIT 4

static inline uint32_t
mul_5_15(uint8_t l, uint16_t r)
// Multiply a 5-bit with a 15-bit polynomial over GF(2) (stored in uint16_t)
// Return as a 19-bit polynomial (stored in uint32_t)
{
    DBG_ASSERT(l<(1<<5));
    DBG_ASSERT(r<(1<<15));

    uint8_t r0 = r&0x1f;
    r>>=5;
    uint8_t r1 = r&0x1f;
    r>>=5;
    uint8_t r2 = r&0x1f;

    uint32_t p = mul_5_5[l][r2];
    p <<= 5;
    p ^= mul_5_5[l][r1];
    p <<= 5;
    p ^= mul_5_5[l][r0];

    return p;
}

static inline uint32_t
mul_15_15(uint16_t l, uint16_t r)
// Multiply two unsigned 15-bit polynomials over GF(2) (stored in uint16_t)
// Return as a 29-bit polynomial (stored in uint32_t)
{
    DBG_ASSERT(l<(1<<15));
    DBG_ASSERT(r<(1<<15));

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


static inline uint64_t
mul_5_30(uint8_t l, uint32_t r)
// Multiply a 5-bit with a 30-bit polynomial over GF(2) (stored in uint16_t)
// Return as a 19-bit polynomial (stored in uint32_t)
{
    DBG_ASSERT(l<(1<<5));
    DBG_ASSERT(r<(1<<30));

    uint8_t r0 = r&0x1f;
    r>>=5;
    uint8_t r1 = r&0x1f;
    r>>=5;
    uint8_t r2 = r&0x1f;
    r>>=5;
    uint8_t r3 = r&0x1f;
    r>>=5;
    uint8_t r4 = r&0x1f;
    r>>=5;
    uint8_t r5 = r&0x1f;

    uint64_t p = mul_5_5[l][r5];
    p <<= 5;
    p ^= mul_5_5[l][r4];
    p <<= 5;
    p ^= mul_5_5[l][r3];
    p <<= 5;
    p ^= mul_5_5[l][r2];
    p <<= 5;
    p ^= mul_5_5[l][r1];
    p <<= 5;
    p ^= mul_5_5[l][r0];

    return p;
}

static inline uint64_t
mul_15_30(uint16_t l, uint32_t r)
// Multiply a 15-bit with a 30-bit polynomial over GF(2) (stored in uint16_t)
// Return as a 44-bit polynomial (stored in uint32_t)
{
    DBG_ASSERT(l<(1<<15));
    DBG_ASSERT(r<(1<<30));

    uint8_t l0 = l&0x1f;
    l>>=5;
    uint8_t l1 = l&0x1f;
    l>>=5;
    uint8_t l2 = l&0x1f;
    uint8_t r0 = r&0x1f;
    r>>=5;
    uint8_t r1 = r&0x1f;
    r>>=5;
    uint8_t r2 = r&0x1f;
    r>>=5;
    uint8_t r3 = r&0x1f;
    r>>=5;
    uint8_t r4 = r&0x1f;
    r>>=5;
    uint8_t r5 = r&0x1f;

    uint64_t p = mul_5_5[l2][r5];
    p <<= 5;
    p ^= mul_5_5[l1][r5] ^ mul_5_5[l2][r4];
    p <<= 5;
    p ^= mul_5_5[l0][r5] ^ mul_5_5[l1][r4] ^ mul_5_5[l2][r3];
    p <<= 5;
    p ^= mul_5_5[l0][r4] ^ mul_5_5[l1][r3] ^ mul_5_5[l2][r2];
    p <<= 5;
    p ^= mul_5_5[l0][r3] ^ mul_5_5[l1][r2] ^ mul_5_5[l2][r1];
    p <<= 5;
    p ^= mul_5_5[l0][r2] ^ mul_5_5[l1][r1] ^ mul_5_5[l2][r0];
    p <<= 5;
    p ^= mul_5_5[l0][r1] ^ mul_5_5[l1][r0];
    p <<= 5;
    p ^= mul_5_5[l0][r0];

    return p;
}

static inline uint64_t
mul_30_30(uint32_t l, uint32_t r) 
// Multiply two unsigned 30-bit polynomials over GF(2) (stored in uint32_t)
// Return as a 59-bit polynomial (stored in uint64_t)
{
    DBG_ASSERT(l<(1<<30));
    DBG_ASSERT(r<(1<<30));

    // Use Karatsubas formula and mul_15_15
    uint16_t ll = l &0x7fff;
    uint16_t lh = (l >> 15);
    uint16_t rl = r &0x7fff;
    uint16_t rh = (r >> 15);

    uint32_t z0 = mul_15_15(ll,rl);
    uint32_t z2 = mul_15_15(lh,rh);
    uint32_t z1 = mul_15_15(ll ^ lh, rl ^ rh) ^z2 ^z0;

    return ((((uint64_t)z2 << 15) ^ (uint64_t)z1 ) << 15) ^ (uint64_t)z0;
}


static inline uint32_t
sqr_15(uint16_t f)
{
    return ((uint32_t)sqr_8[f >> 8] << 16) ^ (uint32_t)sqr_8[f &0xff];
}

static void
square_n(digit * restrict result, const digit *fdigits, int ndigs_f)
//
// Compute square into p
// p += f^2
//
// The length of p must be twice ndigs_f. This may mean one extra digit in p at the most significant end
// in order to make the algorithm simpler and faster
//
{
    int idp=0;
    for(int id=0; id<ndigs_f; id++) {
        digit ic = fdigits[id];
#if (PyLong_SHIFT == 15)
        digit pd0 = sqr_8[ic&0xff];
        ic>>=8;
        digit pd1 = sqr_8[ic&0x7f];
        pd1 <<= 1;
#elif (PyLong_SHIFT == 30)              
        uint16_t pc0 = sqr_8[ic&0xff];
        ic>>=8;
        uint16_t pc1 = sqr_8[ic&0x7f];
        digit pd0 = (pc1 << 16) ^ pc0;
        ic>>=7;
        uint16_t pc2 = sqr_8[ic&0xff];
        ic>>=8;
        uint16_t pc3 = sqr_8[ic&0x7f];
        digit pd1 = (pc3 << 16) ^ pc2;
#else
#error
#endif
        result[idp++] = pd0;
        result[idp++] = pd1;
    }
}

static void mul_nl_nr_IMPL(digit * restrict p,
                           const digit * restrict const l0, int nl,
                           const digit * restrict const r0, int nr)
//
// Compute product and it to p
// p += l*r
//
{
    twodigits pi = 0;
    for(int ip=0; ip<nl+nr-1; ip++) {
        for(int il=GF2X_MAX(0, ip-nr+1), ir=ip-il; il<GF2X_MIN(nl, ip+1); il++, ir--) {
            pi ^= mul_digit_digit(l0[il], r0[ir]);
        }
        p[ip] ^= pi & PyLong_MASK;
        pi >>= PyLong_SHIFT;
    }
    p[nl+nr-1] ^= pi;
}
