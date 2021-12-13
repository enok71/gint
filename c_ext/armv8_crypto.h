/* -*- mode: c; c-basic-offset: 4; -*- */
/*******************************************************************************
 *
 * Copyright (c) 2020 Oskar Enoksson. All rights reserved.
 * Licensed under the MIT license. See LICENSE file in the project root for details.
 *
 * Description:
 * ArmV8 Cryptography extension acceleration functions for polynomials over GF(2)
 *
 *******************************************************************************/
#include <stdint.h>
#include <arm_neon.h>

#define PYGF2X_USE_ARMV8_CRYPTO
#define KARATSUBA_LIMIT 16

#define ATOM 8
#define mul_ATOM_15(a,b) mul_8_15(a,b)
#define mul_ATOM_30(a,b) mul_8_30(a,b)

static inline uint32_t
mul_8_15(uint8_t l, uint16_t r)
// Multiply a 8-bit with a 15-bit polynomial over GF(2) (stored in uint16_t)
// Return as a 22-bit polynomial (stored in uint32_t)
{
    DBG_ASSERT(l<(1<<8));
    DBG_ASSERT(r<(1<<15));

    return vmull_p64((poly64_t)l,(poly64_t)r);
}

static inline uint32_t
mul_15_15(uint16_t l, uint16_t r)
// Multiply two unsigned 15-bit polynomials over GF(2) (stored in uint16_t)
// Return as a 29-bit polynomial (stored in uint32_t)
{
    DBG_ASSERT(l<(1<<15));
    DBG_ASSERT(r<(1<<15));

    return vmull_p64((poly64_t)l,(poly64_t)r);
}


static inline uint64_t
mul_8_30(uint8_t l, uint32_t r) 
// Multiply a 8-bit with a 30-bit polynomial over GF(2) (stored in uint32_t)
// Return as a 37-bit polynomial (stored in uint64_t)
{
    DBG_ASSERT(l<(1<<8));
    DBG_ASSERT(r<(1<<30));

    return vmull_p64((poly64_t)l,(poly64_t)r);
}

static inline uint64_t
mul_15_30(uint16_t l, uint32_t r) 
// Multiply a 15-bit with a 30-bit polynomial over GF(2) (stored in uint32_t)
// Return as a 59-bit polynomial (stored in uint64_t)
{
    DBG_ASSERT(l<(1<<15));
    DBG_ASSERT(r<(1<<30));

    return vmull_p64((poly64_t)l,(poly64_t)r);
}

static inline uint64_t
mul_30_30(uint32_t l, uint32_t r) 
// Multiply two unsigned 30-bit polynomials over GF(2) (stored in uint32_t)
// Return as a 59-bit polynomial (stored in uint64_t)
{
    DBG_ASSERT(l<(1<<30));
    DBG_ASSERT(r<(1<<30));

    return vmull_p64((poly64_t)l,(poly64_t)r);
}

static inline uint32_t
sqr_15(uint16_t f)
{
    return vmull_p64((poly64_t)f,(poly64_t)f);
}

#if (PyLong_SHIFT==30)
static void
square_n(digit * restrict result, const digit *fdigits, int ndigs_f)
//
// Compute square and add it to p
// p += f^2
//
{
    int idp=0;
    int id=0;
    // Loop 2 at a time, taking advantage of the 64x64 bit multiplication instruction
    for(; id<ndigs_f-1; id+=2) {
        digit ic0 = fdigits[id];
        digit ic1 = fdigits[id+1];

        twodigits ic = ic0 | ((twodigits)ic1 << PyLong_SHIFT);
        uint64x2_t pi = vreinterpretq_u64_p128(vmull_p64((poly64_t)ic,(poly64_t)ic));
        twodigits pi_0 = vgetq_lane_u64(pi, 0);
        result[idp++] = pi_0 & PyLong_MASK;
        pi_0 >>= PyLong_SHIFT;
        result[idp++] = pi_0 & PyLong_MASK;

        // Shift 7*8+4 = 60 bits right
        pi = vreinterpretq_u64_u8(vextq_u8(vreinterpretq_u8_u64(pi), (uint8x16_t){0}, 7));
        pi = vshrq_n_u64(pi, 4);

        twodigits pi_1 = vgetq_lane_u64(pi, 0);
        result[idp++] = pi_1 & PyLong_MASK;
        pi_1 >>= PyLong_SHIFT;
        result[idp++] = pi_1 & PyLong_MASK;
    }
    for(; id<ndigs_f; id++) {
        digit ic = fdigits[id];
        twodigits pi = vmull_p64((poly64_t)ic,(poly64_t)ic);
        result[idp++] = pi & PyLong_MASK;
        pi >>= PyLong_SHIFT;
        result[idp++] = pi & PyLong_MASK;
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
    uint64x2_t pi = {0};
    int ip = 0;
    // Loop 2 digits at a time, taking advantage of the 64x64 bit multiplication instruction
    for(; ip<nl+nr-2; ip+=2) {
        int il = GF2X_MAX(0, (ip-(nr-2))&-2), ir = ip-il;
        DBG_PRINTF("nl=%d, nr=%d, ip=%d, il=%d, ir=%d\n",nl,nr,ip,il,ir);
        if(ir == nr-1) {
            DBG_PRINTF("A%d%d*B%d\n",il,il+1,ir);
            poly64_t li = l0[il] | ((twodigits)l0[il+1] << PyLong_SHIFT);
            poly64_t ri = r0[ir];
            pi = veorq_u64(pi, vreinterpretq_u64_p128(vmull_p64(li,ri)));
            il+=2;
            ir-=2;
        }
        for(; il < GF2X_MIN(nl-1, ip+1); il+=2, ir-=2) {
            DBG_PRINTF("A%d%d*B%d%d\n",il,il+1,ir,ir+1);
            poly64_t li = l0[il] | ((twodigits)l0[il+1] << PyLong_SHIFT);
            poly64_t ri = r0[ir] | ((twodigits)r0[ir+1] << PyLong_SHIFT);
            pi = veorq_u64(pi, vreinterpretq_u64_p128(vmull_p64(li,ri)));
        }
        if(il == nl-1 && ir>=0) {
            DBG_PRINTF("A%d*B%d%d\n",il,ir,ir+1);
            poly64_t li = l0[il];
            poly64_t ri = r0[ir] | ((twodigits)r0[ir+1] << PyLong_SHIFT);
            pi = veorq_u64(pi, vreinterpretq_u64_p128(vmull_p64(li,ri)));
        }
        //twodigits pi_0 = _mm_extract_epi64(pi, 0);
        twodigits pi_0 = vgetq_lane_u64(pi, 0);
        p[ip] ^= pi_0 & PyLong_MASK;
        p[ip+1] ^= (pi_0 >> PyLong_SHIFT) & PyLong_MASK;

        // Shift 7*8+4 = 60 bits right
        pi = vreinterpretq_u64_u8(vextq_u8(vreinterpretq_u8_u64(pi), (uint8x16_t){0}, 7));
        pi = vshrq_n_u64(pi, 4);
    }
    // Compute most significant digit(s)
    DBG_PRINTF("nl=%d, nr=%d, ip=%d, %016lx %016lx\n",nl,nr,ip,(uint64_t)pi[0],(uint64_t)pi[1]);

    // If both nl and nr are odd, then one single-digit multiplication remains
    if((nl&1) && (nr&1)) {
        DBG_PRINTF("A%d*B%d\n",nl-1,nr-1);
        poly64_t li = l0[nl-1];
        poly64_t ri = r0[nr-1];
        pi = veorq_u64(pi, vreinterpretq_u64_p128(vmull_p64(li,ri)));
    }

    twodigits pi_0 = vgetq_lane_u64(pi, 0);
    p[ip++] ^= pi_0 & PyLong_MASK;

    if(ip<nl+nr) {
        DBG_PRINTF("ip=%d, nl+nr-1=%d\n",ip,nl+nr-1);
        pi_0 >>= PyLong_SHIFT;
        p[ip++] ^= pi_0;
    }
    DBG_ASSERT(ip == nl+nr);
    DBG_ASSERT(pi_0 < (1<<PyLong_SHIFT));
}
#endif
