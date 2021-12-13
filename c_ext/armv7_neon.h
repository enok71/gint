/* -*- mode: c; c-basic-offset: 4; -*- */
/*******************************************************************************
 *
 * Copyright (c) 2020 Oskar Enoksson. All rights reserved.
 * Licensed under the MIT license. See LICENSE file in the project root for details.
 *
 * Description:
 * ArmV7 Neon extension acceleration functions for polynomials over GF(2)
 *
 *******************************************************************************/

#include <stdint.h>
#include <arm_neon.h>

#define PYGF2X_USE_ARMV7_NEON
#define KARATSUBA_LIMIT 8

#define ATOM 8
#define mul_ATOM_15(a,b) mul_8_15(a,b)
#define mul_ATOM_30(a,b) mul_8_30(a,b)

static inline uint64_t
mul_30_30(uint32_t l, uint32_t r)
{
    poly8x8_t l0 = vreinterpret_p8_u32(vdup_n_u32(l));           // 0 1 2 3 0 1 2 3
    l0 = vzip_p8(l0,l0).val[0];                                  // 0 0 1 1 2 2 3 3
    poly16x4x2_t l1 = vzip_p16(vreinterpret_p16_p8(l0),          // 0 0 0 0 1 1 1 1 2 2 2 2 3 3 3 3
                               vreinterpret_p16_p8(l0));
    poly8x8_t r0 = vreinterpret_p8_u32(vdup_n_u32(r));           // 0 1 2 3 0 1 2 3

    poly16x8_t  p00 = vmull_p8(vreinterpret_p8_p16(l1.val[0]),   // 0 1 1 2 2 3 3 4 1 2 2 3 3 4 4 5
                               r0);
    poly16x8_t  p01 = vmull_p8(vreinterpret_p8_p16(l1.val[1]),   // 2 3 3 4 4 5 5 6 3 4 4 5 5 6 6 7
                               r0);

    poly16x4x2_t p10 = vuzp_p16(vget_low_p16(p00),               // 0 1 2 3 1 2 3 4 1 2 3 4 2 3 4 5
                                vget_high_p16(p00));
    poly16x4x2_t p11 = vuzp_p16(vget_low_p16(p01),               // 2 3 4 5 3 4 5 6 3 4 5 6 4 5 6 7
                                vget_high_p16(p01));

    uint64x2_t p20 = vmovl_u32(vreinterpret_u32_p16(p10.val[0]));// 0 1 2 3 _ _ _ _ 1 2 3 4 _ _ _ _
    uint64x2_t p21 = vmovl_u32(vreinterpret_u32_p16(p10.val[1]));// 1 2 3 4 _ _ _ _ 2 3 4 5 _ _ _ _
    uint64x2_t p22 = vmovl_u32(vreinterpret_u32_p16(p11.val[0]));// 2 3 4 5 _ _ _ _ 3 4 5 6 _ _ _ _
    uint64x2_t p23 = vmovl_u32(vreinterpret_u32_p16(p11.val[1]));// 3 4 5 6 _ _ _ _ 4 5 6 7 _ _ _ _

    p20 = veorq_u64(p20, vshlq_n_u64(p21, 8));                   // 0 1 2 3 4 _ _ _ 1 2 3 4 5 _ _ _
    p22 = veorq_u64(p22, vshlq_n_u64(p23, 8));                   // 2 3 4 5 6 _ _ _ 3 4 5 6 7 _ _ _
    p20 = veorq_u64(p20, vshlq_n_u64(p22, 16));                  // 0 1 2 3 4 5 6 _ 1 2 3 4 5 6 7 _

    return vget_lane_u64(veor_u64(vget_low_u64(p20),
                                  vshl_n_u64(vget_high_u64(p20),8)),0);   // 0 1 2 3 4 5 6 7
}


static inline uint64_t
mul_15_30(uint16_t l, uint32_t r)
{
    poly8x8_t l0 = vreinterpret_p8_u16(vdup_n_u16(l));           // 0 1 0 1 0 1 0 1
    l0 = vzip_p8(l0,l0).val[0];                                  // 0 0 1 1 0 0 1 1
    poly16x4_t l1 = vzip_p16(vreinterpret_p16_p8(l0),            // 0 0 0 0 1 1 1 1
                             vreinterpret_p16_p8(l0)).val[0];
    poly8x8_t r0 = vreinterpret_p8_u32(vdup_n_u32(r));           // 0 1 2 3 0 1 2 3

    poly16x8_t p0 = vmull_p8(vreinterpret_p8_p16(l1),r0);        // 0 1 1 2 2 3 3 4 1 2 2 3 3 4 4 5

    poly16x4x2_t p1 = vuzp_p16(vget_low_p16(p0),                 // 0 1 2 3 1 2 3 4 1 2 3 4 2 3 4 5
                               vget_high_p16(p0));

    uint64x2_t p20 = vmovl_u32(vreinterpret_u32_p16(p1.val[0])); // 0 1 2 3 _ _ _ _ 1 2 3 4 _ _ _ _
    uint64x2_t p21 = vmovl_u32(vreinterpret_u32_p16(p1.val[1])); // 1 2 3 4 _ _ _ _ 2 3 4 5 _ _ _ _

    p20 = veorq_u64(p20, vshlq_n_u64(p21, 8));                   // 0 1 2 3 4 _ _ _ 1 2 3 4 5 _ _ _

    return vget_lane_u64(veor_u64(vget_low_u64(p20),
                                  vshl_n_u64(vget_high_u64(p20),8)),0);   // 0 1 2 3 4 5 _ _
}


static inline uint64_t
mul_8_30(uint8_t l, uint32_t r)
{
    poly8x8_t l0 = vdup_n_p8(l);                                       // 0 0 0 0 0 0 0 0
    poly8x8_t r0 = vreinterpret_p8_u32(vdup_n_u32(r));                 // 0 1 2 3 0 1 2 3

    poly16x4_t p0 = vget_low_p16(vmull_p8(l0,r0));                     // 0 1 1 2 2 3 3 4
    poly16x4x2_t p1 = vuzp_p16(p0, p0);                                // 0 1 2 3 0 1 2 3 1 2 3 4 1 2 3 4
    uint32x2_t p2 = vzip_u32(vreinterpret_u32_p16(p1.val[0]),          // 0 1 2 3 1 2 3 4
                             vreinterpret_u32_p16(p1.val[1])).val[0];
    uint64x2_t p3 = vmovl_u32(p2);                                     // 0 1 2 3 _ _ _ _ 1 2 3 4 _ _ _ _
    return vget_lane_u64(veor_u64(vget_low_u64(p3),
                                  vshl_n_u64(vget_high_u64(p3), 8)),0);
}


static inline uint32_t
mul_15_15(uint16_t l, uint16_t r)
{
    poly8x8_t l0 = vreinterpret_p8_u16(vdup_n_u16(l));           // 0 1 0 1 0 1 0 1
    poly8x8x2_t l1 = vzip_p8(l0,l0);                             // 0 0 1 1 0 0 1 1
    poly8x8_t ri = vreinterpret_p8_u16(vdup_n_u16(r));           // 0 1 0 1 0 1 0 1

    poly16x4_t  p0 = vget_low_p16(vmull_p8(l1.val[0],ri));       // 0 1 1 2 1 2 2 3
    uint32x4_t  p1 = vmovl_u16(vreinterpret_u16_p16(p0));        // 0 1 _ _ 1 2 _ _ 1 2 _ _ 2 3 _ _
    int32x4_t  s = { 0, 8, 8, 16 };
    p1 = vshlq_u32(p1, s);
    uint32x2_t  p2 = veor_u32(vget_low_u32(p1),                  // 0 1 2 _ _ 1 2 3
                              vget_high_u32(p1));
    return p2[0]^p2[1];
}

static inline uint32_t
mul_8_15(uint8_t l, uint16_t r)
{
    poly8x8_t l0 = vdup_n_p8(l);                                       // 0 0 0 0 0 0 0 0
    poly8x8_t r0 = vreinterpret_p8_p16(vdup_n_p16(r));                 // 0 1 0 1 0 1 0 1

    poly16x4_t  p0 = vget_low_p16(vmull_p8(l0,r0));                    // 0   1   0   1
    uint32x2_t  p1 = vget_low_u32(vmovl_u16(vreinterpret_u16_p16(p0)));

    return vget_lane_u32(p1,0) ^ (vget_lane_u32(p1,1)<<8);
}

static inline uint32_t
sqr_15(uint16_t f)
{
    poly8x8_t f0 = vreinterpret_p8_p16(vdup_n_p16(f));
    poly16x8_t p0 = vmull_p8(f0,f0);
    return vreinterpret_u32_p16(vget_low_p16(p0))[0];
}

static void
square_n(digit * restrict result, const digit *fdigits, int ndigs_f)
//
// Compute square and add it to p
// p += f^2
//
{
    int idp=0;
    int id=0;
#if (PyLong_SHIFT == 15)
    for(; id<ndigs_f-3; id+=4) {
        // Loop over 4 digits at a time, taking advantage of the 8x8 x 8x8 multiplication instruction
        poly8x8_t f = vreinterpret_p8_u16(vld1_u16(fdigits + id));
        uint16x8_t pi128 = vreinterpretq_u16_p16(vmull_p8(f,f));
        uint16x8_t pi128_1 = vshlq_n_u16(pi128, 1);
        pi128 = vzipq_u16(vuzpq_u16(pi128,pi128).val[0],
                          vuzpq_u16(pi128_1,pi128_1).val[1]).val[0];
        vst1q_u16(result+idp, pi128);
        idp+=8;
    }
#elif (PyLong_SHIFT == 30)
    for(; id<ndigs_f-1; id+=2) {
        // Loop over 2 digits at a time, taking advantage of the 8x8 x 8x8 multiplication instruction
        poly8x8_t f = vreinterpret_p8_u32(vld1_u32(fdigits + id));
        uint32x4_t pi128 = vreinterpretq_u32_p16(vmull_p8(f,f));
        uint32x4_t pi128_1 = vreinterpretq_u32_u64(vshrq_n_u64(vreinterpretq_u64_u32(pi128), PyLong_SHIFT));
        pi128 = vshrq_n_u32(vshlq_n_u32(pi128,2),2);
        pi128 = vzipq_u32(vuzpq_u32(pi128,pi128).val[0],vuzpq_u32(pi128_1,pi128_1).val[0]).val[0];
        vst1q_u32(result+idp, pi128);
        idp+=4;
    }
#else
#error
#endif
    // Remaining digits
    if(id<ndigs_f) {
#if (PyLong_SHIFT == 15)
        digit ftmp[4]={0}, ptmp[8];
        memcpy(ftmp, fdigits+id, (ndigs_f-id)*sizeof(digit));
        poly8x8_t f = vreinterpret_p8_u16(vld1_u16(ftmp));
        uint16x8_t pi128 = vreinterpretq_u16_p16(vmull_p8(f,f));
        uint16x8_t pi128_1 = vshlq_n_u16(pi128, 1);
        pi128 = vzipq_u16(vuzpq_u16(pi128,pi128).val[0],
                          vuzpq_u16(pi128_1,pi128_1).val[1]).val[0];
        vst1q_u16(ptmp, pi128);
        memcpy(result+idp, ptmp, 2*(ndigs_f-id)*sizeof(digit));
#elif (PyLong_SHIFT == 30)              
        digit ftmp[2]={0}, ptmp[4];
        memcpy(ftmp, fdigits+id, (ndigs_f-id)*sizeof(digit));
        poly8x8_t f = vreinterpret_p8_u32(vld1_u32(ftmp));
        uint32x4_t pi128 = vreinterpretq_u32_p16(vmull_p8(f,f));
        uint32x4_t pi128_1 = vreinterpretq_u32_u64(vshrq_n_u64(vreinterpretq_u64_u32(pi128), PyLong_SHIFT));
        pi128 = vshrq_n_u32(vshlq_n_u32(pi128,2),2);
        pi128 = vzipq_u32(vuzpq_u32(pi128,pi128).val[0],vuzpq_u32(pi128_1,pi128_1).val[0]).val[0];
        vst1q_u32(ptmp, pi128);
        memcpy(result+idp, ptmp, 2*(ndigs_f-id)*sizeof(digit));
#else
#error
#endif
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
#if (PyLong_SHIFT == 15)
    uint64_t pi = 0;
    int ip=0;

    // Loop 2 digits at a time, taking advantage of the 8x8x8x8 bit multiplication instruction
    for(; ip<nl+nr-2; ip+=2) {
        int il = GF2X_MAX(0, (ip-(nr-2))&-2), ir = ip-il;
        DBG_PRINTF("nl=%d, nr=%d, ip=%d, il=%d, ir=%d\n",nl,nr,ip,il,ir);
        if(ir == nr-1) {
            DBG_PRINTF("A%d%d*B%d\n",il,il+1,ir);
            pi ^= mul_15_30(r0[ir], l0[il] | ((uint32_t)l0[il+1] << PyLong_SHIFT));
            il+=2;
            ir-=2;
        }
        for(; il < GF2X_MIN(nl-1, ip+1); il+=2, ir-=2) {
            DBG_PRINTF("A%d%d*B%d%d\n",il,il+1,ir,ir+1);
            pi ^= mul_30_30(r0[ir] | ((uint32_t)r0[ir+1] << PyLong_SHIFT),
                            l0[il] | ((uint32_t)l0[il+1] << PyLong_SHIFT));
        }
        if(il == nl-1 && ir>=0) {
            DBG_PRINTF("A%d*B%d%d\n",il,ir,ir+1);
            pi ^= mul_15_30(l0[il], r0[ir] | ((uint32_t)r0[ir+1] << PyLong_SHIFT));
        }
        twodigits pi_0 = (twodigits)pi;
        p[ip] ^= pi_0 & PyLong_MASK;
        p[ip+1] ^= (pi_0 >> PyLong_SHIFT) & PyLong_MASK;
        pi >>= 2*PyLong_SHIFT;
    }
    // Compute most significant digit(s)
    DBG_PRINTF("nl=%d, nr=%d, ip=%d, %08x %08x\n",nl,nr,ip,(uint32_t)(pi&PyLong_MASK),(uint32_t)(pi>>PyLong_SHIFT));

    twodigits pi_0 = (twodigits)pi;
    // If both nl and nr are odd, then one single-digit multiplication remains
    if((nl&1) && (nr&1)) {
        DBG_PRINTF("A%d*B%d\n",nl-1,nr-1);
        pi_0 ^= mul_15_15(l0[nl-1], r0[nr-1]);
        DBG_PRINTF("%08x\n",pi_0);
    }

    p[ip++] ^= pi_0 & PyLong_MASK;

    if(ip<nl+nr) {
        DBG_PRINTF("ip=%d, nl+nr-1=%d\n",ip,nl+nr-1);
        pi_0 >>= PyLong_SHIFT;
        p[ip++] ^= pi_0;
    }
    DBG_ASSERT(ip == nl+nr);
    DBG_ASSERT(pi_0 < (1<<PyLong_SHIFT));
#elif (PyLong_SHIFT==30)
    twodigits pi = 0;
    for(int ip=0; ip<nl+nr-1; ip++) {
        for(int il=GF2X_MAX(0, ip-nr+1), ir=ip-il; il<GF2X_MIN(nl, ip+1); il++, ir--) {
            pi ^= mul_digit_digit(l0[il], r0[ir]);
        }
        p[ip] ^= pi & PyLong_MASK;
        pi >>= PyLong_SHIFT;
    }
    p[nl+nr-1] ^= pi;
#else
#error
#endif
}
