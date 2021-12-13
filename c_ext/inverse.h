/* -*- mode: c; c-basic-offset: 4; -*- */
/*******************************************************************************
 *
 * Copyright (c) 2021 Oskar Enoksson. All rights reserved.
 * Licensed under the MIT license. See LICENSE file in the project root for details.
 *
 * Description:
 * Implementation of multiplicative inverse on unlimited polynomials over GF(2)
 *
 *******************************************************************************/

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

static void
inverse(digit *restrict e_digits, int ndigs_e, int nbits_e,
        const digit * restrict d_digits, int ndigs_d, int nbits_d)
//
// Compute GF2[x] inverse e to d such that
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

    const bool use_heap = (ndigs_e > STATIC_LIMIT);
    
    // Shift the entire d to the left so that it is left-aligne, i.e. the most significant
    // digit has most significant bit =1
    // Also truncate it, or fill it with zero, from the right, so that it has ndigs_e digits.
    digit d_static[STATIC_LIMIT];
    digit * restrict const d = use_heap ? malloc(ndigs_e*sizeof(digit)) : d_static;
    memset(d,0,ndigs_e*sizeof(digit));
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
    //
    // Find initial approximate inverse using table
    //
    if(nbits_e <= 8) {
        // Compute the whole inverse using table
        uint16_t dh = d[ndigs_e-1] >> (PyLong_SHIFT-nbits_e);
        // Invert dh using tablulated inverse
        e_digits[ndigs_e-1] = inv_8[(dh << (8-nbits_e)) - (1 << (8-1))] >> (8-nbits_e);
        DBG_ASSERT(e_digits[ndigs_e-1] < (1u<<nbits_e));
        return;
    }
    {
        // Extract the highest 8-bit chunk of denominator
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
        // Extract the highest 15-bit chunk of denominator
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
        uint32_t x2 = sqr_15(e_digits[ndigs_e-1]);
        // x2 is 2*15-1
        e_digits[ndigs_e-1] = mul_30_30(x2, dh) >> 28; // nbits_e + 29 -1 - nbits_e = 28
        DBG_ASSERT(e_digits[ndigs_e-1] < (1u<<nbits_e));
        return;
    }
    {
        // Invert the highest 30-bit chunk
        uint32_t dh = d[ndigs_e-1];
        uint32_t x2 = sqr_15(e_digits[ndigs_e-1]);
        e_digits[ndigs_e-1] = mul_30_30(x2, dh) >> 28; // 30 + 29 -1 - 30 = 28
        DBG_ASSERT(e_digits[ndigs_e-1]<(1<<30));
    }
    // e now contains 30 correct bits
    DBG_PRINTF_DIGITS("e=", e_digits, ndigs_e);
#endif
    //
    // e now contains one full correct digit
    //
    // Repeat Newton-steps.
    // In each step the number of correct digits is doubled
    //
    // Prepare a sequence of precisions that avoids costly last Newton steps.
    // E.g. to achieve 75 digits precision we would most naturally achieve that through
    // the steps 1 2 4 8 16 32 64 75 (doubling precision in each iteration, except the last step)
    // but it is more efficient to do it as 1 2 3 5 10 19 38 75.
    // If multiplication is O(n^1.6) the efficiency gain for the multiplication is 30% for the
    // 75-digit example above. If O(n*ln(n)) it is 28%.
    // Each bit in the double_mask variable below will be used to decide if precision will
    // be doubled ncorrect*2 or ncorrect*2-1 in each Newton step.
    //
    int double_mask = 0;
    int ncorrect;
    for(ncorrect=ndigs_e; ncorrect>1;) {
        double_mask = (double_mask << 1) | (ncorrect & 1);
        ncorrect = (ncorrect >> 1) + (ncorrect & 1);
    }

    const int x2len = (ndigs_e&1)+ndigs_e;
    digit x2_static[(STATIC_LIMIT&1)+STATIC_LIMIT];
    digit etmp_static[((STATIC_LIMIT&1)+STATIC_LIMIT)<<1];
    digit * restrict const x2 = use_heap ? malloc(x2len*sizeof(digit)) : x2_static;
    digit * restrict const etmp = use_heap ? malloc((x2len<<1)*sizeof(digit)) : etmp_static;
    
    for(ncorrect=1; ncorrect<ndigs_e; ) {
        DBG_PRINTF("ncorrect=%d\n",ncorrect);
        // Determine number correct digits after the current iteration, according to the plan
        // controlled by bits in double_mask
        const int ncorrect_new = (ncorrect << 1) - (double_mask & 1);
        double_mask >>= 1;
        
        const int nx2 = ncorrect<<1;
        DBG_ASSERT(nx2 <= x2len);
        square_n(x2, &e_digits[ndigs_e-ncorrect], ncorrect);  // The highest bit of x2 is now 0
        DBG_PRINTF_DIGITS("x2=", x2, nx2);
    
        const int nn = ncorrect_new + nx2;
        DBG_ASSERT(nn <= (x2len<<1));
        memset(etmp, 0, nn*sizeof(digit));
        //
        // The reason why nx2 digits from x2 and ncorrect_new from d are enough
        // to correctly form ncorrect_new correct digits in etmp by the
        // multiplication below is based on the knowledge
        // that nx2 is an even number, that the most significant bit is zero, and
        // that every second bit in x2 is zero because x2 is a square.
        //
        mul_nl_nr(etmp, &d[ndigs_e-ncorrect_new], ncorrect_new, x2, nx2);
        // The 2 highest bits of etmp is now 0
        DBG_PRINTF_DIGITS("etmp=", etmp, nn);

        // Discard lowest nx2*PyLong_SHIFT-1 bits of etmp
        // Also don't bother with the first ncorrect digits, because they are already correct
        for(int i=ncorrect+1; i<=ncorrect_new; i++) {
            // Shift away leading zero bits.
            e_digits[ndigs_e-i] = ((etmp[nn-i] << 2) &PyLong_MASK) | (etmp[nn-1-i] >> (PyLong_SHIFT-2));
        }
        DBG_PRINTF_DIGITS("e=", e_digits, ndigs_e);
        
        ncorrect = ncorrect_new;
    }
    //
    // e_digits now contains <ncorrect> correct digits
    //
    DBG_ASSERT(ncorrect == ndigs_e);

    if(use_heap) {
        free(etmp);
        free(x2);
        free(d);
    }

    // Shift e_digits from left-aligned to properly right-aligned
    const int shift = (PyLong_SHIFT-1) - (nbits_e -1)%PyLong_SHIFT;
    for(int i=0; i<ndigs_e-1; i++)
        e_digits[i] = (e_digits[i]>>shift) | ((e_digits[i+1]<<(PyLong_SHIFT-shift)) &PyLong_MASK);
    e_digits[ndigs_e-1] = e_digits[ndigs_e-1]>>shift;
    DBG_PRINTF_DIGITS("e=", e_digits, ndigs_e);

    return;
}
