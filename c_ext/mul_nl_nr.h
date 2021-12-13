/* -*- mode: c; c-basic-offset: 4; -*- */
/*******************************************************************************
 *
 * Copyright (c) 2021 Oskar Enoksson. All rights reserved.
 * Licensed under the MIT license. See LICENSE file in the project root for details.
 *
 * Description:
 * Implementation of multiplication on unlimited polynomials over GF(2)
 *
 *******************************************************************************/
static void mul_nl_nr(digit * restrict p,
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
        if(l0[0] < (1 << ATOM))
            mul_ATOM_nr(p, l0[0], r0, nr);
#if (PyLong_SHIFT == 30)
        else if(l0[0] < (1 << 15))
            mul_15_nr(p, l0[0], r0, nr);
#endif
        else
            mul_digit_nr(p, l0[0], r0, nr);
    } else if(nr == 1) {
        // Stop recursing
        DBG_PRINTF("%-2d:[0,%d],[0,%d]\n",depth,nl,nr);
        if(r0[0] < (1 << ATOM))
            mul_ATOM_nr(p, r0[0], l0, nl);
#if (PyLong_SHIFT == 30)
        else if(r0[0] < (1 << 15))
            mul_15_nr(p, r0[0], l0, nl);
#endif
        else
            mul_digit_nr(p, r0[0], l0, nl);
    } else if(nr < KARATSUBA_LIMIT && nl < KARATSUBA_LIMIT) {
        // Perform standard multiplication
        mul_nl_nr_IMPL(p, l0, nl, r0, nr);
    } else if(nl > 2*nr) {
        // Divide l to form more equal sized pieces
        int nc = nl/nr; // Number of chunks
        for(int ic=0; ic<nc; ic++) {
            int icu = (ic+1)*nl/nc;
            int icl = ic*nl/nc;
            mul_nl_nr(p+icl, l0+icl, icu-icl, r0, nr);
        }
    } else if(nr > 2*nl) {
        // Divide r to form more equal sized pieces
        int nc = nr/nl; // Number of chunks
        for(int ic=0; ic<nc; ic++) {
            int icu = (ic+1)*nr/nc;
            int icl = ic*nr/nc;
            mul_nl_nr(p+icl, l0, nl, r0+icl, icu-icl);
        }
    } else if(nl>1 && nr>1) {
        // Use Karatsuba
        // The choice of m is not obvious
        // Below 
        const int m = (GF2X_MIN(nl,nr) + (abs(nl-nr)&1)) >>1;
        const int nl1 = nl-m;
        const int nr1 = nr-m;
        const digit * restrict l1 = l0+m;
        const digit * restrict r1 = r0+m;
        DBG_PRINTF("%-2d:[0,%d,%d],[0,%d,%d]\n",depth,m,nl,m,nr);

        const int nr01 = GF2X_MAX(m, nr1);
        const int nl01 = GF2X_MAX(m, nl1);
        const int nz0 = 2*m;
        const int nz1 = nl01+nr01;
        const int nz2 = nl1+nr1;

        // Allocate all needed memory in one malloc, for performance
        const int nbuf = nl01+nr01+nz0+nz1+nz2;
        digit bufs[STATIC_LIMIT*8];
        const bool use_heap = (size_t)nbuf > sizeof(bufs)/sizeof(digit);
        digit * const buf0 = use_heap ? malloc(nbuf*sizeof(digit)) : bufs;
        digit * buf = buf0;
        
        digit * restrict const r01 = buf; buf += nr01;   // r01 = r0^r1
        if(m>nr1) {
            for(int i=0; i<nr1; i++)
                r01[i] = r0[i] ^ r1[i];
            for(int i=nr1; i<m; i++)
                r01[i] = r0[i];
        } else {
            for(int i=0; i<m; i++)
                r01[i] = r0[i] ^ r1[i];
            for(int i=m; i<nr1; i++)
                r01[i] = r1[i];
        }
        
        digit * restrict const l01 = buf; buf += nl01;   // l01 = l0^l1
        if(m>nl1) {
            for(int i=0; i<nl1; i++)
                l01[i] = l0[i] ^ l1[i];
            for(int i=nl1; i<m; i++)
                l01[i] = l0[i];
        } else {
            for(int i=0; i<m; i++)
                l01[i] = l0[i] ^ l1[i];
            for(int i=m; i<nl1; i++)
                l01[i] = l1[i];
        }

#ifdef DEBUG_PYGF2X
        depth += 1;
#endif
        memset(buf,0,(nz0+nz2)*sizeof(digit));
        digit * restrict const z0 = buf; buf += nz0;  // z0 = l0*r0, z2 = l1*r1
        digit * restrict const z2 = buf; buf += nz2;
        mul_nl_nr(z0, l0, m, r0, m);
        mul_nl_nr(z2, l1, nl1, r1, nr1);

        digit * restrict const z1 = buf; buf += nz1;   // z1 = l01*r01
        if(nz0>nz2) {
            for(int i=0; i<nz2; i++)
                z1[i] = z0[i] ^z2[i];
            for(int i=nz2; i<nz0; i++)
                z1[i] = z0[i];
            memset(z1+nz0, 0, (nz1-nz0)*sizeof(digit));
        } else {
            for(int i=0; i<nz0; i++)
                z1[i] = z0[i] ^z2[i];
            for(int i=nz0; i<nz2; i++)
                z1[i] = z2[i];
            memset(z1+nz2, 0, (nz1-nz2)*sizeof(digit));
        }
        mul_nl_nr(z1, l01, nl01, r01, nr01);

#ifdef DEBUG_PYGF2X
        depth -= 1;
#endif
        const digit * z020 = z0; // The fact that z2 immediately succeeds z0 in buf is used.
        const digit * z10 = z1;

        // p += z0 + (z0+z1+z2) x^m + z2 x^(2m)
        for(int id_p=0; id_p<m; id_p++)
            *p++ ^= *z020++;

        for(int id_p=0; id_p<nz1; id_p++)
            *p++ ^= *z10++ ^*z020++;
        
        for(int id_p=nz1-m; id_p<nz2; id_p++)
            *p++ ^= *z020++;

        DBG_ASSERT(buf-buf0 == nbuf);
        if(use_heap)
            free(buf0);
    }
}
