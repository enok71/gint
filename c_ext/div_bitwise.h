/* -*- mode: c; c-basic-offset: 4; -*- */
/*******************************************************************************
 *
 * Copyright (c) 2021 Oskar Enoksson. All rights reserved.
 * Licensed under the MIT license. See LICENSE file in the project root for details.
 *
 * Description:
 * Implementation of a simple division algorightm on unlimited polynomials over GF(2)
 *
 *******************************************************************************/
static void
//
// A very simple bitwise Euclidean division implementation
// For comparison, or for for very small numerator/denominator
//
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

