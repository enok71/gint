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

  poly16x4x2_t p10 = vzip_p16(vget_low_p16(p00),               // 0 1 2 3 1 2 3 4 1 2 3 4 2 3 4 5
			      vget_high_p16(p00));
  poly16x4x2_t p11 = vzip_p16(vget_low_p16(p01),               // 2 3 4 5 3 4 5 6 3 4 5 6 4 5 6 7
			      vget_high_p16(p01));

  uint64x2_t p20 = vmovl_u32(vreinterpret_u32_p16(p10.val[0]));// 0 1 2 3 _ _ _ _ 1 2 3 4 _ _ _ _
  uint64x2_t p21 = vmovl_u32(vreinterpret_u32_p16(p10.val[1]));// 1 2 3 4 _ _ _ _ 2 3 4 5 _ _ _ _
  uint64x2_t p22 = vmovl_u32(vreinterpret_u32_p16(p11.val[0]));// 2 3 4 5 _ _ _ _ 3 4 5 6 _ _ _ _
  uint64x2_t p23 = vmovl_u32(vreinterpret_u32_p16(p11.val[1]));// 3 4 5 6 _ _ _ _ 4 5 6 7 _ _ _ _

  p20 = veorq_u64(p20, vshlq_n_u64(p21, 8));                   // 0 1 2 3 4 _ _ _ 1 2 3 4 5 _ _ _
  p22 = veorq_u64(p22, vshlq_n_u64(p23, 8));                   // 2 3 4 5 6 _ _ _ 3 4 5 6 7 _ _ _
  p20 = veorq_u64(p20, vshlq_n_u64(p22, 16));                  // 0 1 2 3 4 5 6 _ 1 2 3 4 5 6 7 _

  return veor_u64(p20[0], vshl_n_u64(p20[1],8));               // 0 1 2 3 4 5 6 7
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

  return veor_u64(p20[0], vshl_n_u64(p20[1],8));
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
  return veor_u64(p3[0], vshl_n_u64(p3[1], 8));
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
  
  return p1[0] ^ (p1[1]<<8);
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
    for(int id=0; id<ndigs_f; id++) {
	digit ic = fdigits[id];
#if (PyLong_SHIFT == 15)
	poly8x8_t f = vreinterpret_p8_p16(vdup_n_p16(ic));
	poly16x8_t  pi128 = vmull_p8(f,f);
	digit pd0 = pi128[0];
	digit pd1 = pi128[1] << 1;
#elif (PyLong_SHIFT == 30)		
	poly8x8_t f  = vreinterpret_p8_u32(vdup_n_32(l));
	uint32x4_t pi128 = vreinterpret_u32_p16(vmull_p8(f,f));
	digit pd0 = pi128[0];
	pi128 = vreinterpret_u32_u64(vshl_n_u64(vreinterpret_u64_u32(pi128,2)));
	digit pd1 = pi128[1];
#else
#error
#endif
	if(pd0) {
	    result[idp] ^= pd0;
	}
	idp++;
	if(pd1) {
	    result[idp] ^= pd1;
	}
	idp++;
    }
}
