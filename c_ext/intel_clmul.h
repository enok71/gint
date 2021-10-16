#include <immintrin.h>

#define PYGF2X_USE_SSE_CLMUL
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

    __m128i li = {l,0};
    __m128i ri = {r,0};
    __m128i pi = _mm_clmulepi64_si128(li,ri,0);
    return pi[0];
}

static inline uint32_t
mul_15_15(uint16_t l, uint16_t r)
// Multiply two unsigned 15-bit polynomials over GF(2) (stored in uint16_t)
// Return as a 29-bit polynomial (stored in uint32_t)
{
    DBG_ASSERT(l<(1<<15));
    DBG_ASSERT(r<(1<<15));

    __m128i li = {l,0};
    __m128i ri = {r,0};
    __m128i pi = _mm_clmulepi64_si128(li,ri,0);
    return pi[0];
}


static inline uint64_t
mul_8_30(uint8_t l, uint32_t r) 
// Multiply a 8-bit with a 30-bit polynomial over GF(2) (stored in uint32_t)
// Return as a 37-bit polynomial (stored in uint64_t)
{
    DBG_ASSERT(l<(1<<8));
    DBG_ASSERT(r<(1<<30));

    __m128i li = {l,0};
    __m128i ri = {r,0};
    __m128i pi = _mm_clmulepi64_si128(li,ri,0);
    return pi[0];
}

static inline uint64_t
mul_15_30(uint16_t l, uint32_t r) 
// Multiply a 15-bit with a 30-bit polynomial over GF(2) (stored in uint32_t)
// Return as a 59-bit polynomial (stored in uint64_t)
{
    DBG_ASSERT(l<(1<<15));
    DBG_ASSERT(r<(1<<30));

    __m128i li = {l,0};
    __m128i ri = {r,0};
    __m128i pi = _mm_clmulepi64_si128(li,ri,0);
    return pi[0];
}

static inline uint64_t
mul_30_30(uint32_t l, uint32_t r) 
// Multiply two unsigned 30-bit polynomials over GF(2) (stored in uint32_t)
// Return as a 59-bit polynomial (stored in uint64_t)
{
    DBG_ASSERT(l<(1<<30));
    DBG_ASSERT(r<(1<<30));

    __m128i li = {l,0};
    __m128i ri = {r,0};
    __m128i pi = _mm_clmulepi64_si128(li,ri,0);
    return pi[0];
}

static inline uint32_t
sqr_15(uint16_t f)
{
  __m128i f64 = {f,0};
  __m128i p128 = _mm_clmulepi64_si128(f64,f64,0);
  return p128[0];
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
	__m128i li = {ic,0};
	__m128i pi128 = _mm_clmulepi64_si128(li,li,0);
	twodigits pi = pi128[0];
	digit pd0 = pi&PyLong_MASK;
	digit pd1 = (pi>>PyLong_SHIFT);

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
