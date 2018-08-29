#include <xmmintrin.h> /* for _mm_shuffle_pi16 and _MM_SHUFFLE */
#include <emmintrin.h> /* for SSE2 intrinsics */
#include "pixman-private.h"
#include "pixman-combine32.h"
#include "pixman-inlines.h"

__m128i mask_0080;
__m128i mask_00ff;
__m128i mask_0101;
__m128i mask_ffff;
__m128i mask_ff000000;
__m128i mask_alpha;

__m128i mask_565_r;
__m128i mask_565_g1, mask_565_g2;
__m128i mask_565_b;
__m128i mask_red;
__m128i mask_green;
__m128i mask_blue;

__m128i mask_565_fix_rb;
__m128i mask_565_fix_g;

__m128i mask_565_rb;
__m128i mask_565_pack_multiplier;

static force_inline __m128i
unpack_32_1x128 (uint32_t data)
{
    return _mm_unpacklo_epi8 (_mm_cvtsi32_si128 (data), _mm_setzero_si128 ());
}

static force_inline void
unpack_128_2x128 (__m128i data, __m128i* data_lo, __m128i* data_hi)
{
    *data_lo = _mm_unpacklo_epi8 (data, _mm_setzero_si128 ());
    *data_hi = _mm_unpackhi_epi8 (data, _mm_setzero_si128 ());
}

static force_inline __m128i
unpack_565_to_8888 (__m128i lo)
{
    __m128i r, g, b, rb, t;

    r = _mm_and_si128 (_mm_slli_epi32 (lo, 8), mask_red);
    g = _mm_and_si128 (_mm_slli_epi32 (lo, 5), mask_green);
    b = _mm_and_si128 (_mm_slli_epi32 (lo, 3), mask_blue);

    rb = _mm_or_si128 (r, b);
    t  = _mm_and_si128 (rb, mask_565_fix_rb);
    t  = _mm_srli_epi32 (t, 5);
    rb = _mm_or_si128 (rb, t);

    t  = _mm_and_si128 (g, mask_565_fix_g);
    t  = _mm_srli_epi32 (t, 6);
    g  = _mm_or_si128 (g, t);

    return _mm_or_si128 (rb, g);
}

static force_inline void
unpack_565_128_4x128 (__m128i  data,
                      __m128i* data0,
                      __m128i* data1,
                      __m128i* data2,
                      __m128i* data3)
{
    __m128i lo, hi;

    lo = _mm_unpacklo_epi16 (data, _mm_setzero_si128 ());
    hi = _mm_unpackhi_epi16 (data, _mm_setzero_si128 ());

    lo = unpack_565_to_8888 (lo);
    hi = unpack_565_to_8888 (hi);

    unpack_128_2x128 (lo, data0, data1);
    unpack_128_2x128 (hi, data2, data3);
}

static force_inline uint16_t
pack_565_32_16 (uint32_t pixel)
{
    return (uint16_t) (((pixel >> 8) & 0xf800) |
		       ((pixel >> 5) & 0x07e0) |
		       ((pixel >> 3) & 0x001f));
}

static force_inline __m128i
pack_2x128_128 (__m128i lo, __m128i hi)
{
    return _mm_packus_epi16 (lo, hi);
}

static force_inline __m128i
pack_565_2packedx128_128 (__m128i lo, __m128i hi)
{
    __m128i rb0 = _mm_and_si128 (lo, mask_565_rb);
    __m128i rb1 = _mm_and_si128 (hi, mask_565_rb);

    __m128i t0 = _mm_madd_epi16 (rb0, mask_565_pack_multiplier);
    __m128i t1 = _mm_madd_epi16 (rb1, mask_565_pack_multiplier);

    __m128i g0 = _mm_and_si128 (lo, mask_green);
    __m128i g1 = _mm_and_si128 (hi, mask_green);

    t0 = _mm_or_si128 (t0, g0);
    t1 = _mm_or_si128 (t1, g1);

    /* Simulates _mm_packus_epi32 */
    t0 = _mm_slli_epi32 (t0, 16 - 5);
    t1 = _mm_slli_epi32 (t1, 16 - 5);
    t0 = _mm_srai_epi32 (t0, 16);
    t1 = _mm_srai_epi32 (t1, 16);
    return _mm_packs_epi32 (t0, t1);
}

static force_inline __m128i
pack_565_2x128_128 (__m128i lo, __m128i hi)
{
    __m128i data;
    __m128i r, g1, g2, b;

    data = pack_2x128_128 (lo, hi);

    r  = _mm_and_si128 (data, mask_565_r);
    g1 = _mm_and_si128 (_mm_slli_epi32 (data, 3), mask_565_g1);
    g2 = _mm_and_si128 (_mm_srli_epi32 (data, 5), mask_565_g2);
    b  = _mm_and_si128 (_mm_srli_epi32 (data, 3), mask_565_b);

    return _mm_or_si128 (_mm_or_si128 (_mm_or_si128 (r, g1), g2), b);
}

static force_inline __m128i
pack_565_4x128_128 (__m128i* xmm0, __m128i* xmm1, __m128i* xmm2, __m128i* xmm3)
{
    return _mm_packus_epi16 (pack_565_2x128_128 (*xmm0, *xmm1),
			     pack_565_2x128_128 (*xmm2, *xmm3));
}

static force_inline int
is_opaque (__m128i x)
{
    __m128i ffs = _mm_cmpeq_epi8 (x, x);

    return (_mm_movemask_epi8 (_mm_cmpeq_epi8 (x, ffs)) & 0x8888) == 0x8888;
}

static force_inline int
is_zero (__m128i x)
{
    return _mm_movemask_epi8 (
	_mm_cmpeq_epi8 (x, _mm_setzero_si128 ())) == 0xffff;
}

static force_inline int
is_transparent (__m128i x)
{
    return (_mm_movemask_epi8 (
		_mm_cmpeq_epi8 (x, _mm_setzero_si128 ())) & 0x8888) == 0x8888;
}

static force_inline __m128i
expand_pixel_32_1x128 (uint32_t data)
{
    return _mm_shuffle_epi32 (unpack_32_1x128 (data), _MM_SHUFFLE (1, 0, 1, 0));
}

static force_inline __m128i
expand_alpha_1x128 (__m128i data)
{
    return _mm_shufflehi_epi16 (_mm_shufflelo_epi16 (data,
						     _MM_SHUFFLE (3, 3, 3, 3)),
				_MM_SHUFFLE (3, 3, 3, 3));
}

static force_inline void
expand_alpha_2x128 (__m128i  data_lo,
                    __m128i  data_hi,
                    __m128i* alpha_lo,
                    __m128i* alpha_hi)
{
    __m128i lo, hi;

    lo = _mm_shufflelo_epi16 (data_lo, _MM_SHUFFLE (3, 3, 3, 3));
    hi = _mm_shufflelo_epi16 (data_hi, _MM_SHUFFLE (3, 3, 3, 3));

    *alpha_lo = _mm_shufflehi_epi16 (lo, _MM_SHUFFLE (3, 3, 3, 3));
    *alpha_hi = _mm_shufflehi_epi16 (hi, _MM_SHUFFLE (3, 3, 3, 3));
}

static force_inline void
expand_alpha_rev_2x128 (__m128i  data_lo,
                        __m128i  data_hi,
                        __m128i* alpha_lo,
                        __m128i* alpha_hi)
{
    __m128i lo, hi;

    lo = _mm_shufflelo_epi16 (data_lo, _MM_SHUFFLE (0, 0, 0, 0));
    hi = _mm_shufflelo_epi16 (data_hi, _MM_SHUFFLE (0, 0, 0, 0));
    *alpha_lo = _mm_shufflehi_epi16 (lo, _MM_SHUFFLE (0, 0, 0, 0));
    *alpha_hi = _mm_shufflehi_epi16 (hi, _MM_SHUFFLE (0, 0, 0, 0));
}

static force_inline void
pix_multiply_2x128 (__m128i* data_lo,
                    __m128i* data_hi,
                    __m128i* alpha_lo,
                    __m128i* alpha_hi,
                    __m128i* ret_lo,
                    __m128i* ret_hi)
{
    __m128i lo, hi;

    lo = _mm_mullo_epi16 (*data_lo, *alpha_lo);
    hi = _mm_mullo_epi16 (*data_hi, *alpha_hi);
    lo = _mm_adds_epu16 (lo, mask_0080);
    hi = _mm_adds_epu16 (hi, mask_0080);
    *ret_lo = _mm_mulhi_epu16 (lo, mask_0101);
    *ret_hi = _mm_mulhi_epu16 (hi, mask_0101);
}

static force_inline void
pix_add_multiply_2x128 (__m128i* src_lo,
                        __m128i* src_hi,
                        __m128i* alpha_dst_lo,
                        __m128i* alpha_dst_hi,
                        __m128i* dst_lo,
                        __m128i* dst_hi,
                        __m128i* alpha_src_lo,
                        __m128i* alpha_src_hi,
                        __m128i* ret_lo,
                        __m128i* ret_hi)
{
    __m128i t1_lo, t1_hi;
    __m128i t2_lo, t2_hi;

    pix_multiply_2x128 (src_lo, src_hi, alpha_dst_lo, alpha_dst_hi, &t1_lo, &t1_hi);
    pix_multiply_2x128 (dst_lo, dst_hi, alpha_src_lo, alpha_src_hi, &t2_lo, &t2_hi);

    *ret_lo = _mm_adds_epu8 (t1_lo, t2_lo);
    *ret_hi = _mm_adds_epu8 (t1_hi, t2_hi);
}

static force_inline void
negate_2x128 (__m128i  data_lo,
              __m128i  data_hi,
              __m128i* neg_lo,
              __m128i* neg_hi)
{
    *neg_lo = _mm_xor_si128 (data_lo, mask_00ff);
    *neg_hi = _mm_xor_si128 (data_hi, mask_00ff);
}

static force_inline void
invert_colors_2x128 (__m128i  data_lo,
                     __m128i  data_hi,
                     __m128i* inv_lo,
                     __m128i* inv_hi)
{
    __m128i lo, hi;

    lo = _mm_shufflelo_epi16 (data_lo, _MM_SHUFFLE (3, 0, 1, 2));
    hi = _mm_shufflelo_epi16 (data_hi, _MM_SHUFFLE (3, 0, 1, 2));
    *inv_lo = _mm_shufflehi_epi16 (lo, _MM_SHUFFLE (3, 0, 1, 2));
    *inv_hi = _mm_shufflehi_epi16 (hi, _MM_SHUFFLE (3, 0, 1, 2));
}

static force_inline void
over_2x128 (__m128i* src_lo,
            __m128i* src_hi,
            __m128i* alpha_lo,
            __m128i* alpha_hi,
            __m128i* dst_lo,
            __m128i* dst_hi)
{
    __m128i t1, t2;

    negate_2x128 (*alpha_lo, *alpha_hi, &t1, &t2);

    pix_multiply_2x128 (dst_lo, dst_hi, &t1, &t2, dst_lo, dst_hi);

    *dst_lo = _mm_adds_epu8 (*src_lo, *dst_lo);
    *dst_hi = _mm_adds_epu8 (*src_hi, *dst_hi);
}

static force_inline void
over_rev_non_pre_2x128 (__m128i  src_lo,
                        __m128i  src_hi,
                        __m128i* dst_lo,
                        __m128i* dst_hi)
{
    __m128i lo, hi;
    __m128i alpha_lo, alpha_hi;

    expand_alpha_2x128 (src_lo, src_hi, &alpha_lo, &alpha_hi);

    lo = _mm_or_si128 (alpha_lo, mask_alpha);
    hi = _mm_or_si128 (alpha_hi, mask_alpha);

    invert_colors_2x128 (src_lo, src_hi, &src_lo, &src_hi);

    pix_multiply_2x128 (&src_lo, &src_hi, &lo, &hi, &lo, &hi);

    over_2x128 (&lo, &hi, &alpha_lo, &alpha_hi, dst_lo, dst_hi);
}

static force_inline void
in_over_2x128 (__m128i* src_lo,
               __m128i* src_hi,
               __m128i* alpha_lo,
               __m128i* alpha_hi,
               __m128i* mask_lo,
               __m128i* mask_hi,
               __m128i* dst_lo,
               __m128i* dst_hi)
{
    __m128i s_lo, s_hi;
    __m128i a_lo, a_hi;

    pix_multiply_2x128 (src_lo,   src_hi, mask_lo, mask_hi, &s_lo, &s_hi);
    pix_multiply_2x128 (alpha_lo, alpha_hi, mask_lo, mask_hi, &a_lo, &a_hi);

    over_2x128 (&s_lo, &s_hi, &a_lo, &a_hi, dst_lo, dst_hi);
}

/* load 4 pixels from a 16-byte boundary aligned address */
static force_inline __m128i
load_128_aligned (__m128i* src)
{
    return _mm_load_si128 (src);
}

/* load 4 pixels from a unaligned address */
static force_inline __m128i
load_128_unaligned (const __m128i* src)
{
    return _mm_loadu_si128 (src);
}

/* save 4 pixels using Write Combining memory on a 16-byte
 * boundary aligned address
 */
static force_inline void
save_128_write_combining (__m128i* dst,
                          __m128i  data)
{
    _mm_stream_si128 (dst, data);
}

/* save 4 pixels on a 16-byte boundary aligned address */
static force_inline void
save_128_aligned (__m128i* dst,
                  __m128i  data)
{
    _mm_store_si128 (dst, data);
}

/* save 4 pixels on a unaligned address */
static force_inline void
save_128_unaligned (__m128i* dst,
                    __m128i  data)
{
    _mm_storeu_si128 (dst, data);
}

static force_inline __m128i
load_32_1x128 (uint32_t data)
{
    return _mm_cvtsi32_si128 (data);
}

static force_inline __m128i
expand_alpha_rev_1x128 (__m128i data)
{
    return _mm_shufflelo_epi16 (data, _MM_SHUFFLE (0, 0, 0, 0));
}

static force_inline __m128i
expand_pixel_8_1x128 (uint8_t data)
{
    return _mm_shufflelo_epi16 (
	unpack_32_1x128 ((uint32_t)data), _MM_SHUFFLE (0, 0, 0, 0));
}

static force_inline __m128i
pix_multiply_1x128 (__m128i data,
		    __m128i alpha)
{
    return _mm_mulhi_epu16 (_mm_adds_epu16 (_mm_mullo_epi16 (data, alpha),
					    mask_0080),
			    mask_0101);
}

static force_inline __m128i
pix_add_multiply_1x128 (__m128i* src,
			__m128i* alpha_dst,
			__m128i* dst,
			__m128i* alpha_src)
{
    __m128i t1 = pix_multiply_1x128 (*src, *alpha_dst);
    __m128i t2 = pix_multiply_1x128 (*dst, *alpha_src);

    return _mm_adds_epu8 (t1, t2);
}

static force_inline __m128i
negate_1x128 (__m128i data)
{
    return _mm_xor_si128 (data, mask_00ff);
}

static force_inline __m128i
invert_colors_1x128 (__m128i data)
{
    return _mm_shufflelo_epi16 (data, _MM_SHUFFLE (3, 0, 1, 2));
}

static force_inline __m128i
over_1x128 (__m128i src, __m128i alpha, __m128i dst)
{
    return _mm_adds_epu8 (src, pix_multiply_1x128 (dst, negate_1x128 (alpha)));
}

static force_inline __m128i
in_over_1x128 (__m128i* src, __m128i* alpha, __m128i* mask, __m128i* dst)
{
    return over_1x128 (pix_multiply_1x128 (*src, *mask),
		       pix_multiply_1x128 (*alpha, *mask),
		       *dst);
}

static force_inline __m128i
over_rev_non_pre_1x128 (__m128i src, __m128i dst)
{
    __m128i alpha = expand_alpha_1x128 (src);

    return over_1x128 (pix_multiply_1x128 (invert_colors_1x128 (src),
					   _mm_or_si128 (alpha, mask_alpha)),
		       alpha,
		       dst);
}

static force_inline uint32_t
pack_1x128_32 (__m128i data)
{
    return _mm_cvtsi128_si32 (_mm_packus_epi16 (data, _mm_setzero_si128 ()));
}

static force_inline __m128i
expand565_16_1x128 (uint16_t pixel)
{
    __m128i m = _mm_cvtsi32_si128 (pixel);

    m = unpack_565_to_8888 (m);

    return _mm_unpacklo_epi8 (m, _mm_setzero_si128 ());
}

static force_inline uint32_t
core_combine_over_u_pixel_sse2 (uint32_t src, uint32_t dst)
{
    uint8_t a;
    __m128i xmms;

    a = src >> 24;

    if (a == 0xff)
    {
	return src;
    }
    else if (src)
    {
	xmms = unpack_32_1x128 (src);
	return pack_1x128_32 (
	    over_1x128 (xmms, expand_alpha_1x128 (xmms),
			unpack_32_1x128 (dst)));
    }

    return dst;
}

static force_inline uint32_t
combine1 (const uint32_t *ps, const uint32_t *pm)
{
    uint32_t s = *ps;

    if (pm)
    {
	__m128i ms, mm;

	mm = unpack_32_1x128 (*pm);
	mm = expand_alpha_1x128 (mm);

	ms = unpack_32_1x128 (s);
	ms = pix_multiply_1x128 (ms, mm);

	s = pack_1x128_32 (ms);
    }

    return s;
}
