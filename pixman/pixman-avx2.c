#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <immintrin.h> /* for AVX2 intrinsics */
#include "pixman-private.h"
#include "pixman-combine32.h"
#include "pixman-inlines.h"
#include "pixman-sse2.h"

#define MASK_0080_AVX2 _mm256_set1_epi16(0x0080)
#define MASK_00FF_AVX2 _mm256_set1_epi16(0x00ff)
#define MASK_0101_AVX2 _mm256_set1_epi16(0x0101)

static force_inline __m256i
load_256_aligned (__m256i* src)
{
    return _mm256_load_si256(src);
}

static force_inline void
negate_2x256 (__m256i  data_lo,
	      __m256i  data_hi,
	      __m256i* neg_lo,
	      __m256i* neg_hi)
{
    *neg_lo = _mm256_xor_si256 (data_lo, MASK_00FF_AVX2);
    *neg_hi = _mm256_xor_si256 (data_hi, MASK_00FF_AVX2);
}

static force_inline __m256i
pack_2x256_256 (__m256i lo, __m256i hi)
{
    return _mm256_packus_epi16 (lo, hi);
}
 
static force_inline void
pix_multiply_2x256 (__m256i* data_lo,
		    __m256i* data_hi,
		    __m256i* alpha_lo,
		    __m256i* alpha_hi,
		    __m256i* ret_lo,
		    __m256i* ret_hi)
{
    __m256i lo, hi;

    lo = _mm256_mullo_epi16 (*data_lo, *alpha_lo);
    hi = _mm256_mullo_epi16 (*data_hi, *alpha_hi);
    lo = _mm256_adds_epu16 (lo, MASK_0080_AVX2);
    hi = _mm256_adds_epu16 (hi, MASK_0080_AVX2);
    *ret_lo = _mm256_mulhi_epu16 (lo, MASK_0101_AVX2);
    *ret_hi = _mm256_mulhi_epu16 (hi, MASK_0101_AVX2);
}
 
static force_inline void
over_2x256 (__m256i* src_lo,
	    __m256i* src_hi,
	    __m256i* alpha_lo,
	    __m256i* alpha_hi,
	    __m256i* dst_lo,
	    __m256i* dst_hi)
{
    __m256i t1, t2;

    negate_2x256 (*alpha_lo, *alpha_hi, &t1, &t2);

    pix_multiply_2x256 (dst_lo, dst_hi, &t1, &t2, dst_lo, dst_hi);

    *dst_lo = _mm256_adds_epu8 (*src_lo, *dst_lo);
    *dst_hi = _mm256_adds_epu8 (*src_hi, *dst_hi);
}

static force_inline void
expand_alpha_2x256 (__m256i  data_lo,
		    __m256i  data_hi,
		    __m256i* alpha_lo,
		    __m256i* alpha_hi)
{
    __m256i lo, hi;

    lo = _mm256_shufflelo_epi16 (data_lo, _MM_SHUFFLE (3, 3, 3, 3));
    hi = _mm256_shufflelo_epi16 (data_hi, _MM_SHUFFLE (3, 3, 3, 3));

    *alpha_lo = _mm256_shufflehi_epi16 (lo, _MM_SHUFFLE (3, 3, 3, 3));
    *alpha_hi = _mm256_shufflehi_epi16 (hi, _MM_SHUFFLE (3, 3, 3, 3));
}

static force_inline  void
unpack_256_2x256 (__m256i data, __m256i* data_lo, __m256i* data_hi)
{
    *data_lo = _mm256_unpacklo_epi8 (data, _mm256_setzero_si256 ());
    *data_hi = _mm256_unpackhi_epi8 (data, _mm256_setzero_si256 ());
}

/* save 4 pixels on a 16-byte boundary aligned address */
static force_inline void
save_256_aligned (__m256i* dst,
		  __m256i  data)
{
    _mm256_store_si256 (dst, data);
}

static force_inline int
is_opaque_256 (__m256i x)
{
    __m256i ffs = _mm256_cmpeq_epi8 (x, x);

    return (_mm256_movemask_epi8
	    (_mm256_cmpeq_epi8 (x, ffs)) & 0x88888888) == 0x88888888;
}

static force_inline int
is_zero_256 (__m256i x)
{
    return _mm256_movemask_epi8 (
	_mm256_cmpeq_epi8 (x, _mm256_setzero_si256 ())) == 0xffffffff;
}

static force_inline int
is_transparent_256 (__m256i x)
{
    return (_mm256_movemask_epi8 (
		_mm256_cmpeq_epi8 (x, _mm256_setzero_si256 ())) & 0x88888888)
                == 0x88888888;
}


/* load 4 pixels from a unaligned address */
static force_inline __m256i
load_256_unaligned (const __m256i* src)
{
    return _mm256_loadu_si256 (src);
}

static force_inline __m256i
combine8 (const __m256i *ps, const __m256i *pm)
{
    __m256i ymm_src_lo, ymm_src_hi;
    __m256i ymm_msk_lo, ymm_msk_hi;
    __m256i s;

    if (pm)
    {
	ymm_msk_lo = load_256_unaligned (pm);

	if (is_transparent_256 (ymm_msk_lo))
	    return _mm256_setzero_si256 ();
    }

    s = load_256_unaligned (ps);

    if (pm)
    {
	unpack_256_2x256 (s, &ymm_src_lo, &ymm_src_hi);
	unpack_256_2x256 (ymm_msk_lo, &ymm_msk_lo, &ymm_msk_hi);

	expand_alpha_2x256 (ymm_msk_lo, ymm_msk_hi, &ymm_msk_lo, &ymm_msk_hi);

	pix_multiply_2x256 (&ymm_src_lo, &ymm_src_hi,
			    &ymm_msk_lo, &ymm_msk_hi,
			    &ymm_src_lo, &ymm_src_hi);

	s = pack_2x256_256 (ymm_src_lo, ymm_src_hi);
    }

    return s;
}

static force_inline void
core_combine_over_u_avx2_mask (uint32_t *	  pd,
			       const uint32_t*	  ps,
			       const uint32_t*	  pm,
			       int		  w)
{
    uint32_t s, d;
    while (w && ((uintptr_t)pd & 31))
    {
	d = *pd;
	s = combine1 (ps, pm);

	if (s)
	    *pd = core_combine_over_u_pixel_sse2 (s, d);
	pd++;
	ps++;
	pm++;
	w--;
    }

    /*
     * dst is 32 byte aligned, and w >=8 means the next 256 bits
     * contain relevant data
    */
    
    while (w >= 8)
    {
	__m256i mask = load_256_unaligned ((__m256i *)pm);

	if (!is_zero_256 (mask))
	{
	    __m256i src;
	    __m256i src_hi, src_lo;
	    __m256i mask_hi, mask_lo;
	    __m256i alpha_hi, alpha_lo;

	    src = load_256_unaligned ((__m256i *)ps);

	    if (is_opaque_256 (_mm256_and_si256 (src, mask)))
	    {
	        save_256_aligned ((__m256i *)pd, src);
	    }
	    else
	    {
	        __m256i dst = load_256_aligned ((__m256i *)pd);
	        __m256i dst_hi, dst_lo;

	        unpack_256_2x256 (mask, &mask_lo, &mask_hi);
	        unpack_256_2x256 (src, &src_lo, &src_hi);

	        expand_alpha_2x256 (mask_lo, mask_hi, &mask_lo, &mask_hi);
	        pix_multiply_2x256 (&src_lo, &src_hi,
				    &mask_lo, &mask_hi,
				    &src_lo, &src_hi);
		
		unpack_256_2x256 (dst, &dst_lo, &dst_hi);
		expand_alpha_2x256 (src_lo, src_hi,
				    &alpha_lo, &alpha_hi);

		over_2x256 (&src_lo, &src_hi, &alpha_lo, &alpha_hi,
			    &dst_lo, &dst_hi);

		save_256_aligned (
		    (__m256i *)pd,
		    pack_2x256_256 (dst_lo, dst_hi));
	    }
	}
	pm += 8;
	ps += 8;
	pd += 8;
	w -= 8;
    }

    while (w)
    {
	d = *pd;
	s = combine1 (ps, pm);

	if (s)
	    *pd = core_combine_over_u_pixel_sse2 (s, d);
	pd++;
	ps++;
	pm++;
	w--;
    }
}

static force_inline void
core_combine_over_u_avx2_no_mask (uint32_t *	     pd,
				  const uint32_t*    ps,
				  int		     w)
{
    uint32_t s, d;

    /* Align dst on a 16-byte boundary */
    while (w && ((uintptr_t)pd & 31))
    {
	d = *pd;
	s = *ps;

	if (s)
	    *pd = core_combine_over_u_pixel_sse2 (s, d);
	pd++;
	ps++;
	w--;
    }

    while (w >= 8)
    {
	__m256i src;
	__m256i src_hi, src_lo, dst_hi, dst_lo;
	__m256i alpha_hi, alpha_lo;

	src = load_256_unaligned ((__m256i *)ps);

	if (!is_zero_256 (src))
	{
	    if (is_opaque_256 (src))
	    {
		save_256_aligned ((__m256i *)pd, src);
	    }
	    else
	    {
		__m256i dst = load_256_aligned ((__m256i *)pd);

		unpack_256_2x256 (src, &src_lo, &src_hi);
		unpack_256_2x256 (dst, &dst_lo, &dst_hi);

		expand_alpha_2x256 (src_lo, src_hi,
				    &alpha_lo, &alpha_hi);
		over_2x256 (&src_lo, &src_hi, &alpha_lo, &alpha_hi,
			    &dst_lo, &dst_hi);

		save_256_aligned (
		    (__m256i *)pd,
		    pack_2x256_256 (dst_lo, dst_hi));
	    }
	}

	ps += 8;
	pd += 8;
	w -= 8;
    }
    while (w)
    {
	d = *pd;
	s = *ps;

	if (s)
	    *pd = core_combine_over_u_pixel_sse2 (s, d);
	pd++;
	ps++;
	w--;
    }
}

static force_inline void
avx2_combine_over_u (pixman_implementation_t *imp,
		     pixman_op_t	      op,
		     uint32_t *		      pd,
		     const uint32_t *	      ps,
		     const uint32_t *	      pm,
		     int		      w)
{
    if (pm)
	core_combine_over_u_avx2_mask (pd, ps, pm, w);
    else
	core_combine_over_u_avx2_no_mask (pd, ps, w);
}

static void
avx2_combine_over_reverse_u (pixman_implementation_t *imp,
			     pixman_op_t	      op,
			     uint32_t *		      pd,
			     const uint32_t *	      ps,
			     const uint32_t *	      pm,
			     int		      w)
{
    uint32_t s, d;

    __m256i ymm_dst_lo, ymm_dst_hi;
    __m256i ymm_src_lo, ymm_src_hi;
    __m256i ymm_alpha_lo, ymm_alpha_hi;

    /* Align dst on a 16-byte boundary */
    while (w &&
	   ((uintptr_t)pd & 31))
    {
	d = *pd;
	s = combine1 (ps, pm);

	*pd++ = core_combine_over_u_pixel_sse2 (d, s);
	w--;
	ps++;
	if (pm)
	    pm++;
    }

    while (w >= 8)
    {
	ymm_src_hi = combine8 ((__m256i*)ps, (__m256i*)pm);
	ymm_dst_hi = load_256_aligned ((__m256i*) pd);

	unpack_256_2x256 (ymm_src_hi, &ymm_src_lo, &ymm_src_hi);
	unpack_256_2x256 (ymm_dst_hi, &ymm_dst_lo, &ymm_dst_hi);

	expand_alpha_2x256 (ymm_dst_lo, ymm_dst_hi,
			    &ymm_alpha_lo, &ymm_alpha_hi);

	over_2x256 (&ymm_dst_lo, &ymm_dst_hi,
		    &ymm_alpha_lo, &ymm_alpha_hi,
		    &ymm_src_lo, &ymm_src_hi);

	/* rebuid the 4 pixel data and save*/
	save_256_aligned ((__m256i*)pd,
			  pack_2x256_256 (ymm_src_lo, ymm_src_hi));

	w -= 8;
	ps += 8;
	pd += 8;

	if (pm)
	    pm += 8;
    }

    while (w)
    {
	d = *pd;
	s = combine1 (ps, pm);

	*pd++ = core_combine_over_u_pixel_sse2 (d, s);
	ps++;
	w--;
	if (pm)
	    pm++;
    }
}

static void
avx2_composite_over_8888_8888 (pixman_implementation_t *imp,
                               pixman_composite_info_t *info)
{
    PIXMAN_COMPOSITE_ARGS (info);
    int dst_stride, src_stride;
    uint32_t    *dst_line, *dst;
    uint32_t    *src_line, *src;

    PIXMAN_IMAGE_GET_LINE (
	dest_image, dest_x, dest_y, uint32_t, dst_stride, dst_line, 1);
    PIXMAN_IMAGE_GET_LINE (
	src_image, src_x, src_y, uint32_t, src_stride, src_line, 1);

    dst = dst_line;
    src = src_line;

    while (height--)
    {
	avx2_combine_over_u (imp, op, dst, src, NULL, width);

	dst += dst_stride;
	src += src_stride;
    }
}
static const pixman_fast_path_t avx2_fast_paths[] =
{
    PIXMAN_STD_FAST_PATH (OVER, a8r8g8b8, null, a8r8g8b8, avx2_composite_over_8888_8888),
    PIXMAN_STD_FAST_PATH (OVER, a8r8g8b8, null, x8r8g8b8, avx2_composite_over_8888_8888),
    PIXMAN_STD_FAST_PATH (OVER, a8b8g8r8, null, a8b8g8r8, avx2_composite_over_8888_8888),
    PIXMAN_STD_FAST_PATH (OVER, a8b8g8r8, null, x8b8g8r8, avx2_composite_over_8888_8888),
    { PIXMAN_OP_NONE },
};

static const pixman_iter_info_t avx2_iters[] =
{
    { PIXMAN_null },
};

#if defined(__GNUC__) && !defined(__x86_64__) && !defined(__amd64__)
__attribute__((__force_align_arg_pointer__))
#endif
pixman_implementation_t *
_pixman_implementation_create_avx2 (pixman_implementation_t *fallback)
{
    pixman_implementation_t *imp = _pixman_implementation_create (fallback, avx2_fast_paths);

    /* Set up function pointers */
    imp->combine_32[PIXMAN_OP_OVER] = avx2_combine_over_u;
    imp->combine_32[PIXMAN_OP_OVER_REVERSE] = avx2_combine_over_reverse_u;
    
    imp->iter_info = avx2_iters;

    return imp;
}
