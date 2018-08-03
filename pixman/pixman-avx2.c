#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <immintrin.h> /* for AVX2 intrinsics */
#include "pixman-private.h"
#include "pixman-combine32.h"
#include "pixman-inlines.h"

static const pixman_fast_path_t avx2_fast_paths[] =
{
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
    imp->iter_info = avx2_iters;

    return imp;
}
