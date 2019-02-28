#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// Fast merge function using Intel SIMD instrinsics, takes two sorted arrays of eight 16-bit integers
// and produces two vectors (vecMin, vecMax) containing the sixteen integer inputs in a sorted sequence, with
// the eight smallest integers in vecMin, and eight largest in vecMax.
unsafe fn sse_merge(vInput1: __m128i, vInput2: __m128i) -> (__m128i, __m128i) {
    let mut vecTmp = _mm_min_epu16(vInput1, vInput2);
    let mut vecMax = _mm_max_epu16(vInput1, vInput2);
    vecTmp = _mm_alignr_epi8(vecTmp, vecTmp, 2);
    let mut vecMin = _mm_min_epu16(vecTmp, vecMax);
    vecMax = _mm_max_epu16(vecTmp, vecMax);
    vecTmp = _mm_alignr_epi8(vecMin, vecMin, 2);
    vecMin = _mm_min_epu16(vecTmp, vecMax);
    vecMax = _mm_max_epu16(vecTmp, vecMax);
    vecTmp = _mm_alignr_epi8(vecMin, vecMin, 2);
    vecMin = _mm_min_epu16(vecTmp, vecMax);
    vecMax = _mm_max_epu16(vecTmp, vecMax);
    vecTmp = _mm_alignr_epi8(vecMin, vecMin, 2);
    vecMin = _mm_min_epu16(vecTmp, vecMax);
    vecMax = _mm_max_epu16(vecTmp, vecMax);
    vecTmp = _mm_alignr_epi8(vecMin, vecMin, 2);
    vecMin = _mm_min_epu16(vecTmp, vecMax);
    vecMax = _mm_max_epu16(vecTmp, vecMax);
    vecTmp = _mm_alignr_epi8(vecMin, vecMin, 2);
    vecMin = _mm_min_epu16(vecTmp, vecMax);
    vecMax = _mm_max_epu16(vecTmp, vecMax);
    vecTmp = _mm_alignr_epi8(vecMin, vecMin, 2);
    vecMin = _mm_min_epu16(vecTmp, vecMax);
    vecMax = _mm_max_epu16(vecTmp, vecMax);
    vecMin = _mm_alignr_epi8(vecMin, vecMin, 2);
    (vecMin, vecMax)
}

#[test]
fn test_sse_merge() {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    use std::mem;
    unsafe {
        let v1 = _mm_set_epi16(
            10, 12, 13, 15, 16, 19, 23, 31, 
        );
        let v2 = _mm_set_epi16(
            13, 17, 33, 50, 58, 59, 60, 62,
        );
        let (r1, r2) = sse_merge(v1, v2);
        assert_eq!(
            mem::transmute::<__m128i, [i16; 8]>(r1),
            [13, 17, 19, 16, 15, 13, 12, 10]
        );
        assert_eq!(
            mem::transmute::<__m128i, [i16; 8]>(r2),
            [62, 60, 59, 58, 50, 33, 23, 31]
        );
    }
}
