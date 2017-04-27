#pragma once
#include <immintrin.h>

inline
float sum8f_avx(const __m256 val) {
	const __m128 valupper = _mm256_extractf128_ps(val, 1);
	const __m128 vallower = _mm256_extractf128_ps(val, 0);
	_mm256_zeroupper();
	const __m128 valval = _mm_add_ps(valupper,
		vallower);
	__m128 valsum = _mm_add_ps(_mm_permute_ps(valval, 0x1B), valval);
	__m128 res = _mm_add_ps(_mm_permute_ps(valsum, 0xB1), valval);
	return _mm_cvtss_f32(res);
}

inline
double sum4d_avx(const __m256d val)
{
	const __m128d valupper = _mm256_extractf128_pd(val, 1);
	const __m128d vallower = _mm256_castpd256_pd128(val);
	_mm256_zeroupper();
	const __m128d valval = _mm_add_pd(valupper, vallower);
	const __m128d res = _mm_add_pd(_mm_permute_pd(valval, 1), valval);
	return _mm_cvtsd_f64(res);
}

inline
float sum4f_sse2(const __m128 val) {
	const __m128 val02_13_20_31 = _mm_add_ps(val, _mm_movehl_ps(val, val));
	const __m128 res = _mm_add_ss(val02_13_20_31, _mm_shuffle_ps(val02_13_20_31, val02_13_20_31, 1));
	return _mm_cvtss_f32(res);
}

inline
double sum2d_sse2(const __m128d val) {
	const __m128d res = _mm_add_pd(val, _mm_shuffle_pd(val, val, 1));
	return _mm_cvtsd_f64(res);
}