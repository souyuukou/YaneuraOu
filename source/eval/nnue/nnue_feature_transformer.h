// A class that converts the input features of the NNUE evaluation function
// NNUE評価関数の入力特徴量の変換を行うクラス

#ifndef CLASSIC_NNUE_FEATURE_TRANSFORMER_H_INCLUDED
#define CLASSIC_NNUE_FEATURE_TRANSFORMER_H_INCLUDED

#include "../../config.h"

#if defined(EVAL_NNUE)

#if defined(SFNNwoPSQT)
#define USE_ELEMENT_WISE_MULTIPLY
#endif

#include "nnue_common.h"
#include "nnue_architecture.h"
#include "features/index_list.h"

#include <algorithm>  // std::clamp
#include <cstring>  // std::memset()

namespace YaneuraOu {
namespace Eval::NNUE {

// If vector instructions are enabled, we update and refresh the
// accumulator tile by tile such that each tile fits in the CPU's
// vector registers.
// ベクトル命令が有効な場合、変数のタイルを、
// 各タイルがCPUのベクトルレジスタに収まるように、更新してリフレッシュする。
#define VECTOR

#if defined(USE_AVX512)
using vec_t = __m512i;
#define vec_load(a) _mm512_load_si512(a)
#define vec_store(a, b) _mm512_store_si512(a, b)
#define vec_add_16(a, b) _mm512_add_epi16(a, b)
#define vec_sub_16(a, b) _mm512_sub_epi16(a, b)
#define vec_mulhi_16(a, b) _mm512_mulhi_epi16(a, b)
#define vec_set_16(a) _mm512_set1_epi16(a)
#define vec_max_16(a, b) _mm512_max_epi16(a, b)
#define vec_min_16(a, b) _mm512_min_epi16(a, b)
#define vec_slli_16(a, b) _mm512_slli_epi16(a, b)
#define vec_packus_16(a, b) _mm512_packus_epi16(a, b)
#define vec_zero() _mm512_setzero_si512()
static constexpr IndexType kNumRegs = 8;  // only 8 are needed

#elif defined(USE_AVX2)
using vec_t = __m256i;
#define vec_load(a) _mm256_load_si256(a)
#define vec_store(a, b) _mm256_store_si256(a, b)
#define vec_add_16(a, b) _mm256_add_epi16(a, b)
#define vec_sub_16(a, b) _mm256_sub_epi16(a, b)
#define vec_mulhi_16(a, b) _mm256_mulhi_epi16(a, b)
#define vec_set_16(a) _mm256_set1_epi16(a)
#define vec_max_16(a, b) _mm256_max_epi16(a, b)
#define vec_min_16(a, b) _mm256_min_epi16(a, b)
#define vec_slli_16(a, b) _mm256_slli_epi16(a, b)
#define vec_packus_16(a, b) _mm256_packus_epi16(a, b)
#define vec_zero() _mm256_setzero_si256()
static constexpr IndexType kNumRegs = 16;

#elif defined(USE_SSE2)
using vec_t = __m128i;
#define vec_load(a) (*(a))
#define vec_store(a, b) *(a) = (b)
#define vec_add_16(a, b) _mm_add_epi16(a, b)
#define vec_sub_16(a, b) _mm_sub_epi16(a, b)
#define vec_mulhi_16(a, b) _mm_mulhi_epi16(a, b)
#define vec_set_16(a) _mm_set1_epi16(a)
#define vec_max_16(a, b) _mm_max_epi16(a, b)
#define vec_min_16(a, b) _mm_min_epi16(a, b)
#define vec_slli_16(a, b) _mm_slli_epi16(a, b)
#define vec_packus_16(a, b) _mm_packus_epi16(a, b)
#define vec_zero() _mm_setzero_si128()
static constexpr IndexType kNumRegs = Is64Bit ? 16 : 8;

#elif defined(USE_MMX)
using vec_t = __m64;
#define vec_load(a) (*(a))
#define vec_store(a, b) *(a) = (b)
#define vec_add_16(a, b) _mm_add_pi16(a, b)
#define vec_sub_16(a, b) _mm_sub_pi16(a, b)
#define vec_zero() _mm_setzero_si64()
static constexpr IndexType kNumRegs = 8;

#elif defined(USE_NEON)
using vec_t = int16x8_t;
#define vec_load(a) (*(a))
#define vec_store(a, b) *(a) = (b)
#define vec_add_16(a, b) vaddq_s16(a, b)
#define vec_sub_16(a, b) vsubq_s16(a, b)
#define vec_mulhi_16(a, b) vqdmulhq_s16(a, b)
#define vec_set_16(a) vdupq_n_s16(a)
#define vec_max_16(a, b) vmaxq_s16(a, b)
#define vec_min_16(a, b) vminq_s16(a, b)
#define vec_slli_16(a, b) vshlq_s16(a, vec_set_16(b))
#define vec_packus_16(a, b) reinterpret_cast<vec_t>(vcombine_u8(vqmovun_s16(a), vqmovun_s16(b)))
#define vec_zero() \
	vec_t { 0 }
static constexpr IndexType kNumRegs = 16;

#else
#undef VECTOR

#endif

/*
 例) SFNNwop-1536のときのkNumChunksの計算

┌─────────┬───────────────┬─────────────────┬────────────┐
│  SIMD            │ sizeof(vec_t)                │ / sizeof(int16)                  │ kNumChunks             │
├─────────┼───────────────┼─────────────────┼────────────┤
│ AVX-512          │ 64                           │ 32                               │ 1536/32=48             │
├─────────┼───────────────┼─────────────────┼────────────┤
│ AVX2             │ 32                           │ 16                               │ 1536/16=96             │
├─────────┼───────────────┼─────────────────┼────────────┤
│ SSE2             │ 16                           │ 8                                │ 1536/8=192             │
├─────────┼───────────────┼─────────────────┼────────────┤
│ NEON             │ 16                           │ 8                                │ 1536/8=192             │
└─────────┴───────────────┴─────────────────┴────────────┘
*/

constexpr IndexType MaxChunkSize = 16;

// Input feature converter
// 入力特徴量変換器
class FeatureTransformer {
   private:
	// Number of output dimensions for one side
	// 片側分の出力の次元数
	static constexpr IndexType kHalfDimensions = kTransformedFeatureDimensions;

#if defined(VECTOR)
	//static constexpr IndexType kTileHeight = kNumRegs * sizeof(vec_t) / 2;
	//static_assert(kHalfDimensions % kTileHeight == 0, "kTileHeight must divide kHalfDimensions");
	// ⇨  AVX-512でこの制約守れないっぽ。
#endif

   public:
	// Output type
	// 出力の型
	using OutputType = TransformedFeatureType;
	using BiasType   = std::int16_t;
	using WeightType = std::int16_t;

	// Number of input/output dimensions
	// 入出力の次元数
	static constexpr IndexType kInputDimensions  = RawFeatures::kDimensions;
#if defined(USE_ELEMENT_WISE_MULTIPLY)
	static constexpr IndexType kOutputDimensions = kHalfDimensions;
#else
	static constexpr IndexType kOutputDimensions = kHalfDimensions * 2;
#endif

	// Size of forward propagation buffer
	// 順伝播用バッファのサイズ
	static constexpr std::size_t kBufferSize = kOutputDimensions * sizeof(OutputType);

	// Hash value embedded in the evaluation file
	// 評価関数ファイルに埋め込むハッシュ値
	static constexpr std::uint32_t GetHashValue() {
#if defined(SFNNwoPSQT)
		// 学習部と整合性とるの面倒なのでSFNNwoPSQTのときはこれに固定しておく。
		return 0x5f134ab8u;
#else
		return RawFeatures::kHashValue ^ kOutputDimensions;
#endif
	}

	// A string that represents the structure
	// 構造を表す文字列
	static std::string GetStructureString() {
		return RawFeatures::GetName() + "[" + std::to_string(kInputDimensions) + "->"
		       + std::to_string(kHalfDimensions) + "x2]";
	}

	// Read network parameters
	// パラメータを読み込む
	Tools::Result ReadParameters(std::istream& stream) {
#if defined(SFNNwoPSQT_V2)
		// rshogi/nnue-pytorch 互換形式: LEB128 2ブロック読み込み、scale/permute なし
		// Block 1: biases (kHalfDimensions 要素)
		// Block 2: weights (kHalfDimensions * kInputDimensions 要素)
		read_leb_128<WeightType>(stream, biases_, kHalfDimensions);
		read_leb_128<WeightType>(stream, weights_, kHalfDimensions * kInputDimensions);
		// V2: scale_weightsは行わないが、permute_weightsはSIMD packus_16のために必要
#if defined(VECTOR)
		permute_weights(inverse_order_packs);
#endif
#elif defined(USE_ELEMENT_WISE_MULTIPLY)
		// bullet-shogi single-block format: biases + weights in one LEB128 block
		{
			const size_t total = kHalfDimensions + kHalfDimensions * kInputDimensions;
			auto* tmp = new WeightType[total];
			read_leb_128<WeightType>(stream, tmp, total);
			std::memcpy(biases_, tmp, kHalfDimensions * sizeof(WeightType));
			std::memcpy(weights_, tmp + kHalfDimensions, kHalfDimensions * kInputDimensions * sizeof(WeightType));
			delete[] tmp;
		}

#if defined(VECTOR)
		permute_weights(inverse_order_packs);
#endif
		scale_weights(true);
#else
		for (std::size_t i = 0; i < kHalfDimensions; ++i) biases_[i] = read_little_endian<BiasType>(stream);
		for (std::size_t i = 0; i < kHalfDimensions * kInputDimensions; ++i)
			weights_[i] = read_little_endian<WeightType>(stream);
#endif
		return !stream.fail() ? Tools::ResultCode::Ok : Tools::ResultCode::FileReadError;
	}

	// Write network parameters
	// パラメータを書き込む
	bool WriteParameters(std::ostream& stream) const {
		stream.write(reinterpret_cast<const char*>(biases_), kHalfDimensions * sizeof(BiasType));
		stream.write(reinterpret_cast<const char*>(weights_), kHalfDimensions * kInputDimensions * sizeof(WeightType));
		return !stream.fail();
	}

	// Proceed with the difference calculation if possible
	// 可能なら差分計算を進める
	bool UpdateAccumulatorIfPossible(const Position& pos) const {
		const auto now = pos.state();
		if (now->accumulator.computed_accumulation) {
			return true;
		}
#if defined(SFNNwoPSQT)
		// SFNNwoPSQT: MAX_DEPTH=4 手前まで探索して計算済み祖先を探す
		static constexpr int kMaxAncestorDepth = 4;
		StateInfo* path[kMaxAncestorDepth];
		int depth = 0;
		auto* st = now;

		while (st->previous && depth < kMaxAncestorDepth) {
			if (st->dirtyPiece.pieceNo[0] == PIECE_NUMBER_KING + BLACK ||
			    st->dirtyPiece.pieceNo[0] == PIECE_NUMBER_KING + WHITE)
				break;
			path[depth++] = st;
			st = st->previous;
			if (st->accumulator.computed_accumulation) {
				update_accumulator_multi(pos, st, path, depth);
				return true;
			}
		}
#else
		// 非SFNNwoPSQT: 1手前のみ
		const auto prev = now->previous;
		if (prev && prev->accumulator.computed_accumulation) {
			update_accumulator(pos);
			return true;
		}
#endif
		return false;
	}

#if defined(SFNNwoPSQT)
	// キャッシュ付き版: 可能なら差分計算を進める (SFNNwoPSQT専用)
	// reset が必要な視点ではキャッシュ経由で refresh する
	bool UpdateAccumulatorIfPossible(const Position& pos, AccumulatorCaches& cache) const {
		const auto now = pos.state();
		if (now->accumulator.computed_accumulation) {
			return true;
		}

		// MAX_DEPTH 手前まで探索して計算済み祖先を探す
		static constexpr int kMaxAncestorDepth = 4;
		StateInfo* path[kMaxAncestorDepth];
		int depth = 0;
		auto* st = now;

		while (st->previous && depth < kMaxAncestorDepth) {
			if (st->dirtyPiece.pieceNo[0] == PIECE_NUMBER_KING + BLACK ||
			    st->dirtyPiece.pieceNo[0] == PIECE_NUMBER_KING + WHITE)
				break;
			path[depth++] = st;
			st = st->previous;
			if (st->accumulator.computed_accumulation) {
				update_accumulator_multi_with_cache(pos, st, path, depth, cache);
				return true;
			}
		}
		return false;
	}
#endif // defined(SFNNwoPSQT)

	// Convert input features
	// 入力特徴量を変換する
	void Transform(const Position& pos, OutputType* output, bool refresh) const {
		if (refresh || !UpdateAccumulatorIfPossible(pos)) {
			refresh_accumulator(pos);
		}
		const auto& accumulation = pos.state()->accumulator.accumulation;

#if defined(USE_ELEMENT_WISE_MULTIPLY)

#if defined(VECTOR)
			// Packed output is sizeof(vec_t) bytes for each SIMD register
			constexpr IndexType OutputChunkSize = kSimdWidth;
		static_assert((kHalfDimensions / 2) % OutputChunkSize == 0);
		constexpr IndexType NumOutputChunks = kHalfDimensions / 2 / OutputChunkSize;

#if defined(SFNNwoPSQT_V2)
		// V2 (rshogi互換): clamp(0,127) → mullo → >>7 → packus
		// rshogi の sqr_clipped_relu_transform と同一アルゴリズム
		vec_t Zero = vec_zero();
		vec_t One = vec_set_16(127);

		const Color perspectives[2] = { pos.side_to_move(), ~pos.side_to_move() };
		for (IndexType p = 0; p < 2; ++p) {
			const IndexType offset = (kHalfDimensions / 2) * p;

			const vec_t* in0 = reinterpret_cast<const vec_t*>(&(accumulation[perspectives[p]][0][0]));
			const vec_t* in1 = reinterpret_cast<const vec_t*>(&(accumulation[perspectives[p]][0][kHalfDimensions / 2]));
			vec_t* out = reinterpret_cast<vec_t*>(output + offset);

			for (IndexType j = 0; j < NumOutputChunks; ++j)
			{
				// clamp(a, 0, 127), clamp(b, 0, 127)
				const vec_t a0 = vec_min_16(vec_max_16(in0[j * 2 + 0], Zero), One);
				const vec_t a1 = vec_min_16(vec_max_16(in0[j * 2 + 1], Zero), One);
				const vec_t b0 = vec_min_16(vec_max_16(in1[j * 2 + 0], Zero), One);
				const vec_t b1 = vec_min_16(vec_max_16(in1[j * 2 + 1], Zero), One);

				// mullo(a, b) → a*b (max 127*127=16129, fits in i16)
				// srli(product, 7) → a*b >> 7 (max 126, fits in u8 via packus)
#if defined(USE_AVX512)
				const vec_t pa2 = _mm512_srli_epi16(_mm512_mullo_epi16(a0, b0), 7);
				const vec_t pb2 = _mm512_srli_epi16(_mm512_mullo_epi16(a1, b1), 7);
#elif defined(USE_AVX2)
				const vec_t pa2 = _mm256_srli_epi16(_mm256_mullo_epi16(a0, b0), 7);
				const vec_t pb2 = _mm256_srli_epi16(_mm256_mullo_epi16(a1, b1), 7);
#elif defined(USE_SSE2)
				const vec_t pa2 = _mm_srli_epi16(_mm_mullo_epi16(a0, b0), 7);
				const vec_t pb2 = _mm_srli_epi16(_mm_mullo_epi16(a1, b1), 7);
#elif defined(USE_NEON)
				const vec_t pa2 = vshrq_n_s16(vmulq_s16(a0, b0), 7);
				const vec_t pb2 = vshrq_n_s16(vmulq_s16(a1, b1), 7);
#endif
				out[j] = vec_packus_16(pa2, pb2);
			}
		}
#else
		// V1 (既存): scale_weights済みアキュムレータ用 mulhi 方式
		vec_t Zero = vec_zero();
		vec_t One = vec_set_16(127 * 2);

		const Color perspectives[2] = { pos.side_to_move(), ~pos.side_to_move() };
		for (IndexType p = 0; p < 2; ++p) {
			const IndexType offset = (kHalfDimensions / 2) * p;

			const vec_t* in0 = reinterpret_cast<const vec_t*>(&(accumulation[perspectives[p]][0][0]));
			const vec_t* in1 = reinterpret_cast<const vec_t*>(&(accumulation[perspectives[p]][0][kHalfDimensions / 2]));
			vec_t* out = reinterpret_cast<vec_t*>(output + offset);

			constexpr int shift =
#if defined(USE_SSE2)
				7;
#else
				6;
#endif

			for (IndexType j = 0; j < NumOutputChunks; ++j)
			{
				const vec_t sum0a =
					vec_slli_16(vec_max_16(vec_min_16(in0[j * 2 + 0], One), Zero), shift);
				const vec_t sum0b =
					vec_slli_16(vec_max_16(vec_min_16(in0[j * 2 + 1], One), Zero), shift);
				const vec_t sum1a = vec_min_16(in1[j * 2 + 0], One);
				const vec_t sum1b = vec_min_16(in1[j * 2 + 1], One);

				const vec_t pa = vec_mulhi_16(sum0a, sum1a);
				const vec_t pb = vec_mulhi_16(sum0b, sum1b);

				out[j] = vec_packus_16(pa, pb);
			}
		}
#endif

#else
		const Color perspectives[2] = { pos.side_to_move(), ~pos.side_to_move() };
		for (IndexType p = 0; p < 2; ++p) {
			const IndexType offset = (kHalfDimensions / 2) * p;

			for (IndexType j = 0; j < kHalfDimensions / 2; ++j)
			{
				BiasType sum0 = accumulation[perspectives[p]][0][j];
				BiasType sum1 = accumulation[perspectives[p]][0][j + kHalfDimensions / 2];
#if defined(SFNNwoPSQT_V2)
				sum0 = std::clamp<BiasType>(sum0, 0, 127);
				sum1 = std::clamp<BiasType>(sum1, 0, 127);
				output[offset + j] = static_cast<OutputType>((unsigned(sum0) * unsigned(sum1)) >> 7);
#else
				sum0 = std::clamp<BiasType>(sum0, 0, 127 * 2);
				sum1 = std::clamp<BiasType>(sum1, 0, 127 * 2);
				output[offset + j] = static_cast<OutputType>(unsigned(sum0 * sum1) / 512);
#endif
			}

		}
#endif

#else

		// 以下は旧NNUEのコード。
		// ループ本体がx86とNEONで異なる（2入力→1出力 vs 1入力→1出力）ため、
		// kNumChunksの意味自体がアーキテクチャごとに違うため、共通化しにくい。触らないことにする。

#if defined(USE_AVX512)
		constexpr IndexType kNumChunks = kHalfDimensions / kSimdWidth;
		static_assert(kHalfDimensions % kSimdWidth == 0);
		const __m512i kControl = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);
		const __m512i kZero    = _mm512_setzero_si512();

#elif defined(USE_AVX2)
		constexpr IndexType kNumChunks = kHalfDimensions / kSimdWidth;
		constexpr int       kControl   = 0b11011000;
		const __m256i       kZero      = _mm256_setzero_si256();

#elif defined(USE_SSE2)
		constexpr IndexType kNumChunks = kHalfDimensions / kSimdWidth;
#if defined(USE_SSE41)
		const __m128i kZero = _mm_setzero_si128();
#else  // SSE41非対応だがSSE2は使える環境
		const __m128i k0x80s = _mm_set1_epi8(-128);
#endif

#elif defined(USE_MMX)
		// USE_MMX を config.h では現状、有効化することがないので dead code
		constexpr IndexType kNumChunks = kHalfDimensions / kSimdWidth;
		const __m64         k0x80s     = _mm_set1_pi8(-128);

#elif defined(USE_NEON)
		constexpr IndexType kNumChunks = kHalfDimensions / (kSimdWidth / 2);
		const int8x8_t      kZero      = {0};
#endif
		const Color perspectives[2] = {pos.side_to_move(), ~pos.side_to_move()};
		for (IndexType p = 0; p < 2; ++p) {
			const IndexType offset = kHalfDimensions * p;
#if defined(USE_AVX512)
			auto out = reinterpret_cast<__m512i*>(&output[offset]);
			for (IndexType j = 0; j < kNumChunks; ++j) {
				__m512i sum0 =
				    _mm512_load_si512(&reinterpret_cast<const __m512i*>(accumulation[perspectives[p]][0])[j * 2 + 0]);
				__m512i sum1 =
				    _mm512_load_si512(&reinterpret_cast<const __m512i*>(accumulation[perspectives[p]][0])[j * 2 + 1]);
				for (IndexType i = 1; i < kRefreshTriggers.size(); ++i) {
					sum0 = _mm512_add_epi16(
					    sum0,
					    reinterpret_cast<const __m512i*>(accumulation[perspectives[p]][i])[j * 2 + 0]);
					sum1 = _mm512_add_epi16(
					    sum1,
					    reinterpret_cast<const __m512i*>(accumulation[perspectives[p]][i])[j * 2 + 1]);
				}
				_mm512_store_si512(&out[j], _mm512_permutexvar_epi64(
								 kControl, _mm512_max_epi8(_mm512_packs_epi16(sum0, sum1), kZero)));
			}

#elif defined(USE_AVX2)
			auto out = reinterpret_cast<__m256i*>(&output[offset]);
			for (IndexType j = 0; j < kNumChunks; ++j) {
					__m256i sum0 =
					    _mm256_loadu_si256(&reinterpret_cast<const __m256i*>(accumulation[perspectives[p]][0])[j * 2 + 0]);
					__m256i sum1 =
					    _mm256_loadu_si256(&reinterpret_cast<const __m256i*>(accumulation[perspectives[p]][0])[j * 2 + 1]);
					for (IndexType i = 1; i < kRefreshTriggers.size(); ++i) {
						sum0 = _mm256_add_epi16(
							sum0,
							_mm256_loadu_si256(&reinterpret_cast<const __m256i*>(accumulation[perspectives[p]][i])[j * 2 + 0]));
						sum1 = _mm256_add_epi16(
							sum1,
							_mm256_loadu_si256(&reinterpret_cast<const __m256i*>(accumulation[perspectives[p]][i])[j * 2 + 1]));
					}
					_mm256_store_si256(&out[j], _mm256_permute4x64_epi64(
									 _mm256_max_epi8(_mm256_packs_epi16(sum0, sum1), kZero), kControl));
			}

#elif defined(USE_SSE2)
			auto out = reinterpret_cast<__m128i*>(&output[offset]);
			for (IndexType j = 0; j < kNumChunks; ++j) {
				__m128i sum0 =
				    _mm_load_si128(&reinterpret_cast<const __m128i*>(accumulation[perspectives[p]][0])[j * 2 + 0]);
				__m128i sum1 =
				    _mm_load_si128(&reinterpret_cast<const __m128i*>(accumulation[perspectives[p]][0])[j * 2 + 1]);
				for (IndexType i = 1; i < kRefreshTriggers.size(); ++i) {
					sum0 = _mm_add_epi16(sum0,
					                     reinterpret_cast<const __m128i*>(accumulation[perspectives[p]][i])[j * 2 + 0]);
					sum1 = _mm_add_epi16(sum1,
					                     reinterpret_cast<const __m128i*>(accumulation[perspectives[p]][i])[j * 2 + 1]);
				}

				const __m128i packedbytes = _mm_packs_epi16(sum0, sum1);
				_mm_store_si128(&out[j],
#if defined(USE_SSE41)
				                _mm_max_epi8(packedbytes, kZero)
#else  // SSE41非対応だがSSE2は使える環境
				                _mm_subs_epi8(_mm_adds_epi8(packedbytes, k0x80s), k0x80s)
#endif
				);
			}

#elif defined(USE_MMX)
			// USE_MMX を config.h では現状、有効化することがないので dead code
			auto out = reinterpret_cast<__m64*>(&output[offset]);
			for (IndexType j = 0; j < kNumChunks; ++j) {
				__m64       sum0 = *(&reinterpret_cast<const __m64*>(accumulation[perspectives[p]][0])[j * 2 + 0]);
				__m64       sum1 = *(&reinterpret_cast<const __m64*>(accumulation[perspectives[p]][0])[j * 2 + 1]);
				const __m64 packedbytes = _mm_packs_pi16(sum0, sum1);
				out[j]                  = _mm_subs_pi8(_mm_adds_pi8(packedbytes, k0x80s), k0x80s);
			}

#elif defined(USE_NEON)
			const auto out = reinterpret_cast<int8x8_t*>(&output[offset]);
			for (IndexType j = 0; j < kNumChunks; ++j) {
				int16x8_t sum = reinterpret_cast<const int16x8_t*>(accumulation[perspectives[p]][0])[j];
				for (IndexType i = 1; i < kRefreshTriggers.size(); ++i) {
					sum = vaddq_s16(sum, reinterpret_cast<const int16x8_t*>(accumulation[perspectives[p]][i])[j]);
				}
				out[j] = vmax_s8(vqmovn_s16(sum), kZero);
			}
#else
			for (IndexType j = 0; j < kHalfDimensions; ++j) {
				BiasType sum = accumulation[perspectives[p]][0][j];
				for (IndexType i = 1; i < kRefreshTriggers.size(); ++i) {
					sum += accumulation[perspectives[p]][i][j];
				}
				output[offset + j] = static_cast<OutputType>(std::clamp<int>(sum, 0, 127));
			}
#endif
		}
#if defined(USE_MMX)
		// USE_MMX を config.h では現状、有効化することがないので dead code
		_mm_empty();
#endif
#endif
	}

#if defined(SFNNwoPSQT)
	// キャッシュ付き版: 入力特徴量を変換する (SFNNwoPSQT専用)
	void Transform(const Position& pos, OutputType* output, bool refresh, AccumulatorCaches& cache) const {
		if (refresh || !UpdateAccumulatorIfPossible(pos, cache)) {
			refresh_accumulator_with_cache(pos, cache);
		}
		const auto& accumulation = pos.state()->accumulator.accumulation;

		// USE_ELEMENT_WISE_MULTIPLY は SFNNwoPSQT で常に定義される
#if defined(VECTOR)
		constexpr IndexType OutputChunkSize = kSimdWidth;
		static_assert((kHalfDimensions / 2) % OutputChunkSize == 0);
		constexpr IndexType NumOutputChunks = kHalfDimensions / 2 / OutputChunkSize;

		vec_t Zero = vec_zero();
		const Color perspectives[2] = { pos.side_to_move(), ~pos.side_to_move() };
		for (IndexType p = 0; p < 2; ++p) {
			const IndexType offset = (kHalfDimensions / 2) * p;
			const vec_t* in0 = reinterpret_cast<const vec_t*>(&(accumulation[perspectives[p]][0][0]));
			const vec_t* in1 = reinterpret_cast<const vec_t*>(&(accumulation[perspectives[p]][0][kHalfDimensions / 2]));
			vec_t* out = reinterpret_cast<vec_t*>(output + offset);

#if defined(SFNNwoPSQT_V2)
			// V2: rshogi互換 clamp(0,127) → mullo → >>7 → packus
			vec_t One = vec_set_16(127);
			for (IndexType j = 0; j < NumOutputChunks; ++j) {
				const vec_t a0 = vec_min_16(vec_max_16(in0[j * 2 + 0], Zero), One);
				const vec_t a1 = vec_min_16(vec_max_16(in0[j * 2 + 1], Zero), One);
				const vec_t b0 = vec_min_16(vec_max_16(in1[j * 2 + 0], Zero), One);
				const vec_t b1 = vec_min_16(vec_max_16(in1[j * 2 + 1], Zero), One);
#if defined(USE_AVX512)
				const vec_t pa = _mm512_srli_epi16(_mm512_mullo_epi16(a0, b0), 7);
				const vec_t pb = _mm512_srli_epi16(_mm512_mullo_epi16(a1, b1), 7);
#elif defined(USE_AVX2)
				const vec_t pa = _mm256_srli_epi16(_mm256_mullo_epi16(a0, b0), 7);
				const vec_t pb = _mm256_srli_epi16(_mm256_mullo_epi16(a1, b1), 7);
#elif defined(USE_SSE2)
				const vec_t pa = _mm_srli_epi16(_mm_mullo_epi16(a0, b0), 7);
				const vec_t pb = _mm_srli_epi16(_mm_mullo_epi16(a1, b1), 7);
#elif defined(USE_NEON)
				const vec_t pa = vshrq_n_s16(vmulq_s16(a0, b0), 7);
				const vec_t pb = vshrq_n_s16(vmulq_s16(a1, b1), 7);
#endif
				out[j] = vec_packus_16(pa, pb);
			}
#else
			// V1: scale_weights済み mulhi 方式
			vec_t One = vec_set_16(127 * 2);
			constexpr int shift =
#if defined(USE_SSE2)
				7;
#else
				6;
#endif
			for (IndexType j = 0; j < NumOutputChunks; ++j) {
				const vec_t sum0a = vec_slli_16(vec_max_16(vec_min_16(in0[j * 2 + 0], One), Zero), shift);
				const vec_t sum0b = vec_slli_16(vec_max_16(vec_min_16(in0[j * 2 + 1], One), Zero), shift);
				const vec_t sum1a = vec_min_16(in1[j * 2 + 0], One);
				const vec_t sum1b = vec_min_16(in1[j * 2 + 1], One);
				out[j] = vec_packus_16(vec_mulhi_16(sum0a, sum1a), vec_mulhi_16(sum0b, sum1b));
			}
#endif
		}
#else
		// スカラーフォールバック
		const Color perspectives[2] = { pos.side_to_move(), ~pos.side_to_move() };
		for (IndexType p = 0; p < 2; ++p) {
			const IndexType offset = (kHalfDimensions / 2) * p;
			for (IndexType j = 0; j < kHalfDimensions / 2; ++j) {
				BiasType sum0 = accumulation[perspectives[p]][0][j];
				BiasType sum1 = accumulation[perspectives[p]][0][j + kHalfDimensions / 2];
#if defined(SFNNwoPSQT_V2)
				sum0 = std::clamp<BiasType>(sum0, 0, 127);
				sum1 = std::clamp<BiasType>(sum1, 0, 127);
				output[offset + j] = static_cast<OutputType>((unsigned(sum0) * unsigned(sum1)) >> 7);
#else
				sum0 = std::clamp<BiasType>(sum0, 0, 127 * 2);
				sum1 = std::clamp<BiasType>(sum1, 0, 127 * 2);
				output[offset + j] = static_cast<OutputType>(unsigned(sum0 * sum1) / 512);
#endif
			}
		}
#endif
	}
#endif // defined(SFNNwoPSQT) — キャッシュ付きTransform

   private:
	static void order_packs([[maybe_unused]] uint64_t* v) {
#if defined(USE_AVX512)  // _mm512_set_epi32 packs in the order [15 11 7 3 14 10 6 2 13 9 5 1 12 8 4 0]
		uint64_t tmp0 = v[4], tmp1 = v[5];
		v[4] = v[6], v[5] = v[7];
		v[6] = tmp0, v[7] = tmp1;
		tmp0 = v[8], tmp1 = v[9];
		v[8] = v[12], v[9] = v[13];
		v[12] = v[10], v[13] = v[11];
		v[10] = tmp0, v[11] = tmp1;
#elif defined(USE_AVX2)  // _mm256_set_epi32 packs in the order [7 3 6 2 5 1 4 0]
		uint64_t tmp0 = v[2], tmp1 = v[3];
		v[2] = v[4], v[3] = v[5];
		v[4] = tmp0, v[5] = tmp1;
#endif
	}

	static void inverse_order_packs([[maybe_unused]] uint64_t* v) {
#if defined(USE_AVX512)
		uint64_t tmp0 = v[2], tmp1 = v[3];
		v[2] = v[4], v[3] = v[5];
		v[4] = v[8], v[5] = v[9];
		v[8] = tmp0, v[9] = tmp1;
		tmp0 = v[6], tmp1 = v[7];
		v[6] = v[12], v[7] = v[13];
		v[12] = v[10], v[13] = v[11];
		v[10] = tmp0, v[11] = tmp1;
#elif defined(USE_AVX2)  // Inverse _mm256_packs_epi16 ordering
		uint64_t tmp0 = v[2], tmp1 = v[3];
		v[2] = v[4], v[3] = v[5];
		v[4] = tmp0, v[5] = tmp1;
#endif
	}

	void permute_weights([[maybe_unused]] void (*order_fn)(uint64_t*)) const {
#if defined(USE_AVX2)
#if defined(USE_AVX512)
		constexpr IndexType di = 16;
#else
		constexpr IndexType di = 8;
#endif
		uint64_t* b = reinterpret_cast<uint64_t*>(const_cast<BiasType*>(&biases_[0]));
		for (IndexType i = 0; i < kHalfDimensions * sizeof(BiasType) / sizeof(uint64_t); i += di)
			order_fn(&b[i]);

		for (IndexType j = 0; j < kInputDimensions; ++j)
		{
			uint64_t* w =
				reinterpret_cast<uint64_t*>(const_cast<WeightType*>(&weights_[j * kHalfDimensions]));
			for (IndexType i = 0; i < kHalfDimensions * sizeof(WeightType) / sizeof(uint64_t);
					i += di)
				order_fn(&w[i]);
		}
#endif
	}

	inline void scale_weights(bool read) const {
		for (IndexType j = 0; j < kInputDimensions; ++j)
		{
			WeightType* w = const_cast<WeightType*>(&weights_[j * kHalfDimensions]);
			for (IndexType i = 0; i < kHalfDimensions; ++i)
				w[i] = read ? w[i] * 2 : w[i] / 2;
		}

		BiasType* b = const_cast<BiasType*>(biases_);
		for (IndexType i = 0; i < kHalfDimensions; ++i)
			b[i] = read ? b[i] * 2 : b[i] / 2;
	}

	// Calculate cumulative value without using difference calculation
	// 差分計算を用いずに累積値を計算する
	void refresh_accumulator(const Position& pos) const {
		auto& accumulator = pos.state()->accumulator;
		for (IndexType i = 0; i < kRefreshTriggers.size(); ++i) {
			Features::IndexList active_indices[2];
			RawFeatures::AppendActiveIndices(pos, kRefreshTriggers[i], active_indices);
			for (Color perspective : {BLACK, WHITE}) {
#if defined(VECTOR)
				if (i == 0) {
					std::memcpy(accumulator.accumulation[perspective][i], biases_, kHalfDimensions * sizeof(BiasType));
				} else {
					std::memset(accumulator.accumulation[perspective][i], 0, kHalfDimensions * sizeof(BiasType));
				}
				for (const auto index : active_indices[perspective]) {
					const IndexType offset = kHalfDimensions * index;
					auto accumulation      = reinterpret_cast<vec_t*>(&accumulator.accumulation[perspective][i][0]);
					auto column            = reinterpret_cast<const vec_t*>(&weights_[offset]);
					constexpr IndexType kNumChunks = kHalfDimensions / (sizeof(vec_t) / sizeof(BiasType));
					for (IndexType j = 0; j < kNumChunks; ++j) {
						accumulation[j] = vec_add_16(accumulation[j], column[j]);
					}
				}
#else
				if (i == 0) {
					std::memcpy(accumulator.accumulation[perspective][i], biases_, kHalfDimensions * sizeof(BiasType));
				} else {
					std::memset(accumulator.accumulation[perspective][i], 0, kHalfDimensions * sizeof(BiasType));
				}
				for (const auto index : active_indices[perspective]) {
					const IndexType offset = kHalfDimensions * index;

					for (IndexType j = 0; j < kHalfDimensions; ++j) {
						accumulator.accumulation[perspective][i][j] += weights_[offset + j];
					}
				}
#endif
			}
		}

		accumulator.computed_accumulation = true;
		// Stockfishでは fc27d15(2020-09-07) にcomputed_scoreが排除されているので確認
		accumulator.computed_score = false;
	}

	// Calculate cumulative value using difference calculation
	// 差分計算を用いて累積値を計算する
	void update_accumulator(const Position& pos) const {
		const auto prev_accumulator = pos.state()->previous->accumulator;
		auto&      accumulator      = pos.state()->accumulator;
		for (IndexType i = 0; i < kRefreshTriggers.size(); ++i) {
			Features::IndexList removed_indices[2], added_indices[2];
			bool                reset[2];
			RawFeatures::AppendChangedIndices(pos, kRefreshTriggers[i], removed_indices, added_indices, reset);
			for (Color perspective : {BLACK, WHITE}) {
#if defined(VECTOR)
				constexpr IndexType kNumChunks = kHalfDimensions / (sizeof(vec_t) / sizeof(BiasType));
				auto accumulation              = reinterpret_cast<vec_t*>(&accumulator.accumulation[perspective][i][0]);
#endif
				if (reset[perspective]) {
					if (i == 0) {
						std::memcpy(accumulator.accumulation[perspective][i], biases_,
						            kHalfDimensions * sizeof(BiasType));
					} else {
						std::memset(accumulator.accumulation[perspective][i], 0, kHalfDimensions * sizeof(BiasType));
					}
				} else {
					// Difference calculation for the feature amount changed from 1 to 0
					// 1から0に変化した特徴量に関する差分計算
					std::memcpy(accumulator.accumulation[perspective][i], prev_accumulator.accumulation[perspective][i],
					            kHalfDimensions * sizeof(BiasType));
					for (const auto index : removed_indices[perspective]) {
						const IndexType offset = kHalfDimensions * index;
#if defined(VECTOR)
						auto column = reinterpret_cast<const vec_t*>(&weights_[offset]);
						for (IndexType j = 0; j < kNumChunks; ++j) {
							accumulation[j] = vec_sub_16(accumulation[j], column[j]);
						}
#else
						for (IndexType j = 0; j < kHalfDimensions; ++j) {
							accumulator.accumulation[perspective][i][j] -= weights_[offset + j];
						}
#endif
					}
				}
				{
					// Difference calculation for features that changed from 0 to 1
					// 0から1に変化した特徴量に関する差分計算
					for (const auto index : added_indices[perspective]) {
						const IndexType offset = kHalfDimensions * index;
#if defined(VECTOR)
						auto column = reinterpret_cast<const vec_t*>(&weights_[offset]);
						for (IndexType j = 0; j < kNumChunks; ++j) {
							accumulation[j] = vec_add_16(accumulation[j], column[j]);
						}
#else
						for (IndexType j = 0; j < kHalfDimensions; ++j) {
							accumulator.accumulation[perspective][i][j] += weights_[offset + j];
						}
#endif
					}
				}
			}
		}

		accumulator.computed_accumulation = true;
		// Stockfishでは fc27d15(2020-09-07) にcomputed_scoreが排除されているので確認
		accumulator.computed_score = false;
	}

#if defined(SFNNwoPSQT)
	// 複数手分の差分を適用してアキュムレータを更新する（祖先探索版、SFNNwoPSQT専用）
	// ancestor: 計算済みの祖先 StateInfo
	// path[0..depth-1]: ancestor+1 から now までの各 StateInfo（逆順: path[0]=now, path[depth-1]=ancestorの次）
	void update_accumulator_multi(const Position& pos, const StateInfo* ancestor,
		StateInfo* path[], int depth) const
	{
		auto& accumulator = pos.state()->accumulator;

		// 玉位置を現在の局面から取得（パス中に玉は動いていない）
		Square sq_k_black, sq_k_white;
		{
			auto* fb = pos.eval_list()->piece_list_fb();
			auto* fw = pos.eval_list()->piece_list_fw();
			sq_k_black = static_cast<Square>((fb[PIECE_NUMBER_KING + BLACK] - f_king) % SQ_NB);
			sq_k_white = static_cast<Square>((fw[PIECE_NUMBER_KING + WHITE] - f_king) % SQ_NB);
		}

		for (IndexType i = 0; i < kRefreshTriggers.size(); ++i) {
			for (Color perspective : {BLACK, WHITE}) {
				// 祖先の accumulation をコピー
				std::memcpy(accumulator.accumulation[perspective][i],
					ancestor->accumulator.accumulation[perspective][i],
					kHalfDimensions * sizeof(BiasType));

#if defined(VECTOR)
				constexpr IndexType kNumChunks = kHalfDimensions / (kSimdWidth / 2);
				auto acc = reinterpret_cast<vec_t*>(&accumulator.accumulation[perspective][i][0]);
#endif

				Square sq_k = (perspective == BLACK) ? sq_k_black : sq_k_white;

				// path は逆順（path[0]=now, path[depth-1]=ancestorの次）
				// ancestorの次から now まで順に適用
				for (int d = depth - 1; d >= 0; --d) {
					const auto& dp = path[d]->dirtyPiece;
					if (dp.dirty_num == 0) continue;

					// 各変化駒の差分を適用
					// 玉が動いていないことは呼び出し元で保証済み
					for (int k = 0; k < dp.dirty_num; ++k) {
						// removed (old)
						const auto old_p = static_cast<BonaPiece>(
							dp.changed_piece[k].old_piece.from[perspective]);
						const IndexType old_idx = Features::HalfKA_hm2<Features::Side::kFriend>::MakeIndex(sq_k, old_p);
						sub_weight(accumulator.accumulation[perspective][i], old_idx);

						// added (new)
						const auto new_p = static_cast<BonaPiece>(
							dp.changed_piece[k].new_piece.from[perspective]);
						const IndexType new_idx = Features::HalfKA_hm2<Features::Side::kFriend>::MakeIndex(sq_k, new_p);
						add_weight(accumulator.accumulation[perspective][i], new_idx);
					}
				}
			}
		}

		accumulator.computed_accumulation = true;
		accumulator.computed_score = false;
	}

	// キャッシュ付き版: 複数手分の差分を適用
	void update_accumulator_multi_with_cache(const Position& pos, const StateInfo* ancestor,
		StateInfo* path[], int depth, AccumulatorCaches& cache) const
	{
		// 祖先が1手前の場合はキャッシュ版の通常更新が使える
		if (depth == 1) {
			update_accumulator_with_cache(pos, cache);
			return;
		}
		// 複数手の場合は通常のmulti更新（キャッシュは refresh 時のみ使用）
		update_accumulator_multi(pos, ancestor, path, depth);
	}

	// キャッシュ付き版: 差分計算を用いて累積値を計算する
	// reset が必要な視点ではキャッシュ経由で refresh する
	void update_accumulator_with_cache(const Position& pos, AccumulatorCaches& cache) const {
		const auto prev_accumulator = pos.state()->previous->accumulator;
		auto&      accumulator      = pos.state()->accumulator;
		for (IndexType i = 0; i < kRefreshTriggers.size(); ++i) {
			Features::IndexList removed_indices[2], added_indices[2];
			bool                reset[2];
			RawFeatures::AppendChangedIndices(pos, kRefreshTriggers[i], removed_indices, added_indices, reset);
			for (Color perspective : {BLACK, WHITE}) {
#if defined(VECTOR)
				constexpr IndexType kNumChunks = kHalfDimensions / (kSimdWidth / 2);
				auto accumulation              = reinterpret_cast<vec_t*>(&accumulator.accumulation[perspective][i][0]);
#endif
				if (reset[perspective]) {
					// 玉が移動した → キャッシュ経由で refresh
					refresh_perspective_with_cache(pos, perspective, i,
						accumulator.accumulation[perspective][i], cache);
				} else {
					// 差分計算（通常と同じ）
					std::memcpy(accumulator.accumulation[perspective][i], prev_accumulator.accumulation[perspective][i],
					            kHalfDimensions * sizeof(BiasType));
					for (const auto index : removed_indices[perspective]) {
						const IndexType offset = kHalfDimensions * index;
#if defined(VECTOR)
						auto column = reinterpret_cast<const vec_t*>(&weights_[offset]);
						for (IndexType j = 0; j < kNumChunks; ++j) {
							accumulation[j] = vec_sub_16(accumulation[j], column[j]);
						}
#else
						for (IndexType j = 0; j < kHalfDimensions; ++j) {
							accumulator.accumulation[perspective][i][j] -= weights_[offset + j];
						}
#endif
					}
				}
				{
					if (!reset[perspective]) {
						for (const auto index : added_indices[perspective]) {
							const IndexType offset = kHalfDimensions * index;
#if defined(VECTOR)
							auto column = reinterpret_cast<const vec_t*>(&weights_[offset]);
							for (IndexType j = 0; j < kNumChunks; ++j) {
								accumulation[j] = vec_add_16(accumulation[j], column[j]);
							}
#else
							for (IndexType j = 0; j < kHalfDimensions; ++j) {
								accumulator.accumulation[perspective][i][j] += weights_[offset + j];
							}
#endif
						}
					}
				}
			}
		}

		accumulator.computed_accumulation = true;
		accumulator.computed_score = false;
	}

	// キャッシュ付き版: 差分計算を用いずに累積値を計算する（両視点）
	void refresh_accumulator_with_cache(const Position& pos, AccumulatorCaches& cache) const {
		auto& accumulator = pos.state()->accumulator;
		for (IndexType i = 0; i < kRefreshTriggers.size(); ++i) {
			for (Color perspective : {BLACK, WHITE}) {
				refresh_perspective_with_cache(pos, perspective, i,
					accumulator.accumulation[perspective][i], cache);
			}
		}

		accumulator.computed_accumulation = true;
		accumulator.computed_score = false;
	}

	// 単一視点のキャッシュ経由 refresh
	// アクティブ特徴量を取得・ソートし、キャッシュとのマージベース差分を適用する。
	void refresh_perspective_with_cache(
		const Position& pos, Color perspective, IndexType trigger_idx,
		BiasType* accumulation_out, AccumulatorCaches& cache) const
	{
		// アクティブ特徴量を取得
		Features::IndexList active_list[2];
		RawFeatures::AppendActiveIndices(pos, kRefreshTriggers[trigger_idx], active_list);

		// trigger_idx == 0 のみキャッシュ対象（SFNNwoPSQT では常に 0）
		if (trigger_idx != 0) {
			std::memset(accumulation_out, 0, kHalfDimensions * sizeof(BiasType));
			for (const auto index : active_list[perspective]) {
				const IndexType offset = kHalfDimensions * index;
				for (IndexType j = 0; j < kHalfDimensions; ++j) {
					accumulation_out[j] += weights_[offset + j];
				}
			}
			return;
		}

		// ソート済みアクティブインデックスを作成
		std::uint32_t sorted_active[kMaxActiveFeatures];
		int num_active = 0;
		for (const auto index : active_list[perspective]) {
			if (num_active < kMaxActiveFeatures)
				sorted_active[num_active++] = static_cast<std::uint32_t>(index);
		}
		std::sort(sorted_active, sorted_active + num_active);

		// 玉位置を取得（キャッシュのキーとして使用）
		// HalfKA_hm2 では perspective の玉を使う
		BonaPiece* pieces;
		Square sq_target_k;
		{
			auto* pl = (perspective == BLACK) ?
				pos.eval_list()->piece_list_fb() :
				pos.eval_list()->piece_list_fw();
			const PieceNumber target =
				static_cast<PieceNumber>(PIECE_NUMBER_KING + perspective);
			sq_target_k = static_cast<Square>((pl[target] - f_king) % SQ_NB);
			(void)pieces;
		}

		auto& entry = cache.entries[sq_target_k][perspective];

		if (entry.valid) {
			// キャッシュヒット → マージベース差分
			std::memcpy(accumulation_out, entry.accumulation, kHalfDimensions * sizeof(BiasType));
			apply_cache_diff(accumulation_out,
				entry.active_indices, entry.num_active,
				sorted_active, num_active);
		} else {
			// キャッシュミス → バイアスから full refresh
			std::memcpy(accumulation_out, biases_, kHalfDimensions * sizeof(BiasType));
			for (int k = 0; k < num_active; ++k) {
				const IndexType offset = kHalfDimensions * sorted_active[k];
#if defined(VECTOR)
				auto acc  = reinterpret_cast<vec_t*>(accumulation_out);
				auto col  = reinterpret_cast<const vec_t*>(&weights_[offset]);
				constexpr IndexType kNumChunks = kHalfDimensions / (kSimdWidth / 2);
				for (IndexType j = 0; j < kNumChunks; ++j) {
					acc[j] = vec_add_16(acc[j], col[j]);
				}
#else
				for (IndexType j = 0; j < kHalfDimensions; ++j) {
					accumulation_out[j] += weights_[offset + j];
				}
#endif
			}
		}

		// キャッシュを更新
		std::memcpy(entry.accumulation, accumulation_out, kHalfDimensions * sizeof(BiasType));
		int n = std::min(num_active, (int)kMaxActiveFeatures);
		std::memcpy(entry.active_indices, sorted_active, n * sizeof(std::uint32_t));
		entry.num_active = static_cast<std::uint16_t>(n);
		entry.valid = true;
	}

	// ソート済み配列のマージベース差分を適用（O(n+m)）
	void apply_cache_diff(
		BiasType* accumulation,
		const std::uint32_t* cached, int cached_len,
		const std::uint32_t* current, int current_len) const
	{
		int ci = 0, ni = 0;

		while (ci < cached_len && ni < current_len) {
			std::uint32_t c = cached[ci];
			std::uint32_t n = current[ni];
			if (c < n) {
				// cached にあって current にない → 重み減算
				sub_weight(accumulation, c);
				ci++;
			} else if (c > n) {
				// current にあって cached にない → 重み加算
				add_weight(accumulation, n);
				ni++;
			} else {
				// 両方にある → 変化なし
				ci++;
				ni++;
			}
		}

		while (ci < cached_len) {
			sub_weight(accumulation, cached[ci++]);
		}

		while (ni < current_len) {
			add_weight(accumulation, current[ni++]);
		}
	}

	// 1特徴量の重み加算
	void add_weight(BiasType* accumulation, std::uint32_t index) const {
		const IndexType offset = kHalfDimensions * index;
#if defined(VECTOR)
		auto acc = reinterpret_cast<vec_t*>(accumulation);
		auto col = reinterpret_cast<const vec_t*>(&weights_[offset]);
		constexpr IndexType kNumChunks = kHalfDimensions / (kSimdWidth / 2);
		for (IndexType j = 0; j < kNumChunks; ++j) {
			acc[j] = vec_add_16(acc[j], col[j]);
		}
#else
		for (IndexType j = 0; j < kHalfDimensions; ++j) {
			accumulation[j] += weights_[offset + j];
		}
#endif
	}

	// 1特徴量の重み減算
	void sub_weight(BiasType* accumulation, std::uint32_t index) const {
		const IndexType offset = kHalfDimensions * index;
#if defined(VECTOR)
		auto acc = reinterpret_cast<vec_t*>(accumulation);
		auto col = reinterpret_cast<const vec_t*>(&weights_[offset]);
		constexpr IndexType kNumChunks = kHalfDimensions / (kSimdWidth / 2);
		for (IndexType j = 0; j < kNumChunks; ++j) {
			acc[j] = vec_sub_16(acc[j], col[j]);
		}
#else
		for (IndexType j = 0; j < kHalfDimensions; ++j) {
			accumulation[j] -= weights_[offset + j];
		}
#endif
	}
#endif // defined(SFNNwoPSQT) — cache/ancestor/multi 関連メソッド群

	// parameter type
	// パラメータの型

	// parameter
	// パラメータ
	alignas(kCacheLineSize) BiasType biases_[kHalfDimensions];
	alignas(kCacheLineSize) WeightType weights_[kHalfDimensions * kInputDimensions];
};

} // namespace Eval::NNUE
} // namespace YaneuraOu

#endif  // defined(EVAL_NNUE)

#endif  // #ifndef NNUE_FEATURE_TRANSFORMER_H_INCLUDED
