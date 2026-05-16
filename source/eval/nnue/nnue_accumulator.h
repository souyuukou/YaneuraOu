// NNUE評価関数の差分計算用のクラス

#ifndef CLASSIC_NNUE_ACCUMULATOR_H_
#define CLASSIC_NNUE_ACCUMULATOR_H_

#include "../../config.h"

#if defined(EVAL_NNUE)

#include "nnue_architecture.h"
#include <algorithm>
#include <cstring>

namespace YaneuraOu {
namespace Eval::NNUE {

// 入力特徴量をアフィン変換した結果を保持するクラス
// 最終的な出力である評価値も一緒に持たせておく
// AVX-512命令を使用する場合に64bytesのアライメントが要求される。
struct alignas(64) Accumulator {
  std::int16_t
      accumulation[2][kRefreshTriggers.size()][kTransformedFeatureDimensions];
  Value score = VALUE_ZERO;
  bool computed_accumulation = false;
  bool computed_score = false;
};

// ============================================================
//     AccumulatorCaches (Finny Tables)
// ============================================================

// 1視点のアクティブ特徴量の最大数。
// HalfKA_hm2は PIECE_NUMBER_NB(=40) 個の特徴量を生成する。
static constexpr int kMaxActiveFeatures = 40;

// 1エントリ = ある玉位置・ある視点でのキャッシュ
struct alignas(64) AccCacheEntry {
  // キャッシュされたアキュムレータ値 (refresh trigger index 0 のみ)
  std::int16_t accumulation[kTransformedFeatureDimensions];
  // キャッシュ時点のアクティブ特徴インデックス（ソート済み）
  std::uint32_t active_indices[kMaxActiveFeatures];
  // active_indices の有効数
  std::uint16_t num_active;
  // 有効フラグ
  bool valid;
};

// AccumulatorCaches: 81マス × 2視点 = 162 エントリ
// full refresh が必要な場合に、前回同じ玉位置で計算した
// アキュムレータとの差分のみを適用することで高速化する。
struct AccumulatorCaches {

  AccCacheEntry entries[SQ_NB][2]; // [king_sq][perspective]

  AccumulatorCaches() { invalidate(); }

  void invalidate() {
    for (auto& sq_entries : entries)
      for (auto& entry : sq_entries)
        entry.valid = false;
  }
};

} // namespace Eval::NNUE
} // namespace YaneuraOu

#endif  // defined(EVAL_NNUE)

#endif
