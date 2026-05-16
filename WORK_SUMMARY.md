# YaneuraOu AccumulatorCaches + progress8kpabs 実装サマリ

## 実装した機能

### 1. progress8kpabs バケット選択
- **ファイル**: `source/eval/nnue/evaluate_nnue.cpp`
- USIオプション `LS_BUCKET_MODE` (`kingrank9` | `progress8kpabs`) 追加
- USIオプション `LS_PROGRESS_COEFF` (progress.binパス) 追加
- `progress.bin` (f64[81][1548]) 読み込み → f32 変換
- `stack_index_for_nnue()` に progress8kpabs 分岐追加
- 閾値テーブルで sigmoid 不要のバケット計算

### 2. AccumulatorCaches (Finny Tables)
- **ファイル**: `source/eval/nnue/nnue_accumulator.h`, `source/eval/nnue/nnue_feature_transformer.h`, `source/eval/nnue/evaluate_nnue.cpp`
- `AccCacheEntry` (accumulation + ソート済みインデックス + valid) × 162エントリ (81マス × 2視点)
- `thread_local AccumulatorCaches` でスレッドローカル管理
- `refresh_perspective_with_cache()`: キャッシュヒット時はマージベース O(n) 差分
- `update_accumulator_with_cache()`: reset 視点のみキャッシュ経由 refresh
- `apply_cache_diff()`: ソート済み配列の2ポインタマージ
- `add_weight()` / `sub_weight()`: SIMD対応ヘルパー

### 3. MAX_DEPTH=4 祖先探索
- **ファイル**: `source/eval/nnue/nnue_feature_transformer.h`
- `kMaxAncestorDepth = 4` (既存は1手前のみ)
- `UpdateAccumulatorIfPossible()`: 4手前まで遡って計算済み祖先を探す
- 玉移動検出で早期打ち切り
- `update_accumulator_multi()`: 祖先から現在まで全 diff を一括適用 (HalfKA_hm2::MakeIndex使用)

### 4. 新規アーキテクチャ SFNNwoPSQT_V2 (rshogi/nnue-pytorch互換)
- **ファイル**: `source/eval/nnue/architectures/sfnnwop-1536-v2.h`, `source/Makefile`
- FT ReadParameters: 2-block LEB128読み込み、`scale_weights` なし、`permute_weights` あり
- Transform: `One = 127` (rshogi互換、max output ≈ 31)
- FV_SCALE: ヘッダ arch 文字列から自動検出 (`fv_scale=28`)
- Network層は V1 と同一

### 5. その他
- NNUEヘッダ version チェック → 警告に緩和
- EOF trailing data チェック → 警告に緩和

## 変更ファイル一覧

| ファイル | 変更内容 |
|---------|---------|
| `source/eval/nnue/evaluate_nnue.cpp` | progress8kpabs, AccumulatorCaches TLS, FV_SCALE自動検出, version/EOF緩和 |
| `source/eval/nnue/nnue_accumulator.h` | AccCacheEntry, AccumulatorCaches 構造体 |
| `source/eval/nnue/nnue_feature_transformer.h` | MAX_DEPTH祖先探索, キャッシュ付きTransform/refresh/update, V2分岐 |
| `source/eval/nnue/architectures/sfnnwop-1536-v2.h` | 新規: rshogi互換アーキテクチャ |
| `source/Makefile` | `YANEURAOU_ENGINE_NNUE_SFNNwoP1536_V2` ターゲット追加 |

## ビルド方法

```bash
cd source

# V1 (既存 bullet-shogi 形式モデル用)
make clean && make -j$(nproc) YANEURAOU_EDITION=YANEURAOU_ENGINE_NNUE_SFNNwoP1536

# V2 (rshogi/nnue-pytorch 形式モデル用)
make clean && make -j$(nproc) YANEURAOU_EDITION=YANEURAOU_ENGINE_NNUE_SFNNwoP1536_V2
```

## NPS ベンチマーク結果

v82-300 eval, bench コマンド, Threads=1, FV_SCALE=28, kingrank9 モード。
V1 ビルド (scale_weights あり = 評価値は不正だが計算パスは同一なので NPS は有効)。

| 構成 | NPS | ベースライン比 |
|------|-----|---------------|
| ベースライン (MaxDepth=1, キャッシュなし) | 484,472 | — |
| + AccumulatorCaches (MaxDepth=1) | 707,329 | **+46.0%** |
| + AccumulatorCaches + MaxDepth=4 | 743,750 | **+53.5%** |

rshogi での同等実装: +50% NPS (L1=1536)。ほぼ同等の改善。

## 正しさ検証 (V2 ビルド)

### 検証用ファイル
- eval: `/mnt/nvme1/development/bullet-shogi/checkpoints/v82/v82-300/quantised.bin`
- progress.bin: `/mnt/nvme1/development/bullet-shogi/data/progress/nodchip_progress_e1_f1_cuda.bin`
- rshogi バイナリ: `/mnt/nvme1/development/rshogi/target/release/rshogi-usi`
- eval symlink: `source/eval/nn.bin → v82-300/quantised.bin`

### 計測コマンド

#### rshogi (kingrank9, v82-300, depth 12)
```bash
{ printf "usi\nsetoption name Threads value 1\nsetoption name Hash value 256\nsetoption name EvalFile value /mnt/nvme1/development/bullet-shogi/checkpoints/v82/v82-300/quantised.bin\nsetoption name LS_BUCKET_MODE value kingrank9\nisready\nposition startpos\ngo depth 12\n"; sleep 30; printf "quit\n"; } | timeout 35 /mnt/nvme1/development/rshogi/target/release/rshogi-usi 2>&1 | grep -E "info depth 12 |^bestmove"
```

#### YaneuraOu V2 (kingrank9, v82-300, depth 12)
```bash
cd /mnt/nvme1/development/YaneuraOu/source
ln -sf /mnt/nvme1/development/bullet-shogi/checkpoints/v82/v82-300/quantised.bin eval/nn.bin
{ printf "usi\nsetoption name Threads value 1\nsetoption name USI_Hash value 256\nisready\nposition startpos\ngo depth 12\n"; sleep 30; printf "quit\n"; } | timeout 35 ./YaneuraOu-by-gcc 2>&1 | grep -E "info depth 12 |^bestmove|FV_SCALE"
```

#### YaneuraOu V2 (progress8kpabs, v82-300, depth 12)
```bash
{ printf "usi\nsetoption name Threads value 1\nsetoption name USI_Hash value 256\nsetoption name LS_BUCKET_MODE value progress8kpabs\nsetoption name LS_PROGRESS_COEFF value /mnt/nvme1/development/bullet-shogi/data/progress/nodchip_progress_e1_f1_cuda.bin\nisready\nposition startpos\ngo depth 12\n"; sleep 30; printf "quit\n"; } | timeout 35 ./YaneuraOu-by-gcc 2>&1 | grep -E "info depth 12 |^bestmove|FV_SCALE"
```

#### rshogi (progress8kpabs, v82-300, depth 12)
```bash
{ printf "usi\nsetoption name Threads value 1\nsetoption name Hash value 256\nsetoption name EvalFile value /mnt/nvme1/development/bullet-shogi/checkpoints/v82/v82-300/quantised.bin\nsetoption name LS_BUCKET_MODE value progress8kpabs\nsetoption name LS_PROGRESS_COEFF value /mnt/nvme1/development/bullet-shogi/data/progress/nodchip_progress_e1_f1_cuda.bin\nisready\nposition startpos\ngo depth 12\n"; sleep 30; printf "quit\n"; } | timeout 35 /mnt/nvme1/development/rshogi/target/release/rshogi-usi 2>&1 | grep -E "info depth 12 |^bestmove"
```

#### NPS ベンチ (V1, bench コマンド)
```bash
cd /mnt/nvme1/development/YaneuraOu/source
# FV_SCALE=28 は v82 用。別の eval なら適宜変更
{ printf "usi\nsetoption name Threads value 1\nsetoption name USI_Hash value 256\nsetoption name FV_SCALE value 28\nisready\nbench\n"; sleep 60; printf "quit\n"; } | timeout 65 ./YaneuraOu-by-gcc 2>&1 | grep -E "Nodes/second|Total time|Nodes searched"
```

#### 中盤局面での比較
```bash
SFEN="l6nl/5+P1gk/2np1S3/p1p4Pp/3P2Sp1/1PPb2P1P/P5GS1/R8/LN4bKL w RGgsn5p 1"

# YO V2
{ printf "usi\nsetoption name Threads value 1\nsetoption name USI_Hash value 256\nisready\nposition sfen $SFEN\ngo depth 10\n"; sleep 30; printf "quit\n"; } | timeout 35 ./YaneuraOu-by-gcc 2>&1 | grep -E "info depth 10 |^bestmove"

# rshogi
{ printf "usi\nsetoption name Threads value 1\nsetoption name Hash value 256\nsetoption name EvalFile value /mnt/nvme1/development/bullet-shogi/checkpoints/v82/v82-300/quantised.bin\nsetoption name LS_BUCKET_MODE value kingrank9\nisready\nposition sfen $SFEN\ngo depth 10\n"; sleep 30; printf "quit\n"; } | timeout 35 /mnt/nvme1/development/rshogi/target/release/rshogi-usi 2>&1 | grep -E "info depth 10 |^bestmove"
```

### 検証結果 (SIMD Transform 修正後)

#### 初期局面 depth 12 (kingrank9) — **完全一致**
| エンジン | score cp | bestmove | nodes |
|---------|----------|----------|-------|
| rshogi | 36 | 2g2f | 25,461 |
| YO V2 | 36 | 2g2f | 25,461 |

### レビュー指摘と修正 (2024-03-26)

1. **高: 非SFNNwoPSQTビルド破壊** → cache/ancestor コードを全て `#if defined(SFNNwoPSQT)` ガード。HalfKP ビルド確認済み。
2. **高: V2 SIMD Transform 不一致** → rshogi 式 `mullo_epi16 + srli_epi16(7)` に修正。旧コードは `mulhi` 方式で 4 倍小さい値を出していた。
3. **中: ローダ緩和** → version/hash/EOF チェックは V2 (`SFNNwoPSQT_V2`) のみ warning に緩和。非 V2 は元の error を維持。
4. **中: TLS cache invalidate** → `load_eval()` 完了時にも `InvalidateAccumulatorCaches()` を呼ぶよう修正。
