// NNUE評価関数の計算に関するコード

#include "../../config.h"

#if defined(EVAL_NNUE)

#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <cstring>

#define INCBIN_SILENCE_BITCODE_WARNING
#include "../../incbin/incbin.h"

#include "../../types.h"
#include "../../evaluate.h"
#include "../../position.h"
#include "../../memory.h"
#include "../../usi.h"

#if defined(USE_EVAL_HASH)
#include "../evalhash.h"
#endif

#include "evaluate_nnue.h"

namespace YaneuraOu::Eval::NNUE {
extern int FV_SCALE;
}

// ============================================================
//         progress8kpabs バケット選択
// ============================================================

#if defined(SFNNwoPSQT)
namespace {

// バケットモード
enum class LSBucketMode { KingRank9, KingColor9, Progress8KPAbs };
LSBucketMode ls_bucket_mode = LSBucketMode::KingRank9;

// ルックアップテーブル共通構造体
// f[sq]: 先手視点正規化済み sq に対する自玉側バケット寄与値
// e[sq]: 先手視点正規化済み sq に対する敵玉側バケット寄与値
struct BucketTable { int8_t f[81]; int8_t e[81]; };

// stm(手番) と両玉位置の組み合わせから最終バケット(0..8)を直接引くテーブル。
// [stm][f_king_sq][e_king_sq]
struct alignas(64) KingPairBucketIndexTable { uint8_t v[YaneuraOu::COLOR_NB][81][81]; };

// KingRank9 テーブル（コンパイル時生成）
//   f = (r/3)*3  (r=sq%9)    → 段0-2:0 / 段3-5:3 / 段6-8:6
//   e = (8-r)/3              → 段0-2:2 / 段3-5:1 / 段6-8:0
static constexpr BucketTable kBucketKingRank9 = []() constexpr {
    BucketTable t{};
    for (int sq = 0; sq < 81; ++sq) {
        const int r = sq % 9;
        t.f[sq] = static_cast<int8_t>((r / 3) * 3);
        t.e[sq] = static_cast<int8_t>((8 - r) / 3);
    }
    return t;
}();

// KingColor9 テーブル（コンパイル時生成）
//   sq = file*9 + rank なので sq%2 = (file+rank)%2 = 市松模様色(四つ角=0=白)
//   Inv(sq)=80-sq, 80は偶数 → Inv(sq)%2 == sq%2（先後反転で色不変）
//   f = r>=6 ? 3*(c+1) : 0  → 自陣後ろ3段のみ非ゼロ
//   e = r<=2 ? (c+1)   : 0  → 相手陣後ろ3段のみ非ゼロ
static constexpr BucketTable kBucketKingColor9 = []() constexpr {
    BucketTable t{};
    for (int sq = 0; sq < 81; ++sq) {
        const int r = sq % 9;
        const int c = sq % 2;  // 0=白マス, 1=黒マス
        t.f[sq] = static_cast<int8_t>(r >= 6 ? 3 * (c + 1) : 0);
        t.e[sq] = static_cast<int8_t>(r <= 2 ?     (c + 1) : 0);
    }
    return t;
}();

static constexpr KingPairBucketIndexTable make_king_pair_bucket_index_table(const BucketTable& tbl) {
    KingPairBucketIndexTable t{};
    for (int stm = 0; stm < YaneuraOu::COLOR_NB; ++stm) {
        for (int f_king = 0; f_king < 81; ++f_king) {
            for (int e_king = 0; e_king < 81; ++e_king) {
                // 後手番では先手視点へ正規化した値で参照する。
                const int f_sq = (stm == YaneuraOu::BLACK)
                    ? f_king
                    : static_cast<int>(YaneuraOu::Inv(static_cast<YaneuraOu::Square>(f_king)));
                const int e_sq = (stm == YaneuraOu::BLACK)
                    ? e_king
                    : static_cast<int>(YaneuraOu::Inv(static_cast<YaneuraOu::Square>(e_king)));
                t.v[stm][f_king][e_king] = static_cast<uint8_t>(tbl.f[f_sq] + tbl.e[e_sq]);
            }
        }
    }
    return t;
}

// hand-crafted bucket モード用の最終バケットLUT。
static constexpr KingPairBucketIndexTable kKingPairBucketKingRank9 =
    make_king_pair_bucket_index_table(kBucketKingRank9);
static constexpr KingPairBucketIndexTable kKingPairBucketKingColor9 =
    make_king_pair_bucket_index_table(kBucketKingColor9);

// 非 progress8kpabs 時に使用する有効LUT。
// progress8kpabs で progress.bin 未ロード時は kingrank9 へフォールバックする。
const KingPairBucketIndexTable* active_king_pair_bucket_table = &kKingPairBucketKingRank9;

// progress8kpabs の重み (81 * fe_old_end floats)
// progress.bin = f64[81][fe_old_end], 読み込み時に f32 に変換
constexpr int PROGRESS_KP_ABS_NUM_WEIGHTS = 81 * YaneuraOu::Eval::fe_old_end;
float* progress_kpabs_weights = nullptr;

void set_ls_bucket_mode(const std::string& mode) {
    if (mode == "progress8kpabs") {
        ls_bucket_mode = LSBucketMode::Progress8KPAbs;
        // progress.bin 未ロード時の互換フォールバック
        active_king_pair_bucket_table = &kKingPairBucketKingRank9;
    } else if (mode == "kingcolor9") {
        ls_bucket_mode = LSBucketMode::KingColor9;
        active_king_pair_bucket_table = &kKingPairBucketKingColor9;
    } else {
        ls_bucket_mode = LSBucketMode::KingRank9;
        active_king_pair_bucket_table = &kKingPairBucketKingRank9;
    }
}

// sigmoid(x)*8 = k となる x の閾値 (k=1..7)
// x = ln(k / (8-k))
constexpr float PROGRESS_BUCKET_THRESHOLDS[7] = {
    -1.9459101f, // ln(1/7)
    -1.0986123f, // ln(2/6)
    -0.5108256f, // ln(3/5)
     0.0000000f, // ln(4/4) = 0
     0.5108256f, // ln(5/3)
     1.0986123f, // ln(6/2)
     1.9459101f, // ln(7/1)
};

// progress_sum から bucket index (0..7) を計算
inline int progress_sum_to_bucket(float sum) {
    int bucket = 0;
    for (auto t : PROGRESS_BUCKET_THRESHOLDS)
        if (sum >= t) bucket++;
    return bucket;
}

// progress8kpabs の重み付き和を全駒スキャンで計算
float compute_progress8kpabs_sum(const YaneuraOu::Position& pos) {
    using namespace YaneuraOu;
    using namespace YaneuraOu::Eval;

    const int sq_bk = pos.square<KING>(BLACK);
    const int sq_wk = Inv(pos.square<KING>(WHITE));

    float sum = 0.0f;
    const BonaPiece* fb = pos.eval_list()->piece_list_fb();
    const BonaPiece* fw = pos.eval_list()->piece_list_fw();
    for (int i = 0; i < PIECE_NUMBER_KING; ++i) {
        if (fb[i] != BONA_PIECE_ZERO && fb[i] < fe_old_end)
            sum += progress_kpabs_weights[sq_bk * fe_old_end + fb[i]];
        if (fw[i] != BONA_PIECE_ZERO && fw[i] < fe_old_end)
            sum += progress_kpabs_weights[sq_wk * fe_old_end + fw[i]];
    }
    return sum;
}

// progress8kpabs バケット計算
int compute_progress8kpabs_bucket(const YaneuraOu::Position& pos) {
    float sum = compute_progress8kpabs_sum(pos);
    return progress_sum_to_bucket(sum);
}

// progress.bin を読み込む (f64[81][fe_old_end] -> f32)
bool load_progress_bin(const std::string& path) {
    using namespace YaneuraOu;

    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) return false;

    const size_t expected_bytes = PROGRESS_KP_ABS_NUM_WEIGHTS * sizeof(double);
    // ファイルサイズチェック
    ifs.seekg(0, std::ios::end);
    auto file_size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    if (file_size != static_cast<std::streamoff>(expected_bytes)) {
        sync_cout << "info string progress.bin size mismatch: got " << file_size
                  << " bytes, expected " << expected_bytes << sync_endl;
        return false;
    }

    // 既存の重みを解放
    delete[] progress_kpabs_weights;
    progress_kpabs_weights = new float[PROGRESS_KP_ABS_NUM_WEIGHTS];

    // f64 で読み込んで f32 に変換
    for (int i = 0; i < PROGRESS_KP_ABS_NUM_WEIGHTS; ++i) {
        double val;
        ifs.read(reinterpret_cast<char*>(&val), sizeof(double));
        progress_kpabs_weights[i] = static_cast<float>(val);
    }

    if (!ifs) {
        sync_cout << "info string progress.bin read error" << sync_endl;
        delete[] progress_kpabs_weights;
        progress_kpabs_weights = nullptr;
        return false;
    }

    sync_cout << "info string loaded progress.bin: " << PROGRESS_KP_ABS_NUM_WEIGHTS
              << " weights from " << path << sync_endl;
    return true;
}

} // anonymous namespace
#endif // defined(SFNNwoPSQT)
 
// ============================================================
//              旧評価関数のためのヘルパー
// ============================================================

#if defined(USE_CLASSIC_EVAL)
using namespace YaneuraOu;
void add_options_(OptionsMap& options, ThreadPool& threads);

namespace {
YaneuraOu::OptionsMap* options_ptr;
YaneuraOu::ThreadPool* threads_ptr;
}

// 📌 旧Options、旧Threadsとの互換性のための共通のマクロ 📌
#define Options (*options_ptr)
#define Threads (*threads_ptr)

namespace YaneuraOu::Eval {
void add_options(OptionsMap& options, ThreadPool& threads) {
    options_ptr = &options;
    threads_ptr = &threads;
    add_options_(options, threads);
}
}
// ============================================================

// 評価関数を読み込み済みであるか
bool        eval_loaded   = false;
std::string last_eval_dir = "None";

// 📌 この評価関数で追加したいエンジンオプションはここで追加する。
void add_options_(OptionsMap& options, ThreadPool& threads) {

#if defined(NNUE_EMBEDDING_OFF)
    const char* default_eval_dir = "eval";
#else
	// メモリから読み込む。
    const char* default_eval_dir = "<internal>";
#endif
    Options.add("EvalDir", Option(default_eval_dir, [](const Option& o) {
                    std::string eval_dir = std::string(o);
                    if (last_eval_dir != eval_dir)
                    {
                        // 評価関数フォルダ名の変更に際して、評価関数ファイルの読み込みフラグをクリアする。
                        last_eval_dir = eval_dir;
                        eval_loaded   = false;
                    }
                    return std::nullopt;
                }));

    // NNUEのFV_SCALEの値
    Options.add("FV_SCALE", Option(16, 1, 128, [&](const Option& o) {
                    YaneuraOu::Eval::NNUE::FV_SCALE = int(o);
                    return std::nullopt;
                }));

#if defined(SFNNwoPSQT)
    // LayerStacks バケット選択モード
    Options.add("LS_BUCKET_MODE", Option("kingrank9", [](const Option& o) {
                    set_ls_bucket_mode(std::string(o));
                    return std::nullopt;
                }));

    // progress8kpabs の progress.bin パス
    Options.add("LS_PROGRESS_COEFF", Option("", [](const Option& o) {
                    std::string path = std::string(o);
                    if (!path.empty()) {
                        if (!load_progress_bin(path)) {
                            sync_cout << "info string Warning: failed to load progress.bin: " << path << sync_endl;
                        }
                    }
                    return std::nullopt;
                }));
#endif
}
#endif

// Macro to embed the default efficiently updatable neural network (NNUE) file
// data in the engine binary (using incbin.h, by Dale Weiler).
// This macro invocation will declare the following three variables
//     const unsigned char        gEmbeddedNNUEData[];  // a pointer to the embedded data
//     const unsigned char *const gEmbeddedNNUEEnd;     // a marker to the end
//     const unsigned int         gEmbeddedNNUESize;    // the size of the embedded file
// Note that this does not work in Microsoft Visual Studio.

// デフォルトの効率的に更新可能なニューラルネットワーク（NNUE）ファイルの
// データをエンジンのバイナリに埋め込むためのマクロ
// （Dale Weiler 氏の incbin.h を使用）。
// このマクロを使うことで、以下の3つの変数が宣言されます：
//     const unsigned char        gEmbeddedNNUEData[];  // 埋め込まれたデータへのポインタ
//     const unsigned char *const gEmbeddedNNUEEnd;     // データの終端を示すマーカー
//     const unsigned int         gEmbeddedNNUESize;    // 埋め込まれたファイルのサイズ
// なお、この方法は Microsoft Visual Studio では動作しません。

#if !defined(_MSC_VER) && !defined(NNUE_EMBEDDING_OFF)
INCBIN(EmbeddedNNUE, EvalFileDefaultName);
#else
const unsigned char        gEmbeddedNNUEData[1] = { 0x0 };
const unsigned char* const gEmbeddedNNUEEnd = &gEmbeddedNNUEData[1];
const unsigned int         gEmbeddedNNUESize = 1;
#endif

// NNUEの埋め込みデータ型

namespace {

	struct EmbeddedNNUE {
		EmbeddedNNUE(const unsigned char* embeddedData,
			const unsigned char* embeddedEnd,
			const unsigned int   embeddedSize) :
			data(embeddedData),
			end(embeddedEnd),
			size(embeddedSize) {
		}
		const unsigned char* data;
		const unsigned char* end;
		const unsigned int   size;
	};

	//EmbeddedNNUE get_embedded(EmbeddedNNUEType type) {
	//	if (type == EmbeddedNNUEType::BIG)
	//		return EmbeddedNNUE(gEmbeddedNNUEBigData, gEmbeddedNNUEBigEnd, gEmbeddedNNUEBigSize);
	//	else
	//		return EmbeddedNNUE(gEmbeddedNNUESmallData, gEmbeddedNNUESmallEnd, gEmbeddedNNUESmallSize);
	//}

	// ⇨  StockfishはNNUEとして大きなnetworkと小さなnetworkがある。

	EmbeddedNNUE get_embedded() {
		return EmbeddedNNUE(gEmbeddedNNUEData, gEmbeddedNNUEEnd, gEmbeddedNNUESize);
	}
}


namespace YaneuraOu {
namespace Eval {
namespace NNUE {

	int FV_SCALE = 16; // 水匠5では24がベストらしいのでエンジンオプション"FV_SCALE"で変更可能にした。

    // NNUE評価関数パラメーター（共有メモリまたはローカルメモリ上に配置）
    SystemWideSharedConstant<NnueNetworks> shared_networks;

    // 評価関数ファイル名
    const char* const kFileName = EvalFileDefaultName;

    // 評価関数の構造を表す文字列を取得する
    std::string GetArchitectureString() {
        const std::string base = "Features=" + FeatureTransformer::GetStructureString() +
			",Network=" + Network::GetStructureString();
#if defined(SFNNwoPSQT)
		return "ModelType=SFNNWithoutPsqt;" + base + "{LayerStack=" + std::to_string(kLayerStacks) + "}";
#else
		return base;
#endif
    }

namespace {
	namespace Detail {

		// 評価関数パラメータを読み込む（参照版）
		template <typename T>
		Tools::Result ReadParameters(std::istream& stream, T& obj) {
			std::uint32_t header;
			stream.read(reinterpret_cast<char*>(&header), sizeof(header));
			if (!stream) return Tools::ResultCode::FileReadError;
			// hash値、古い評価関数ファイルに対して一致するとは限らないので、警告に変更する。
			if (header != T::GetHashValue())
				sync_cout << "info string Warning : nn.bin hash mismatch." << sync_endl;
			return obj.ReadParameters(stream);
		}

		// 評価関数パラメータを書き込む（参照版）
		template <typename T>
		bool WriteParameters(std::ostream& stream, const T& obj) {
			constexpr std::uint32_t header = T::GetHashValue();
			stream.write(reinterpret_cast<const char*>(&header), sizeof(header));
			return obj.WriteParameters(stream);
		}

	}  // namespace Detail

	// テンポラリにパラメータを読み込み、共有メモリに配置する。
	// 同じパラメータを持つ他プロセスが既に共有メモリを作成済みなら、そちらを参照する。
	Tools::Result LoadAndShare(std::istream& stream) {
		// テンポラリ領域にパラメータを読み込む
		auto tmp = make_unique_large_page<NnueNetworks>();

		std::uint32_t hash_value;
		std::string architecture;
		std::uint32_t version = 0;
		Tools::Result result = ReadHeader(stream, &hash_value, &architecture, &version);
		if (result.is_not_ok()) return result;
		if (hash_value != kHashValue) {
			sync_cout << "info string Warning: NNUE hash mismatch: expected " << kHashValue
				<< " got " << hash_value
				<< " arch_in_file=" << architecture
				<< " arch_expected=" << GetArchitectureString()
				<< sync_endl;
		}

		result = Detail::ReadParameters<FeatureTransformer>(stream, tmp->feature_transformer);
		if (result.is_not_ok()) {
			sync_cout << "info string NNUE feature params read failed: " << result.to_string() << sync_endl;
			return result;
		}
		for (int i = 0; i < kLayerStacks; ++i) {
			result = Detail::ReadParameters<Network>(stream, tmp->network[i]);
			if (result.is_not_ok()) {
				sync_cout << "info string NNUE network params read failed at stack " << i << ": " << result.to_string() << sync_endl;
				return result;
			}
		}

#if defined(SFNNwoPSQT_V2)
		if (stream && stream.peek() != std::ios::traits_type::eof())
			sync_cout << "info string Warning: NNUE file has trailing data (ignored)" << sync_endl;
#else
		if (!stream || stream.peek() != std::ios::traits_type::eof())
			return Tools::ResultCode::FileCloseError;
#endif

		// 共有メモリに配置（同一ハッシュの共有メモリが既に存在すればそちらを参照）
		shared_networks = SystemWideSharedConstant<NnueNetworks>(*tmp);

		return Tools::ResultCode::Ok;
	}

	}  // namespace
    // ヘッダを読み込む
    Tools::Result ReadHeader(std::istream& stream,
        std::uint32_t* hash_value, std::string* architecture, std::uint32_t* version_out) {
        std::uint32_t version = 0, size = 0;
        stream.read(reinterpret_cast<char*>(&version), sizeof(version));
        stream.read(reinterpret_cast<char*>(hash_value), sizeof(*hash_value));
        stream.read(reinterpret_cast<char*>(&size), sizeof(size));
		if (!stream) return Tools::ResultCode::FileReadError;
		if (version_out)
			*version_out = version;
#if defined(SFNNwoPSQT_V2)
        if (version != kVersion) {
			sync_cout << "info string Warning: NNUE header version mismatch: expected " << kVersion
				<< " got " << version << " (continuing anyway)" << sync_endl;
		}
#else
        if (version != kVersion) {
			sync_cout << "info string NNUE header version mismatch: expected " << kVersion
				<< " got " << version << sync_endl;
			return Tools::ResultCode::FileMismatch;
		}
#endif
        architecture->resize(size);
        stream.read(&(*architecture)[0], size);

        // 学習側でファイルヘッダーにバケット数(4バイト)を書き込むようになったため、
        // アーキテクチャ文字列の後に4バイトのバケット数がある場合がある。
        // 次の4バイトをピークし、バケット数(小整数)であれば読み飛ばす。
        {
            std::streampos pos = stream.tellg();
            std::uint32_t peek_val = 0;
            stream.read(reinterpret_cast<char*>(&peek_val), sizeof(peek_val));
            if (stream) {
                // バケット数は kLayerStacks(1〜16 程度) の小整数。
                // FeatureTransformer ハッシュは通常 0x7Fxxxxxx 等の大きい値。
                // peek_val が kLayerStacks 以下であればバケット数と判断する。
                if (peek_val <= static_cast<std::uint32_t>(kLayerStacks)) {
                    // バケット数を読み飛ばした (seekg は既に進んでいるので何もしない)
                } else {
                    // バケット数ではないので戻す
                    stream.clear();
                    stream.seekg(pos);
                }
            }
        }

		return !stream.fail() ? Tools::ResultCode::Ok : Tools::ResultCode::FileReadError;
    }

    // ヘッダを書き込む
    bool WriteHeader(std::ostream& stream,
        std::uint32_t hash_value, const std::string& architecture) {
        stream.write(reinterpret_cast<const char*>(&kVersion), sizeof(kVersion));
        stream.write(reinterpret_cast<const char*>(&hash_value), sizeof(hash_value));
        const std::uint32_t size = static_cast<std::uint32_t>(architecture.size());
        stream.write(reinterpret_cast<const char*>(&size), sizeof(size));
        stream.write(architecture.data(), size);
        return !stream.fail();
    }

    	// 評価関数パラメータを読み込む
    	Tools::Result ReadParameters(std::istream& stream) {
    		return LoadAndShare(stream);
    	}
    // 評価関数パラメータを書き込む
    bool WriteParameters(std::ostream& stream) {
        if (!WriteHeader(stream, kHashValue, GetArchitectureString())) return false;
        if (!Detail::WriteParameters<FeatureTransformer>(stream, networks().feature_transformer)) return false;
        for (int i = 0; i < kLayerStacks; ++i) {
            if (!Detail::WriteParameters<Network>(stream, networks().network[i])) return false;
        }
        return !stream.fail();
    }

    // 差分計算ができるなら進める
    static void UpdateAccumulatorIfPossible(const Position& pos) {
        networks().feature_transformer.UpdateAccumulatorIfPossible(pos);
    }

#if defined(SFNNwoPSQT)
    // スレッドローカルな AccumulatorCaches
    static thread_local AccumulatorCaches tls_acc_cache;

    // キャッシュ付き版: 差分計算ができるなら進める
    static void UpdateAccumulatorIfPossibleWithCache(const Position& pos) {
        networks().feature_transformer.UpdateAccumulatorIfPossible(pos, tls_acc_cache);
    }

    // AccumulatorCaches を無効化する（新しい局面が設定されたときに呼ぶ）
    static void InvalidateAccumulatorCaches() {
        tls_acc_cache.invalidate();
    }
    // レイヤースタックの選択。
    // kingrank9   : 双方の玉の段に応じて9通りに分岐させる。
    // kingcolor9  : 自陣後ろ3段か否か × 玉のマス色(市松模様)で9通りに分岐させる。
    // progress8kpabs: KP-absolute 進行度に応じて8通りに分岐させる。
    //
    // kingrank9 / kingcolor9 はルックアップテーブルで高速化。
    // 後手番のとき両玉を Inv(sq)=80-sq で先手視点に正規化してテーブルを共用する。
    static int stack_index_for_nnue(const Position& pos) {
        if (ls_bucket_mode == LSBucketMode::Progress8KPAbs && progress_kpabs_weights != nullptr) {
            int bucket = compute_progress8kpabs_bucket(pos);
            // progress8kpabs は 0..7 の 8バケット (LayerStacks=9 のうち 0..7 を使用)
            if (bucket < 0) bucket = 0;
            if (bucket >= kLayerStacks) bucket = kLayerStacks - 1;
            return bucket;
        }

        const auto stm = pos.side_to_move();
        const int stm_i = static_cast<int>(stm);
        const int f_king = static_cast<int>(pos.square<KING>(stm));
        const int e_king = static_cast<int>(pos.square<KING>(~stm));
        return static_cast<int>(active_king_pair_bucket_table->v[stm_i][f_king][e_king]);
    }
#endif

    // 評価値を計算する
    static Value ComputeScore(const Position& pos, bool refresh = false) {
        auto& accumulator = pos.state()->accumulator;
        if (!refresh && accumulator.computed_score) {
            return accumulator.score;
        }

        alignas(kCacheLineSize) TransformedFeatureType
            transformed_features[FeatureTransformer::kBufferSize];
#if defined(SFNNwoPSQT)
        networks().feature_transformer.Transform(pos, transformed_features, refresh, tls_acc_cache);
#else
        networks().feature_transformer.Transform(pos, transformed_features, refresh);
#endif
        alignas(kCacheLineSize) char buffer[Network::kBufferSize];
#if defined(SFNNwoPSQT)
        const auto bucket = stack_index_for_nnue(pos);
        const auto output = networks().network[bucket].Propagate(transformed_features, buffer);
#else
        const auto output = networks().network[0].Propagate(transformed_features, buffer);
#endif

        // VALUE_MAX_EVALより大きな値が返ってくるとaspiration searchがfail highして
        // 探索が終わらなくなるのでVALUE_MAX_EVAL以下であることを保証すべき。

        // この現象が起きても、対局時に秒固定などだとそこで探索が打ち切られるので、
        // 1つ前のiterationのときの最善手がbestmoveとして指されるので見かけ上、
        // 問題ない。このVALUE_MAX_EVALが返ってくるような状況は、ほぼ詰みの局面であり、
        // そのような詰みの局面が出現するのは終盤で形勢に大差がついていることが多いので
        // 勝敗にはあまり影響しない。

        // しかし、教師生成時などdepth固定で探索するときに探索から戻ってこなくなるので
        // そのスレッドの計算時間を無駄にする。またdepth固定対局でtime-outするようになる。

        auto score = static_cast<Value>(output[0] / FV_SCALE);

        // 1) ここ、下手にclipすると学習時には影響があるような気もするが…。
        // 2) accumulator.scoreは、差分計算の時に用いないので書き換えて問題ない。
        score = Math::clamp(score, -VALUE_MAX_EVAL, VALUE_MAX_EVAL);

        accumulator.score = score;
        accumulator.computed_score = true;
        return accumulator.score;
    }

}  // namespace NNUE

#if defined(USE_EVAL_HASH)

// HashTableに評価値を保存するために利用するクラス
struct alignas(16) ScoreKeyValue {
#if defined(USE_SSE2)
    ScoreKeyValue() = default;
    ScoreKeyValue(const ScoreKeyValue & other) {
        static_assert(sizeof(ScoreKeyValue) == sizeof(__m128i),
            "sizeof(ScoreKeyValue) should be equal to sizeof(__m128i)");
        _mm_store_si128(&as_m128i, other.as_m128i);
    }
    ScoreKeyValue& operator=(const ScoreKeyValue & other) {
        _mm_store_si128(&as_m128i, other.as_m128i);
        return *this;
    }
#endif

    // evaluate hashでatomicに操作できる必要があるのでそのための操作子
    void encode() {
#if defined(USE_SSE2)
        // ScoreKeyValue は atomic にコピーされるので key が合っていればデータも合っている。
#else
        key ^= score;
#endif
    }
    // decode()はencode()の逆変換だが、xorなので逆変換も同じ変換。
    void decode() { encode(); }

    union {
        struct {
            std::uint64_t key;
            std::uint64_t score;
        };
#if defined(USE_SSE2)
        __m128i as_m128i;
#endif
    };
};

// evaluateしたものを保存しておくHashTable(俗にいうehash)

struct EvaluateHashTable : HashTable<ScoreKeyValue> {};

EvaluateHashTable g_evalTable;
void EvalHash_Resize(size_t mbSize) { g_evalTable.resize(mbSize); }
void EvalHash_Clear() { g_evalTable.clear(); };

// prefetchする関数も用意しておく。
void prefetch_evalhash(const Key key) {
    constexpr auto mask = ~((u64)0x1f);
    prefetch((void*)((u64)g_evalTable[key] & mask));
}
#endif

// 評価関数ファイルを読み込む
void load_eval() {
    // 評価関数パラメーターを読み込み済みであるなら帰る。
    if (eval_loaded)
        return;

    {
        const std::string dir_name = Options["EvalDir"];
    #if !defined(__EMSCRIPTEN__)
		const std::string file_name = NNUE::kFileName;
#else
		// WASM
        const std::string file_name = Options["EvalFile"];
    #endif
        const Tools::Result result = [&] {
            if (dir_name != "<internal>") {
                auto abs_eval_path = Path::Combine(Directory::GetBinaryFolder(), dir_name);
                const std::string file_path = Path::Combine(abs_eval_path, file_name);
                std::ifstream stream(file_path, std::ios::binary);
                sync_cout << "info string loading eval file : " << file_path << sync_endl;
				if (!stream.is_open())
					return Tools::Result(Tools::ResultCode::FileNotFound);

                return NNUE::ReadParameters(stream);
            }
            else {
                // C++ way to prepare a buffer for a memory stream
                class MemoryBuffer : public std::basic_streambuf<char> {
                    public: MemoryBuffer(char* p, size_t n) {
                        std::streambuf::setg(p, p, p + n);
                        std::streambuf::setp(p, p + n);
                    }
                };

			    const auto embedded = get_embedded(/* embeddedType */);

                MemoryBuffer buffer(
                              const_cast<char*>(reinterpret_cast<const char*>(embedded.data)),
                              size_t(embedded.size));

                std::istream stream(&buffer);
                sync_cout << "info string loading eval file : <internal>" << sync_endl;

                return NNUE::ReadParameters(stream);
            }
        }();

        //      ASSERT(result);

        if (result.is_not_ok())
        {
            // 読み込みエラーのとき終了してくれないと困る。
            sync_cout << "Error! : failed to read " << file_name << " : " << result.to_string() << sync_endl;
            Tools::exit();
        }

		// 評価関数ファイルの読み込みが完了した。
		eval_loaded = true;

#if defined(SFNNwoPSQT)
		// eval ロード時にキャッシュを無効化（重みが変わったため）
		NNUE::InvalidateAccumulatorCaches();
#endif
    }
}


// 評価関数。差分計算ではなく全計算する。
// Position::set()で一度だけ呼び出される。(以降は差分計算)
// 手番側から見た評価値を返すので注意。(他の評価関数とは設計がこの点において異なる)
// なので、この関数の最適化は頑張らない。
Value compute_eval(const Position& pos) {
#if defined(SFNNwoPSQT)
    // 新しい局面が設定されたのでキャッシュを無効化
    NNUE::InvalidateAccumulatorCaches();
#endif
    return NNUE::ComputeScore(pos, true);
}

// 評価関数
Value evaluate(const Position& pos) {
    const auto& accumulator = pos.state()->accumulator;
    if (accumulator.computed_score) {
        return accumulator.score;
    }

#if defined(USE_GLOBAL_OPTIONS)
    // GlobalOptionsでeval hashを用いない設定になっているなら
    // eval hashへの照会をskipする。
    if (!GlobalOptions.use_eval_hash) {
        ASSERT_LV5(pos.state()->materialValue == Eval::material(pos));
        return NNUE::ComputeScore(pos);
    }
#endif

#if defined(USE_EVAL_HASH)
    // evaluate hash tableにはあるかも。
    const Key key = pos.state()->key();
    ScoreKeyValue entry = *g_evalTable[key];
    entry.decode();
    if (entry.key == key) {
        // あった！
        return Value(entry.score);
    }
#endif

    Value score = NNUE::ComputeScore(pos);
#if defined(USE_EVAL_HASH)
    // せっかく計算したのでevaluate hash tableに保存しておく。
    entry.key = key;
    entry.score = score;
    entry.encode();
    *g_evalTable[key] = entry;
#endif

    return score;
}

// 差分計算ができるなら進める
void evaluate_with_no_return(const Position& pos) {
#if defined(SFNNwoPSQT)
    NNUE::UpdateAccumulatorIfPossibleWithCache(pos);
#else
    NNUE::UpdateAccumulatorIfPossible(pos);
#endif
}

// 現在の局面の評価値の内訳を表示する
void print_eval_stat(Position& /*pos*/) {
    std::cout << "--- EVAL STAT: not implemented" << std::endl;
}

} // namespace Eval
} // namespace YaneuraOu

#endif  // defined(EVAL_NNUE)
