#!/usr/bin/env python3
"""Needle In A Haystack 用の合成データを生成する。

決定論的な日本語フィラーテキストを target_chars 程度生成し、
指定位置(position-pct)に needle(秘密の数字)を埋め込み、最後に質問文を付ける。

使い方:
    python generate.py --chars 50000 --position-pct 0.5 --output 32k_p50.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

# 自然文のソースとなる短文集。トピックは雑多。
# 機械生成のテキストに「秘密の数字」が偶然含まれない範囲で意味のある日本語を使う。
FILLER_SENTENCES = [
    "東京は日本の首都であり、人口や経済の中心としての役割を持つ都市です。",
    "京都は古くから多くの寺院や神社が残る都市で、観光客が世界中から訪れます。",
    "梅雨の時期になると、湿度が高くなり室内のカビ対策が重要になります。",
    "夏季の気温上昇に伴い、エアコンの設定温度と冷却効率のバランスが課題です。",
    "コンピュータの性能は、CPU と GPU だけでなくメモリ帯域にも左右されます。",
    "プログラミング言語の選定は、用途・チーム規模・既存資産を踏まえて行います。",
    "オープンソースソフトウェアは、コミュニティによる改善とレビューが特徴です。",
    "クラウドコンピューティングは、初期投資を抑えつつスケーラビリティを得る手段です。",
    "機械学習モデルの訓練には、大量のデータと計算資源が必要になります。",
    "自然言語処理は、検索・翻訳・要約など幅広い応用が知られています。",
    "ネットワークの遅延は、地理的距離と中間ホップ数の双方に依存します。",
    "セキュリティ対策は、認証・認可・通信暗号化を多層で組み合わせるのが原則です。",
    "データベースの設計では、正規化と非正規化のバランスを考慮します。",
    "アジャイル開発では、短いサイクルで価値を届けて学習を回します。",
    "ドキュメントは、最初から完璧を目指さず、必要に応じて更新するのが現実的です。",
    "気象観測には、地上の測候所だけでなく衛星やレーダーも組み合わせて使われます。",
    "鉄道網は、路線の密度と接続性が利便性に直結する社会インフラです。",
    "電力供給は、需要と供給を常に同期させなければ周波数が乱れます。",
    "農業の生産性は、土壌・気候・品種・栽培方法の組み合わせで決まります。",
    "海洋の塩分濃度は、緯度や海流によって異なる分布を持ちます。",
    "音楽の聴覚的な印象は、メロディだけでなくリズムと音色にも依存します。",
    "歴史的な建築物の保存には、構造補強と意匠の維持を両立させる必要があります。",
    "暦の閏年判定は、4 で割れて 100 で割れず、または 400 で割れる年です。",
    "言語の文法は、語順・助詞・時制などの観点で大きく異なります。",
    "宇宙の観測は、可視光だけでなく電波・赤外線・X 線など多波長で行われます。",
]


def generate_filler(target_chars: int, seed: int) -> str:
    rng = random.Random(seed)
    out: list[str] = []
    total = 0
    para_idx = 1
    while total < target_chars:
        para_sentences = rng.sample(FILLER_SENTENCES, k=rng.randint(3, 6))
        para = f"【段落 {para_idx}】 " + "".join(para_sentences)
        out.append(para)
        total += len(para) + 2
        para_idx += 1
    return "\n\n".join(out)


def insert_needle(text: str, needle: str, position_pct: float) -> str:
    pos = max(0.0, min(1.0, position_pct))
    target = int(len(text) * pos)
    # 段落境界に揃える(段落の途中に挿入されないように)
    if target == 0:
        return needle + "\n\n" + text
    boundary = text.find("\n\n", target)
    if boundary == -1:
        return text + "\n\n" + needle
    return text[:boundary] + "\n\n" + needle + "\n\n" + text[boundary + 2 :]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chars", type=int, required=True, help="目標の文書文字数(おおよそ)")
    parser.add_argument("--position-pct", type=float, default=0.5, help="needle の挿入位置(0.0-1.0)")
    parser.add_argument("--needle-id", default=None, help="6桁の needle ID。未指定ならランダム生成")
    parser.add_argument("--seed", type=int, default=42, help="フィラー生成のシード")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rng = random.Random(args.seed + int(args.position_pct * 1000))
    needle_id = args.needle_id or f"{rng.randint(100000, 999999)}"
    needle = f"【重要メモ】秘密の数字は {needle_id} です。この数字は記憶しておいてください。"

    filler = generate_filler(args.chars, seed=args.seed)
    with_needle = insert_needle(filler, needle, args.position_pct)
    question = (
        "\n\n---\n\n"
        "上記の文書のどこかに「秘密の数字」が登場しています。"
        "その数字を答えてください。答えは数字のみ(6桁)で簡潔に。"
    )
    prompt = with_needle + question

    data = {
        "needle_id": needle_id,
        "position_pct": args.position_pct,
        "approximate_chars": args.chars,
        "actual_chars": len(prompt),
        "seed": args.seed,
        "prompt": prompt,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    print(
        f"Generated: {args.output} "
        f"(needle_id={needle_id}, pos={args.position_pct}, chars={len(prompt)})"
    )


if __name__ == "__main__":
    main()
