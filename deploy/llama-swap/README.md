# llama-swap + ccshim: Claude Code バックエンドの稼働構成

RTX 4080 SUPER (16 GB) 上で llama.cpp (llama-swap) を動かし、別 PC の Claude Code から `ANTHROPIC_BASE_URL` 経由で使うための構成。

**採用構成と選定理由・実測値は [experiments/2026-08-claude-code-backend](../../experiments/2026-08-claude-code-backend/README.md) を参照。** 本 README は「動かし方」だけを書く。

| エントリ                  | 用途                                                     | 根拠                                                                                                   |
| ------------------------- | -------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| `qwen36-35b-iq4xs`(B)     | 推奨。UD-IQ4_XS / ncmoe=10 / 128k / KV q8_0              | 実タスク 4/4 PASS、総所要最短                                                                          |
| `qwen36-35b-iq3s`(C)      | 初回応答優先。UD-IQ3_S / オフロードなし / 128k / KV q8_0 | prefill 3.1 倍速いが手数が増える                                                                       |
| `qwen36-35b-iq3s-256k`(D) | 長文のみ。UD-IQ3_S / オフロードなし / 256k / KV q4_0     | [256k 化の検証](../../experiments/2026-08-claude-code-backend/README.md#12-256k-化ccdの検証2026-08-22) |

## 1. 構成

```
別 PC の Claude Code
   ↓ ANTHROPIC_BASE_URL
ccshim (:11436)          ← messages 内の role:"system" をトップレベルに畳み込む
   ↓
llama-swap (:11435)      ← model 名を見てモデルを入れ替える
   ↓
llama-server             ← llama.cpp b10438 / CUDA 12.8
```

`docker compose up -d` で両方が起動する。`restart: unless-stopped`。

### ccshim が必須である理由

Claude Code は **トップレベル `system` とは別に、`messages` 配列へ `role:"system"` のメッセージを注入**する(中身は Agent タイプ一覧)。Qwen 系の Jinja チャットテンプレートはこれを

```
raise_exception('System message must be at the beginning')
```

で拒否するため、**llama-server は全リクエストに 500 を返す**。最小再現も確認済みで、`messages` に `role:"system"` が 1 つでもあれば位置に関係なく失敗する。ccshim はこれをトップレベル `system` の末尾ブロックに移すだけで、内容は失われない。

**SSE は逐次転送しなければならない。** バッファすると thinking が長いときにクライアントから無応答の接続に見え、アイドルタイムアウトで再送が多重化して破綻する(実測: thinking 26,000 トークンで 282 秒ごとに再送、待ち行列が 502→680→842→1,004 秒と膨張)。

### なぜ Docker で公開するか

ホストのプロセスとして動かすと ufw(`DEFAULT_INPUT_POLICY=DROP`)に阻まれて LAN から届かない。Docker の publish は iptables の `DOCKER` チェーンに入るため **ufw の設定変更なしで LAN に出せる**。llama-swap (:11435) が最初から届いていたのはこのため。

---

## 2. クライアント設定(別 PC)

`zshrc-snippet.sh` を `~/.zshrc` から `source` する。

```bash
ccb          # B(UD-IQ4_XS)で起動 ← 推奨
ccc          # C(UD-IQ3_S)で起動
ccd          # D(UD-IQ3_S、256k・KV q4_0)で起動。長文コンテキストが必要なときのみ([256k 化の検証](../../experiments/2026-08-claude-code-backend/README.md#12-256k-化ccdの検証2026-08-22) 参照)
ccstat       # ロード中のモデルを表示
```

手動で起動する場合:

```bash
ANTHROPIC_BASE_URL=http://192.168.0.144:11436 \
ANTHROPIC_API_KEY=dummy \
ANTHROPIC_AUTH_TOKEN=dummy \
ANTHROPIC_MODEL=qwen36-35b-iq4xs \
API_TIMEOUT_MS=1800000 \
CLAUDE_CODE_MAX_CONTEXT_TOKENS=131072 \
claude
```

| 設定                                                 | 理由                                                                                               |
| ---------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| ポート **11436**(shim)                               | 11435 に直結すると全リクエストが 500 になる                                                        |
| `ANTHROPIC_API_KEY` と `ANTHROPIC_AUTH_TOKEN` の両方 | どちらのヘッダが使われるかはバージョン依存                                                         |
| `ANTHROPIC_MODEL`                                    | 未設定だと Claude Code の既定名が送られ 404。`claude-opus-5` / `claude-sonnet-5` は alias 登録済み |
| `API_TIMEOUT_MS`                                     | 既定 600 秒。thinking が伸びると 1 リクエストが 500 秒を超える                                     |
| `CLAUDE_CODE_MAX_CONTEXT_TOKENS`                     | 未知のモデル名だと **200k と誤認**する。実際は 131,072                                             |
| 起動前のウォームアップ                               | モデル切替中に `/v1/messages` を叩くと connection reset で切られる                                 |

**claude-code-router は不要かつ有害。** llama-server が `/v1/messages` をネイティブ実装しているため変換層は要らない。router は使えるエンドポイントを探して `model` 無しの探索リクエストを撃つため、llama-swap が全部 404 を返し認証エラーとして表面化する。

---

## 3. 落とし穴

| 事象                              | 原因 / 対処                                                                                    |
| --------------------------------- | ---------------------------------------------------------------------------------------------- |
| 全リクエストが 500                | Claude Code の `messages` 内 `role:"system"`。**ccshim 必須**                                  |
| thinking が長いと再送が多重化     | シムが SSE をバッファしていた。**逐次転送にする**                                              |
| LAN から届かない                  | ufw が `DROP`。**Docker で publish する**(ufw を迂回)                                          |
| `no model id could be identified` | ボディが JSON として解釈されていない。**`Content-Type: application/json` を付ける**            |
| `no router for requested model`   | `ANTHROPIC_MODEL` 未設定、またはモデル名の誤り(`qwen3.6-` ではなく **`qwen36-`**)              |
| コンテキスト超過                  | Claude Code が 200k と誤認。**`CLAUDE_CODE_MAX_CONTEXT_TOKENS=131072`**                        |
| モデル切替直後に connection reset | ロード中の `/v1/messages` は待たずに切られる。**先に `/v1/chat/completions` でウォームアップ** |
| VRAM 測定が壊れる                 | llama-swap は `ttl` 未指定で常駐する。**計測前に停止する**                                     |
| `-ctk q4_0`                       | 公式が tool calling 品質を有意に落とすと明記。**q8_0 まで**                                    |

---

## 4. ファイル

| ファイル           | 役割                                                                   |
| ------------------ | ---------------------------------------------------------------------- |
| `compose.yml`      | llama-swap(:11435)と ccshim(B:11436 / C:11437 / D:11438)を起動         |
| `config.yaml`      | llama-swap のモデル定義。編集すると `-watch-config` で再起動なしに反映 |
| `ccshim.py`        | Claude Code の `role:"system"` をトップレベル `system` に畳み込むシム  |
| `zshrc-snippet.sh` | クライアント側の `ccb` / `ccc` / `ccd` / `ccstat`                      |
| `.env`             | `HF_TOKEN` / ポート上書き(gitignore)                                   |
