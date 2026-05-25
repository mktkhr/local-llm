# local-llm

オンプレ環境でローカルLLMを **長コンテキスト・GPUフルロード(CPUオフロード回避)** で動かすためのナレッジ集。

特定モデルの選定記録ではなく、以下を残すことを目的とします。

- どのモデルがどのGPUで速度を出せるのか
- 量子化(モデル重み・KVキャッシュ)をどう調整するか
- 計測結果のどこを見て調整するか
- 個別ハードウェアでの実測リファレンス

推論ランタイムは原則 **Ollama**(llama.cpp ベース)を前提にしています。プリビルドバイナリがCPU命令セット差・CUDAランタイム差を吸収するため、古いハードウェアでも動かしやすいことが理由です。

## ドキュメント

| ファイル                                     | 内容                                                                                   |
| -------------------------------------------- | -------------------------------------------------------------------------------------- |
| [docs/01-setup.md](docs/01-setup.md)         | 環境確認の手順、Ollama導入、systemd経由の環境変数設定                                  |
| [docs/02-benchmark.md](docs/02-benchmark.md) | 計測の原則と手順、`ollama ps` / `nvidia-smi` / API レスポンスのどこを見るか            |
| [docs/03-tuning.md](docs/03-tuning.md)       | VRAM予算の見積もり、量子化レベル選定、Flash Attention / KVキャッシュ量子化の判定フロー |
| [docs/04-results.md](docs/04-results.md)     | GPU別・モデル別の動作実績テーブル                                                      |
| [docs/05-auth-gate.md](docs/05-auth-gate.md) | Caddy + 静的Bearer による外部公開用の認証ゲート                                        |

## クイックスタート

1. [docs/01-setup.md](docs/01-setup.md) で Ollama と計測用の環境変数を整える
2. [docs/02-benchmark.md](docs/02-benchmark.md) の計測スクリプトで目当てのモデルを評価
3. CPUオフロードが発生していたら [docs/03-tuning.md](docs/03-tuning.md) の判定フローで設定を絞る
4. 安定した組み合わせを [docs/04-results.md](docs/04-results.md) に追記して資産化

## 設計方針

- **汎用ナレッジ(方法論)と固有実績(数値)を分離**: GPU依存の数値は results に集約し、setup / benchmark / tuning はハードウェア非依存に書く
- **計測の再現性を最優先**: 1モデルロード固定、並行リクエスト固定、`keep_alive` 明示制御
- **判定フローは「観測 → 調整 → 再計測」のループ**: 量子化やFAは試行錯誤前提
