# 05. 認証ゲート(外部公開)

Ollama を外部から認証付きで呼び出せるようにするためのリバースプロキシ層。1人用前提で **Caddy + 静的トークン(`X-Api-Key` ヘッダー)**による認証を行う。

**外部到達経路はこのゲートのスコープ外**。frp / cloudflared / Tailscale Funnel / SSH ポートフォワード / ルーターでのポート開放 + TLS リバースプロキシ などから環境に合わせて任意の手段を選ぶ。本書は Caddy が `127.0.0.1:4180` で受けることだけを前提とする。

> **ヘッダー名に `X-Api-Key` を採用する理由**: 前段に nginx 等のリバースプロキシ層がある場合、`Authorization` ヘッダーが Basic Auth として解釈されて Bearer トークンが弾かれることがある(`401`/`403`)。`X-*` 系のカスタムヘッダーはそうした特別扱いを受けにくく、また Anthropic 公式 SDK の `x-api-key` とも互換性があるためクライアント側の対応もしやすい。

## 構成

```
[ クライアント (opencode / Claude Code) ]
       ↓ HTTPS + X-Api-Key: <token>
[ 外部公開・TLS終端(任意のトンネル / リバースプロキシ) ]
       ↓ HTTP
[ Caddy :4180 ] ─→ [ think-injector :11500 ] ─→ [ Ollama :11434 ]
       ↑ X-Api-Key 検証              ↑ JSON body に think:false を注入
       ↑ 不一致 → 401                ↑ Host を 127.0.0.1:11434 に書き換え
```

- Ollama は systemd で `127.0.0.1:11434` で待ち受け(変更不要)
- Caddy が `127.0.0.1:4180` で待ち受け、`X-Api-Key: <token>` を完全一致でチェック
- think-injector が `127.0.0.1:11500` で待ち受け、リクエストボディの JSON に `think: false` を注入してから Ollama に転送する(後述)
- 外部からの到達経路は別途用意する(後述のサンプル参照)

### think-injector の役割

Qwen3.5 などの thinking モデルは、デフォルトで「思考」を大量に生成するため、短い応答でも数十秒〜数分を要することがある。Ollama 0.23 系は `PARAMETER think` を Modelfile に書けず、リクエストごとに `"think": false` を渡す必要があるが、クライアント側(`opencode` の `extraBody`、Claude Code 等)から確実に乗せる手段が乏しい。そこで Caddy と Ollama の間に薄いプロキシを挟み、JSON ボディに `think:false` を機械的に注入する。

- 注入条件: `Content-Type: application/json` の POST のみ
- 上書き挙動: クライアント側が `think` を明示している場合は上書きしない
- 影響範囲: Ollama ネイティブ (`/api/generate`, `/api/chat`) / OpenAI 互換 (`/v1/chat/completions`) / Anthropic 互換 (`/v1/messages`) 等すべての JSON POST

> **落とし穴: Ollama の Origin/Host チェック**
> Ollama はデフォルトで Host が `localhost` / `127.0.0.1` / `0.0.0.0` 等のローカル系でないリクエストを **403** で弾く(CORS 保護機構)。`think-injector` は Host ヘッダーを破棄して Ollama 接続用に再生成(`127.0.0.1:11434`)するため、外部 Host を含むリクエストでも 403 にならない。

## 採用バージョン

- Caddy: `caddy:2.11.3-alpine`(Docker Official Image、パッチ完全固定)
- think-injector: `python:3.12-slim` + `aiohttp ~= 3.11`(`deploy/auth-gate/think-injector/` 配下でビルド)

## 1. トークン生成と `.env` 配置

サーバー側で 64文字のランダムトークンを生成して `.env` に書き出す。

```bash
cd deploy/auth-gate
TOKEN=$(openssl rand -hex 32)
cat > .env <<EOF
OLLAMA_API_TOKEN=$TOKEN
EOF
echo "Generated token: $TOKEN"  # クライアント側で控える
```

`.env` は `.gitignore` で除外済み。コミット禁止。`.env.example` は設定キーの参照用。

## 2. 起動

```bash
cd deploy/auth-gate
docker compose up -d --build
docker compose ps
docker compose logs --tail 30
```

期待: `ollama-think-injector` と `ollama-auth-gate`(Caddy)の両方が `Up`(injector 側は `healthy`)。初回は `think-injector` の Docker イメージビルドに1〜2分かかる。

## 3. ローカル動作確認

ホスト上で直接叩く。

```bash
TOKEN=$(grep ^OLLAMA_API_TOKEN deploy/auth-gate/.env | cut -d= -f2)

# 認証なし → 401
curl -i http://127.0.0.1:4180/api/tags

# 認証あり → 200(Ollama の /api/tags レスポンス)
curl -i -H "X-Api-Key: $TOKEN" http://127.0.0.1:4180/api/tags
```

両方期待値が返れば、認証ゲートは正常動作している。

## 4. 外部公開(任意のトンネル / リバースプロキシ)

Caddy は `127.0.0.1:4180` でしか待ち受けないので、外部からこのポートに到達できる経路を別途用意する。手段は環境に合わせて選ぶ:

- リバーストンネル(frp、cloudflared、Tailscale Funnel など)
- SSH ポートフォワーディング
- ルーター/FW でのポート開放 + TLS 終端する別レイヤー(nginx、Caddy 別インスタンス、L7 ロードバランサ等)

どの手段でも以下を満たせば良い:

1. 外部からの HTTPS を TLS 終端する
2. `127.0.0.1:4180`(Caddy)に HTTP で転送する

### サンプル(frp を使う場合)

```toml
[[proxies]]
name = "ollama"
type = "http"
localIP = "127.0.0.1"
localPort = 4180
customDomains = ["ollama.your-domain.example"]
```

## 5. 外部からの疎通確認

クライアント側マシンで実行。

```bash
TOKEN="<上で生成したトークン>"
HOST="https://ollama.your-domain.example"

# 認証なし → 401
curl -i $HOST/api/tags

# 認証あり → 200
curl -i -H "X-Api-Key: $TOKEN" $HOST/api/tags

# 生成API(Anthropic互換)
curl -i -H "X-Api-Key: $TOKEN" \
  -H "Content-Type: application/json" \
  $HOST/v1/messages \
  -d '{"model":"qwen3.5:2b","max_tokens":64,"messages":[{"role":"user","content":"hi"}]}'
```

## 6. クライアント設定例

ヘッダー名が `X-Api-Key` 固定なので、クライアント側はこのヘッダーをリクエストに乗せる手段が必要。

### opencode

`~/.config/opencode/config.json` 等にプロバイダを定義し、カスタムヘッダー設定で `X-Api-Key` を送る。

```json
{
  "provider": {
    "ollama-remote": {
      "npm": "@ai-sdk/openai-compatible",
      "options": {
        "baseURL": "https://ollama.your-domain.example/v1",
        "headers": {
          "X-Api-Key": "<TOKEN>"
        }
      },
      "models": {
        "qwen3.5:2b": {},
        "qwen3.5:4b": {}
      }
    }
  }
}
```

> SDK のバージョンによっては `apiKey` 経由で送れる先が `Authorization` ヘッダー固定の場合がある。その時は上記のように `headers` で明示的に `X-Api-Key` を指定する。

### Claude Code

```bash
export ANTHROPIC_BASE_URL=https://ollama.your-domain.example
export ANTHROPIC_API_KEY=<TOKEN>
claude --model qwen3.5:2b
```

`ANTHROPIC_API_KEY` を使うと Anthropic SDK の慣習で **`x-api-key` ヘッダーが乗る**(`ANTHROPIC_AUTH_TOKEN` は `Authorization: Bearer` 系)。Caddy 側 matcher と一致するので、こちら経由がスムーズ。

> **落とし穴: 自動リトライによるリクエスト多重化(GPUスペックに依存)**
>
> Claude Code は API レスポンスが遅い・タイムアウト判定された場合に **10回固定でリトライ**する仕様(2026年5月時点、`settings.json` や環境変数で抑制不可。参考: [Issue #23115](https://github.com/anthropics/claude-code/issues/23115))。ローカル LLM の応答時間が Claude Code のタイムアウト閾値を超えると、同じリクエストが何重にも積まれて Ollama 側で詰まる。
>
> 発生するかは **GPU スペックとモデルサイズの組み合わせ次第**:
>
> | 構成                                      | 想定応答時間(thinking 含む) | リトライ発火                                                                          |
> | ----------------------------------------- | --------------------------- | ------------------------------------------------------------------------------------- |
> | RTX 4080 SUPER (16GB) + 中型モデル        | 秒オーダー                  | しない                                                                                |
> | RTX 3090/4090 (24GB) + 7B〜14B + thinking | 秒〜10秒台                  | しない                                                                                |
> | GTX 1080 (8GB) + 2B/4B + thinking ON      | 数十秒〜分単位              | **する**(`/v1/messages` 経由は injector の `think:false` が効かず thinking ON のまま) |
>
> 当該 GPU で「Retrying in Xs · attempt X/10」が出る場合、Claude Code 経由での実用は厳しい。対処の選択肢:
>
> 1. **より高速な GPU に換装**(根本)
> 2. **Claude Code 以外のクライアントを使う**(opencode の `/api` 経路など、injector の `think:false` 注入が効くパスを使えば応答時間が大幅短縮)
> 3. **injector に Anthropic SSE 互換の `ping` keep-alive を実装**(将来課題)— Claude Code に「応答中」と認識させて タイムアウト判定を遅延させる
>
> 結論: Claude Code は応答時間が秒オーダーで返る前提で設計されている。ローカル LLM 運用では「GPU 性能 × モデルサイズ × thinking 有無」の積が Claude Code のタイムアウト内に収まるかが分かれ目になる。

### Cline(VS Code 拡張)

Cline は GUI 上のプロバイダ設定で2系統使い分けられる。**Anthropic 互換経由が最も素直**(Cline は Claude を主用途とした設計で、`x-api-key` ヘッダーを標準で送るため Caddy の matcher と一致する)。

**Anthropic 互換経由**:

| 項目         | 値                                   |
| ------------ | ------------------------------------ |
| API Provider | Anthropic                            |
| Base URL     | `https://ollama.your-domain.example` |
| API Key      | `<TOKEN>`                            |
| Model ID     | `qwen3.5:2b` または `qwen3.5:4b`     |

API Key 入力欄に直接 `<TOKEN>` を入れれば、Cline 内部の Anthropic SDK が `x-api-key` ヘッダーに乗せて送る。Ollama の `/v1/messages` を叩く。

**OpenAI 互換経由(代替手段)**:

| 項目           | 値                                                |
| -------------- | ------------------------------------------------- |
| API Provider   | OpenAI Compatible                                 |
| Base URL       | `https://ollama.your-domain.example/v1`           |
| API Key        | `<TOKEN>`(必須項目を埋めるため。実体は使われない) |
| Custom Headers | `X-Api-Key: <TOKEN>`                              |
| Model ID       | `qwen3.5:2b` または `qwen3.5:4b`                  |

OpenAI Compatible は `Authorization: Bearer <key>` を送るが、これは前段のリバースプロキシで Basic Auth として解釈されて弾かれる可能性があるため、Custom Headers で `X-Api-Key` を明示的に追加する。

### 共通: ヘッダー確認とトラブルシュート

最初の1回はクライアントが実際に送っているヘッダー名を確認すると確実:

```bash
# サーバー側で
docker compose exec caddy tail -f /var/log/caddy/access.log
# クライアント側で叩いて、Caddy のアクセスログから X-Api-Key ヘッダー有無を見る
```

もし送られているヘッダー名が `X-Api-Key` 以外(`Authorization: Bearer`、別の独自ヘッダー等)の場合、Caddyfile の matcher を該当ヘッダーに合わせて変更する。

```caddyfile
# 例: Authorization: Bearer 形式を受けたい場合(前段プロキシが Authorization を弾かない時のみ可)
@authorized header Authorization "Bearer {env.OLLAMA_API_TOKEN}"
```

> **既知の制約: thinking モードによる遅延**
> Qwen3.5 系は thinking モードがデフォルトで有効。`/api/*` パス経由(`think-injector` が `think: false` を注入できる経路)では数秒で応答するが、`/v1/chat/completions` や `/v1/messages` 経由は Ollama 側で `think` フィールドが内部に伝わらず、思考トークンを大量生成して応答が遅くなる。Cline や Claude Code は `/v1/*` パスを叩くため、現状この経路は遅さを許容する形になる。injector を OpenAI/Anthropic 互換 ↔ Ollama ネイティブのプロトコル変換まで拡張すれば解決可能(将来課題)。

## 7. 運用

### トークンローテーション

漏洩を疑った時、または定期更新時:

```bash
cd deploy/auth-gate
NEW_TOKEN=$(openssl rand -hex 32)
cat > .env <<EOF
OLLAMA_API_TOKEN=$NEW_TOKEN
EOF
docker compose up -d --force-recreate caddy
echo "New token: $NEW_TOKEN"
```

クライアント側設定も同じトークンに更新する。

### アクセスログ確認

```bash
docker compose exec caddy cat /var/log/caddy/access.log | tail -50
# または volume を直接見る
docker volume inspect auth-gate_caddy-logs
```

JSON 形式で出るので `jq` で絞り込み可能。

```bash
docker compose exec caddy cat /var/log/caddy/access.log \
  | jq 'select(.status >= 400) | {ts, request: .request.uri, status}'
```

### Caddy 設定の再読み込み(無停止)

```bash
docker compose exec caddy caddy reload --config /etc/caddy/Caddyfile
```

## 8. 制約事項とリスク

| 項目             | 内容                                                                                 |
| ---------------- | ------------------------------------------------------------------------------------ |
| 認証強度         | 静的トークン1個。漏れたら即無効化(ローテーション)                                    |
| ユーザー識別     | 不可(全クライアントが同一トークンを使う)                                             |
| ロール/権限      | 不可(全部 or なし)                                                                   |
| 監査ログ         | Caddy アクセスログのみ(Ollama 側ログと突合可能)                                      |
| トークン管理     | クライアント側 環境変数 / 設定ファイルでの平文保持。シェル履歴・スクリーンシェア注意 |
| 拡張(チーム展開) | このゲートでは難しい。OIDC ベース(Keycloak + oauth2-proxy)に移行する                 |

## 9. C構成(Keycloak+oauth2-proxy)への将来移行

- Ollama と frpc の設定は無傷で済む
- `deploy/auth-gate/` を `deploy/auth-gate-oidc/` 等に置き換え、Compose を Keycloak+oauth2-proxy 構成に作り直す
- frpc の向き先(`local_port`)を新ゲートのポートに変更
- クライアント側はトークン取得方法が `openssl rand` から `client_credentials grant` に変わる程度
