# open-webui + SearXNG: iPhone 向けチャット UI

llama-swap(`deploy/llama-swap/`)を推論バックエンドにした open-webui と、ウェブ検索用の SearXNG。Tailscale 経由で iPhone から使う前提で、ホストの 127.0.0.1 にだけ公開する。経路と設定理由は `compose.yml` のコメントを参照。

初回のみ `searxng/settings.example.yml` を `searxng/settings.yml` にコピーし、`secret_key` を `openssl rand -hex 32` の値に置き換える(`settings.yml` は gitignore)。
