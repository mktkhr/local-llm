# GPU 電力リミットの永続化(gpu-power-limit)

RTX 4080 SUPER の電力リミットを起動時に 280 W に固定する systemd unit。`nvidia-smi -pl` は再起動やドライバの再ロードで既定値(320 W)に戻るため、oneshot サービスで毎回設定する。persistence mode も同時に有効にして、プロセスが居ないときにドライバがアンロードされて設定が消えるのを防ぐ。

320 W だとファンの音が大きい。280 W なら画像・動画生成の速度はほぼ変わらず(演算律速のステップで数 % 落ちる程度)、騒音と温度が下がる。LLM 推論の電力感度は [experiments/2026-10-power-limit](../../experiments/2026-10-power-limit/README.md) を参照。

## 導入(GPU 機で、sudo が要る)

```bash
sudo install -m 644 ~/nvidia-power-limit.service /etc/systemd/system/nvidia-power-limit.service
```

```bash
sudo systemctl daemon-reload && sudo systemctl enable --now nvidia-power-limit.service
```

確認:

```bash
nvidia-smi --query-gpu=power.limit,persistence_mode --format=csv
```

`280.00 W, Enabled` と出れば完了。値を変えるときは unit の `POWER_LIMIT_W` を書き換えて `daemon-reload` → `restart`。範囲は `nvidia-smi -q -d POWER` の Min/Max(この個体は 150〜370 W)。

## ファイル

| パス                         | 内容                                                             |
| ---------------------------- | ---------------------------------------------------------------- |
| `nvidia-power-limit.service` | oneshot unit。`-pm 1` → `-pl 280`。`/etc/systemd/system/` に置く |
