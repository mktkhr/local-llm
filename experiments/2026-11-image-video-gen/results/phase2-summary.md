| 構成 | 成功/失敗 | 初回(モデル読込込み) | 2 回目以降 中央値 | VRAM ピーク | 平均電力 | 電力制限 |
| --- | --- | --- | --- | --- | --- | --- |
| wan22-5b-t2v-720p | 4/0 | 285.63 s | 244.56 s | 15904 MiB | 289.3 W | 320.0 W |
| wan22-14b-t2v-fp8 | 4/0 | 743.63 s | 658.16 s | 14758 MiB | 224.9 W | 320.0 W |
| hunyuan15-t2v-720p-fp16 | 4/2 | 2708.7 s | 2689.67 s | 15910 MiB | 246.7 W | 320.0 W |
| wan22-14b-t2v-fp8-720p | 1/0 | 2447.84 s | - | 11846 MiB | 245.2 W | 250.0 W |

成功 0 件(未取得・未実行): hunyuan15-t2v-720p-sr1080p

失敗:
- hunyuan15-t2v-720p-fp16: V1-human-motion s1: TimeoutError('prompt 4f769a2e-1b14-41ec-8c6c-c5ec4180aa06 did not finish in 1800s')
- hunyuan15-t2v-720p-fp16: V2-camera-move s1: RuntimeError('{"status_str": "error", "completed": false, "messages": [["execution_start", {"prompt_id": "7854c1e1-f31c-427c-888d-abf1857df848", "timestamp": 17
