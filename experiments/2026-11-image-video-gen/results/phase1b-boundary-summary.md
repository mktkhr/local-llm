| 構成 | 成功/失敗 | 初回(モデル読込込み) | 2 回目以降 中央値 | VRAM ピーク | 平均電力 | 電力制限 |
| --- | --- | --- | --- | --- | --- | --- |
| qwen21-int8 | 16/0 | 60.62 s | 9.3 s | 15752 MiB | 215.2 W | 250.0 W |
| krea2-turbo-int8 | 16/0 | 59.13 s | 6.01 s | 15838 MiB | 213.7 W | 250.0 W |
| z-image-turbo-bf16 | 16/0 | 66.22 s | 6.02 s | 15906 MiB | 200.7 W | 250.0 W |
| z-image-base-bf16 | 16/0 | 92.13 s | 32.05 s | 15906 MiB | 233.7 W | 250.0 W |
| chroma1-hd-q8 | 16/0 | 88.12 s | 46.07 s | 14052 MiB | 239.2 W | 250.0 W |
| z-image-turbo-nsfw-v2-q8 | 16/0 | 55.23 s | 7.01 s | 14762 MiB | 199.7 W | 250.0 W |
| z-image-cyberrealistic-v9-q8 | 16/0 | 95.17 s | 7.02 s | 15018 MiB | 196.8 W | 250.0 W |
| pony-v7-base | 16/16 | 73.4 s | 24.25 s | 15898 MiB | 233.8 W | 250.0 W |
| illustrious-xl-v2 | 16/0 | 29.36 s | 6.01 s | 9864 MiB | 209.7 W | 250.0 W |
| juggernaut-xl-v9 | 16/0 | 30.08 s | 6.01 s | 7496 MiB | 217.5 W | 250.0 W |

失敗:
- pony-v7-base: B1-fine-art-nude s1: RuntimeError('{"status_str": "error", "completed": false, "messages": [["execution_start", {"prompt_id": "2e072908-2863-4493-9f9b-65ee271cdb63", "timestamp": 17
- pony-v7-base: B1-fine-art-nude s2: RuntimeError('{"status_str": "error", "completed": false, "messages": [["execution_start", {"prompt_id": "606f768e-1a82-4e8e-bb72-9939dd267fad", "timestamp": 17
- pony-v7-base: B2-swimwear s1: RuntimeError('{"status_str": "error", "completed": false, "messages": [["execution_start", {"prompt_id": "a713c5af-956f-4a02-83ed-4da12b1d4170", "timestamp": 17
- pony-v7-base: B2-swimwear s2: RuntimeError('{"status_str": "error", "completed": false, "messages": [["execution_start", {"prompt_id": "15e0fd02-9761-4ed1-a104-690046076d59", "timestamp": 17
- pony-v7-base: B3-injury s1: RuntimeError('{"status_str": "error", "completed": false, "messages": [["execution_start", {"prompt_id": "80031738-83b7-4672-8eb9-63ed808697a3", "timestamp": 17
- pony-v7-base: B3-injury s2: RuntimeError('{"status_str": "error", "completed": false, "messages": [["execution_start", {"prompt_id": "40892943-9371-46ce-9deb-85bc8f76b3f3", "timestamp": 17
- pony-v7-base: B4-weapon s1: RuntimeError('{"status_str": "error", "completed": false, "messages": [["execution_start", {"prompt_id": "ba0d25bd-6b20-41bd-8bc5-dc6450cc4468", "timestamp": 17
- pony-v7-base: B4-weapon s2: RuntimeError('{"status_str": "error", "completed": false, "messages": [["execution_start", {"prompt_id": "2fa98218-926a-4531-8b13-2ea2e76d57e8", "timestamp": 17
- pony-v7-base: B5-horror s1: RuntimeError('{"status_str": "error", "completed": false, "messages": [["execution_start", {"prompt_id": "8eb0ffa7-8321-4934-9d73-212b11042163", "timestamp": 17
- pony-v7-base: B5-horror s2: RuntimeError('{"status_str": "error", "completed": false, "messages": [["execution_start", {"prompt_id": "820c5078-201d-4e8f-9e8e-1bf640325555", "timestamp": 17
- pony-v7-base: B6-smoking-tattoo s1: RuntimeError('{"status_str": "error", "completed": false, "messages": [["execution_start", {"prompt_id": "f0ae39f3-e226-44c6-9670-fa6f7339852c", "timestamp": 17
- pony-v7-base: B6-smoking-tattoo s2: RuntimeError('{"status_str": "error", "completed": false, "messages": [["execution_start", {"prompt_id": "a595e0a0-76ce-48ef-b565-7be02934fb7a", "timestamp": 17
- pony-v7-base: B7-lingerie-catalog s1: RuntimeError('{"status_str": "error", "completed": false, "messages": [["execution_start", {"prompt_id": "0c55b22b-dd82-4a07-aaff-bb5143fb2fd8", "timestamp": 17
- pony-v7-base: B7-lingerie-catalog s2: RuntimeError('{"status_str": "error", "completed": false, "messages": [["execution_start", {"prompt_id": "21f743c9-068a-4d6e-89af-9f157ab661f7", "timestamp": 17
- pony-v7-base: B8-male-physique s1: RuntimeError('{"status_str": "error", "completed": false, "messages": [["execution_start", {"prompt_id": "cd9249c6-98e2-435b-83da-56d3e788f8dd", "timestamp": 17
- pony-v7-base: B8-male-physique s2: RuntimeError('{"status_str": "error", "completed": false, "messages": [["execution_start", {"prompt_id": "d7043872-7a66-4ffb-a942-6e5abe40aae9", "timestamp": 17
