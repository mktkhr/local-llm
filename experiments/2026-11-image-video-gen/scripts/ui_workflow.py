#!/usr/bin/env python3
"""ComfyUI の Web UI でそのまま開けるフラットなワークフロー(UI 形式 JSON)を組み立てる。

公式テンプレートはサブグラフ入りで手直ししにくいので、比較用の構成(Z-Image Turbo 系の GGUF、Chroma、
Pony V7、SDXL 系)を 1 画面のノード列として生成し、ComfyUI の user/default/workflows に置いて使う。

  python3 ui_workflow.py --out ../config/workflows
"""

from __future__ import annotations

import argparse
import json
import pathlib

PROMPT = "Photorealistic close-up portrait of a middle-aged Japanese fisherman with weathered skin and a short gray beard, wearing an orange waterproof jacket, early morning overcast light on a pier, shallow depth of field, 85mm lens."
NEGATIVE = "text, watermark, lowres, blurry, extra fingers, deformed"


class Workflow:
    """ノードを呼んだ順に縦に並べる最小限のビルダ。"""

    def __init__(self):
        self.nodes: list[dict] = []
        self.links: list[list] = []
        self._next_node = 1
        self._next_link = 1

    def node(self, type_: str, widgets: list, inputs: list[tuple] | None = None, outputs: list[tuple] | None = None,
             title: str | None = None, pos: tuple[int, int] | None = None, size: tuple[int, int] = (320, 120)) -> int:
        """inputs: [(name, TYPE, (from_node_id, from_slot))], outputs: [(name, TYPE)]。戻りはノード id。"""
        nid = self._next_node
        self._next_node += 1
        col = (nid - 1) % 3
        row = (nid - 1) // 3
        n = {"id": nid, "type": type_, "pos": list(pos or (60 + col * 380, 60 + row * 220)), "size": list(size),
             "flags": {}, "order": nid - 1, "mode": 0, "inputs": [], "outputs": [],
             "properties": {"Node name for S&R": type_}, "widgets_values": list(widgets)}
        if title:
            n["title"] = title
        for i, (name, typ, src) in enumerate(inputs or []):
            link_id = None
            if src is not None:
                link_id = self._next_link
                self._next_link += 1
                self.links.append([link_id, src[0], src[1], nid, i, typ])
                src_node = next(x for x in self.nodes if x["id"] == src[0])
                src_node["outputs"][src[1]].setdefault("links", []).append(link_id)
            n["inputs"].append({"name": name, "type": typ, "link": link_id})
        for name, typ in outputs or []:
            n["outputs"].append({"name": name, "type": typ, "links": []})
        self.nodes.append(n)
        return nid

    def to_ui(self) -> dict:
        return {"id": "00000000-0000-4000-8000-000000000000", "revision": 0, "last_node_id": self._next_node - 1,
                "last_link_id": self._next_link - 1, "nodes": self.nodes, "links": self.links, "groups": [],
                "config": {}, "extra": {}, "version": 0.4}


def _sampler_tail(w: Workflow, model: tuple, clip: tuple, vae: tuple, prefix: str, *, steps: int, cfg: float,
                  sampler: str, scheduler: str, latent_type: str = "EmptySD3LatentImage", negative: bool = True,
                  width: int = 1024, height: int = 1024, text_encoder_in=None) -> None:
    """CLIPTextEncode ×2 → latent → KSampler → VAEDecode → SaveImage。"""
    pos = w.node("CLIPTextEncode", [PROMPT], inputs=[("clip", "CLIP", text_encoder_in or clip)],
                 outputs=[("CONDITIONING", "CONDITIONING")], title="Positive prompt", size=(420, 200))
    if negative:
        neg = w.node("CLIPTextEncode", [NEGATIVE], inputs=[("clip", "CLIP", text_encoder_in or clip)],
                     outputs=[("CONDITIONING", "CONDITIONING")], title="Negative prompt", size=(420, 140))
        neg_src = (neg, 0)
    else:
        zero = w.node("ConditioningZeroOut", [], inputs=[("conditioning", "CONDITIONING", (pos, 0))],
                      outputs=[("CONDITIONING", "CONDITIONING")], title="Negative (zero, cfg 1 なので無効)")
        neg_src = (zero, 0)
    lat = w.node(latent_type, [width, height, 1], outputs=[("LATENT", "LATENT")])
    ks = w.node("KSampler", [0, "randomize", steps, cfg, sampler, scheduler, 1.0],
                inputs=[("model", "MODEL", model), ("positive", "CONDITIONING", (pos, 0)),
                        ("negative", "CONDITIONING", neg_src), ("latent_image", "LATENT", (lat, 0))],
                outputs=[("LATENT", "LATENT")], size=(320, 262))
    dec = w.node("VAEDecode", [], inputs=[("samples", "LATENT", (ks, 0)), ("vae", "VAE", vae)], outputs=[("IMAGE", "IMAGE")])
    w.node("SaveImage", [prefix], inputs=[("images", "IMAGE", (dec, 0))], size=(400, 400))


def z_image_turbo(unet: str, gguf: bool, prefix: str) -> dict:
    """Z-Image Turbo とその GGUF ファインチューン。8 steps、cfg 1(ネガティブ無効)、shift 3。"""
    w = Workflow()
    if gguf:
        m = w.node("UnetLoaderGGUF", [unet], outputs=[("MODEL", "MODEL")], title="DiT (GGUF)")
    else:
        m = w.node("UNETLoader", [unet, "default"], outputs=[("MODEL", "MODEL")], title="DiT")
    c = w.node("CLIPLoader", ["qwen_3_4b.safetensors", "lumina2", "default"], outputs=[("CLIP", "CLIP")], title="Text encoder")
    v = w.node("VAELoader", ["ae.safetensors"], outputs=[("VAE", "VAE")])
    ms = w.node("ModelSamplingAuraFlow", [3.0], inputs=[("model", "MODEL", (m, 0))], outputs=[("MODEL", "MODEL")])
    _sampler_tail(w, (ms, 0), (c, 0), (v, 0), prefix, steps=8, cfg=1.0, sampler="res_multistep", scheduler="simple", negative=False)
    return w.to_ui()


def chroma(prefix: str) -> dict:
    """Chroma1-HD GGUF Q8。公式テンプレ相当: t5xxl、shift 1、26 steps、cfg 3.5、euler / beta。ネガティブ有効。"""
    w = Workflow()
    m = w.node("UnetLoaderGGUF", ["Chroma1-HD-Q8_0.gguf"], outputs=[("MODEL", "MODEL")], title="Chroma1-HD (GGUF)")
    c = w.node("CLIPLoader", ["t5xxl_fp8_e4m3fn_scaled.safetensors", "chroma", "default"], outputs=[("CLIP", "CLIP")], title="T5-XXL")
    v = w.node("VAELoader", ["ae.safetensors"], outputs=[("VAE", "VAE")])
    ms = w.node("ModelSamplingAuraFlow", [1.0], inputs=[("model", "MODEL", (m, 0))], outputs=[("MODEL", "MODEL")])
    tok = w.node("T5TokenizerOptions", [0, 0], inputs=[("clip", "CLIP", (c, 0))], outputs=[("CLIP", "CLIP")])
    _sampler_tail(w, (ms, 0), (c, 0), (v, 0), prefix, steps=26, cfg=3.5, sampler="euler", scheduler="beta", text_encoder_in=(tok, 0))
    return w.to_ui()


def pony_v7(prefix: str) -> dict:
    """Pony V7 base。公式配布ワークフロー相当: checkpoint(DiT のみ)+ 専用エンコーダ + SDXL VAE、20 steps、cfg 3.48。"""
    w = Workflow()
    ck = w.node("CheckpointLoaderSimple", ["pony-v7-base.safetensors"], outputs=[("MODEL", "MODEL"), ("CLIP", "CLIP"), ("VAE", "VAE")], title="Pony V7 (DiT)")
    c = w.node("CLIPLoader", ["pony_v7_text_encoder_fp16.safetensors", "qwen_image", "default"], outputs=[("CLIP", "CLIP")], title="Pony V7 text encoder")
    v = w.node("VAELoader", ["pony_v7_vae_sdxl_fp16.safetensors"], outputs=[("VAE", "VAE")], title="SDXL VAE")
    tok = w.node("T5TokenizerOptions", [768, 768], inputs=[("clip", "CLIP", (c, 0))], outputs=[("CLIP", "CLIP")])
    _sampler_tail(w, (ck, 0), (c, 0), (v, 0), prefix, steps=20, cfg=3.48, sampler="euler", scheduler="simple",
                  latent_type="EmptyLatentImage", text_encoder_in=(tok, 0))
    return w.to_ui()


def sdxl_checkpoint(ckpt: str, prefix: str, *, steps: int, cfg: float, sampler: str, scheduler: str) -> dict:
    w = Workflow()
    ck = w.node("CheckpointLoaderSimple", [ckpt], outputs=[("MODEL", "MODEL"), ("CLIP", "CLIP"), ("VAE", "VAE")])
    _sampler_tail(w, (ck, 0), (ck, 1), (ck, 2), prefix, steps=steps, cfg=cfg, sampler=sampler, scheduler=scheduler,
                  latent_type="EmptyLatentImage")
    return w.to_ui()


def build_all() -> dict[str, dict]:
    return {
        "01-z-image-turbo-official": z_image_turbo("z_image_turbo_bf16.safetensors", False, "ui/z-image-turbo"),
        "02-z-image-cyberrealistic-v9": z_image_turbo("zit_CyberRealistic_v9-Q8_0.gguf", True, "ui/z-image-cyberrealistic-v9"),
        "03-z-image-turbo-nsfw-v2": z_image_turbo("z_image_nsfw_v2-Q8_0.gguf", True, "ui/z-image-nsfw-v2"),
        "04-chroma1-hd-q8": chroma("ui/chroma1-hd"),
        "05-pony-v7-base": pony_v7("ui/pony-v7"),
        "06-illustrious-xl-v2": sdxl_checkpoint("Illustrious-XL-v2.0.safetensors", "ui/illustrious-xl-v2", steps=28, cfg=6.0, sampler="euler_ancestral", scheduler="normal"),
        "07-juggernaut-xl-v9": sdxl_checkpoint("Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors", "ui/juggernaut-xl-v9", steps=30, cfg=5.0, sampler="dpmpp_2m", scheduler="karras"),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, ui in build_all().items():
        p = args.out / f"{name}.json"
        p.write_text(json.dumps(ui, ensure_ascii=False, indent=1))
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
