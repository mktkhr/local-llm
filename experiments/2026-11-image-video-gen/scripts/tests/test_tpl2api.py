import json
import pathlib
import sys
import unittest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from comfy_api import graph_z_image_turbo  # noqa: E402
from tpl2api import apply_overrides, to_api  # noqa: E402

DATA = HERE.parent / "data"
OI = json.loads((DATA / "object_info_subset.json").read_text())


def load(name):
    return json.loads((DATA / "templates" / name).read_text())


def shape(api):
    """node id を無視して (class_type, widget 入力) の集合で比べる。"""
    return sorted((v["class_type"], tuple(sorted((k, json.dumps(x)) for k, x in v["inputs"].items() if not isinstance(x, list))))
                  for v in api.values())


class ZImageTurbo(unittest.TestCase):
    def test_matches_hand_built_graph(self):
        api = to_api(load("image_z_image_turbo.json"), OI)
        api = apply_overrides(api, prompt="P", seed=42, width=1024, height=1024)
        hand = graph_z_image_turbo("P", 42, 1024, 1024)
        self.assertEqual(shape(api), shape(hand))

    def test_links_resolve_to_real_nodes(self):
        api = to_api(load("image_z_image_turbo.json"), OI)
        for k, v in api.items():
            for name, x in v["inputs"].items():
                if isinstance(x, list):
                    self.assertIn(x[0], api, f"{k}.{name} -> {x}")
        self.assertTrue(any(v["class_type"] == "SaveImage" for v in api.values()))


class AllTemplates(unittest.TestCase):
    def test_every_template_converts(self):
        for p in sorted((DATA / "templates").glob("*.json")):
            with self.subTest(p.name):
                api = to_api(json.loads(p.read_text()), OI)
                self.assertTrue(api, p.name)
                for k, v in api.items():
                    for name, x in v["inputs"].items():
                        if isinstance(x, list):
                            self.assertIn(x[0], api, f"{p.name}: {k}.{name} -> {x}")

    def test_override_set(self):
        api = to_api(load("image_z_image_turbo.json"), OI)
        api = apply_overrides(api, sets=["KSampler.steps=4", "KSampler.cfg=1.5"])
        ks = [v for v in api.values() if v["class_type"] == "KSampler"][0]
        self.assertEqual(ks["inputs"]["steps"], 4)
        self.assertEqual(ks["inputs"]["cfg"], 1.5)
        with self.assertRaises(ValueError):
            apply_overrides(api, sets=["Nope.x=1"])


if __name__ == "__main__":
    unittest.main()


class Edges(unittest.TestCase):
    """公式テンプレートに実在する厄介なケース。"""

    def by_type(self, api, t):
        return [v for v in api.values() if v["class_type"] == t]

    def test_bypassed_node_is_passed_through(self):
        # hunyuan t2v: BasicScheduler.model -> EasyCache(bypass) -> UNETLoader
        api = to_api(load("video_hunyuan_video_1.5_720p_t2v.json"), OI)
        sched = self.by_type(api, "BasicScheduler")[0]
        self.assertEqual(api[sched["inputs"]["model"][0]]["class_type"], "UNETLoader")
        self.assertFalse(self.by_type(api, "EasyCache"))

    def test_bypassed_source_without_matching_input_drops_link(self):
        # wan2.2 5B ti2v: LoadImage(bypass) -> start_image は未接続になる
        api = to_api(load("video_wan2_2_5B_ti2v.json"), OI)
        lat = self.by_type(api, "Wan22ImageToVideoLatent")[0]
        self.assertNotIn("start_image", lat["inputs"])

    def test_subgraph_inputs_are_mapped_by_name(self):
        # hidream: サブグラフ定義の入力順 (replace, clip_name, image) とノード側の接続順 (replace, image) が違う
        api = to_api(load("image_hidream_o1_dev.json"), OI, prune=False)
        tg = self.by_type(api, "TextGenerate")[0]
        self.assertEqual(api[tg["inputs"]["image"][0]]["class_type"], "LoadImage")
        loader = [v for v in self.by_type(api, "CLIPLoader") if v["inputs"]["type"] == "stable_diffusion"][0]
        self.assertEqual(loader["inputs"]["clip_name"], "gemma4_e4b_it_fp8_scaled.safetensors")

    def test_dynamic_combo_widgets_are_flattened(self):
        api = to_api(load("image_hidream_o1_dev.json"), OI, prune=False)
        tg = self.by_type(api, "TextGenerate")[0]["inputs"]
        self.assertEqual(tg["sampling_mode"], "on")
        self.assertEqual(tg["sampling_mode.temperature"], 0.7)
        self.assertEqual(tg["sampling_mode.top_k"], 64)
        self.assertIs(tg["thinking"], False)

    def test_constant_switch_is_folded_and_unreachable_pruned(self):
        # hidream dev: 参照画像スイッチ(154)を False にすると LoadImage とプロンプト拡張(TextGenerate)が消える
        api = to_api(load("image_hidream_o1_dev.json"), OI, sets=["154:value=false"])
        types = {v["class_type"] for v in api.values()}
        self.assertNotIn("ComfySwitchNode", types)
        self.assertNotIn("LoadImage", types)
        self.assertNotIn("TextGenerate", types)
        self.assertIn("SaveImage", types)
        sampler = self.by_type(api, "SamplerCustom")[0]
        self.assertEqual(api[sampler["inputs"]["positive"][0]]["class_type"], "CLIPTextEncode")
        self.assertEqual(api[sampler["inputs"]["latent_image"][0]]["class_type"], "EmptyHiDreamO1LatentImage")

    def test_prompt_override_follows_positive_path(self):
        # hidream: 正プロンプトは CLIPTextEncode.text <- PreviewAny <- PrimitiveStringMultiline
        api = to_api(load("image_hidream_o1_dev.json"), OI, sets=["154:value=false"])
        api = apply_overrides(api, prompt="HELLO")
        prim = self.by_type(api, "PrimitiveStringMultiline")[0]
        self.assertEqual(prim["inputs"]["value"], "HELLO")
        neg = [v for v in self.by_type(api, "CLIPTextEncode") if isinstance(v["inputs"]["text"], str)]
        self.assertTrue(all(v["inputs"]["text"] == "" for v in neg), "negative を上書きしてはいけない")

    def test_qwen21_default_has_no_prompt_enhancer(self):
        api = to_api(load("image_qwen_image_2_1_t2i.json"), OI)
        types = {v["class_type"] for v in api.values()}
        self.assertNotIn("TextGenerate", types)
        self.assertIn("TextEncodeQwenImage21", types)
        api = apply_overrides(api, prompt="HELLO", seed=1)
        enc = self.by_type(api, "TextEncodeQwenImage21")[0]
        self.assertEqual(enc["inputs"]["prompt"], "HELLO")


class PromotedWidgets(unittest.TestCase):
    """サブグラフノードの widgets_values は、サブグラフ定義の inputs(widget 型のもの)の並び順。"""

    def test_qwen21_sampler_values_come_from_subgraph_node(self):
        api = to_api(load("image_qwen_image_2_1_t2i.json"), OI)
        ks = [v for v in api.values() if v["class_type"] == "KSampler"][0]["inputs"]
        self.assertEqual(ks["steps"], 25)
        self.assertEqual(ks["sampler_name"], "euler")
        self.assertEqual(ks["scheduler"], "simple")
        self.assertEqual(ks["cfg"], 1)
        self.assertEqual(ks["seed"], 447606998181262)

    def test_flux3_base_loader_names(self):
        api = to_api(load("image_flux3_klein_t2i.json"), OI)
        unet = [v for v in api.values() if v["class_type"] == "UNETLoader"][0]["inputs"]
        self.assertTrue(unet["unet_name"].startswith("flux3-image-klein-base"), unet)
        vae = [v for v in api.values() if v["class_type"] == "VAELoader"][0]["inputs"]
        self.assertEqual(vae["vae_name"], "flux2-vae.safetensors")
        ks = [v for v in api.values() if v["class_type"] == "KSampler"][0]["inputs"]
        self.assertEqual((ks["steps"], ks["cfg"]), (50, 3.5))


class GgufSwap(unittest.TestCase):
    def test_unet_loader_is_replaced_by_gguf_loader(self):
        from tpl2api import swap_unet_loader_gguf
        api = to_api(load("image_krea2_turbo_t2i.json"), OI)
        n = swap_unet_loader_gguf(api, "krea2-turbo-Q4_K_M.gguf")
        self.assertEqual(n, 1)
        loaders = [v for v in api.values() if v["class_type"] in ("UNETLoader", "UnetLoaderGGUF")]
        self.assertEqual([v["class_type"] for v in loaders], ["UnetLoaderGGUF"])
        self.assertEqual(loaders[0]["inputs"], {"unet_name": "krea2-turbo-Q4_K_M.gguf"})
        # 下流(KSampler.model 等)の参照はそのまま同じキーを指す
        key = next(k for k, v in api.items() if v["class_type"] == "UnetLoaderGGUF")
        self.assertTrue(any(x == [key, 0] for v in api.values() for x in v["inputs"].values() if isinstance(x, list)))


class SetParsing(unittest.TestCase):
    def test_string_values_starting_like_json_tokens_stay_strings(self):
        api = to_api(load("image_z_image_turbo.json"), OI,
                     sets=["UNETLoader.unet_name=flux3-image-klein-bf16.safetensors", "KSampler.steps=4", "KSampler.cfg=1.5"])
        unet = [v for v in api.values() if v["class_type"] == "UNETLoader"][0]["inputs"]
        self.assertEqual(unet["unet_name"], "flux3-image-klein-bf16.safetensors")
        ks = [v for v in api.values() if v["class_type"] == "KSampler"][0]["inputs"]
        self.assertEqual((ks["steps"], ks["cfg"]), (4, 1.5))


class EnableBypassed(unittest.TestCase):
    def test_enable_turns_bypassed_super_resolution_branch_on(self):
        tpl = load("video_hunyuan_video_1.5_720p_t2v.json")
        base = to_api(tpl, OI)
        self.assertNotIn("HunyuanVideo15SuperResolution", {v["class_type"] for v in base.values()})
        sr_ids = [n["id"] for n in tpl["nodes"] if n.get("mode") == 4 and n["type"] not in ("EasyCache", "Note")]
        api = to_api(load("video_hunyuan_video_1.5_720p_t2v.json"), OI, enable=sr_ids)
        types = {v["class_type"] for v in api.values()}
        self.assertIn("HunyuanVideo15SuperResolution", types)
        self.assertIn("HunyuanVideo15LatentUpscaleWithModel", types)
        self.assertEqual(sum(v["class_type"] == "SaveVideo" for v in api.values()), 2)
        self.assertNotIn("EasyCache", types)
        for k, v in api.items():
            for name, x in v["inputs"].items():
                if isinstance(x, list):
                    self.assertIn(x[0], api, f"{k}.{name} -> {x}")


class PreviewToSave(unittest.TestCase):
    def test_preview_image_nodes_become_save_image(self):
        from tpl2api import preview_to_save
        api = {"1": {"class_type": "PreviewImage", "inputs": {"images": ["0", 0]}},
               "2": {"class_type": "SaveImage", "inputs": {"images": ["0", 0], "filename_prefix": "x"}}}
        self.assertEqual(preview_to_save(api, "p/q"), 1)
        self.assertEqual(api["1"], {"class_type": "SaveImage", "inputs": {"images": ["0", 0], "filename_prefix": "p/q"}})
        self.assertEqual(api["2"]["inputs"]["filename_prefix"], "x")
