import json
import pathlib
import sys
import unittest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from tpl2api import apply_overrides, to_api  # noqa: E402
from ui_workflow import Workflow, build_all  # noqa: E402

OI = json.loads((HERE.parent / "data" / "object_info_subset.json").read_text())


class Builder(unittest.TestCase):
    def test_nodes_links_and_widgets_round_trip_through_tpl2api(self):
        w = Workflow()
        u = w.node("UnetLoaderGGUF", ["m.gguf"], outputs=[("MODEL", "MODEL")])
        c = w.node("CLIPLoader", ["enc.safetensors", "lumina2", "default"], outputs=[("CLIP", "CLIP")])
        t = w.node("CLIPTextEncode", ["hello"], inputs=[("clip", "CLIP", (c, 0))], outputs=[("CONDITIONING", "CONDITIONING")])
        ui = w.to_ui()
        self.assertEqual({n["type"] for n in ui["nodes"]}, {"UnetLoaderGGUF", "CLIPLoader", "CLIPTextEncode"})
        self.assertEqual(len(ui["links"]), 1)
        api = to_api(ui, OI, prune=False)
        self.assertEqual(api[str(t)]["inputs"], {"text": "hello", "clip": [str(c), 0]})
        self.assertEqual(api[str(u)]["inputs"], {"unet_name": "m.gguf"})

    def test_build_all_produces_runnable_graphs(self):
        flows = build_all()
        self.assertGreaterEqual(len(flows), 7)
        for name, ui in flows.items():
            with self.subTest(name):
                api = to_api(ui, OI)
                api = apply_overrides(api, prompt="P", seed=1)
                types = {v["class_type"] for v in api.values()}
                self.assertIn("SaveImage", types, name)
                self.assertTrue(types & {"KSampler", "SamplerCustomAdvanced"}, name)
                for k, v in api.items():
                    for inp, x in v["inputs"].items():
                        if isinstance(x, list):
                            self.assertIn(x[0], api, f"{name}: {k}.{inp}")
