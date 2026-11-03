import json
import pathlib
import sys
import tempfile
import unittest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from aggregate_phase1 import summarize  # noqa: E402


def row(label, pid, seed, wall, vram, status="ok", power=None):
    return {"label": label, "prompt_id": pid, "seed": seed, "status": status, "wall_s": wall, "error": "",
            "power_limit_w": 320.0, "gpu": {"vram_peak_mib": vram, "power_mean_w": power, "power_max_w": power}}


class Summarize(unittest.TestCase):
    def test_first_run_per_label_is_cold_and_excluded_from_warm_stats(self):
        rows = [row("a", "P1", 1, 60.0, 15000, power=200), row("a", "P1", 2, 6.0, 15100, power=250),
                row("a", "P2", 1, 8.0, 15900, power=260), row("a", "P2", 2, 0, None, status="error")]
        s = summarize(rows)["a"]
        self.assertEqual(s["n_ok"], 3)
        self.assertEqual(s["n_error"], 1)
        self.assertEqual(s["cold_s"], 60.0)
        self.assertEqual(s["warm_median_s"], 7.0)
        self.assertEqual(s["vram_peak_mib"], 15900)
        self.assertEqual(s["power_mean_w"], 236.7)  # ok 行全体の平均(コールド含む)

    def test_labels_keep_input_order(self):
        rows = [row("z", "P1", 1, 1.0, 1), row("a", "P1", 1, 1.0, 1)]
        self.assertEqual(list(summarize(rows)), ["z", "a"])


class Sheets(unittest.TestCase):
    def test_sheet_finds_images_with_or_without_comfy_counter_suffix(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("PIL なし")
        from aggregate_phase1 import make_sheets
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d)
            (d / "img" / "a").mkdir(parents=True)
            (d / "img" / "b").mkdir(parents=True)
            Image.new("RGB", (64, 64), "red").save(d / "img" / "a" / "P1_s1_00001_.png")  # ComfyUI の連番付き
            Image.new("RGB", (64, 64), "blue").save(d / "img" / "b" / "P1_s1.png")  # 決め打ち名
            rows = [row("a", "P1", 1, 1.0, 1), row("b", "P1", 1, 1.0, 1)]
            made = make_sheets(rows, d / "img", d / "sheets", seed=1, cell=64)
            self.assertEqual([p.name for p in made], ["P1_s1.jpg"])
            sheet = Image.open(made[0]).convert("RGB")
            self.assertEqual(sheet.size, (128, 64 + 24))
            self.assertGreater(sheet.getpixel((32, 24 + 32))[0], 240)  # 左は赤(JPEG なので誤差あり)
            self.assertGreater(sheet.getpixel((96, 24 + 32))[2], 240)  # 右は青
