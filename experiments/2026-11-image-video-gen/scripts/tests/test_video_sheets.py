import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from video_sheets import ffmpeg_args, plan  # noqa: E402


class Plan(unittest.TestCase):
    def test_groups_videos_by_prompt_in_config_order(self):
        files = ["b/V1_s1.mp4", "a/V1_s1.mp4", "a/V2_s1.mp4"]
        got = plan([pathlib.Path(f) for f in files], labels=["a", "b"])
        self.assertEqual(list(got), ["V1_s1", "V2_s1"])
        self.assertEqual([p.parts[0] for p in got["V1_s1"]], ["a", "b"])

    def test_ffmpeg_args_select_every_nth_frame_and_tile(self):
        args = ffmpeg_args(pathlib.Path("x.mp4"), pathlib.Path("o.jpg"), frames=6, total_frames=121, width=320)
        self.assertEqual(args[0], "ffmpeg")
        vf = args[args.index("-vf") + 1]
        self.assertIn("not(mod(n,24))", vf)  # 121/6 → 24 フレームおき
        self.assertIn("tile=6x1", vf)
        self.assertIn("scale=320:-1", vf)
