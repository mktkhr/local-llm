import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from turn_probe import anthropic_to_oai, plan, variant  # noqa: E402

SAMPLE = {
    "model": "x",
    "max_tokens": 32000,
    "metadata": {"user_id": "u"},
    "context_management": {"edits": []},
    "output_config": {"effort": "high"},
    "thinking": {"type": "adaptive"},
    "system": [
        {"type": "text", "text": "A"},
        {"type": "text", "text": "B", "cache_control": {"type": "ephemeral"}},
    ],
    "tools": [
        {"name": "Read", "description": "read", "input_schema": {"type": "object"}},
        {"name": "Bash", "description": "bash", "input_schema": {"type": "object"}},
        {"name": "Edit", "description": "edit"},
    ],
    "messages": [
        {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "t1"},
                {"type": "text", "text": "ok"},
                {"type": "tool_use", "id": "c1", "name": "Read", "input": {"p": 1}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "c1", "content": "body"},
                {"type": "tool_result", "tool_use_id": "c2", "content": [{"type": "text", "text": "x"}, {"type": "text", "text": "y"}]},
                {"type": "tool_result", "tool_use_id": "c3"},
            ],
        },
        {"role": "assistant", "content": [{"type": "tool_use", "id": "c4", "name": "Bash", "input": {}}]},
        {"role": "user", "content": "plain"},
    ],
}


class TestAnthropicToOai(unittest.TestCase):
    def setUp(self):
        self.oai = anthropic_to_oai(SAMPLE)

    def test_system_blocks_joined_without_separator(self):
        self.assertEqual(self.oai["messages"][0], {"role": "system", "content": "AB"})

    def test_assistant_blocks_split_into_content_tool_calls_reasoning(self):
        m = self.oai["messages"][2]
        self.assertEqual(m["role"], "assistant")
        self.assertEqual(m["content"], [{"type": "text", "text": "ok"}])
        self.assertEqual(m["reasoning_content"], "t1")
        self.assertEqual(m["tool_calls"], [{"id": "c1", "type": "function",
                                            "function": {"name": "Read", "arguments": '{"p":1}'}}])

    def test_tool_results_become_tool_messages_after_user_text(self):
        tools = self.oai["messages"][3:6]
        self.assertEqual(tools, [
            {"role": "tool", "tool_call_id": "c1", "content": "body"},
            {"role": "tool", "tool_call_id": "c2", "content": "xy"},
            {"role": "tool", "tool_call_id": "c3", "content": ""},
        ])

    def test_tool_use_only_assistant_gets_empty_string_content(self):
        m = self.oai["messages"][6]
        self.assertEqual(m["content"], "")
        self.assertEqual(m["tool_calls"][0]["function"]["arguments"], "{}")

    def test_string_content_passes_through(self):
        self.assertEqual(self.oai["messages"][7], {"role": "user", "content": "plain"})

    def test_tools_wrapped_as_functions_with_parameters(self):
        self.assertEqual(self.oai["tools"][0], {"type": "function", "function": {
            "name": "Read", "description": "read", "parameters": {"type": "object"}}})
        self.assertEqual(self.oai["tools"][2]["function"]["parameters"], {})

    def test_tool_choice_any_maps_to_required(self):
        oai = anthropic_to_oai({**SAMPLE, "tool_choice": {"type": "any"}})
        self.assertEqual(oai["tool_choice"], "required")

    def test_max_tokens_and_passthrough_keys(self):
        oai = anthropic_to_oai({**SAMPLE, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}})
        self.assertEqual(oai["max_tokens"], 32000)
        self.assertEqual(oai["temperature"], 0)
        self.assertEqual(oai["chat_template_kwargs"], {"enable_thinking": False})


class TestVariant(unittest.TestCase):
    def test_truncates_tools_and_messages(self):
        v = variant(SAMPLE, n_tools=2, n_msgs=3, max_tokens=1)
        self.assertEqual([t["name"] for t in v["tools"]], ["Read", "Bash"])
        self.assertEqual(len(v["messages"]), 3)
        self.assertEqual(len(SAMPLE["tools"]), 3)  # 元は壊さない

    def test_zero_tools_drops_key(self):
        self.assertNotIn("tools", variant(SAMPLE, n_tools=0, n_msgs=None, max_tokens=1))

    def test_fixes_sampling_and_strips_build_dependent_fields(self):
        v = variant(SAMPLE, n_tools=None, n_msgs=None, max_tokens=7)
        self.assertEqual(v["max_tokens"], 7)
        self.assertEqual(v["temperature"], 0)
        self.assertIs(v["stream"], False)
        self.assertEqual(v["thinking"], {"type": "disabled"})
        self.assertEqual(v["chat_template_kwargs"], {"enable_thinking": False})
        for k in ("metadata", "context_management", "output_config"):
            self.assertNotIn(k, v)

    def test_tool_choice_any_sets_field(self):
        v = variant(SAMPLE, n_tools=None, n_msgs=None, max_tokens=1, tool_choice="any")
        self.assertEqual(v["tool_choice"], {"type": "any"})


class TestPlan(unittest.TestCase):
    def test_tools_axis_caps_at_available_and_ends_with_all(self):
        src = {"tools": [{}] * 40, "messages": []}
        self.assertEqual(plan(src, "tools"), [(0, None), (8, None), (16, None), (33, None), (40, None)])

    def test_msgs_axis_snaps_to_user_messages(self):
        roles = ["user", "assistant"] * 30  # 60 件、ユーザー発話は奇数番目
        src = {"messages": [{"role": r} for r in roles]}
        got = [m for _, m in plan(src, "msgs")]
        self.assertTrue(all(src["messages"][m - 1]["role"] == "user" for m in got))
        self.assertEqual(got[0], 1)
        self.assertEqual(got[-1], 59)


if __name__ == "__main__":
    unittest.main()
