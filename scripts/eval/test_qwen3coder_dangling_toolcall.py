#!/usr/bin/env python3
"""Regression guard for the qwen3_coder trailing-<tool_call> content leak.

Background: on the turn after a tool result, Qwen3-Coder can emit a *dangling*
tool-call start marker (e.g. "…light rain.\n<tool_call>") with no complete
`<function=…></function>` inside and finish_reason=stop. The old symptom
(README next-step, validated 2026-07-21 on v0.5.15) was that this marker leaked
into user-facing `message.content`.

On v0.5.16 the serving auto-path runs the parser and returns the parser's
stripped `normal_text` when no structured call survives, so the marker never
reaches content. This test locks that in by exercising the exact pieces the
OpenAI serving layer uses (`FunctionCallParser.has_tool_call` +
`parse_non_stream`) with real tools present, plus the raw detector.

Offline / GPU-free. Run:  python scripts/eval/test_qwen3coder_dangling_toolcall.py
"""
import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.function_call.qwen3_coder_detector import Qwen3CoderDetector

TOOLS = [
    Tool(
        type="function",
        function=Function(
            name="get_weather",
            parameters={
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        ),
    )
]

# Raw model outputs that carry a tool-call START marker but no complete call.
DANGLING = [
    "The weather in Paris is light rain.\n<tool_call>",
    "The weather in Paris is light rain.\n<tool_call>\n",
    "Sure, let me check.\n<tool_call>\n<function=get_weather",
    "Here is the summary.<tool_call></tool_call>",
]

COMPLETE = (
    "Let me look that up.\n"
    "<tool_call>\n<function=get_weather>\n<parameter=location>\nParis\n</parameter>\n"
    "</function>\n</tool_call>"
)


def _serving_content(parser: FunctionCallParser, text: str):
    """Reproduce the OpenAI serving auto-path content decision exactly."""
    if not parser.has_tool_call(text):
        return text, []  # branch skipped -> raw text is the content
    stripped, calls = parser.parse_non_stream(text)
    return (stripped if not calls else stripped), calls


class Qwen3CoderDanglingToolCall(unittest.TestCase):
    def setUp(self):
        self.parser = FunctionCallParser(TOOLS, "qwen3_coder")
        self.detector = Qwen3CoderDetector()

    def test_detector_strips_dangling_marker(self):
        for text in DANGLING:
            r = self.detector.detect_and_parse(text, TOOLS)
            self.assertEqual(len(r.calls), 0, f"unexpected call for {text!r}")
            self.assertNotIn("<tool_call>", r.normal_text, f"marker leaked: {text!r}")
            self.assertNotIn("<function", r.normal_text, f"marker leaked: {text!r}")

    def test_serving_path_never_leaks_marker(self):
        for text in DANGLING:
            content, calls = _serving_content(self.parser, text)
            self.assertEqual(len(calls), 0, f"unexpected call for {text!r}")
            self.assertNotIn("<tool_call>", content, f"CONTENT LEAK for {text!r}: {content!r}")
            self.assertNotIn("<function", content, f"CONTENT LEAK for {text!r}: {content!r}")

    def test_complete_call_still_parses_and_content_is_clean(self):
        content, calls = _serving_content(self.parser, COMPLETE)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "get_weather")
        self.assertIn("Paris", calls[0].parameters)
        self.assertNotIn("<tool_call>", content)
        self.assertNotIn("<function", content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
