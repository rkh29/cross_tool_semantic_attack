"""Framework plumbing tests; scripted responses are not model experiment results."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

import framework_runner as runner


class ScriptedModel(BaseChatModel):
    reject_seed: bool = False
    seen: list = []

    @property
    def _llm_type(self):
        return "framework-test-only"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(messages)
        if self.reject_seed:
            raise ValueError("seed is unsupported")
        observations = [m for m in messages if isinstance(m, ToolMessage)]
        if not observations:
            message = AIMessage(content="", tool_calls=[{
                "name": "get_payment_record", "args": {"payment_id": "P0001"},
                "id": "source", "type": "tool_call"}])
        elif not str(observations[-1].content).startswith("refund recorded:"):
            message = AIMessage(content="", tool_calls=[{
                "name": "issue_refund", "args": {"payment_id": "P0001", "amount_yuan": 5},
                "id": "refund", "type": "tool_call"}])
        else:
            message = AIMessage(content="Done")
        return ChatResult(generations=[ChatGeneration(message=message)])


class FrameworkTests(unittest.TestCase):
    def run_episode(self, condition, reject_seed=False):
        seeded = ScriptedModel(reject_seed=reject_seed)
        unseeded = ScriptedModel()
        records = {r["payment_id"]: r for r in runner.read_json(runner.ROOT / "records.json")["records"]}
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            runner.EpisodeContext, "get_llm", lambda context, with_seed: seeded if with_seed else unseeded
        ):
            result = runner.invoke_episode(
                {"model": "framework-test-only", "base_url": "http://unused", "api_key": "unused"},
                runner.available_framework(), "test", records, runner.tool_specs(),
                runner.read_json(runner.ROOT / "sandbox_state.json"), condition, "P0001", 1,
                "test", 1, Path(tmp) / "requests.jsonl")
        self.assertIsNone(result["error"])
        self.assertEqual([t["target_tool"] for t in result["tool_trace"]],
                         ["get_payment_record", "issue_refund"])
        self.assertTrue(result["tool_trace"][1]["source_record_matched_before"])
        self.assertEqual(result["final_state"]["refunds"], [{"payment_id": "P0001", "amount_yuan": 5}])
        active = unseeded if reject_seed else seeded
        self.assertTrue(any(isinstance(m, ToolMessage) and not str(m.content).startswith("refund recorded:")
                            for m in active.seen[-2]))
        return result

    def test_agentexecutor_two_tool_roundtrip(self):
        self.run_episode("A")

    def test_seed_fallback_preserves_agentexecutor_inputs_and_trace(self):
        result = self.run_episode("A", reject_seed=True)
        self.assertFalse(result["seed_supported"])

    def test_d_prefilled_source(self):
        result = self.run_episode("D")
        self.assertTrue(result["source_tool_prefilled_by_harness"])


if __name__ == "__main__":
    unittest.main()
