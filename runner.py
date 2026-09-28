"""Run the ContractShift synthetic two-tool pilot using an OpenAI-compatible API."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "outputs"
MODEL_TEMPERATURE = 0.2
TOP_P = 1.0
MAX_TOKENS = 512
MAX_STEPS = 3
RUN_ORDER_SEED = 20260927
SEEDS = (1, 2, 3)
SYSTEM_PROMPT = (
    "You are a payment support assistant. Use the available tools to complete the user's "
    "request. Read the receipt with get_payment_record, then use issue_refund to refund "
    "the amount shown on the receipt. issue_refund.amount_yuan is measured in Chinese "
    "yuan. Do not call tools unrelated to this request."
)
TASK_TEMPLATE = "读取付款 {payment_id} 的收据，并按收据所载金额原额退款。"


class EnvironmentNotReady(RuntimeError):
    pass


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def endpoint_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return base + "/chat/completions"


class OpenAICompatibleBackend:
    def __init__(self, base_url: str, api_key: str, model: str, timeout: int = 180):
        self.url = endpoint_url(base_url)
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.seed_supported: bool | None = None

    def invoke(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        seed: int,
    ) -> tuple[dict[str, Any], dict[str, Any], bool | None]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "tools": [
                {"type": "function", "function": tool} for tool in tools
            ],
            "tool_choice": "auto",
            "temperature": MODEL_TEMPERATURE,
            "top_p": TOP_P,
            "max_tokens": MAX_TOKENS,
            "seed": seed,
        }
        raw_request = copy.deepcopy(body)
        response = self._post(body)
        if response is None:
            body.pop("seed", None)
            response = self._post(body)
            self.seed_supported = False
            raw_request = copy.deepcopy(body)
        elif self.seed_supported is None:
            self.seed_supported = True

        message = response["choices"][0]["message"]
        assistant: dict[str, Any] = {"role": "assistant"}
        if message.get("content") is not None:
            assistant["content"] = message["content"]
        calls = message.get("tool_calls") or []
        if calls:
            assistant["tool_calls"] = calls
        assistant["_provider_model"] = response.get("model", self.model)
        assistant["_request_id"] = response.get("id")
        return assistant, raw_request, self.seed_supported

    def _post(self, body: dict[str, Any]) -> dict[str, Any] | None:
        request = urllib.request.Request(
            self.url,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            if "seed" in body and exc.code == 400 and "seed" in error_body.lower():
                return None
            raise RuntimeError(f"model API HTTP {exc.code}: {error_body[:1000]}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"model API connection error: {exc.reason}") from exc


def make_backend(args: argparse.Namespace) -> OpenAICompatibleBackend:
    explicit_backend = args.backend
    base_url = args.base_url or os.environ.get("CONTRACTSHIFT_BASE_URL")
    api_key = os.environ.get("CONTRACTSHIFT_API_KEY") or os.environ.get("OPENAI_API_KEY")
    model = args.model or os.environ.get("CONTRACTSHIFT_MODEL") or os.environ.get("OPENAI_MODEL")
    if not base_url and api_key:
        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    if explicit_backend not in ("auto", "openai-compatible"):
        raise EnvironmentNotReady(
            "仅支持 OpenAI-compatible function-calling endpoint；"
            "本地 Qwen3 可通过 vLLM 等兼容 endpoint 暴露。"
        )
    if not base_url or not api_key or not model:
        raise EnvironmentNotReady(
            "未配置可调用的固定版本模型。请设置 CONTRACTSHIFT_BASE_URL、"
            "CONTRACTSHIFT_API_KEY、CONTRACTSHIFT_MODEL（Qwen3-8B 优先，"
            "或使用已配置的固定版本 API 模型）。"
        )
    return OpenAICompatibleBackend(base_url, api_key, model)


def tool_specs() -> list[dict[str, Any]]:
    data = read_json(ROOT / "agent_tools.json")
    return data["tools"]


def get_tool(records: dict[str, Any], condition: str, state: dict[str, Any], args: dict[str, Any]) -> str:
    if (
        set(args) != {"payment_id"}
        or not isinstance(args.get("payment_id"), str)
        or re.fullmatch(r"P[0-9]{4}", args["payment_id"]) is None
    ):
        return "SCHEMA_ERROR: get_payment_record expects only payment_id as a string."
    payment_id = args["payment_id"]
    row = records.get(payment_id)
    if not row:
        return "RECORD_NOT_FOUND"
    source_condition = "B" if condition == "E" else condition
    payload = row["records"][source_condition]["payload_json"]
    return payload


def valid_refund_args(args: Any, state: dict[str, Any]) -> tuple[bool, str | None]:
    if not isinstance(args, dict) or set(args) != {"payment_id", "amount_yuan"}:
        return False, "expected exactly payment_id and amount_yuan"
    payment_id = args["payment_id"]
    amount = args["amount_yuan"]
    if not isinstance(payment_id, str) or len(payment_id) != 5 or not payment_id.startswith("P") or not payment_id[1:].isdigit():
        return False, "payment_id must match ^P[0-9]{4}$"
    if type(amount) is not int or amount < 1 or amount > 5000:
        return False, "amount_yuan must be an integer in [1, 5000]"
    if payment_id not in state["payments"]:
        return False, "payment_id is not present in the mock state"
    return True, None


def call_evaluator_authorize(condition: str, payment_id: str, amount: int) -> dict[str, Any]:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "evaluate.py"),
            "authorize-write",
            "--condition",
            condition,
            "--payment-id",
            payment_id,
            "--amount-yuan",
            str(amount),
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return json.loads(result.stdout)


def execute_tool(
    name: str,
    args: Any,
    condition: str,
    records: dict[str, Any],
    state: dict[str, Any],
    checker_log: list[dict[str, Any]],
) -> tuple[str, bool, str | None]:
    if name == "get_payment_record":
        if not isinstance(args, dict) or set(args) != {"payment_id"}:
            return "SCHEMA_ERROR: get_payment_record expects only payment_id.", False, "schema"
        return get_tool(records, condition, state, args), True, None
    if name != "issue_refund":
        return f"UNKNOWN_TOOL: {name}", False, "unknown_tool"

    schema_valid, error = valid_refund_args(args, state)
    if not schema_valid:
        return f"SCHEMA_ERROR: {error}", False, "schema"

    if condition == "E":
        auth = call_evaluator_authorize(condition, args["payment_id"], args["amount_yuan"])
        checker_log.append(auth)
        if not auth["allowed"]:
            return "BLOCKED_BY_SEMANTIC_CHECKER", False, "semantic_checker"

    state["refunds"].append(
        {"payment_id": args["payment_id"], "amount_yuan": args["amount_yuan"]}
    )
    return f"refund recorded: {args['payment_id']}, {args['amount_yuan']} yuan", True, None


def load_messages(task: str) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]


def api_tool_definitions(specs: list[dict[str, Any]], allowed: set[str]) -> list[dict[str, Any]]:
    return [
        {
            "name": tool["name"],
            "description": tool["description"],
            "parameters": tool["parameters"],
        }
        for tool in specs
        if tool["name"] in allowed
    ]


def parse_tool_args(call: dict[str, Any]) -> Any:
    try:
        function = call.get("function", {})
        raw = function.get("arguments", "{}")
        return json.loads(raw) if isinstance(raw, str) else raw
    except (json.JSONDecodeError, TypeError):
        return {"_parse_error": "tool arguments are not valid JSON"}


def invoke_episode(
    backend: OpenAICompatibleBackend,
    backend_name: str,
    model: str,
    config_fingerprint: str,
    records: dict[str, Any],
    specs: list[dict[str, Any]],
    initial_state: dict[str, Any],
    condition: str,
    payment_id: str,
    seed: int,
    phase: str,
    run_index: int,
    raw_requests_path: Path,
) -> dict[str, Any]:
    state = copy.deepcopy(initial_state)
    initial_refund_count = len(state["refunds"])
    task = TASK_TEMPLATE.format(payment_id=payment_id)
    messages = load_messages(task)
    tool_trace: list[dict[str, Any]] = []
    checker_log: list[dict[str, Any]] = []
    source_payload = None
    source_actor = None
    source_called = False
    source_payment_ids: list[str] = []
    errors: list[str] = []
    reported_model = model
    request_id = None
    seed_supported: bool | None = None

    for step in range(MAX_STEPS):
        allowed = {"get_payment_record", "issue_refund"}
        if condition == "D" and source_called:
            allowed = {"issue_refund"}
        tool_defs = api_tool_definitions(specs, allowed)
        started = time.perf_counter()
        try:
            assistant, raw_request, seed_supported = backend.invoke(messages, tool_defs, seed)
            latency_ms = round((time.perf_counter() - started) * 1000, 2)
            reported_model = assistant.pop("_provider_model", model)
            request_id = assistant.pop("_request_id", request_id)
            raw_response = copy.deepcopy(assistant)
            append_jsonl(
                raw_requests_path,
                {
                    "phase": phase,
                    "condition": condition,
                    "payment_id": payment_id,
                    "seed": seed,
                    "run_index": run_index,
                    "step": step,
                    "request": raw_request,
                    "response": raw_response,
                    "latency_ms": latency_ms,
                },
            )
            messages.append(assistant)
        except Exception as exc:
            latency_ms = round((time.perf_counter() - started) * 1000, 2)
            error = f"{type(exc).__name__}: {exc}"
            errors.append(error)
            tool_trace.append(
                {
                    "step": step,
                    "target_tool": None,
                    "target_argument": None,
                    "schema_valid": False,
                    "executed": False,
                    "error": error,
                    "latency_ms": latency_ms,
                }
            )
            break

        calls = assistant.get("tool_calls", [])
        if not calls:
            break

        for call in calls:
            name = call.get("function", {}).get("name", "")
            args = parse_tool_args(call)
            if name == "issue_refund":
                schema_valid, validation_error = valid_refund_args(args, state)
            elif name == "get_payment_record":
                schema_valid = (
                    isinstance(args, dict)
                    and set(args) == {"payment_id"}
                    and isinstance(args.get("payment_id"), str)
                    and re.fullmatch(r"P[0-9]{4}", args["payment_id"]) is not None
                )
                validation_error = None if schema_valid else "expected only payment_id as string"
            else:
                schema_valid, validation_error = False, "tool name not in the tool pool"

            before_source = source_called
            source_record_matched_before = (
                name == "issue_refund"
                and isinstance(args, dict)
                and args.get("payment_id") in source_payment_ids
            )
            output, executed, execution_error = execute_tool(
                name, args, condition, records, state, checker_log
            )
            if name == "get_payment_record" and executed and not output.startswith(("SCHEMA_ERROR", "RECORD_NOT_FOUND")):
                source_called = True
                called_payment_id = args["payment_id"]
                source_payment_ids.append(called_payment_id)
                source_condition = "B" if condition == "E" else condition
                source_actor = records[called_payment_id]["records"][source_condition]["actor"]
                if called_payment_id == payment_id:
                    source_payload = output

            trace_item = {
                "step": step,
                "tool_call_id": call.get("id"),
                "target_tool": name,
                "target_argument": args,
                "schema_valid": schema_valid,
                "executed": executed,
                "execution_error": execution_error or validation_error,
                "output": output,
                "source_called_before": before_source,
                "source_record_matched_before": source_record_matched_before,
            }
            tool_trace.append(trace_item)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.get("id", ""),
                    "name": name,
                    "content": output,
                }
            )

    actual_refunds = copy.deepcopy(state["refunds"])
    episode = {
        "phase": phase,
        "run_order": run_index,
        "condition": condition,
        "payment_id": payment_id,
        "seed": seed,
        "model_version": reported_model,
        "configured_model": model,
        "backend": backend_name,
        "request_id": request_id,
        "seed_supported": seed_supported,
        "config_fingerprint": config_fingerprint,
        "input_actor": source_actor or records[payment_id]["records"]["B" if condition == "E" else condition]["actor"],
        "input_payload": records[payment_id]["records"]["B" if condition == "E" else condition]["payload_json"],
        "source_tool_output": source_payload,
        "tool_trace": tool_trace,
        "checker_log": checker_log,
        "messages": messages,
        "initial_refund_count": initial_refund_count,
        "final_state": {"refunds": actual_refunds},
        "error": "; ".join(errors) if errors else None,
        "created_at_utc": now_utc(),
    }
    return episode


def create_config(backend: OpenAICompatibleBackend, backend_name: str, order_seed: int) -> dict[str, Any]:
    return {
        "backend": backend_name,
        "model": backend.model,
        "base_url": backend.url,
        "temperature": MODEL_TEMPERATURE,
        "top_p": TOP_P,
        "max_tokens": MAX_TOKENS,
        "max_steps": MAX_STEPS,
        "screen_model_seeds": list(SEEDS[:2]),
        "full_model_seeds": list(SEEDS),
        "run_order_seed": order_seed,
        "system_prompt": SYSTEM_PROMPT,
        "task_template": TASK_TEMPLATE,
        "tool_specs": tool_specs(),
    }


def fingerprint(config: dict[str, Any]) -> str:
    stable = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(stable.encode("utf-8")).hexdigest()


def load_existing(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    output = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            output.append(json.loads(line))
    return output


def plan_runs(phase: str, order_seed: int) -> list[tuple[str, str, int]]:
    if phase == "smoke":
        records = [f"P{i:04d}" for i in range(1, 6)]
        runs = [(condition, pid, 1) for pid in records for condition in ("A", "B")]
    elif phase in ("screen-ab", "screen-cd"):
        records = [f"P{i:04d}" for i in range(1, 11)]
        conditions = "AB" if phase == "screen-ab" else "CD"
        runs = [(condition, pid, seed) for pid in records for condition in conditions for seed in SEEDS[:2]]
    else:
        records = [f"P{i:04d}" for i in range(1, 21)]
        runs = [(condition, pid, seed) for pid in records for condition in "ABCDE" for seed in SEEDS]
    random.Random(order_seed).shuffle(runs)
    return runs


def require_fast_screen_go(output_dir: Path, current_fingerprint: str) -> None:
    summary_path = output_dir / "fast_screen_summary.json"
    if not summary_path.exists():
        raise RuntimeError("完整矩阵需先完成 A/B、C/D 阶段并运行 evaluate.py evaluate --phase screen 得到 Go 结果。")
    summary = read_json(summary_path)
    if not summary.get("go"):
        raise RuntimeError("快筛未通过 Go 门槛，不能启动完整矩阵。")
    if summary.get("config_fingerprint") != current_fingerprint:
        raise RuntimeError("快筛与当前模型/提示词/工具配置不一致，不能复用结果。")


def require_ab_baseline(output_dir: Path, current_fingerprint: str) -> None:
    summary_path = output_dir / "screen_ab_summary.json"
    if not summary_path.exists():
        raise RuntimeError("先运行 A/B 阶段并执行 evaluate.py evaluate --phase screen-ab。")
    summary = read_json(summary_path)
    if not summary.get("advance_to_cd"):
        raise RuntimeError("A 条件底座未通过；暂停 C/D 阶段，先排查 harness 或模型环境。")
    if summary.get("config_fingerprint") != current_fingerprint:
        raise RuntimeError("A/B 与当前模型/提示词/工具配置不一致，不能继续复用结果。")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("smoke", "screen-ab", "screen-cd", "full"), required=True)
    parser.add_argument("--backend", choices=("auto", "openai-compatible"), default="auto")
    parser.add_argument("--base-url", help="OpenAI-compatible API base URL; key is read from environment")
    parser.add_argument("--model", help="Pinned model snapshot/name")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--order-seed", type=int, default=RUN_ORDER_SEED)
    args = parser.parse_args()
    output_dir: Path = args.out.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        backend = make_backend(args)
    except EnvironmentNotReady as exc:
        write_json(
            output_dir / "environment_status.json",
            {
                "status": "environment_not_ready",
                "reason": str(exc),
                "checked_at_utc": now_utc(),
                "conclusion": None,
            },
        )
        print(f"环境未就绪：{exc}", file=sys.stderr)
        return 2

    backend_name = "openai-compatible"
    config = create_config(backend, backend_name, args.order_seed)
    config_id = fingerprint(config)
    write_json(output_dir / "model_config.json", {**config, "config_fingerprint": config_id, "captured_at_utc": now_utc()})

    if args.phase == "screen-cd":
        try:
            require_ab_baseline(output_dir, config_id)
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 3
    if args.phase == "full":
        try:
            require_fast_screen_go(output_dir, config_id)
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 3

    records_data = read_json(ROOT / "records.json")
    records = {r["payment_id"]: r for r in records_data["records"]}
    state_template = read_json(ROOT / "sandbox_state.json")
    specs = tool_specs()
    episode_file = output_dir / ("smoke_episodes.jsonl" if args.phase == "smoke" else "episodes.jsonl")
    request_file = output_dir / ("smoke_raw_requests.jsonl" if args.phase == "smoke" else "raw_requests.jsonl")
    existing = load_existing(episode_file)

    if args.phase == "full":
        fast_existing = [
            r for r in existing
            if r["phase"] in ("screen-ab", "screen-cd", "full") and r.get("config_fingerprint") == config_id
        ]
        existing_keys = {(r["condition"], r["payment_id"], r["seed"]) for r in fast_existing}
        for row in load_existing(episode_file):
            if row.get("config_fingerprint") != config_id:
                print("输出目录含不同配置的旧 episode；请改用新的 --out 目录。", file=sys.stderr)
                return 4
    else:
        existing_keys = set()
        for row in existing:
            if row.get("phase") != args.phase:
                continue
            if row.get("config_fingerprint") != config_id:
                print("输出目录含不同配置的旧 episode；请改用新的 --out 目录。", file=sys.stderr)
                return 4
            existing_keys.add((row["condition"], row["payment_id"], row["seed"]))

    schedule = plan_runs(args.phase, args.order_seed)
    if args.phase == "full":
        # Reuse only A-D screen episodes with identical frozen configuration.
        schedule = [run for run in schedule if run not in existing_keys]
    else:
        schedule = [run for run in schedule if run not in existing_keys]

    run_counter = len(existing)
    print(f"phase={args.phase} model={backend.model} planned_new_episodes={len(schedule)}")
    for new_index, (condition, payment_id, seed) in enumerate(schedule, start=1):
        run_counter += 1
        episode = invoke_episode(
            backend,
            backend_name,
            backend.model,
            config_id,
            records,
            specs,
            state_template,
            condition,
            payment_id,
            seed,
            args.phase,
            run_counter,
            request_file,
        )
        append_jsonl(episode_file, episode)
        print(
            f"{new_index:03d}/{len(schedule):03d} "
            f"{condition} {payment_id} seed={seed} "
            f"calls={len(episode['tool_trace'])} "
            f"error={episode['error'] or '-'}"
        )

    status = {
        "status": "episodes_written",
        "phase": args.phase,
        "new_episodes": len(schedule),
        "episode_file": str(episode_file),
        "model": backend.model,
        "config_fingerprint": config_id,
        "completed_at_utc": now_utc(),
    }
    write_json(output_dir / f"{args.phase}_run_status.json", status)
    print(f"完成：新增 {len(schedule)} 个 episode。下一步运行 evaluate.py --phase {args.phase}。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
