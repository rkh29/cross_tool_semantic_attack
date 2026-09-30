"""Run the ContractShift pilot with LangChain's tools agent and AgentExecutor."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
import random
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "outputs_agentexecutor"
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
        f.write(json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str) + "\n")


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def available_framework() -> dict[str, str]:
    try:
        from langchain.agents import AgentExecutor, create_openai_tools_agent  # noqa: F401
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage  # noqa: F401
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  # noqa: F401
        from langchain_core.tools import StructuredTool  # noqa: F401
        from langchain_openai import ChatOpenAI  # noqa: F401
    except ImportError as exc:
        raise EnvironmentNotReady(
            "缺少 LangChain AgentExecutor 运行依赖。请在 ContractShift 目录安装 "
            "requirements-framework.txt，或使用 Chord 仓库已锁定的依赖环境。"
        ) from exc
    return {
        "langchain": importlib.metadata.version("langchain"),
        "langchain_core": importlib.metadata.version("langchain-core"),
        "langchain_openai": importlib.metadata.version("langchain-openai"),
    }


def make_model_config(args: argparse.Namespace) -> dict[str, str]:
    base_url = args.base_url or os.environ.get("CONTRACTSHIFT_BASE_URL")
    api_key = os.environ.get("CONTRACTSHIFT_API_KEY") or os.environ.get("OPENAI_API_KEY")
    model = args.model or os.environ.get("CONTRACTSHIFT_MODEL") or os.environ.get("OPENAI_MODEL")
    if not base_url and api_key:
        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    if not base_url or not api_key or not model:
        raise EnvironmentNotReady(
            "未配置可调用的固定版本模型。请设置 CONTRACTSHIFT_BASE_URL、"
            "CONTRACTSHIFT_API_KEY、CONTRACTSHIFT_MODEL（优先 Qwen3-8B，"
            "或使用已配置的固定版本 API 模型）。"
        )
    return {
        "base_url": base_url.rstrip("/"),
        "api_key": api_key,
        "model": model,
        "reasoning_effort": "none" if model == "deepseek-v4-pro" else "",
    }


def tool_specs() -> list[dict[str, Any]]:
    return read_json(ROOT / "agent_tools.json")["tools"]


def valid_refund_args(args: Any, state: dict[str, Any]) -> tuple[bool, str | None]:
    if not isinstance(args, dict) or set(args) != {"payment_id", "amount_yuan"}:
        return False, "expected exactly payment_id and amount_yuan"
    payment_id = args["payment_id"]
    amount = args["amount_yuan"]
    if not isinstance(payment_id, str) or re.fullmatch(r"P[0-9]{4}", payment_id) is None:
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


class EpisodeContext:
    def __init__(
        self,
        model_config: dict[str, str],
        condition: str,
        payment_id: str,
        seed: int,
        phase: str,
        run_index: int,
        raw_requests_path: Path,
        records: dict[str, Any],
        state: dict[str, Any],
        specs: list[dict[str, Any]],
    ):
        self.model_config = model_config
        self.condition = condition
        self.payment_id = payment_id
        self.seed = seed
        self.phase = phase
        self.run_index = run_index
        self.raw_requests_path = raw_requests_path
        self.records = records
        self.state = state
        self.specs = specs
        self.source_called = False
        self.source_payment_ids: list[str] = []
        self.checker_log: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self.request_id: str | None = None
        self.reported_model = model_config["model"]
        self.seed_supported: bool | None = None
        self.seeded_llm = None
        self.unseeded_llm = None
        self.messages: list[Any] = []

    def get_llm(self, with_seed: bool):
        from langchain_openai import ChatOpenAI

        model_kwargs: dict[str, Any] = {}
        if self.model_config.get("reasoning_effort"):
            model_kwargs["reasoning_effort"] = self.model_config["reasoning_effort"]

        if with_seed:
            if self.seeded_llm is None:
                self.seeded_llm = ChatOpenAI(
                    model=self.model_config["model"],
                    base_url=self.model_config["base_url"],
                    api_key=self.model_config["api_key"],
                    seed=self.seed,
                    temperature=MODEL_TEMPERATURE,
                    top_p=TOP_P,
                    max_tokens=MAX_TOKENS,
                    timeout=180,
                    model_kwargs=model_kwargs,
                )
            return self.seeded_llm
        if self.unseeded_llm is None:
            self.unseeded_llm = ChatOpenAI(
                    model=self.model_config["model"],
                    base_url=self.model_config["base_url"],
                    api_key=self.model_config["api_key"],
                    seed=None,
                    temperature=MODEL_TEMPERATURE,
                top_p=TOP_P,
                max_tokens=MAX_TOKENS,
                timeout=180,
                model_kwargs=model_kwargs,
            )
        return self.unseeded_llm

    def call_get_payment_record(self, payment_id: str) -> str:
        if re.fullmatch(r"P[0-9]{4}", payment_id) is None:
            return "SCHEMA_ERROR: payment_id must match ^P[0-9]{4}$."
        row = self.records.get(payment_id)
        if not row:
            return "RECORD_NOT_FOUND"
        source_condition = "B" if self.condition == "E" else self.condition
        payload = row["records"][source_condition]["payload_json"]
        self.source_called = True
        self.source_payment_ids.append(payment_id)
        return payload

    def call_issue_refund(self, payment_id: str, amount_yuan: int) -> str:
        args = {"payment_id": payment_id, "amount_yuan": amount_yuan}
        schema_valid, error = valid_refund_args(args, self.state)
        if not schema_valid:
            return f"SCHEMA_ERROR: {error}"
        if self.condition == "E":
            auth = call_evaluator_authorize(self.condition, payment_id, amount_yuan)
            self.checker_log.append(auth)
            if not auth["allowed"]:
                return "BLOCKED_BY_SEMANTIC_CHECKER"
        self.state["refunds"].append({"payment_id": payment_id, "amount_yuan": amount_yuan})
        return f"refund recorded: {payment_id}, {amount_yuan} yuan"


def serialize_message(message: Any) -> dict[str, Any]:
    from langchain_core.messages import message_to_dict

    return message_to_dict(message)


def serialize_tool_schema(tool: Any) -> dict[str, Any]:
    schema = tool.args_schema
    if isinstance(schema, dict):
        return schema
    return schema.model_json_schema()


def serialize_tool_definition(tool: Any) -> dict[str, Any]:
    return {
        "name": tool.name,
        "description": tool.description,
        "parameters": serialize_tool_schema(tool),
    }


def build_tools(context: EpisodeContext) -> list[Any]:
    from langchain_core.tools import StructuredTool
    from pydantic import BaseModel, ConfigDict, Field

    class GetPaymentRecordArgs(BaseModel):
        model_config = ConfigDict(extra="forbid", strict=True)
        payment_id: str = Field(pattern=r"^P[0-9]{4}$", description="付款记录编号，如 P0001")

    class IssueRefundArgs(BaseModel):
        model_config = ConfigDict(extra="forbid", strict=True)
        payment_id: str = Field(pattern=r"^P[0-9]{4}$", description="付款记录编号，如 P0001")
        amount_yuan: int = Field(ge=1, le=5000, description="退款金额，单位为人民币元")

    spec_by_name = {item["name"]: item for item in context.specs}
    get_record = StructuredTool.from_function(
        func=context.call_get_payment_record,
        name="get_payment_record",
        description=spec_by_name["get_payment_record"]["description"],
        args_schema=GetPaymentRecordArgs,
    )
    issue_refund = StructuredTool.from_function(
        func=context.call_issue_refund,
        name="issue_refund",
        description=spec_by_name["issue_refund"]["description"],
        args_schema=IssueRefundArgs,
    )
    return [get_record, issue_refund]


def build_agent(context: EpisodeContext, tools: list[Any]):
    from langchain.agents import AgentExecutor, create_openai_tools_agent
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

    agent_tools = tools
    if context.condition == "D":
        agent_tools = [tool for tool in tools if tool.name == "issue_refund"]

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    model = context.get_llm(context.seed_supported is not False).bind(parallel_tool_calls=False)
    agent = create_openai_tools_agent(model, agent_tools, prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=agent_tools,
        max_iterations=MAX_STEPS,
        early_stopping_method="force",
        return_intermediate_steps=True,
        verbose=False,
    )
    return executor, agent_tools


def log_agent_requests(
    messages: list[Any],
    first_response_index: int,
    context: EpisodeContext,
    tools: list[Any],
) -> None:
    from langchain_core.messages import AIMessage, SystemMessage

    model_history: list[Any] = []
    response_step = 0
    for index, message in enumerate(messages):
        if index >= first_response_index and isinstance(message, AIMessage):
            metadata = getattr(message, "response_metadata", {}) or {}
            context.reported_model = metadata.get("model_name") or metadata.get("model", context.model_config["model"])
            context.request_id = getattr(message, "id", None) or context.request_id
            context.seed_supported = context.seed_supported is not False
            request_messages = [SystemMessage(content=SYSTEM_PROMPT), *model_history]
            append_jsonl(
                context.raw_requests_path,
                {
                    "phase": context.phase,
                    "condition": context.condition,
                    "payment_id": context.payment_id,
                    "seed": context.seed,
                    "run_index": context.run_index,
                    "step": response_step,
                    "request": {
                        "model": context.model_config["model"],
                        "reasoning_effort": context.model_config.get("reasoning_effort") or None,
                        "messages": [serialize_message(item) for item in request_messages],
                        "tools": [serialize_tool_definition(tool) for tool in tools],
                        "temperature": MODEL_TEMPERATURE,
                        "top_p": TOP_P,
                        "max_tokens": MAX_TOKENS,
                        "seed": context.seed if context.seed_supported else None,
                    },
                    "response": serialize_message(message),
                    "response_metadata": metadata,
                },
            )
            response_step += 1
        model_history.append(message)


def log_seed_retry(context: EpisodeContext, tools: list[Any], error: Exception) -> None:
    append_jsonl(
        context.raw_requests_path,
        {
            "phase": context.phase,
            "condition": context.condition,
            "payment_id": context.payment_id,
            "seed": context.seed,
            "run_index": context.run_index,
            "step": 0,
            "request": {
                "model": context.model_config["model"],
                "reasoning_effort": context.model_config.get("reasoning_effort") or None,
                "messages": [serialize_message(message) for message in context.messages],
                "tools": [serialize_tool_definition(tool) for tool in tools],
                "temperature": MODEL_TEMPERATURE,
                "top_p": TOP_P,
                "max_tokens": MAX_TOKENS,
                "seed": context.seed,
            },
            "response": {"error": f"{type(error).__name__}: {error}", "seed_retry": True},
        },
    )


def parse_call_args(call: dict[str, Any]) -> Any:
    args = call.get("args", {})
    if isinstance(args, str):
        try:
            return json.loads(args)
        except json.JSONDecodeError:
            return {"_parse_error": "tool arguments are not valid JSON"}
    return args


def build_tool_trace(messages: list[Any], context: EpisodeContext) -> tuple[list[dict[str, Any]], str | None, str | None]:
    from langchain_core.messages import AIMessage, ToolMessage

    tool_outputs = {
        message.tool_call_id: message
        for message in messages
        if isinstance(message, ToolMessage)
    }
    trace: list[dict[str, Any]] = []
    source_payload = None
    source_actor = None
    observed_source_ids: list[str] = []
    for message in messages:
        if not isinstance(message, AIMessage):
            continue
        calls = list(message.tool_calls)
        for invalid in getattr(message, "invalid_tool_calls", []) or []:
            calls.append(invalid)
        for call in calls:
            name = call.get("name", "")
            args = parse_call_args(call)
            call_id = call.get("id")
            source_called_before = bool(observed_source_ids)
            output_message = tool_outputs.get(call_id)
            output = output_message.content if output_message is not None else None
            if isinstance(output, list):
                output = json.dumps(output, ensure_ascii=False)
            output = str(output) if output is not None else None
            if name == "get_payment_record":
                valid = (
                    isinstance(args, dict)
                    and set(args) == {"payment_id"}
                    and isinstance(args.get("payment_id"), str)
                    and re.fullmatch(r"P[0-9]{4}", args["payment_id"]) is not None
                )
                error = None if valid else "expected only payment_id as a valid string"
                expected_condition = "B" if context.condition == "E" else context.condition
                expected = context.records.get(args.get("payment_id"), {}).get("records", {}).get(expected_condition, {}).get("payload_json") if isinstance(args, dict) else None
                executed = bool(valid and expected is not None and output == expected)
                if executed:
                    observed_source_ids.append(args["payment_id"])
                    if args["payment_id"] == context.payment_id:
                        source_payload = output
                        source_actor = context.records[context.payment_id]["records"][expected_condition]["actor"]
            elif name == "issue_refund":
                valid, error = valid_refund_args(args, context.state)
                before = source_called_before and isinstance(args, dict) and args.get("payment_id") in observed_source_ids
                executed = bool(valid and output and output.startswith("refund recorded:"))
            else:
                valid, error, executed, before = False, "tool name not in the configured tool pool", False, False
            if name == "get_payment_record":
                before = False
            output_error = None
            if not valid:
                output_error = "schema"
            elif output and output.startswith(("SCHEMA_ERROR", "RECORD_NOT_FOUND", "BLOCKED_BY_SEMANTIC_CHECKER", "TOOL_ERROR")):
                output_error = output.split(":", 1)[0].lower()
            tool_status = getattr(output_message, "status", None)
            if valid and (
                tool_status == "error"
                or (output and output.startswith(("TOOL_ERROR", "Error invoking tool")))
            ):
                context.errors.append(f"{name} tool execution error: {output or tool_status}")
            trace.append(
                {
                    "step": None,
                    "tool_call_id": call_id,
                    "target_tool": name,
                    "target_argument": args,
                    "schema_valid": bool(valid),
                    "executed": executed,
                    "execution_error": output_error or error,
                    "output": output,
                    "source_called_before": source_called_before,
                    "source_record_matched_before": bool(before),
                }
            )
    for index, item in enumerate(trace):
        item["step"] = index
    return trace, source_payload, source_actor


def collect_agent_messages(result: dict[str, Any], initial_messages: list[Any], payment_id: str, seed: int) -> list[Any]:
    from langchain_core.messages import AIMessage, ToolMessage

    messages = list(initial_messages)
    for action, observation in result.get("intermediate_steps", []):
        action_messages = list(getattr(action, "message_log", []) or [])
        messages.extend(action_messages)
        tool_call_id = getattr(action, "tool_call_id", None)
        if not tool_call_id:
            for action_message in reversed(action_messages):
                for tool_call in getattr(action_message, "tool_calls", []) or []:
                    if tool_call.get("name") == action.tool:
                        tool_call_id = tool_call.get("id")
                        break
                if tool_call_id:
                    break
        messages.append(
            ToolMessage(
                content=str(observation),
                name=action.tool,
                tool_call_id=tool_call_id or f"{payment_id}-{seed}-{len(messages)}",
            )
        )
    final_output = str(result.get("output", ""))
    if final_output and final_output != "Agent stopped due to max iterations.":
        messages.append(AIMessage(content=final_output))
    return messages


def invoke_episode(
    model_config: dict[str, str],
    framework_versions: dict[str, str],
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
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    state = copy.deepcopy(initial_state)
    context = EpisodeContext(
        model_config,
        condition,
        payment_id,
        seed,
        phase,
        run_index,
        raw_requests_path,
        records,
        state,
        specs,
    )
    tools = build_tools(context)
    task = TASK_TEMPLATE.format(payment_id=payment_id)
    chat_history: list[Any] = []
    if condition == "D":
        source_tool = next(tool for tool in tools if tool.name == "get_payment_record")
        source_call_id = f"prefilled-source-{payment_id}-{seed}"
        source_output = source_tool.invoke({"payment_id": payment_id})
        chat_history.extend(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "get_payment_record",
                            "args": {"payment_id": payment_id},
                            "id": source_call_id,
                            "type": "tool_call",
                        }
                    ],
                ),
                ToolMessage(
                    content=str(source_output),
                    name="get_payment_record",
                    tool_call_id=source_call_id,
                ),
            ]
        )
    initial_messages = [*chat_history, HumanMessage(content=task)]
    context.messages = initial_messages
    agent_tools = [tool for tool in tools if condition != "D" or tool.name == "issue_refund"]
    try:
        agent, agent_tools = build_agent(context, tools)
        result = agent.invoke(
            {"input": task, "chat_history": chat_history},
        )
        messages = collect_agent_messages(result, initial_messages, payment_id, seed)
    except Exception as exc:
        if "seed" in str(exc).lower() and context.seed_supported is not False and not state["refunds"]:
            log_seed_retry(context, agent_tools, exc)
            context.seed_supported = False
            state.clear()
            state.update(copy.deepcopy(initial_state))
            context.state = state
            context.source_called = condition == "D"
            context.source_payment_ids = [payment_id] if condition == "D" else []
            context.errors.clear()
            context.checker_log.clear()
            context.request_id = None
            context.reported_model = model_config["model"]
            try:
                agent, agent_tools = build_agent(context, tools)
                result = agent.invoke(
                    {"input": task, "chat_history": chat_history},
                )
                messages = collect_agent_messages(result, initial_messages, payment_id, seed)
            except Exception as retry_exc:
                context.errors.append(f"{type(retry_exc).__name__}: {retry_exc}")
                messages = context.messages
        else:
            context.errors.append(f"{type(exc).__name__}: {exc}")
            messages = context.messages
    context.messages = messages
    log_agent_requests(messages, len(initial_messages), context, agent_tools)
    tool_trace, source_payload, source_actor = build_tool_trace(messages, context)
    source_condition = "B" if condition == "E" else condition
    episode = {
        "phase": phase,
        "run_order": run_index,
        "condition": condition,
        "payment_id": payment_id,
        "seed": seed,
        "model_version": context.reported_model,
        "configured_model": model_config["model"],
        "backend": "langchain-openai+langchain-agents",
        "framework": "LangChain create_openai_tools_agent + AgentExecutor",
        "source_tool_prefilled_by_harness": condition == "D",
        "framework_versions": framework_versions,
        "request_id": context.request_id,
        "seed_supported": context.seed_supported,
        "config_fingerprint": config_fingerprint,
        "input_actor": source_actor or records[payment_id]["records"][source_condition]["actor"],
        "input_payload": records[payment_id]["records"][source_condition]["payload_json"],
        "source_tool_output": source_payload,
        "tool_trace": tool_trace,
        "checker_log": context.checker_log,
        "messages": [serialize_message(message) for message in messages],
        "initial_refund_count": len(initial_state["refunds"]),
        "final_state": {"refunds": copy.deepcopy(state["refunds"])},
        "error": "; ".join(context.errors) if context.errors else None,
        "created_at_utc": now_utc(),
    }
    return episode


def create_config(model_config: dict[str, str], framework_versions: dict[str, str], order_seed: int) -> dict[str, Any]:
    return {
        "backend": "langchain-openai+langchain-agents",
        "framework": "LangChain create_openai_tools_agent + AgentExecutor",
        "agent": "langchain.agents.create_openai_tools_agent",
        "framework_versions": framework_versions,
        "model": model_config["model"],
        "base_url": model_config["base_url"],
        "reasoning_effort": model_config.get("reasoning_effort") or None,
        "temperature": MODEL_TEMPERATURE,
        "top_p": TOP_P,
        "max_tokens": MAX_TOKENS,
        "max_steps": MAX_STEPS,
        "parallel_tool_calls": False,
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
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


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
        raise RuntimeError("快筛与当前模型/提示词/工具/agent 框架配置不一致，不能复用结果。")


def require_ab_baseline(output_dir: Path, current_fingerprint: str) -> None:
    summary_path = output_dir / "screen_ab_summary.json"
    if not summary_path.exists():
        raise RuntimeError("先运行 A/B 阶段并执行 evaluate.py evaluate --phase screen-ab。")
    summary = read_json(summary_path)
    if not summary.get("advance_to_cd"):
        raise RuntimeError("A 条件底座未通过；暂停 C/D 阶段，先排查 harness 或模型环境。")
    if summary.get("config_fingerprint") != current_fingerprint:
        raise RuntimeError("A/B 与当前模型/提示词/工具/agent 框架配置不一致，不能继续复用结果。")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("smoke", "screen-ab", "screen-cd", "full"), required=True)
    parser.add_argument("--base-url", help="OpenAI-compatible API base URL; key is read from environment")
    parser.add_argument("--model", help="Pinned model snapshot/name")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--order-seed", type=int, default=RUN_ORDER_SEED)
    args = parser.parse_args()
    output_dir: Path = args.out.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        framework_versions = available_framework()
        model_config = make_model_config(args)
    except EnvironmentNotReady as exc:
        write_json(
            output_dir / "environment_status.json",
            {"status": "environment_not_ready", "reason": str(exc), "checked_at_utc": now_utc(), "conclusion": None},
        )
        print(f"环境未就绪：{exc}", file=sys.stderr)
        return 2

    config = create_config(model_config, framework_versions, args.order_seed)
    config_id = fingerprint(config)
    write_json(
        output_dir / "model_config.json",
        {**config, "config_fingerprint": config_id, "captured_at_utc": now_utc()},
    )
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
    records = {row["payment_id"]: row for row in records_data["records"]}
    state_template = read_json(ROOT / "sandbox_state.json")
    specs = tool_specs()
    episode_file = output_dir / ("smoke_episodes.jsonl" if args.phase == "smoke" else "episodes.jsonl")
    request_file = output_dir / ("smoke_raw_requests.jsonl" if args.phase == "smoke" else "raw_requests.jsonl")
    existing = load_existing(episode_file)

    if args.phase == "full":
        for row in existing:
            if row.get("config_fingerprint") != config_id:
                print("输出目录含不同配置的旧 episode；请改用新的 --out 目录。", file=sys.stderr)
                return 4
        existing_keys = {
            (row["condition"], row["payment_id"], row["seed"])
            for row in existing
            if row["phase"] in ("screen-ab", "screen-cd", "full")
        }
    else:
        existing_keys = set()
        for row in existing:
            if row.get("phase") != args.phase:
                continue
            if row.get("config_fingerprint") != config_id:
                print("输出目录含不同配置的旧 episode；请改用新的 --out 目录。", file=sys.stderr)
                return 4
            existing_keys.add((row["condition"], row["payment_id"], row["seed"]))

    schedule = [run for run in plan_runs(args.phase, args.order_seed) if run not in existing_keys]
    run_counter = len(existing)
    print(f"phase={args.phase} framework=LangChain AgentExecutor model={model_config['model']} planned_new_episodes={len(schedule)}")
    for new_index, (condition, payment_id, seed) in enumerate(schedule, start=1):
        run_counter += 1
        episode = invoke_episode(
            model_config,
            framework_versions,
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
            f"{new_index:03d}/{len(schedule):03d} {condition} {payment_id} seed={seed} "
            f"calls={len(episode['tool_trace'])} error={episode['error'] or '-'}"
        )

    status = {
        "status": "episodes_written",
        "phase": args.phase,
        "framework": "LangChain create_openai_tools_agent + AgentExecutor",
        "new_episodes": len(schedule),
        "episode_file": str(episode_file),
        "model": model_config["model"],
        "config_fingerprint": config_id,
        "completed_at_utc": now_utc(),
    }
    write_json(output_dir / f"{args.phase}_run_status.json", status)
    print(f"完成：新增 {len(schedule)} 个 episode。下一步运行 evaluate.py --phase {args.phase}。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
