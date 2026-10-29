#!/usr/bin/env python3
"""
s16: Team Protocols — request-response protocol + request_id + dispatch + state machine.

Run:  python s16_team_protocols/code.py
Need: pip install anthropic python-dotenv + .env with ANTHROPIC_API_KEY

Changes from s15:
  - ProtocolState dataclass (request_id, type, sender, status, created_at)
  - pending_requests dict: tracks in-flight protocol requests
  - dispatch_message: routes incoming messages by type to handlers
  - request_shutdown: Lead sends shutdown protocol request
  - request_plan: Lead asks teammate to submit plan
  - handle_shutdown_request / handle_plan_response: teammate receives & responds
  - match_response: Lead correlates response to request via request_id (with type validation)
  - Teammate idle loop: waits for inbox messages instead of exiting after 10 rounds
  - Unified consume_lead_inbox: protocol routing + injection into history
  - 3 new Lead tools: request_shutdown, request_plan, review_plan
  - 1 new teammate tool: submit_plan

ASCII flow:
  Lead: BUS.send("shutdown_request", {request_id}) ──────→ teammate inbox
  Teammate: dispatch → handler → BUS.send("shutdown_response", {request_id}) ─→ Lead inbox
  Lead: consume_lead_inbox → match_response(request_id) → pending_requests[req_id].status = approved
"""

import os, subprocess, json, time, random, threading
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict, field

try:
    import readline
    readline.parse_and_bind('set bind-tty-special-chars off')
except ImportError:
    pass

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv(override=True)
if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

WORKDIR = Path.cwd()
MEMORY_DIR = WORKDIR / ".memory"
MEMORY_INDEX = MEMORY_DIR / "MEMORY.md"
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.environ["MODEL_ID"]

# ── Task System (from s12, synced) ──

TASKS_DIR = WORKDIR / ".tasks"
TASKS_DIR.mkdir(exist_ok=True)


@dataclass
class Task:
    id: str
    subject: str
    description: str
    status: str          # pending | in_progress | completed
    owner: str | None
    blockedBy: list[str]


def _task_path(task_id: str) -> Path:
    return TASKS_DIR / f"{task_id}.json"


def create_task(subject: str, description: str = "",
                blockedBy: list[str] | None = None) -> Task:
    task = Task(
        id=f"task_{int(time.time())}_{random.randint(0, 9999):04d}",
        subject=subject, description=description,
        status="pending", owner=None,
        blockedBy=blockedBy or [],
    )
    save_task(task)
    return task


def save_task(task: Task):
    _task_path(task.id).write_text(json.dumps(asdict(task), indent=2))


def load_task(task_id: str) -> Task:
    return Task(**json.loads(_task_path(task_id).read_text()))


def list_tasks() -> list[Task]:
    return [Task(**json.loads(p.read_text()))
            for p in sorted(TASKS_DIR.glob("task_*.json"))]


def get_task(task_id: str) -> str:
    """Return full task details as JSON."""
    task = load_task(task_id)
    return json.dumps(asdict(task), indent=2)


def can_start(task_id: str) -> bool:
    """Check if all blockedBy dependencies are completed.
    Missing dependencies are treated as blocked."""
    task = load_task(task_id)
    for dep_id in task.blockedBy:
        if not _task_path(dep_id).exists():
            return False
        if load_task(dep_id).status != "completed":
            return False
    return True


def claim_task(task_id: str, owner: str = "agent") -> str:
    task = load_task(task_id)
    if task.status != "pending":
        return f"Task {task_id} is {task.status}, cannot claim"
    if not can_start(task_id):
        deps = [d for d in task.blockedBy
                if not _task_path(d).exists() or load_task(d).status != "completed"]
        return f"Blocked by: {deps}"
    task.owner = owner
    task.status = "in_progress"
    save_task(task)
    print(f"  \033[36m[claim] {task.subject} → in_progress (owner: {owner})\033[0m")
    return f"Claimed {task.id} ({task.subject})"


def complete_task(task_id: str) -> str:
    task = load_task(task_id)
    if task.status != "in_progress":
        return f"Task {task_id} is {task.status}, cannot complete"
    task.status = "completed"
    save_task(task)
    unblocked = [t.subject for t in list_tasks()
                 if t.status == "pending" and t.blockedBy and can_start(t.id)]
    print(f"  \033[32m[complete] {task.subject} ✓\033[0m")
    msg = f"Completed {task.id} ({task.subject})"
    if unblocked:
        msg += f"\nUnblocked: {', '.join(unblocked)}"
        print(f"  \033[33m[unblocked] {', '.join(unblocked)}\033[0m")
    return msg


# ── Prompt Assembly (from s10, synced) ──

PROMPT_SECTIONS = {
    "identity": "You are a coding agent. Act, don't explain.",
    "tools": "Available tools: bash, read_file, write_file, "
             "get_task, create_task, list_tasks, claim_task, complete_task, "
             "spawn_teammate, send_message, check_inbox, "
             "request_shutdown, request_plan, review_plan.",
    "workspace": f"Working directory: {WORKDIR}",
    "memory": "Relevant memories are injected below when available.",
}


def assemble_system_prompt(context: dict) -> str:
    sections = [PROMPT_SECTIONS["identity"],
                PROMPT_SECTIONS["tools"],
                PROMPT_SECTIONS["workspace"]]
    memories = context.get("memories", "")
    if memories:
        sections.append(f"Relevant memories:\n{memories}")
    return "\n\n".join(sections)


_last_context_key, _last_prompt = None, None


def get_system_prompt(context: dict) -> str:
    global _last_context_key, _last_prompt
    key = json.dumps(context, sort_keys=True, ensure_ascii=False, default=str)
    if key == _last_context_key and _last_prompt:
        return _last_prompt
    _last_context_key = key
    _last_prompt = assemble_system_prompt(context)
    return _last_prompt


# ── Tools ──

def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def run_bash(command: str, run_in_background: bool = False) -> str:
    # run_in_background is handled by agent_loop dispatch, not here
    try:
        r = subprocess.run(command, shell=True, cwd=WORKDIR,
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"


def run_read(path: str, limit: int | None = None) -> str:
    try:
        lines = safe_path(path).read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"


def run_write(path: str, content: str) -> str:
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"


# Task tools

def run_create_task(subject: str, description: str = "",
                    blockedBy: list[str] | None = None) -> str:
    task = create_task(subject, description, blockedBy)
    deps = f" (blockedBy: {', '.join(blockedBy)})" if blockedBy else ""
    print(f"  \033[34m[create] {task.subject}{deps}\033[0m")
    return f"Created {task.id}: {task.subject}{deps}"


def run_list_tasks() -> str:
    tasks = list_tasks()
    if not tasks:
        return "No tasks. Use create_task to add some."
    lines = []
    for t in tasks:
        icon = {"pending": "○", "in_progress": "●",
                "completed": "✓"}.get(t.status, "?")
        deps = f" (blockedBy: {', '.join(t.blockedBy)})" if t.blockedBy else ""
        owner = f" [{t.owner}]" if t.owner else ""
        lines.append(f"  {icon} {t.id}: {t.subject} "
                     f"[{t.status}]{owner}{deps}")
    return "\n".join(lines)


def run_get_task(task_id: str) -> str:
    try:
        return get_task(task_id)
    except FileNotFoundError:
        return f"Error: Task {task_id} not found"


def run_claim_task(task_id: str) -> str:
    return claim_task(task_id, owner="agent")


def run_complete_task(task_id: str) -> str:
    return complete_task(task_id)


# ── Background Tasks (from s13, synced) ──

_bg_counter = 0
background_tasks: dict[str, dict] = {}
background_results: dict[str, str] = {}
background_lock = threading.Lock()


def is_slow_operation(tool_name: str, tool_input: dict) -> bool:
    """Fallback heuristic: commands likely to take > 30s."""
    if tool_name != "bash":
        return False
    cmd = tool_input.get("command", "").lower()
    slow_keywords = ["install", "build", "test", "deploy", "compile",
                     "docker build", "pip install", "npm install",
                     "cargo build", "pytest", "make"]
    return any(kw in cmd for kw in slow_keywords)


def should_run_background(tool_name: str, tool_input: dict) -> bool:
    """Model explicit request takes priority; fallback to heuristic."""
    if tool_input.get("run_in_background"):
        return True
    return is_slow_operation(tool_name, tool_input)


def start_background_task(block) -> str:
    """Run tool in a daemon thread. Returns background task ID."""
    global _bg_counter
    _bg_counter += 1
    bg_id = f"bg_{_bg_counter:04d}"
    cmd = block.input.get("command", block.name)

    def worker():
        result = execute_tool(block)
        with background_lock:
            background_tasks[bg_id]["status"] = "completed"
            background_results[bg_id] = result

    with background_lock:
        background_tasks[bg_id] = {
            "tool_use_id": block.id,
            "command": cmd,
            "status": "running",
        }
    threading.Thread(target=worker, daemon=True).start()
    print(f"  \033[33m[background] dispatched {bg_id}: {cmd[:40]}\033[0m")
    return bg_id


def collect_background_results() -> list[str]:
    """Collect completed background results as task_notification messages."""
    with background_lock:
        ready_ids = [bid for bid, task in background_tasks.items()
                     if task["status"] == "completed"]
    notifications = []
    for bg_id in ready_ids:
        with background_lock:
            task = background_tasks.pop(bg_id)
            output = background_results.pop(bg_id, "")
        summary = output[:200] if len(output) > 200 else output
        notifications.append(
            f"<task_notification>\n"
            f"  <task_id>{bg_id}</task_id>\n"
            f"  <status>completed</status>\n"
            f"  <command>{task['command']}</command>\n"
            f"  <summary>{summary}</summary>\n"
            f"</task_notification>")
        print(f"  \033[32m[background done] {bg_id}: "
              f"{task['command'][:40]} ({len(output)} chars)\033[0m")
    return notifications


# ── MessageBus (from s15) ──

MAILBOX_DIR = WORKDIR / ".mailboxes"
MAILBOX_DIR.mkdir(exist_ok=True)


class MessageBus:
    """File-based message bus. Each agent has a .jsonl inbox.
    Read is destructive: read_text + unlink (consumes messages).
    Teaching version: no file locking; real CC uses proper-lockfile."""

    def send(self, from_agent: str, to_agent: str, content: str,
             msg_type: str = "message", metadata: dict = None):
        msg = {"from": from_agent, "to": to_agent,
               "content": content, "type": msg_type,
               "ts": time.time(), "metadata": metadata or {}}
        inbox = MAILBOX_DIR / f"{to_agent}.jsonl"
        with open(inbox, "a") as f:
            f.write(json.dumps(msg) + "\n")
        print(f"  \033[33m[bus] {from_agent} → {to_agent}: "
              f"({msg_type}) {content[:50]}\033[0m")

    def read_inbox(self, agent: str) -> list[dict]:
        inbox = MAILBOX_DIR / f"{agent}.jsonl"
        if not inbox.exists():
            return []
        msgs = [json.loads(line) for line in inbox.read_text().splitlines()
                if line.strip()]
        inbox.unlink()  # consume: read + delete
        return msgs

    def peek_inbox(self, agent: str) -> list[dict]:
        # FIX #2/#3: 非销毁式读取。teammate 改用它来“看”信箱，
        # 这样 .jsonl 文件永远不会被 unlink。
        # 旧 bug 的根因：模型被告知“去读 inbox”，于是 read_file 那个 .jsonl，
        # 但销毁式读取已经把它删了 -> 读到错误 -> 绕圈。peek 让文件始终存在。
        inbox = MAILBOX_DIR / f"{agent}.jsonl"
        if not inbox.exists():
            return []
        return [json.loads(line) for line in inbox.read_text().splitlines()
                if line.strip()]


BUS = MessageBus()
active_teammates: dict[str, bool] = {} # mmc 正在活跃的teammate（线程）

# FIX #2/#3: 每个 teammate 的“已读游标”，记录已经投递给它的信箱消息条数。
# 配合非销毁式 peek_inbox，teammate 永远不会对 .jsonl 做 unlink，
# 因此模型不可能再去“读一个已被删除的文件”。
SEEN: dict[str, int] = {}


def new_messages(agent: str):
    """只返回该 teammate 还没见过的新消息，以及消息总数（用于推进 SEEN 游标）。
    使用非销毁式读取，不会删除文件。"""
    all_msgs = BUS.peek_inbox(agent)
    seen = SEEN.get(agent, 0)
    return all_msgs[seen:], len(all_msgs)

# ── Protocol State (s16 new) ──

@dataclass
class ProtocolState:
    request_id: str
    type: str       # "shutdown" | "plan_approval"
    sender: str
    target: str
    status: str     # pending | approved | rejected
    payload: str    # plan text or shutdown reason
    created_at: float = field(default_factory=time.time)


pending_requests: dict[str, ProtocolState] = {} # mmc 协议相关的消息（"shutdown" | "plan_approval"）投递信箱时，在这里保存一个索引


def new_request_id() -> str:
    return f"req_{random.randint(0, 999999):06d}"


def match_response(response_type: str, request_id: str, approve: bool):
    """Correlate a response to the original request via request_id.
    Validates that response_type matches the request type."""
    state = pending_requests.get(request_id)
    if not state:
        print(f"  \033[31m[protocol] unknown request_id: {request_id}\033[0m")
        return
    # Validate response type matches request type
    if state.type == "shutdown" and response_type != "shutdown_response":
        print(f"  \033[31m[protocol] type mismatch: expected shutdown_response, "
              f"got {response_type}\033[0m")
        return
    if state.type == "plan_approval" and response_type != "plan_approval_response":
        print(f"  \033[31m[protocol] type mismatch: expected plan_approval_response, "
              f"got {response_type}\033[0m")
        return
    if state.status != "pending":
        print(f"  \033[33m[protocol] {request_id} already {state.status}, "
              f"ignoring duplicate\033[0m")
        return
    state.status = "approved" if approve else "rejected"
    icon = "✓" if approve else "✗"
    color = "32" if approve else "31"
    print(f"  \033[{color}m[protocol] {state.type} {icon} "
          f"({request_id}: {state.status})\033[0m")


# ── Unified Lead Inbox Consumer (s16 fix) ──
# Both check_inbox tool and main loop call this function.
# Protocol responses are routed via match_response before returning.

def consume_lead_inbox(route_protocol: bool = True) -> list[dict]:
    """Read Lead's inbox. Route protocol responses, return all messages.
    Called by both run_check_inbox() and main loop to avoid
    messages being consumed without protocol routing."""
    msgs = BUS.read_inbox("lead")
    if not msgs:
        return []
    if route_protocol: # mmc 本代码中的调用，route_protocol都为true
        for msg in msgs:
            meta = msg.get("metadata", {})
            req_id = meta.get("request_id", "")
            msg_type = msg.get("type", "")
            if req_id and msg_type.endswith("_response"):
                approve = meta.get("approve", False)
                match_response(msg_type, req_id, approve) # mmc consume_lead_inbox（L427-434）的路由规则只有一条：带 req_id 且 type 以 _response 结尾的消息才自动 match_response。
    return msgs


# ── Teammate Thread (s16: idle loop + dispatch) ──

def spawn_teammate_thread(name: str, role: str, prompt: str) -> str:
    """Spawn a teammate agent in a background thread.
    Uses idle loop: after each LLM turn, waits for inbox messages
    (shutdown_request, new task) instead of exiting."""
    if name in active_teammates:
        return f"Teammate '{name}' already exists"

    # FIX #1: 明确指令通道，杜绝模型去读 .mailboxes/*.jsonl 文件。
    # 旧措辞 "via your inbox" 暗示了一个物理文件，导致模型去 read_file 一个
    # 已经被销毁式读取删掉的文件。新措辞：指令直接出现在 <inbox> 块里，看到就
    # 执行；想自查就调 check_inbox 工具，绝不 read_file 信箱文件。
    system = (f"You are '{name}', a {role}. "
              f"Use tools to complete tasks.\n"
              f"INSTRUCTIONS: New instructions from the Lead appear inside "
              f"<inbox>...</inbox> blocks in this conversation. When you see an "
              f"<inbox> block, treat its content as your instruction and ACT ON "
              f"IT DIRECTLY (e.g. call write_file) — do NOT read any file and do "
              f"NOT run bash to look for an inbox file.\n"
              f"If you simply want to check whether new messages arrived, call "
              f"the check_inbox tool. Never use read_file on paths under "
              f".mailboxes/.\n"
              f"Check inbox for protocol messages (shutdown_request, etc).")

    def handle_inbox_message(name: str, msg: dict, messages: list) -> bool: # mmc 处理协议相关的消息，有关机请求和计划同意通知两种
        """Dispatch incoming protocol messages by type.
        Returns True if teammate should stop."""
        msg_type = msg.get("type", "message")
        meta = msg.get("metadata", {})
        req_id = meta.get("request_id", "")

        if msg_type == "shutdown_request": # mmc 关机请求
            BUS.send(name, "lead", "Shutting down gracefully.",
                     "shutdown_response",
                     {"request_id": req_id, "approve": True}) # mmc metadata里放自己回复的那条request的id，以及是否同意
            print(f"  \033[35m[protocol] {name} approved shutdown "
                  f"({req_id})\033[0m")
            return True  # stop the loop

        if msg_type == "plan_approval_response": # mmc 计划同意通知
            approve = meta.get("approve", False)
            if approve:
                messages.append({"role": "user",
                    "content": f"[Plan approved] Proceed with the task."})
            else:
                messages.append({"role": "user",
                    "content": f"[Plan rejected] Feedback: {msg['content']}"})

        return False  # continue

    def run():
        messages = [{"role": "user", "content": prompt}]
        sub_tools = [
            {"name": "bash", "description": "Run a shell command.",
             "input_schema": {"type": "object",
                              "properties": {"command": {"type": "string"}},
                              "required": ["command"]}},
            {"name": "read_file", "description": "Read file.",
             "input_schema": {"type": "object",
                              "properties": {"path": {"type": "string"}},
                              "required": ["path"]}},
            {"name": "write_file", "description": "Write file.",
             "input_schema": {"type": "object",
                              "properties": {"path": {"type": "string"},
                                             "content": {"type": "string"}},
                              "required": ["path", "content"]}},
            {"name": "send_message",
             "description": "Send message to another agent.",
             "input_schema": {"type": "object",
                              "properties": {"to": {"type": "string"},
                                             "content": {"type": "string"}},
                              "required": ["to", "content"]}},
            {"name": "submit_plan",
             "description": "Submit a plan for Lead approval.",
             "input_schema": {"type": "object",
                              "properties": {"plan": {"type": "string"}},
                              "required": ["plan"]}},
            {"name": "check_inbox",  # FIX #2: 正规的“查收件箱”手段
             "description": "Check for new messages from the Lead or other agents. "
                            "Returns unseen messages and does NOT read any file. "
                            "Use this instead of reading .mailboxes/*.jsonl.",
             "input_schema": {"type": "object",
                              "properties": {}}},
        ]
        sub_handlers = {
            "bash": run_bash, "read_file": run_read, "write_file": run_write,
            "send_message": lambda to, content: (BUS.send(name, to, content),
                                                  "Sent")[1],
            "submit_plan": lambda plan: _teammate_submit_plan(name, plan), # mmc 注意
            "check_inbox": lambda: _teammate_check_inbox(),  # FIX #2
        }

        def _teammate_check_inbox():
            # FIX #2: 安全、正规的查收件箱方式。使用非销毁式 peek + SEEN 游标，
            # 永远不可能读到“已被删除的文件”。这正是要替代模型旧习惯
            # （read_file .mailboxes/*.jsonl）的手段。
            msgs, total = new_messages(name)
            SEEN[name] = total
            if not msgs:
                return "No new messages."
            lines = [f"From {m.get('from')}: {m.get('content')}" for m in msgs]
            return "New messages:\n" + "\n".join(lines)

        shutdown_requested = False
        first_done = False   # FIX #3: 允许第一轮处理初始 prompt，之后再做“空转”门控
        mid_task = False      # FIX #3: 模型正处于多步任务中（上一轮请求了工具）时为 True
        while not shutdown_requested:
            # FIX #3: 改用非销毁式 new_messages（peek + SEEN 游标），不再 unlink 文件
            inbox, total = new_messages(name)
            should_stop = False
            non_protocol = []   # FIX #4: 每轮重新初始化，避免旧消息跨轮累积重复注入
            for msg in inbox:
                if msg.get("type") in ("shutdown_request", "plan_approval_response"):
                    should_stop = handle_inbox_message(name, msg, messages) # mmc 处理协议相关的消息，有关机请求和计划同意通知两种
                    if should_stop:
                        break
                else:
                    non_protocol.append(msg) # mmc 非协议相关的消息，先收集起来
            if should_stop:
                shutdown_requested = True
                SEEN[name] = total   # 推进游标
                break
            if non_protocol:
                inbox_json = json.dumps(non_protocol)
                messages.append({"role": "user",
                    "content": "<inbox>" + inbox_json + "</inbox>"}) # mmc 非协议相关的消息，发给大模型处理

            # FIX #3: 只有“有新消息要处理”或“正在多步任务中”时才调 LLM；
            # 否则 WAIT，不要拿着空信箱去调 LLM（旧 bug：模型被空转调用后去
            # 探索文件系统、并读那个已被删除的 .jsonl）。
            if not non_protocol and not inbox and first_done and not mid_task:
                time.sleep(1)
                continue
            first_done = True
            SEEN[name] = total   # 推进游标：本轮消息已投递

            # LLM turn
            try: # mmc 由此可以看到这里的外层循环就是react循环，也就是之前代码中的内层循环agent_loop
                response = client.messages.create(
                    model=MODEL, system=system, messages=messages[-20:],
                    tools=sub_tools, max_tokens=8000)
            except Exception:
                break

            messages.append({"role": "assistant", "content": response.content})
            mid_task = (response.stop_reason == "tool_use")  # FIX #3: 记录是否处于多步任务
            if response.stop_reason != "tool_use": # mmc 本来应该直接从agent_loop中return，但是这里并没有，用了一个循环来等待信箱里的新消息，直到读到shutdown_request
                # Idle: wait for inbox messages instead of exiting
                # Real CC sends idle_notification to Lead here
                while not shutdown_requested:
                    time.sleep(1) # mmc 每一秒询问一次
                    non_protocol = []   # FIX #4: 空闲循环每轮重新初始化，避免重复注入
                    inbox, total = new_messages(name)  # FIX #3: 非销毁式
                    delivered = False
                    for msg in inbox:
                        if msg.get("type") in ("shutdown_request", "plan_approval_response"):
                            should_stop = handle_inbox_message(name, msg, messages) # mmc 如果是plan_approval_response，会追加messages
                            delivered = True
                            if should_stop:
                                shutdown_requested = True # mmc shutdown_requested = True时，也会退出外层循环
                                break
                        else:
                            non_protocol.append(msg)
                            delivered = True
                    SEEN[name] = total   # FIX #4: 空闲循环也推进游标，避免协议消息被重复追加
                    if shutdown_requested:
                        break
                    if non_protocol:
                        inbox_json = json.dumps(non_protocol)
                        messages.append({"role": "user", # mmc non_protocol也追加到messages
                            "content": "<inbox>" + inbox_json + "</inbox>"})
                        break  # back to LLM turn with new messages # mmc 走到这里的话，相当于之前的代码里用户输入了新消息，需要开启新一轮agent_loop来处理这个新消息
                    if delivered:
                        # 协议消息（如 plan approved）已追加 -> 重新跑一轮 LLM 处理它
                        break

            # Execute tool calls # mmc 如果进了上面的if response.stop_reason != "tool_use"，那这里的results就为空，因为没有工具调用
            results = []
            for block in response.content:
                if block.type == "tool_use":
                    handler = sub_handlers.get(block.name)
                    output = handler(**block.input) if handler else "Unknown"
                    results.append({"type": "tool_result",
                                    "tool_use_id": block.id,
                                    "content": str(output)})
            messages.append({"role": "user", "content": results})

        # Send final summary to Lead # mmc shutdown了
        summary = "Done."
        for msg in reversed(messages):
            if msg["role"] == "assistant" and isinstance(msg["content"], list):
                for b in msg["content"]:
                    if getattr(b, "type", None) == "text":
                        summary = b.text # mmc 找到最后一条大模型的消息，作为总结发给lead
                        break
                else:
                    continue
                break
        BUS.send(name, "lead", summary, "result")
        active_teammates.pop(name, None)
        print(f"  \033[32m[teammate] {name} finished\033[0m")

    active_teammates[name] = True
    threading.Thread(target=run, daemon=True).start()
    print(f"  \033[36m[teammate] {name} spawned as {role}\033[0m")
    return f"Teammate '{name}' spawned as {role}"


def _teammate_submit_plan(from_name: str, plan: str) -> str:
    """Teammate submits a plan to Lead for approval.

    Note: This is a protocol-level request, not a code-level gate.
    After submitting, the teammate's thread continues running — it can
    still call bash/write/etc. Real enforcement relies on the model
    waiting for the approval response before acting. Code-level tool
    gating would require blocking the teammate's tool dispatch until
    approval arrives.
    """
    req_id = new_request_id() # mmc 协议消息，有req_id
    pending_requests[req_id] = ProtocolState(
        request_id=req_id, type="plan_approval",
        sender=from_name, target="lead",
        status="pending", payload=plan)
    BUS.send(from_name, "lead", plan,
             "plan_approval_request",
             {"request_id": req_id})
    return f"Plan submitted ({req_id}). Waiting for approval..."


# ── Lead Protocol Tools (s16 new) ──

def run_request_shutdown(teammate: str) -> str:
    req_id = new_request_id() # mmc 协议消息，有req_id
    pending_requests[req_id] = ProtocolState(
        request_id=req_id, type="shutdown",
        sender="lead", target=teammate,
        status="pending", payload="")
    BUS.send("lead", teammate, "Please shut down gracefully.",
             "shutdown_request",
             {"request_id": req_id})
    print(f"  \033[35m[protocol] shutdown_request → {teammate} "
          f"({req_id})\033[0m")
    return f"Shutdown request sent to {teammate} (req: {req_id})"


def run_request_plan(teammate: str, task: str) -> str:
    """Lead asks a teammate to submit a plan for a task."""
    BUS.send("lead", teammate, f"Please submit a plan for: {task}",
             "message")
    return f"Asked {teammate} to submit a plan"


def run_review_plan(request_id: str, approve: bool, feedback: str = "") -> str: # mmc 大模型读信箱里的消息，里面有plan和它的request_id，然后调用这个工具，填写它要回复的plan的request_id
    state = pending_requests.get(request_id)
    if not state:
        return f"Request {request_id} not found"
    if state.status != "pending":
        return f"Request {request_id} already {state.status}"
    state.status = "approved" if approve else "rejected"
    BUS.send("lead", state.sender, feedback or ("Approved" if approve else "Rejected"),
             "plan_approval_response",
             {"request_id": request_id, "approve": approve}) # mmc metadata：协议消息的相关信息
    icon = "✓" if approve else "✗"
    print(f"  \033[32m[protocol] plan {icon} ({request_id})\033[0m")
    return f"Plan {'approved' if approve else 'rejected'} ({request_id})"


# ── Other Lead Tool Handlers ──

def run_spawn_teammate(name: str, role: str, prompt: str) -> str:
    return spawn_teammate_thread(name, role, prompt)


def run_send_message(to: str, content: str) -> str:
    BUS.send("lead", to, content)
    return f"Sent to {to}"


def run_check_inbox() -> str:
    """Check Lead's inbox. Routes protocol responses via match_response."""
    msgs = consume_lead_inbox(route_protocol=True)
    if not msgs:
        return "(inbox empty)"
    lines = []
    for m in msgs:
        meta = m.get("metadata", {})
        req_id = meta.get("request_id", "")
        tag = f" [{m['type']} req:{req_id}]" if req_id else f" [{m['type']}]"
        lines.append(f"  [{m['from']}]{tag} {m['content'][:200]}")
    return "\n".join(lines)


# ── Tool Dispatch ──

def execute_tool(block) -> str:
    """Execute a tool call block, return output."""
    handler = {
        "bash": run_bash, "read_file": run_read, "write_file": run_write,
        "create_task": run_create_task, "list_tasks": run_list_tasks,
        "get_task": run_get_task, "claim_task": run_claim_task,
        "complete_task": run_complete_task,
        "spawn_teammate": run_spawn_teammate,
        "send_message": run_send_message, "check_inbox": run_check_inbox,
        "request_shutdown": run_request_shutdown,
        "request_plan": run_request_plan, "review_plan": run_review_plan,
    }.get(block.name)
    if handler:
        return handler(**block.input)
    return f"Unknown tool: {block.name}"


# ── Tool Definitions ──

TOOLS = [
    {"name": "bash", "description": "Run a shell command.",
     "input_schema": {"type": "object",
                      "properties": {
                          "command": {"type": "string"},
                          "run_in_background": {"type": "boolean"}},
                      "required": ["command"]}},
    {"name": "read_file", "description": "Read file contents.",
     "input_schema": {"type": "object",
                      "properties": {"path": {"type": "string"},
                                     "limit": {"type": "integer"}},
                      "required": ["path"]}},
    {"name": "write_file", "description": "Write content to a file.",
     "input_schema": {"type": "object",
                      "properties": {"path": {"type": "string"},
                                     "content": {"type": "string"}},
                      "required": ["path", "content"]}},
    {"name": "create_task",
     "description": "Create a new task with optional blockedBy dependencies.",
     "input_schema": {"type": "object",
                      "properties": {
                          "subject": {"type": "string"},
                          "description": {"type": "string"},
                          "blockedBy": {"type": "array",
                                        "items": {"type": "string"}}},
                      "required": ["subject"]}},
    {"name": "list_tasks",
     "description": "List all tasks with status, owner, and dependencies.",
     "input_schema": {"type": "object", "properties": {},
                      "required": []}},
    {"name": "get_task",
     "description": "Get full details of a specific task by ID.",
     "input_schema": {"type": "object",
                      "properties": {"task_id": {"type": "string"}},
                      "required": ["task_id"]}},
    {"name": "claim_task",
     "description": "Claim a pending task. Sets owner, changes status to in_progress.",
     "input_schema": {"type": "object",
                      "properties": {"task_id": {"type": "string"}},
                      "required": ["task_id"]}},
    {"name": "complete_task",
     "description": "Complete an in-progress task. Reports unblocked downstream tasks.",
     "input_schema": {"type": "object",
                      "properties": {"task_id": {"type": "string"}},
                      "required": ["task_id"]}},
    {"name": "spawn_teammate",
     "description": "Spawn a teammate agent in a background thread.",
     "input_schema": {"type": "object",
                      "properties": {
                          "name": {"type": "string"},
                          "role": {"type": "string"},
                          "prompt": {"type": "string"}},
                      "required": ["name", "role", "prompt"]}},
    {"name": "send_message",
     "description": "Send message to a teammate via MessageBus.",
     "input_schema": {"type": "object",
                      "properties": {"to": {"type": "string"},
                                     "content": {"type": "string"}},
                      "required": ["to", "content"]}},
    {"name": "check_inbox",
     "description": "Check Lead's inbox. Routes protocol responses automatically.",
     "input_schema": {"type": "object", "properties": {},
                      "required": []}},
    {"name": "request_shutdown",
     "description": "Request a teammate to shut down gracefully.",
     "input_schema": {"type": "object",
                      "properties": {"teammate": {"type": "string"}},
                      "required": ["teammate"]}},
    {"name": "request_plan",
     "description": "Ask a teammate to submit a plan for review.",
     "input_schema": {"type": "object",
                      "properties": {"teammate": {"type": "string"},
                                     "task": {"type": "string"}},
                      "required": ["teammate", "task"]}},
    {"name": "review_plan",
     "description": "Approve or reject a submitted plan by request_id.",
     "input_schema": {"type": "object",
                      "properties": {
                          "request_id": {"type": "string"},
                          "approve": {"type": "boolean"},
                          "feedback": {"type": "string"}},
                      "required": ["request_id", "approve"]}},
]


# ── Context ──

def update_context(context: dict, messages: list) -> dict:
    """Derive context from real state."""
    memories = ""
    if MEMORY_INDEX.exists():
        content = MEMORY_INDEX.read_text().strip()
        if content:
            memories = content
    return {
        "enabled_tools": [t["name"] for t in TOOLS],
        "workspace": str(WORKDIR),
        "memories": memories,
    }


# ── Agent Loop ──

def agent_loop(messages: list, context: dict):
    system = get_system_prompt(context)
    while True:
        try:
            response = client.messages.create(
                model=MODEL, system=system, messages=messages,
                tools=TOOLS, max_tokens=8000)
        except Exception as e:
            messages.append({"role": "assistant", "content": [
                {"type": "text",
                 "text": f"[Error] {type(e).__name__}: {e}"}]})
            return

        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            return

        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            print(f"\033[36m> {block.name}\033[0m")

            if should_run_background(block.name, block.input):
                bg_id = start_background_task(block)
                results.append({"type": "tool_result",
                                "tool_use_id": block.id,
                                "content": f"[Background task {bg_id} started] "
                                           f"Result will be available when complete."})
            else:
                output = execute_tool(block)
                print(str(output)[:300])
                results.append({"type": "tool_result",
                                "tool_use_id": block.id,
                                "content": output})

        # Merge background tool results + notifications into one user message
        user_content = list(results)
        bg_notifications = collect_background_results()
        if bg_notifications:
            for notif in bg_notifications:
                user_content.append({"type": "text", "text": notif})
        messages.append({"role": "user", "content": user_content})
        context = update_context(context, messages)
        system = get_system_prompt(context)


if __name__ == "__main__":
    print("s16: team protocols")
    print("Enter a question, press Enter to send. Type q to quit.\n")
    history = []
    context = update_context({}, [])
    while True:
        try:
            query = input("\033[36ms16 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append({"role": "user", "content": query})
        agent_loop(history, context)
        context = update_context(context, history)
        for block in history[-1]["content"]:
            if getattr(block, "type", None) == "text":
                print(block.text)
            elif isinstance(block, dict) and block.get("type") == "text":
                print(block.get("text", ""))

        # Check inbox → route protocol + inject into history
        inbox_msgs = consume_lead_inbox(route_protocol=True)
        if inbox_msgs:
            inbox_text = "\n".join(
                f"From {m['from']}: {m['content'][:200]}" for m in inbox_msgs)
            history.append({"role": "user",
                            "content": f"[Inbox]\n{inbox_text}"})
            print(f"\n\033[33m[Inbox: {len(inbox_msgs)} messages injected]\033[0m")
        print()
