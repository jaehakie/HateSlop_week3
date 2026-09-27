"""
Japan Trip Companion Agent
===========================
Travel itinerary reviewer + Japanese language assistant, all in one agent.

Built with LangChain create_agent + 5 tools (2 shared, 3 custom).
Includes Harness (budget, error handling, trace) and Loop (verification, re-run).

Usage:
    python assignment/jaehakie/agent.py
"""

import json
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# ── Project Setup ──────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")
api_key = os.getenv("OPENAI_API_KEY", "")
if not api_key or api_key == "your-openai-api-key":
    raise RuntimeError("Set OPENAI_API_KEY in .env at project root.")

MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

from tools.read_file import read_file as _read_file
from tools.wikipedia_search import wikipedia_search

# ── Internal Data ──────────────────────────────────────────────

TRAVEL_TIME_DB: dict[tuple[str, str], tuple[str, int]] = {
    ("도쿄", "오사카"): ("도카이도 신칸센", 150),
    ("도쿄", "교토"): ("도카이도 신칸센", 135),
    ("도쿄", "히로시마"): ("산요 신칸센", 240),
    ("도쿄", "후쿠오카"): ("산요 신칸센+특급", 300),
    ("도쿄", "삿포로"): ("비행기", 90),
    ("도쿄", "요코하마"): ("JR 히타치/특급", 30),
    ("도쿄", "나고야"): ("도카이도 신칸센", 100),
    ("도쿄", "하코네"): ("JR 소운+하코네선", 85),
    ("도쿄", "니코"): ("JR 닛코선/특급", 120),
    ("도쿄", "마쓰야마"): ("비행기", 80),
    ("오사카", "교토"): ("JR 특급/신칸센", 30),
    ("오사카", "나라"): ("JR 대화특급", 45),
    ("오사카", "고베"): ("JR 특급", 20),
    ("오사카", "히로시마"): ("산요 신칸센", 90),
    ("오사카", "후쿠오카"): ("산요 신칸센", 150),
    ("오사카", "마쓰야마"): ("신칸센+시코쿠특급", 150),
    ("교토", "나라"): ("JR 나라선", 50),
    ("나고야", "교토"): ("도카이도 신칸센", 35),
    ("나고야", "오사카"): ("도카이도 신칸센", 55),
}

PHRASE_DB: dict[str, dict[str, str]] = {
    "すみません": {
        "읽는_법": "스미마센",
        "의미": "죄송합니다 / 실례합니다 / 저기요",
        "사용_상황": "사과, 주문, 길 묻기, 주의 끌기 등",
        "예문": "すみません、駅はどこですか？ (스미마센, 에키와 도코데스카? → 실례합니다, 역은 어디인가요?)",
    },
    "お願いします": {
        "읽는_법": "오네가이시마스",
        "의미": "부탁드립니다 / 주세요",
        "사용_상황": "주문이나 요청 시 정중한 표현",
        "예문": "コーヒーをお願いします。 (코히오 오네가이시마스 → 커피 주세요)",
    },
    "ありがとうございます": {
        "읽는_법": "아리가토고자이마스",
        "의미": "감사합니다",
        "사용_상황": "정중한 감사 표현",
        "예문": "ありがとうございます。 (아리가토고자이마스 → 감사합니다)",
    },
    "いくらですか": {
        "읽는_법": "이쿠라데스카",
        "의미": "얼마인가요?",
        "사용_상황": "가격 물어볼 때",
        "예문": "これはいくらですか？ (코레와 이쿠라데스카? → 이건 얼마인가요?)",
    },
    "どこですか": {
        "읽는_법": "도코데스카",
        "의미": "어디인가요?",
        "사용_상황": "위치 물어볼 때",
        "예문": "トイレはどこですか？ (토이레와 도코데스카? → 화장실은 어디인가요?)",
    },
    "いただきます": {
        "읽는_법": "이타다키마스",
        "의미": "잘 먹겠습니다 (식사 전 인사)",
        "사용_상황": "식사 시작 전",
        "예문": "いただきます！ (이타다키마스 → 잘 먹겠습니다!)",
    },
    "ごちそうさまでした": {
        "읽는_법": "고치소사마데시타",
        "의미": "잘 먹었습니다 (식사 후 인사)",
        "사용_상황": "식사 마친 후",
        "예문": "ごちそうさまでした。 (고치소사마데시타 → 잘 먹었습니다)",
    },
}

# ── Tool Definitions ──────────────────────────────────────────

@tool
def read_file(path: str) -> str:
    """Read a text file from the project (itineraries, notes, etc.)."""
    return _read_file(path)


@tool
def search_place(query: str) -> str:
    """Search travel destination info via Korean Wikipedia."""
    try:
        result = wikipedia_search(query, timeout_seconds=15.0)
        data = json.loads(result)
        if not data.get("results") or all(
            not r.get("summary", "").strip() for r in data["results"]
        ):
            return json.dumps({"query": query, "results": [],
                "message": f"'{query}' 검색 결과가 없습니다."}, ensure_ascii=False)
        return result
    except Exception as e:
        return json.dumps({"error": f"검색 오류: {str(e)}"}, ensure_ascii=False)


@tool
def calculate_travel_time(origin: str, destination: str) -> str:
    """Calculate travel time between two Japanese cities.

    Uses built-in DB of Shinkansen / limited express / flight times.

    Args:
        origin: Departure city, e.g. '도쿄', '오사카', '마쓰야마'
        destination: Arrival city, e.g. '마쓰야마', '교토', '히로시마'
    Returns:
        JSON with 출발지, 도착지, 교통편, 예상_시간_분, 예상_시간
    """
    org, dst = origin.strip(), destination.strip()
    key, rev = (org, dst), (dst, org)

    def _fmt(transport, minutes, note=""):
        r = {"출발지": org, "도착지": dst, "교통편": transport,
             "예상_시간_분": minutes,
             "예상_시간": f"{minutes//60}시간 {minutes%60}분"
                          if minutes >= 60 else f"{minutes}분"}
        if note:
            r["참고"] = note
        return json.dumps(r, ensure_ascii=False)

    if key in TRAVEL_TIME_DB:
        return _fmt(*TRAVEL_TIME_DB[key])
    if rev in TRAVEL_TIME_DB:
        return _fmt(*TRAVEL_TIME_DB[rev],
                    note=f"'{dst}'→'{org}' 역방향 데이터 사용")
    return json.dumps({"출발지": org, "도착지": dst,
                       "message": f"'{org}'→'{dst}' 미등록 경로"},
                      ensure_ascii=False)


@tool
def lookup_japanese(expression: str) -> str:
    """Look up a Japanese phrase: reading, meaning, usage, example.

    Built-in DB for common travel phrases; falls back to generation.

    Args:
        expression: e.g. 'すみません', 'お願いします', 'いただきます'
    Returns:
        JSON with 표현, 읽는_법, 의미, 사용_상황, 예문
    """
    expr = expression.strip()
    if expr in PHRASE_DB:
        r = {"표현": expr} | PHRASE_DB[expr]
        return json.dumps(r, ensure_ascii=False)

    try:
        lm = ChatOpenAI(model=MODEL, temperature=0.3, timeout=15)
        prompt = (
            f"다음 일본어 표현을 JSON으로 알려주세요 (코드 블록 없이):\n"
            f"표현: {expr}\n\n"
            f'필드: "표현", "읽는_법", "의미", "사용_상황", "예문"\n'
        )
        resp = lm.invoke([HumanMessage(content=prompt)])
        content = resp.content.strip()
        try:
            json.loads(content)
            return content
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", content, re.DOTALL)
            if m:
                return m.group()
            return json.dumps({"표현": expr, "정보": content},
                              ensure_ascii=False)
    except Exception as e:
        return json.dumps({"표현": expr, "오류": str(e)},
                          ensure_ascii=False)


@tool
def check_japanese(text: str) -> str:
    """Check Japanese text for grammar issues and awkward expressions.

    Returns structured feedback with improvement suggestions.

    Args:
        text: Japanese sentence to review, e.g. '私は昨日お店に行きました。'
    Returns:
        JSON with 원본, 문법_오류, 어색한_표현, 개선_제안, 총평
    """
    try:
        cm = ChatOpenAI(model=MODEL, temperature=0.2, timeout=15)
        prompt = (
            f"다음 일본어를 검토하고 JSON으로 답변하세요 (코드 블록 없이):\n"
            f"텍스트: {text}\n\n"
            f'{{\n  "원본": "{text}",\n  "문법_오류": "...",\n'
            f'  "어색한_표현": "...",\n  "개선_제안": "...",\n'
            f'  "총평": "..."\n}}\n'
        )
        resp = cm.invoke([HumanMessage(content=prompt)])
        content = resp.content.strip()
        try:
            json.loads(content)
            return content
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", content, re.DOTALL)
            if m:
                return m.group()
            return json.dumps({"원본": text, "분석": content},
                              ensure_ascii=False)
    except Exception as e:
        return json.dumps({"원본": text, "오류": str(e)},
                          ensure_ascii=False)
# ── Harness: RunState & Execution ──────────────────────────────

@dataclass
class RunState:
    """Execution state: budget, trace, errors, feedback."""
    iteration: int = 0
    max_iterations: int = 3
    step: int = 0
    max_steps: int = 15
    trace: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    passed: bool = False
    feedback: str = ""
    final_answer: str = ""
    budget_exceeded: bool = False


def add_trace(state: RunState, event: str, detail: str = "") -> None:
    state.trace.append({
        "event": event, "detail": detail,
        "step": state.step, "run": state.iteration,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    })


def execute_single_run(agent: Any, task: str, state: RunState,
                       recursion_limit: int = 30) -> str:
    """Execute one agent run within the Harness.

    Harness features:
    1. Stop Condition — agent stops naturally when done
    2. Execution Budget — max_steps limits agent invocations
    3. Error Handling — exceptions caught, logged, returned safely
    4. Trace — every event recorded with timestamp
    """
    state.step += 1
    add_trace(state, "RUN_START", f"Agent invoke #{state.step}")

    try:
        response = agent.invoke(
            {"messages": [HumanMessage(content=task)]},
            config={"recursion_limit": recursion_limit},
        )
    except Exception as exc:
        msg = f"Agent error (step {state.step}): {type(exc).__name__}: {exc}"
        state.errors.append(msg)
        add_trace(state, "ERROR", msg)
        return f"[ERROR] {msg}"

    messages = response.get("messages", [])
    if not messages:
        add_trace(state, "EMPTY", "No messages")
        return "[WARN] Empty response."

    for msg in messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                add_trace(state, "TOOL_CALL",
                          f"{tc['name']}({json.dumps(tc['args'], ensure_ascii=False)})")

    # Last AIMessage without tool_calls = final answer
    for msg in reversed(messages):
        if (hasattr(msg, "content") and msg.content
                and not (hasattr(msg, "tool_calls") and msg.tool_calls)):
            answer = str(msg.content)
            preview = answer[:200] + ("..." if len(answer) > 200 else "")
            add_trace(state, "AGENT_ANSWER", preview)
            return answer

    last = messages[-1]
    fallback = str(getattr(last, "content", ""))
    add_trace(state, "DONE", fallback[:200])
    return fallback
# ── Verification ──────────────────────────────────────────────

def verify_output(answer: str, task: str) -> tuple[bool, list[str]]:
    """Verify the agent's answer against the task.

    - If task mentions travel: check for transport/time keywords
    - If task mentions Japanese: check for Japanese content
    - General quality: min length
    """
    issues: list[str] = []

    has_travel = any(kw in task
                     for kw in ["일정", "여행", "이동", "경로", "관광", "도시", "신칸센"])
    has_japanese = any(kw in task
                       for kw in ["일본어", "표현", "회화", "phrase",
                                  "말", "문장", "일본"])

    if has_travel:
        travel_tokens = ["이동", "시간", "교통", "신칸센", "특급",
                         "JR", "일정", "비행기", "소요"]
        if not any(t in answer for t in travel_tokens):
            issues.append("여행 정보(이동 시간, 교통편 등)가 부족합니다.")

    if has_japanese:
        jp_tokens = ["일본어", "すみません", "ありがとう",
                     "お願い", "발음", "읽는"]
        if not any(t in answer for t in jp_tokens):
            issues.append("일본어 표현/학습 정보가 결과에 없습니다.")

    if len(answer) < 60:
        issues.append("답변이 너무 짧습니다.")

    return (len(issues) == 0, issues)
# ── Loop Engineering ──────────────────────────────────────────

def run_loop(agent: Any, task: str, *,
             max_iterations: int = 3, max_steps: int = 15) -> RunState:
    """Outer verification loop.

    Loop features:
    1. Verification after each run
    2. Conditional stop (PASSED or LIMIT_REACHED)
    3. Feedback built from issues, fed into next run
    4. Re-run with feedback attached to task
    5. Loop limit prevents infinite re-runs
    """
    state = RunState(max_iterations=max_iterations, max_steps=max_steps)

    print(f"\n{'=' * 60}")
    print(f"  🗾  Japan Trip Companion Agent")
    print(f"{'=' * 60}")
    print(f"  Task: {task[:140]}{'...' if len(task) > 140 else ''}\n")

    while state.iteration < state.max_iterations:
        state.iteration += 1
        print(f"{'─' * 50}")
        print(f"  🔄  Run {state.iteration} / {state.max_iterations}")
        print(f"{'─' * 50}")

        current_task = task
        if state.feedback:
            current_task = (
                f"{task}\n\n[Feedback from previous run]\n"
                f"{state.feedback}\n\n"
                f"Please address the issues above and rewrite your answer."
            )
            print(f"  Feedback: {state.feedback[:180]}...\n")

        add_trace(state, "ITERATION_START",
                  f"Outer-loop iteration {state.iteration}")

        answer = execute_single_run(agent, current_task, state)
        state.final_answer = answer

        preview = answer[:400].replace("\n", " ")
        print(f"\n  Response ({len(answer)} chars):\n"
              f"  {preview}{'...' if len(answer) > 400 else ''}\n")

        passed, issues = verify_output(answer, task)
        state.passed = passed

        if passed:
            print(f"  ✅  Verification PASSED")
            add_trace(state, "PASSED", "All checks passed")
            return state

        state.feedback = "Please fix:\n- " + "\n- ".join(issues)
        print(f"  ❌  FAILED ({len(issues)} issue(s)):")
        for i, iss in enumerate(issues, 1):
            print(f"       {i}. {iss}")
        add_trace(state, "FAILED", f"{len(issues)} issues")

    print(f"\n  ⚠️  Max iterations ({state.max_iterations}) reached.")
    add_trace(state, "LIMIT_REACHED", "Outer loop limit reached")
    return state
# ── Agent Factory ─────────────────────────────────────────────

def create_travel_agent() -> Any:
    """Create a Japan Trip Companion Agent via LangChain create_agent.

    5 tools: read_file, search_place, calculate_travel_time,
             lookup_japanese, check_japanese.
    """
    model = ChatOpenAI(model=MODEL, temperature=0.3, timeout=30)

    system_prompt = (
        "You are a Japan Trip Companion Agent. Help users plan trips "
        "to Japan and learn Japanese. Answer in Korean.\n\n"
        "## Tools\n"
        "1. read_file(path) — Read project files\n"
        "2. search_place(query) — Look up destinations via Wikipedia\n"
        "3. calculate_travel_time(origin, dest) — Travel times\n"
        "4. lookup_japanese(expression) — Japanese phrase info\n"
        "5. check_japanese(text) — Review Japanese writing\n\n"
        "## File paths\n"
        "- Itinerary file: assignment/jaehakie/japan_itinerary.md\n"
        "- Use read_file with the full relative path from the project root.\n\n"
        "## Guidelines\n"
        "- Combine tools as needed.\n"
        "- Verify travel times between consecutive destinations.\n"
        "- Include pronunciation and usage context for Japanese.\n"
        "- Never guess — use tools to confirm.\n"
        "- Respond clearly in Korean."
    )

    tools = [read_file, search_place, calculate_travel_time,
             lookup_japanese, check_japanese]

    return create_agent(model=model, tools=tools,
                        system_prompt=system_prompt,
                        name="japan_trip_companion")
# ── Main ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import textwrap

    print("=" * 60)
    print("  🗾  Japan Trip Companion Agent")
    print("  " + "=" * 54)
    print(f"  Model: {MODEL}")
    print("  Tools: read_file, search_place, calculate_travel_time,")
    print("         lookup_japanese, check_japanese")
    print("=" * 60)

    agent = create_travel_agent()
    print(f"  ✅  Agent: {type(agent).__name__}")

    print(f"\n{'=' * 60}")
    print("  📝  Interactive Mode")
    print(f"{'=' * 60}")
    print("  여행 일정, 일본어 표현, 이동 시간 등 자유롭게 물어보세요.")
    print("  종료하려면 exit / quit / 종료 를 입력하세요.")
    print(f"{'─' * 60}")

    while True:
        raw = input("\n  You: ").strip()
        if not raw:
            continue
        if raw.lower() in ("exit", "quit", "종료"):
            print("  👋  Bye!")
            break

        print(f"\n{'─' * 60}")
        final = run_loop(agent, raw, max_iterations=3, max_steps=15)
        print(f"\n{'=' * 60}")
        print("  📊  Trace")
        print(f"{'=' * 60}")
        ICONS = {"ITERATION_START": "🔄", "RUN_START": "▶",
                 "TOOL_CALL": "🔧", "AGENT_ANSWER": "💬",
                 "ERROR": "❌", "PASSED": "✅", "FAILED": "❌",
                 "LIMIT_REACHED": "⛔", "EMPTY": "⚠️", "DONE": "✅"}
        for entry in final.trace:
            icon = ICONS.get(entry["event"], "•")
            d = entry["detail"][:150]
            if len(entry["detail"]) > 150:
                d += "..."
            print(f"  {icon}  [{entry['event']}]  (step {entry['step']})")
            print(f"       {d}")

        verdict = "✅ PASSED" if final.passed else "❌ NOT PASSED"
        print(f"\n  {verdict}  |  Runs: {final.iteration}/{final.max_iterations}"
              f"  |  Steps: {final.step}")

        if final.final_answer:
            print(f"\n  📄  Final Answer:\n")
            for line in textwrap.fill(final.final_answer, width=72).split("\n"):
                print(f"  {line}")