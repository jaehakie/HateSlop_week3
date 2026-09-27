# 🗾 Japan Trip Companion Agent

**"여행 일정을 점검하고, 여행에 필요한 일본어까지 함께 준비해주는 AI 여행 동반자 Agent"**

---

## 1. 아이디어 소개

일본 여행을 준비하거나 여행 중인 사용자를 대상으로, **여행 일정 관리와 일본어 회화 학습을 하나의 Agent에서 지원**합니다.

사용자의 요청에 따라 Agent가 스스로 판단하고 필요한 Tool을 선택하여 작업을 수행합니다.

### 주요 기능

| 기능 | 설명 |
|------|------|
| **여행 일정 검토** | 사용자의 일정 파일을 읽고, 장소 간 이동 시간을 확인하여 현실성 검토 |
| **여행지 정보 조회** | 위키백과를 통해 방문 예정 장소의 기본 정보 조회 |
| **이동 시간 계산** | 일본 내 주요 도시 간 이동 시간과 교통편을 DB 기반으로 조회 |
| **일본어 표현 학습** | 여행 상황에 맞는 일본어 표현의 의미, 발음, 예문 제공 |
| **일본어 문장 검토** | 사용자의 일본어 문장 문법과 자연스러움 확인 |

---

## 2. Tool 정의

총 **5개**의 Tool을 제공합니다 (프로젝트 공유 Tool 2개 + 직접 구현 Tool 3개).

### 2.1 `read_file(path)` — 파일 읽기 (프로젝트 공유)

- **기능**: 프로젝트 내 텍스트 파일을 읽습니다. 여행 일정 파일 로드에 사용
- **사용 예**: `read_file("assignment/jaehakie/japan_itinerary.md")`

### 2.2 `search_place(query)` — 장소 검색 (프로젝트 공유)

- **기능**: 한국어 위키백과를 통해 일본 여행지 정보를 검색
- **사용 예**: `search_place("마쓰야마")` → JSON (title, summary, URL)

### 2.3 `calculate_travel_time(origin, destination)` — 이동 시간 계산 (직접 구현)

- **기능**: 일본 내 두 도시 간 이동 시간과 추천 교통편 조회
- **구현**: 주요 노선(신칸센/특급/비행기) 이동 시간 내장 DB에서 조회
- **사용 예**: `calculate_travel_time("도쿄", "마쓰야마")`
- **출력**: `{"출발지":"도쿄","도착지":"마쓰야마","교통편":"비행기","예상_시간_분":80,...}`

### 2.4 `lookup_japanese(expression)` — 일본어 표현 조회 (직접 구현)

- **기능**: 일본어 표현의 의미, 읽는 법, 사용 상황, 예문 제공
- **구현**: 자주 사용되는 여행 일본어(すみません 등)는 내장 DB에서 즉시 반환, 미등록 표현은 ChatOpenAI로 생성
- **사용 예**: `lookup_japanese("すみません")`

### 2.5 `check_japanese(text)` — 일본어 문장 검토 (직접 구현)

- **기능**: 사용자의 일본어 문장 문법 오류와 어색한 표현 분석 및 개선 제안
- **구현**: ChatOpenAI를 Evaluator로 사용하여 구조화된 피드백 반환
- **사용 예**: `check_japanese("私は昨日お店に行きました。")`

---

## 3. LangChain Agent 구성

`from langchain.agents import create_agent`를 사용하여 Agent를 생성합니다.

```python
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4.1-mini", temperature=0.3)
agent = create_agent(
    model=model,
    tools=[read_file, search_place, calculate_travel_time,
           lookup_japanese, check_japanese],
    system_prompt="You are a Japan Trip Companion Agent...",
    name="japan_trip_companion",
)
```

- **Model**: `gpt-4.1-mini` (temperature 0.3)
- **create_agent**: 내부적으로 LangGraph `CompiledStateGraph`를 생성, Agent Node와 Tool Node 간 Loop 자동 처리
- **System Prompt**: Agent 역할, Tool 목록, 행동 지침 포함

### Agent 호출

```python
response = agent.invoke(
    {"messages": [HumanMessage(content="여행 일정을 확인해줘.")]},
    config={"recursion_limit": 30},
)
```

---

## 4. Harness Engineering

Agent 실행을 안전하게 제어하고 상태를 관찰하는 Harness를 구현했습니다.

### 4.1 `RunState` — 실행 상태

```python
@dataclass
class RunState:
    iteration: int         # Outer Loop 실행 횟수
    max_iterations: int    # 최대 Outer Loop 횟수 (기본 3)
    step: int              # Agent Invoke 횟수
    max_steps: int         # 최대 Agent Invoke 횟수 (기본 15)
    trace: list[dict]      # 실행 이벤트 기록
    errors: list[str]      # 오류 메시지
    passed: bool           # 검증 통과 여부
    feedback: str          # 검증 실패 시 피드백
    final_answer: str      # 최종 응답
```

### 4.2 Stop Condition

Agent Loop는 LangGraph 내부에서 Tool Call이 더 이상 없으면 자연 종료. `recursion_limit`(기본 30)을 안전장치로 설정.

### 4.3 Execution Budget

`max_steps`(기본 15)로 Agent Invoke 횟수 제한. 초과 시 Trace에 기록.

### 4.4 Error Handling

모든 예외를 `try/except`로 포착, state.errors에 저장, Trace에 기록, 오류 메시지 반환.

### 4.5 Trace

모든 이벤트를 timestamp와 함께 기록:
## 5. Loop Engineering

Agent 결과를 검증하고 문제가 있으면 재실행하는 Outer Loop를 구성했습니다.

### 동작 흐름

```text
사용자 요청 → Agent 실행 (Harness) → 검증 → 조건 판단
  ├─ 통과 → 최종 응답 반환
  └─ 실패 → 피드백 구성 → 재실행
```

### Verification 함수

```python
def verify_output(answer, task):
    issues = []
    # 여행 맥락 → 교통/이동 정보 확인
    # 일본어 맥락 → 일본어 표현 확인
    # 기본 품질 확인 (길이)
    return (len(issues) == 0, issues)
```

## 6. Agent Flow (전체 실행 흐름)

```text
사용자 입력 (예: "마쓰야마 2박 3일 일정 확인 + 일본어 표현 알려줘")
    │
    ▼
┌───────────────────────────────────────────────┐
│  Outer Loop 시작 (run_loop)                    │
│  Iteration 1/3, max_steps=15, recursion_limit=30│
└───────────────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────────────┐
│  Agent 실행 (CompiledStateGraph)              │
│  [Step] Model 요청 분석 → Tool 선택/호출       │
│  [Step] Tool 결과 관찰 → 다음 Tool 결정         │
│  [Step] 모든 정보 수집 완료 → 최종 응답 생성    │
└───────────────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────────────┐
│  검증 (verify_output)                         │
│  통과 → 최종 응답 반환                         │
│  실패 → 피드백 → 재실행                         │
└───────────────────────────────────────────────┘
```

## 7. 실행 예시

### 인터랙티브 모드 실행

```text
============================================================
  🗾  Japan Trip Companion Agent
============================================================
  Model: gpt-4.1-mini
  Tools: read_file, search_place, calculate_travel_time,
         lookup_japanese, check_japanese
============================================================

  ✅  Agent: CompiledStateGraph

============================================================
  📝  Interactive Mode
============================================================
  여행 일정, 일본어 표현, 이동 시간 등 자유롭게 물어보세요.
  종료하려면 exit / quit / 종료 를 입력하세요.
────────────────────────────────────────────────────────────

  You: 마쓰야마 2박 3일 일정 확인하고 필요한 일본어 표현 3개 알려줘
```

### 자유 질문 — 여행 일정 검토 + 일본어 표현

```text
  You: 마쓰야마 2박 3일 일정을 확인하고, 필요한 일본어 표현 3개 알려줘

────────────────────────────────────────────────────────────
  🔄  Run 1 / 3
  ─────────────────────────────────────────────────────
  🔧  [TOOL_CALL] read_file({"path": "assignment/jaehakie/japan_itinerary.md"})
  🔧  [TOOL_CALL] search_place({"query": "마쓰야마"})
  🔧  [TOOL_CALL] calculate_travel_time({"origin": "도쿄",
                                         "destination": "마쓰야마"})
  🔧  [TOOL_CALL] lookup_japanese({"expression": "すみません"})
  🔧  [TOOL_CALL] lookup_japanese({"expression": "お願いします"})
  🔧  [TOOL_CALL] lookup_japanese({"expression": "ありがとうございます"})

  💬  [AGENT_ANSWER] (step 1)

  ✅  Verification PASSED

  ✅ PASSED  |  Runs: 1/3  |  Steps: 1

  📄  Final Answer:
  [여행 일정 검토 결과 + 3개 일본어 표현과 발음/사용 상황...]

  ────────────────────────────────────────────────────────────
  You: 도쿄에서 오사카까지 얼마나 걸려?
```

### 자유 질문 — 이동 시간 + 연속 대화

```text
  You: 도쿄에서 오사카까지 얼마나 걸려?

────────────────────────────────────────────────────────────
  🔧  [TOOL_CALL] calculate_travel_time({"origin": "도쿄",
                                         "destination": "오사카"})

  💬  [AGENT_ANSWER] (step 1)
  ✅ PASSED  |  Runs: 1/3  |  Steps: 1

  ────────────────────────────────────────────────────────────
  You: 그럼 오사카에서 교토까지는?
  🔧  [TOOL_CALL] calculate_travel_time({"origin": "오사카",
                                         "destination": "교토"})

  ...
  ────────────────────────────────────────────────────────────
  You: exit
  👋  Bye!
```

### 자유 질문 예시 — 일본어 표현

```text
  You: すみません의 정확한 사용법 알려줘

────────────────────────────────────────────────────────────
  🔧  [TOOL_CALL] lookup_japanese({"expression": "すみません"})

  💬  [AGENT_ANSWER] (step 1)
  ✅ PASSED  |  Runs: 1/3  |  Steps: 1

  📄  Final Answer:
  [すみません의 의미, 읽는 법, 사용 상황, 예문 설명...]

  ────────────────────────────────────────────────────────────
  You: 이 문장 검토해줘 「私は昨日お店に行きました。美味しい料理を食べました。」

  🔧  [TOOL_CALL] check_japanese({"text": "私は昨日お店に行きました。美味しい料理を食べました。"})

  💬  [AGENT_ANSWER] (step 1)
  ✅ PASSED  |  Runs: 1/3  |  Steps: 1
```

---

## 환경 설정 및 실행

### 사전 요구 사항

```bash
# 가상 환경 활성화
.venv\\Scripts\\activate

# 의존성 설치
pip install -r requirements.txt

# .env 파일 (프로젝트 루트)
# OPENAI_API_KEY=your-api-key-here
# OPENAI_MODEL=gpt-4.1-mini
```

### 실행

```bash
python assignment/jaehakie/agent.py
```

### 인터랙티브 명령어

실행 후 표시되는 `You:` 프롬프트에서 자유롭게 질문을 입력하세요.

Agent가 질문 내용을 분석하여 필요한 Tool을 자동으로 선택하고 실행합니다.

| 명령어 | 설명 |
|--------|------|
| `자유 텍스트` | 여행 일정 확인, 일본어 표현 질문, 이동 시간 조회 등 모든 자유 질문 |
| `exit` / `quit` / `종료` | 프로그램 종료 |

**자유 질문 예시:**
```
You: 마쓰야마 2박 3일 일정 확인하고 필요한 일본어 표현 3개 알려줘
  → Agent가 파일 읽기 + 장소 검색 + 이동 시간 계산 + 일본어 표현 조회를 자동으로 수행

You: 도쿄에서 오사카까지 신칸센으로 얼마나 걸려?
  → calculate_travel_time 자동 호출

You: すみません의 정확한 사용법 알려줘
  → lookup_japanese 자동 호출

You: 이 문장 검토해줘 「私は明日京都に行きます。」
  → check_japanese 자동 호출
```

### 과제 충족 항목

| 요구사항 | 구현 위치 |
|----------|-----------|
| LangChain `create_agent` | agent.py 24행: `from langchain.agents import create_agent` |
| Tool 3개 이상 | 5개 Tool (read_file, search_place, calculate_travel_time, lookup_japanese, check_japanese) |
| Harness (Stop/Budget/Error/Trace) | `RunState` + `execute_single_run()` |
| Loop (Verification + Re-run) | `verify_output()` + `run_loop()` |
