import json
import os
import re
import sys
import time
import asyncio
import concurrent.futures
from contextlib import aclosing
from pathlib import Path
from typing import Any

from google import genai
from google.adk.agents.llm_agent import Agent
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.google_search_tool import GoogleSearchTool
from google.genai import types as genai_types

# Add the code directory to the path to ensure imports work
sys.path.insert(0, str(Path(__file__).parent))
from spreadsheet_tool import GoogleSheetsSpreadsheetTool


def _load_env_if_present() -> None:
    """Load key/value pairs from .env if the file exists."""
    env_path = Path(__file__).with_name(".env")
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_env_if_present()

# Shared tools/services
raw_search_tool = GoogleSearchTool(bypass_multi_tools_limit=True)
spreadsheet_tool = GoogleSheetsSpreadsheetTool()
STATE_DIR = Path(__file__).with_name(".state")
STATE_PATH = STATE_DIR / "church_research_state.json"


class WriteChurchesToSheetTool:
    """Wrapper to make the spreadsheet tool compatible with ADK agents."""

    def __init__(self, sheet_tool: GoogleSheetsSpreadsheetTool):
        self.sheet_tool = sheet_tool
        self.__name__ = "write_churches_to_sheet"
        self.__doc__ = "Write JSON church list to Google Sheets"

    def __call__(self, churches_json: str) -> str:
        return self.sheet_tool(churches_json)


write_churches_to_sheet = WriteChurchesToSheetTool(spreadsheet_tool)


class PauseForQuotaTool:
    """Explicit pause tool for root-agent orchestration pacing."""

    def __init__(self):
        self.__name__ = "pause_for_quota"
        self.__doc__ = "Pause for a bounded number of seconds to reduce request bursts"
        self._last_called_at = 0.0

    def __call__(self, seconds: int = 20) -> str:
        now = time.time()
        # Guard against consecutive pause calls.
        if now - self._last_called_at < 2:
            return "Skipped duplicate pause_for_quota call."
        wait = max(1, min(120, int(seconds)))
        time.sleep(wait)
        self._last_called_at = time.time()
        return f"Paused {wait} seconds."


pause_for_quota = PauseForQuotaTool()


class QuotaPauseController:
    """Track search call counts; optional sleep support."""

    def __init__(self, pause_every_calls: int = 3, pause_seconds: int = 60):
        # Allow disabling implicit sleep by setting pause_every_calls <= 0.
        self.pause_every_calls = max(0, int(pause_every_calls))
        self.pause_seconds = max(1, int(pause_seconds))
        self.total_calls = 0

    def register_search_call(self, apply_sleep: bool = True) -> None:
        self.total_calls += 1
        if apply_sleep and self.pause_every_calls > 0 and self.total_calls % self.pause_every_calls == 0:
            time.sleep(self.pause_seconds)


class ProgressStore:
    """Persistent local state so progress survives 429s/restarts."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"search_cache": {}, "church_results": {}}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return {"search_cache": {}, "church_results": {}}
            data.setdefault("search_cache", {})
            data.setdefault("church_results", {})
            return data
        except Exception:
            return {"search_cache": {}, "church_results": {}}

    def save(self, data: dict[str, Any]) -> None:
        try:
            self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass


class RateLimitedGoogleSearchTool:
    """Wrapper that tracks search usage and performs Google-search-backed calls."""

    def __init__(self, inner: GoogleSearchTool, quota: QuotaPauseController):
        self.inner = inner
        self.quota = quota
        self.client = genai.Client()
        self.cooldown_until = 0.0
        self.consecutive_429 = 0
        self.consecutive_504 = 0
        self.last_error = ""
        self.__name__ = "google_search_tool_rate_limited"
        self.__doc__ = "Google search with request counting"

    def __call__(self, query: str) -> Any:
        now = time.time()
        if now < self.cooldown_until:
            # Circuit breaker: do not hit API during cooldown window.
            wait_left = int(self.cooldown_until - now)
            return f"__RATE_LIMITED__ cooldown_active:{wait_left}s"

        # Count and pause on each search call.
        self.quota.register_search_call(apply_sleep=True)
        # NOTE:
        # GoogleSearchTool is an ADK runtime model-built-in and is not directly callable
        # from this local Python orchestration path. To keep your current class structure,
        # we execute an equivalent Google-search-enabled model request here.
        max_attempts = 2  # one immediate retry for transient server/network timeouts
        for attempt in range(1, max_attempts + 1):
            try:
                response = self.client.models.generate_content(
                    model=os.getenv("SEARCH_MODEL", "gemini-2.5-flash"),
                    contents=f"Search web for: {query}",
                    config=genai_types.GenerateContentConfig(
                        tools=[genai_types.Tool(google_search=genai_types.GoogleSearch())],
                        automatic_function_calling=genai_types.AutomaticFunctionCallingConfig(
                            disable=False,
                            maximum_remote_calls=1,
                        ),
                        http_options=genai_types.HttpOptions(timeout=20_000),
                        temperature=0.0,
                    ),
                )
                if response is None:
                    return ""
                text = getattr(response, "text", None)
                if text:
                    self.consecutive_429 = 0
                    self.consecutive_504 = 0
                    self.last_error = ""
                    return text
                self.consecutive_429 = 0
                self.consecutive_504 = 0
                self.last_error = ""
                return str(response)
            except Exception as e:
                msg = str(e).lower()
                self.last_error = str(e)

                # Hard quota handling with cooldown circuit-breaker.
                if "429" in msg or "resource_exhausted" in msg or "quota" in msg or "too many requests" in msg:
                    self.consecutive_429 += 1
                    # Exponential cooldown: 60s, 120s, 240s ... capped at 10 minutes.
                    cooldown = min(600, 60 * (2 ** (self.consecutive_429 - 1)))
                    self.cooldown_until = time.time() + cooldown
                    return f"__RATE_LIMITED__ triggered:{cooldown}s"

                if "504" in msg or "gateway timeout" in msg:
                    self.consecutive_504 += 1
                    # Short cooldown for transient server timeouts: 20s, 40s, 80s, 120s.
                    cooldown = min(120, 20 * (2 ** (self.consecutive_504 - 1)))
                    self.cooldown_until = time.time() + cooldown
                    return f"__TRANSIENT_TIMEOUT__ triggered:{cooldown}s"

                # Retry once on transient timeout/gateway/network errors.
                is_transient = any(
                    k in msg
                    for k in [
                        "504",
                        "gateway timeout",
                        "timed out",
                        "timeout",
                        "connection reset",
                        "temporarily unavailable",
                        "unavailable",
                    ]
                )
                if is_transient and attempt < max_attempts:
                    time.sleep(6)
                    continue
                return ""

        return ""


global_quota_controller = QuotaPauseController(
    pause_every_calls=int(os.getenv("PAUSE_EVERY_SEARCH_CALLS", "0")),
    pause_seconds=int(os.getenv("PAUSE_SECONDS", "30")),
)
search_tool = RateLimitedGoogleSearchTool(inner=raw_search_tool, quota=global_quota_controller)


class ResearchContext:
    """Shared state for one church enrichment run."""

    def __init__(
        self,
        church: dict[str, Any],
        area: str,
        search: Any,
        persisted_cache: dict[str, str] | None = None,
        on_cache_update=None,
    ):
        self.church = dict(church)
        self.area = (area or "").strip()
        self.search = search
        self.cache: dict[str, str] = dict(persisted_cache or {})
        self.on_cache_update = on_cache_update

    @staticmethod
    def as_text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, (int, float, bool)):
            return str(value)
        if isinstance(value, list):
            return " ".join(ResearchContext.as_text(v) for v in value)
        if isinstance(value, dict):
            return " ".join(f"{k} {ResearchContext.as_text(v)}" for k, v in value.items())
        return str(value)

    @staticmethod
    def compact(text: str, max_len: int = 900) -> str:
        return re.sub(r"\s+", " ", text).strip()[:max_len]

    def base_query(self, suffix: str) -> str:
        name = str(self.church.get("name", "")).strip()
        # Force church-specific query scoping. Never run bulk/multi-church lookup prompts.
        if self.area:
            return f"\"{name}\" {self.area} {suffix}".strip()
        return f"\"{name}\" {suffix}".strip()

    def search_query(self, query: str) -> str:
        q = query.strip()
        if not q:
            return ""
        if q in self.cache:
            cached = self.cache[q]
            # Do not treat rate-limit markers/empty placeholders as durable cached answers.
            if cached and cached != "__RATE_LIMITED__":
                return cached

        result: Any = None
        try:
            result = self.search(q)
        except Exception as e:
            msg = str(e).lower()
            if "429" in msg or "resource_exhausted" in msg or "quota" in msg:
                # Keep prior successful cache entry if it exists; do not overwrite with empty.
                return ""
            # Non-quota error: keep prior cache untouched.
            return ""
        if isinstance(result, str) and result.startswith("__RATE_LIMITED__"):
            # Do not persist transient rate-limit sentinel as a cached search answer.
            return "__RATE_LIMITED__"
        text = self.as_text(result)
        if text.strip():
            self.cache[q] = text
            if self.on_cache_update:
                self.on_cache_update(q, text)
        return text


def _column_instruction(
    field_name: str,
    preserve_keys: list[str],
    query_suffixes: list[str],
    extraction_rule: str,
    manual_value: str = "Needs manual review",
) -> str:
    return (
        "You are a single-field enrichment subagent for worship centers.\n"
        "Input is JSON text with:\n"
        "- church: object\n"
        "- area: string\n\n"
        f"Field to produce: {field_name}\n"
        "Preserve existing values if any of these keys is present and non-empty/non-unknown: "
        + json.dumps(preserve_keys, ensure_ascii=False)
        + "\n"
        f"Run at most {max(1, int(os.getenv('MAX_SEARCHES_PER_ATTRIBUTE', '7')))} search calls. Only stop searching when you get given information\n"
        "Build worship-center-specific queries only with center name + area + suffix.\n"
        "Query suffix order: "
        + json.dumps(query_suffixes, ensure_ascii=False)
        + "\n"
        f"Extract value using this rule: {extraction_rule}\n"
        f"If still missing, return \"{manual_value}\".\n\n"
        "Output strict JSON only:\n"
        f'{{"field":"{field_name}","value":"<resolved value>"}}'
    )


def _column_generate_config() -> genai_types.GenerateContentConfig:
    return genai_types.GenerateContentConfig(
        automatic_function_calling=genai_types.AutomaticFunctionCallingConfig(
            disable=False,
            maximum_remote_calls=max(1, int(os.getenv("MAX_SEARCHES_PER_ATTRIBUTE", "7"))),
        ),
        temperature=0.0,
    )


address_subagent_agent_instance = Agent(
    name="address_subagent_agent",
    model=os.getenv("SEARCH_MODEL", "gemini-2.5-flash"),
    description="Resolve address for one church.",
    instruction=_column_instruction(
        "address",
        ["address"],
        ["address location", "google maps address", "official website address", "contact address"],
        'Return full street address. If not fully verified, return "ADDRESS NOT VERIFIED - needs manual review".',
        "ADDRESS NOT VERIFIED - needs manual review",
    ),
    tools=[search_tool],
    generate_content_config=_column_generate_config(),
)
phone_subagent_agent_instance = Agent(
    name="phone_subagent_agent",
    model=os.getenv("SEARCH_MODEL", "gemini-2.5-flash"),
    description="Resolve phone for one church.",
    instruction=_column_instruction(
        "phone",
        ["phone"],
        ["phone number", "contact phone", "google maps phone", "facebook phone"],
        "Return one normalized US phone number if found.",
    ),
    tools=[search_tool],
    generate_content_config=_column_generate_config(),
)
website_subagent_agent_instance = Agent(
    name="website_subagent_agent",
    model=os.getenv("SEARCH_MODEL", "gemini-2.5-flash"),
    description="Resolve website for one church.",
    instruction=_column_instruction(
        "website",
        ["website"],
        ["official website", "website", "homepage", "contact site"],
        "Return one canonical website URL with http/https.",
    ),
    tools=[search_tool],
    generate_content_config=_column_generate_config(),
)
attendance_subagent_agent_instance = Agent(
    name="attendance_subagent_agent",
    model=os.getenv("SEARCH_MODEL", "gemini-2.5-flash"),
    description="Resolve attendance for one church.",
    instruction=_column_instruction(
        "explicit_attendance",
        ["explicit_attendance", "attendance", "members", "families"],
        ["attendance members families", "about us congregation size", "membership count", "average sunday attendance"],
        "Return numeric attendance/membership text only.",
    ),
    tools=[search_tool],
    generate_content_config=_column_generate_config(),
)
google_reviews_subagent_agent_instance = Agent(
    name="google_reviews_subagent_agent",
    model=os.getenv("SEARCH_MODEL", "gemini-2.5-flash"),
    description="Resolve Google reviews for one church.",
    instruction=_column_instruction(
        "google_reviews",
        ["google_reviews", "review_count", "reviews"],
        ["google maps reviews", "google reviews", "maps rating reviews", "business profile reviews"],
        "Return numeric review count only.",
    ),
    tools=[search_tool],
    generate_content_config=_column_generate_config(),
)
yelp_reviews_subagent_agent_instance = Agent(
    name="yelp_reviews_subagent_agent",
    model=os.getenv("SEARCH_MODEL", "gemini-2.5-flash"),
    description="Resolve Yelp reviews for one church.",
    instruction=_column_instruction(
        "yelp_reviews",
        ["yelp_reviews", "yelp_review_count"],
        ["yelp reviews", "site:yelp.com reviews", "yelp rating", "yelp worship center listing"],
        "Return numeric Yelp review count only.",
    ),
    tools=[search_tool],
    generate_content_config=_column_generate_config(),
)
yahoo_local_reviews_subagent_agent_instance = Agent(
    name="yahoo_local_reviews_subagent_agent",
    model=os.getenv("SEARCH_MODEL", "gemini-2.5-flash"),
    description="Resolve Yahoo local reviews for one church.",
    instruction=_column_instruction(
        "yahoo_local_reviews",
        ["yahoo_local_reviews", "yahoo_reviews"],
        ["yahoo local reviews", "site:local.yahoo.com reviews", "yahoo reviews", "yahoo worship center listing"],
        "Return numeric Yahoo local review count only.",
    ),
    tools=[search_tool],
    generate_content_config=_column_generate_config(),
)
facebook_followers_subagent_agent_instance = Agent(
    name="facebook_followers_subagent_agent",
    model=os.getenv("SEARCH_MODEL", "gemini-2.5-flash"),
    description="Resolve Facebook followers for one church.",
    instruction=_column_instruction(
        "facebook_followers",
        ["facebook_followers", "fb_followers", "facebook_likes"],
        ["site:facebook.com followers", "facebook page likes", "facebook worship center page", "facebook ministry profile"],
        "Return follower/like count with optional k/m suffix.",
    ),
    tools=[search_tool],
    generate_content_config=_column_generate_config(),
)
instagram_followers_subagent_agent_instance = Agent(
    name="instagram_followers_subagent_agent",
    model=os.getenv("SEARCH_MODEL", "gemini-2.5-flash"),
    description="Resolve Instagram followers for one church.",
    instruction=_column_instruction(
        "instagram_followers",
        ["instagram_followers", "ig_followers"],
        ["site:instagram.com followers", "instagram profile followers", "instagram worship center account", "instagram ministry profile"],
        "Return follower count with optional k/m suffix.",
    ),
    tools=[search_tool],
    generate_content_config=_column_generate_config(),
)
youtube_subscribers_subagent_agent_instance = Agent(
    name="youtube_subscribers_subagent_agent",
    model=os.getenv("SEARCH_MODEL", "gemini-2.5-flash"),
    description="Resolve YouTube subscribers for one church.",
    instruction=_column_instruction(
        "youtube_subscribers",
        ["youtube_subscribers", "yt_subscribers"],
        ["site:youtube.com subscribers", "youtube channel subscribers", "youtube sermons", "youtube livestream worship center"],
        "Return subscriber count with optional k/m suffix.",
    ),
    tools=[search_tool],
    generate_content_config=_column_generate_config(),
)
activities_subagent_agent_instance = Agent(
    name="activities_subagent_agent",
    model=os.getenv("SEARCH_MODEL", "gemini-2.5-flash"),
    description="Resolve activities for one church.",
    instruction=_column_instruction(
        "activities",
        ["activities"],
        ["podcast livestream", "outreach youth ministry events", "community programs ministries", "news events sermons"],
        "Return comma-separated activity keywords (podcast/livestream/outreach/youth ministry/etc).",
    ),
    tools=[search_tool],
    generate_content_config=_column_generate_config(),
)
condition_subagent_agent_instance = Agent(
    name="condition_subagent_agent",
    model=os.getenv("SEARCH_MODEL", "gemini-2.5-flash"),
    description="Resolve condition notes for one church.",
    instruction=_column_instruction(
        "condition",
        ["condition"],
        ["historic aging new campus multi-generational community", "worship center history generations", "about us history mission", "youth ministry next generation"],
        "Return concise condition notes such as older/aging, very new, multi-generational.",
    ),
    tools=[search_tool],
    generate_content_config=_column_generate_config(),
)
signals_subagent_agent_instance = Agent(
    name="signals_subagent_agent",
    model=os.getenv("SEARCH_MODEL", "gemini-2.5-flash"),
    description="Resolve sale/lifecycle signals for one church.",
    instruction=_column_instruction(
        "signals",
        ["signals"],
        ["for sale closing merger lease declining membership", "worship center property listing", "ceasing operations", "facility rental hall rental"],
        "Return short signal summary text from strongest discovered evidence.",
    ),
    tools=[search_tool],
    generate_content_config=_column_generate_config(),
)

# Wrap all subagents with AgentTool instances (explicit style requested).
address_subagent_tool_instance = AgentTool(agent=address_subagent_agent_instance)
phone_subagent_tool_instance = AgentTool(agent=phone_subagent_agent_instance)
website_subagent_tool_instance = AgentTool(agent=website_subagent_agent_instance)
attendance_subagent_tool_instance = AgentTool(agent=attendance_subagent_agent_instance)
google_reviews_subagent_tool_instance = AgentTool(agent=google_reviews_subagent_agent_instance)
yelp_reviews_subagent_tool_instance = AgentTool(agent=yelp_reviews_subagent_agent_instance)
yahoo_local_reviews_subagent_tool_instance = AgentTool(agent=yahoo_local_reviews_subagent_agent_instance)
facebook_followers_subagent_tool_instance = AgentTool(agent=facebook_followers_subagent_agent_instance)
instagram_followers_subagent_tool_instance = AgentTool(agent=instagram_followers_subagent_agent_instance)
youtube_subscribers_subagent_tool_instance = AgentTool(agent=youtube_subscribers_subagent_agent_instance)
activities_subagent_tool_instance = AgentTool(agent=activities_subagent_agent_instance)
condition_subagent_tool_instance = AgentTool(agent=condition_subagent_agent_instance)
signals_subagent_tool_instance = AgentTool(agent=signals_subagent_agent_instance)

subagent_tool_calls = [
    ("address", address_subagent_tool_instance),
    ("phone", phone_subagent_tool_instance),
    ("website", website_subagent_tool_instance),
    ("explicit_attendance", attendance_subagent_tool_instance),
    ("google_reviews", google_reviews_subagent_tool_instance),
    ("yelp_reviews", yelp_reviews_subagent_tool_instance),
    ("yahoo_local_reviews", yahoo_local_reviews_subagent_tool_instance),
    ("facebook_followers", facebook_followers_subagent_tool_instance),
    ("instagram_followers", instagram_followers_subagent_tool_instance),
    ("youtube_subscribers", youtube_subscribers_subagent_tool_instance),
    ("activities", activities_subagent_tool_instance),
    ("condition", condition_subagent_tool_instance),
    ("signals", signals_subagent_tool_instance),
]


def _strip_code_fence(text: str) -> str:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_+-]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)
    return cleaned.strip()


def _extract_json_object(text: str) -> dict[str, Any]:
    cleaned = _strip_code_fence(text)
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        m = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not m:
            return {}
        try:
            parsed = json.loads(m.group(0))
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}


async def _run_agent_once_async(agent: Agent, request_text: str) -> str:
    timeout_sec = max(10, int(os.getenv("SUBAGENT_TIMEOUT_SEC", "40")))
    event_limit = max(10, int(os.getenv("SUBAGENT_EVENT_LIMIT", "80")))
    deadline = time.time() + timeout_sec
    session_service = InMemorySessionService()
    runner = Runner(
        app_name=agent.name,
        agent=agent,
        session_service=session_service,
        memory_service=InMemoryMemoryService(),
    )
    session = await session_service.create_session(app_name=agent.name, user_id="local_user", state={})
    last_content = None
    content = genai_types.Content(role="user", parts=[genai_types.Part.from_text(text=request_text)])
    async with aclosing(runner.run_async(user_id=session.user_id, session_id=session.id, new_message=content)) as agen:
        seen_events = 0
        async for event in agen:
            seen_events += 1
            if event.content:
                last_content = event.content
            if seen_events >= event_limit or time.time() >= deadline:
                break
    close_result = runner.close()
    if asyncio.iscoroutine(close_result):
        await close_result
    if last_content is None or not getattr(last_content, "parts", None):
        return ""
    return "\n".join(
        p.text for p in last_content.parts if getattr(p, "text", None) and not getattr(p, "thought", False)
    )


def _run_async_safely(coro):
    """
    Run a coroutine from sync code, even if an event loop is already active.
    """
    timeout_sec = max(10, int(os.getenv("SUBAGENT_SYNC_TIMEOUT_SEC", "50")))
    try:
        asyncio.get_running_loop()
        loop_running = True
    except RuntimeError:
        loop_running = False

    if not loop_running:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(lambda: asyncio.run(coro))
            try:
                return future.result(timeout=timeout_sec)
            except concurrent.futures.TimeoutError:
                print(f"[worker] subagent sync timeout after {timeout_sec}s; returning manual review")
                return ""

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(lambda: asyncio.run(coro))
        try:
            return future.result(timeout=timeout_sec)
        except concurrent.futures.TimeoutError:
            print(f"[worker] subagent sync timeout after {timeout_sec}s; returning manual review")
            return ""


def _run_subagent_tool_once(field_name: str, tool: AgentTool, church: dict[str, Any], area: str) -> str:
    invoker_agent = Agent(
        name=f"{field_name}_tool_invoker",
        model=os.getenv("SEARCH_MODEL", "gemini-2.5-flash"),
        description=f"Invoke tool for {field_name}.",
        instruction=(
            f"Call tool `{tool.name}` exactly once using the provided JSON input.\n"
            "Return strict JSON only in this shape:\n"
            f'{{"field":"{field_name}","value":"<resolved value>"}}\n'
            "No markdown."
        ),
        tools=[tool],
        generate_content_config=genai_types.GenerateContentConfig(
            automatic_function_calling=genai_types.AutomaticFunctionCallingConfig(
                disable=False,
                maximum_remote_calls=1,
            ),
            temperature=0.0,
        ),
    )
    payload = {"church": church, "area": area}
    response_text = _run_async_safely(_run_agent_once_async(invoker_agent, json.dumps(payload, ensure_ascii=False)))
    if not str(response_text).strip():
        return "Needs manual review"
    parsed = _extract_json_object(response_text)
    if str(parsed.get("value", "")).strip():
        return str(parsed.get("value")).strip()
    if str(parsed.get(field_name, "")).strip():
        return str(parsed.get(field_name)).strip()
    return "Needs manual review"


def _extract_website(text: str) -> str:
    m = re.search(r"(https?://[^\s)]+)", text, flags=re.IGNORECASE)
    return m.group(1) if m else ""


def _extract_phone(text: str) -> str:
    m = re.search(r"(\+?1?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})", text)
    return m.group(1) if m else ""


def _extract_address(text: str) -> str:
    m = re.search(
        r"(\d{1,6}\s+[a-z0-9.\-'\s]+(?:st|street|ave|avenue|rd|road|blvd|drive|dr|ln|lane|way|ct|court)\b[^,\n]*)",
        text,
        flags=re.IGNORECASE,
    )
    return m.group(1).strip() if m else ""


def _extract_numberish(text: str, keyword: str = "") -> str:
    pat = rf"{keyword}[^0-9]{{0,40}}(\d+(?:\.\d+)?[km]?)" if keyword else r"(\d+(?:\.\d+)?[km]?)"
    m = re.search(pat, text, flags=re.IGNORECASE)
    return m.group(1) if m else ""


def _resolve_field_direct(field_name: str, church: dict[str, Any], area: str, search: Any) -> str:
    name = str(church.get("name", "")).strip()
    min_searches = max(1, int(os.getenv("MIN_SEARCHES_PER_ATTRIBUTE", "3")))
    max_searches = max(min_searches, min(7, int(os.getenv("MAX_SEARCHES_PER_ATTRIBUTE", "7"))))

    preserve_map: dict[str, list[str]] = {
        "address": ["address"],
        "phone": ["phone"],
        "website": ["website"],
        "explicit_attendance": ["explicit_attendance", "attendance", "members", "families"],
        "google_reviews": ["google_reviews", "review_count", "reviews"],
        "yelp_reviews": ["yelp_reviews", "yelp_review_count"],
        "yahoo_local_reviews": ["yahoo_local_reviews", "yahoo_reviews"],
        "facebook_followers": ["facebook_followers", "fb_followers", "facebook_likes"],
        "instagram_followers": ["instagram_followers", "ig_followers"],
        "youtube_subscribers": ["youtube_subscribers", "yt_subscribers"],
        "activities": ["activities"],
        "condition": ["condition"],
        "signals": ["signals"],
    }

    for key in preserve_map.get(field_name, []):
        v = str(church.get(key, "")).strip()
        if v and v.lower() not in {"unknown", "needs manual review"}:
            return v

    suffixes_map: dict[str, list[str]] = {
        "address": ["address location", "google maps address", "official website address"],
        "phone": ["phone number", "contact phone", "google maps phone"],
        "website": ["official website", "homepage", "contact site"],
        "explicit_attendance": ["attendance members families", "congregation size", "average sunday attendance"],
        "google_reviews": ["google maps reviews", "google reviews"],
        "yelp_reviews": ["yelp reviews", "site:yelp.com reviews"],
        "yahoo_local_reviews": ["yahoo local reviews", "site:local.yahoo.com reviews"],
        "facebook_followers": ["site:facebook.com followers", "facebook page likes"],
        "instagram_followers": ["site:instagram.com followers", "instagram profile followers"],
        "youtube_subscribers": ["site:youtube.com subscribers", "youtube channel subscribers"],
        "activities": ["podcast livestream outreach youth ministry", "community programs ministries events"],
        "condition": ["historic aging new campus multi-generational", "worship center history mission youth ministry"],
        "signals": ["for sale closing merger lease", "worship center property listing ceasing operations"],
    }

    extracted_signals: list[str] = []
    found_value = ""
    base_suffixes = suffixes_map.get(field_name, [])
    query_modifiers = ["", "official", "contact", "directory", "latest", "local listing", "community"]
    queries: list[str] = []
    seen: set[str] = set()
    for suffix in base_suffixes:
        for mod in query_modifiers:
            q = f"\"{name}\" {area} {suffix} {mod}".strip()
            if q in seen:
                continue
            seen.add(q)
            queries.append(q)
            if len(queries) >= max_searches:
                break
        if len(queries) >= max_searches:
            break

    for idx, query in enumerate(queries):
        result = search(query)
        text = ResearchContext.as_text(result)
        if not text or text.startswith("__RATE_LIMITED__") or text.startswith("__TRANSIENT_TIMEOUT__"):
            continue

        if field_name == "address":
            val = _extract_address(text)
            if val:
                found_value = val
        elif field_name == "phone":
            val = _extract_phone(text)
            if val:
                found_value = val
        elif field_name == "website":
            val = _extract_website(text)
            if val:
                found_value = val
        elif field_name in {"explicit_attendance", "google_reviews", "yelp_reviews", "yahoo_local_reviews"}:
            val = _extract_numberish(text)
            if val:
                found_value = val
        elif field_name == "facebook_followers":
            val = _extract_numberish(text, "facebook")
            if val:
                found_value = val
        elif field_name == "instagram_followers":
            val = _extract_numberish(text, "instagram")
            if val:
                found_value = val
        elif field_name == "youtube_subscribers":
            val = _extract_numberish(text, "youtube")
            if val:
                found_value = val
        elif field_name == "activities":
            lower = text.lower()
            keys = [k for k in ["podcast", "livestream", "outreach", "youth ministry", "bible study", "community"] if k in lower]
            if keys:
                found_value = ", ".join(sorted(set(keys)))
        elif field_name == "condition":
            lower = text.lower()
            notes: list[str] = []
            if any(k in lower for k in ["historic", "aging", "older", "long-standing"]):
                notes.append("older/aging")
            if any(k in lower for k in ["new worship center", "new campus", "newly opened", "recently opened"]):
                notes.append("very new")
            if any(k in lower for k in ["multi-generational", "multigenerational", "served generations"]):
                notes.append("multi-generational")
            if notes:
                found_value = "; ".join(notes)
        elif field_name == "signals":
            extracted_signals.append(ResearchContext.compact(text, max_len=240))

        # Require at least min_searches attempts; after that, stop as soon as we find the value.
        if found_value and (idx + 1) >= min_searches:
            return found_value

    if field_name == "address":
        return "ADDRESS NOT VERIFIED - needs manual review"
    if found_value:
        return found_value
    if field_name == "signals" and extracted_signals:
        return " | ".join(extracted_signals[:2])
    return "Needs manual review"


def run_worker_subagents(church: dict[str, Any], area: str, search: Any) -> dict[str, Any]:
    merged = dict(church)
    church_name = str(merged.get("name", "")).strip() or "<unknown church>"
    for field_name, _tool in subagent_tool_calls:
        print(f"[worker] {church_name} -> resolving `{field_name}`")
        try:
            value = _resolve_field_direct(field_name=field_name, church=merged, area=area, search=search)
        except Exception:
            value = "Needs manual review"
        if field_name == "address" and value == "Needs manual review":
            value = "ADDRESS NOT VERIFIED - needs manual review"
        merged[field_name] = value
        print(f"[worker] {church_name} -> `{field_name}` done")
    if not str(merged.get("followers", "")).strip():
        merged["followers"] = merged.get("explicit_attendance", "Needs manual review")
    return merged


class EnrichChurchesAndWriteTool:
    """
    Root orchestration tool:
    1) accepts church array from root agent
    2) loops each church and runs Class X
    3) writes enriched output once to spreadsheet
    """

    def __init__(self, search: Any, sheet_tool: GoogleSheetsSpreadsheetTool, quota: QuotaPauseController):
        self.search = search
        self.sheet_tool = sheet_tool
        self.quota = quota
        self.store = ProgressStore(STATE_PATH)
        self.__name__ = "enrich_churches_and_write"
        self.__doc__ = "Enrich each church across all spreadsheet columns, then write once to Google Sheets"

    @staticmethod
    def _church_key(church: dict[str, Any], area: str) -> str:
        name = str(church.get("name", "")).strip().lower()
        addr = str(church.get("address", "")).strip().lower()
        area_norm = (area or "").strip().lower()
        return f"{area_norm}|{name}|{addr}"

    def __call__(self, churches_json: str, area: str = "") -> str:
        try:
            churches = json.loads(churches_json)
            if not isinstance(churches, list):
                return "Invalid churches_json: expected a JSON list."
        except json.JSONDecodeError as e:
            return f"Invalid churches_json: {e}"

        state = self.store.load()
        search_cache: dict[str, str] = state.get("search_cache", {})
        church_results: dict[str, dict[str, Any]] = state.get("church_results", {})

        def on_cache_update(query: str, value: str) -> None:
            search_cache[query] = value

        enriched: list[dict[str, Any]] = []
        max_calls_per_run = max(1, int(os.getenv("MAX_SEARCH_CALLS_PER_RUN", "9000000")))
        for church in churches:
            if not isinstance(church, dict):
                continue
            if self.quota.total_calls >= max_calls_per_run:
                row = {
                    **church,
                    "explicit_attendance": church.get("explicit_attendance", "Needs manual review"),
                    "google_reviews": church.get("google_reviews", "Needs manual review"),
                    "yelp_reviews": church.get("yelp_reviews", "Needs manual review"),
                    "yahoo_local_reviews": church.get("yahoo_local_reviews", "Needs manual review"),
                    "facebook_followers": church.get("facebook_followers", "Needs manual review"),
                    "instagram_followers": church.get("instagram_followers", "Needs manual review"),
                    "youtube_subscribers": church.get("youtube_subscribers", "Needs manual review"),
                    "activities": church.get("activities", "Needs manual review"),
                    "condition": church.get("condition", "Needs manual review"),
                    "signals": "Run capped by MAX_SEARCH_CALLS_PER_RUN",
                    "followers": church.get("followers", "Needs manual review"),
                }
                enriched.append(row)
                continue
            key = self._church_key(church, area)
            use_row_cache = os.getenv("ENABLE_CHURCH_RESULT_CACHE", "0").strip() == "1"
            cached_row = church_results.get(key)
            if use_row_cache and isinstance(cached_row, dict):
                enriched.append(cached_row)
                continue
            if isinstance(self.search, RateLimitedGoogleSearchTool) and time.time() < self.search.cooldown_until:
                wait_left = max(1, int(self.search.cooldown_until - time.time()))
                print(
                    f"[enrich_churches_and_write] Cooldown active ({wait_left}s). "
                    "Waiting and then continuing research for remaining worship centers..."
                )
                time.sleep(wait_left)
            row = run_worker_subagents(church=church, area=area, search=self.search)
            enriched.append(row)
            church_results[key] = row
            self.store.save({"search_cache": search_cache, "church_results": church_results})

        print(f"[enrich_churches_and_write] Enrichment complete. Rows: {len(enriched)}. Starting sheet write...")
        result = self.sheet_tool.write_churches(enriched)
        print(f"[enrich_churches_and_write] Sheet write finished. Result: {result}")
        self.store.save({"search_cache": search_cache, "church_results": church_results})
        return f"{result} (Search calls used: {self.quota.total_calls}; pause_for_quota used explicitly.)"


enrich_churches_and_write = EnrichChurchesAndWriteTool(
    search=search_tool,
    sheet_tool=spreadsheet_tool,
    quota=global_quota_controller,
)


root_agent = Agent(
    name="church_finder_agent",
    model="gemini-2.5-flash",
    description="Find worship centers in a user-provided area and enrich all spreadsheet columns per center.",
    instruction=(
        """
You help find worship centers in a user-provided area.

Workflow (must follow):
1) Discovery:
- Build an array of worship centers for the requested area using search.
- Deduplicate by name + address.
- Keep at least name and address for each worship center.
- Search calls can run back-to-back.
-Also, you may search worship center to help find more options (doesn't have to be strictly church)
- if the user asks for multiple areas look through all of those areas


2) Enrichment delegation:
- Call enrich_churches_and_write once with:
  - churches_json: JSON array from step 1
  - area: user location
- Do not call write_churches_to_sheet directly unless enrich_churches_and_write fails.
- Enrichment research does at least 3 searches per attribute (up to 7), and may stop early only after finding the desired value.
- IMPORTANT: Do per-worship-center, per-attribute lookups only.
- Never run one search intended to fetch addresses/reviews/followers for multiple worship centers at once.
- For each center row, resolve address/reviews/socials/activities independently.
- Pause policy: use pause_for_quota only when needed, and never call pause_for_quota twice in a row.

3) Quota behavior:
- If search errors mention 429/quota/resource exhausted, stop adding new searches.
- Return partial output and state that user can rerun for another pass.

Output requirements for discovery list:
- Every row must include non-empty address.
- If full address cannot be verified: "ADDRESS NOT VERIFIED - needs manual review".
- Keep JSON compact and valid.
        """
    ),
    tools=[search_tool, pause_for_quota, enrich_churches_and_write, write_churches_to_sheet],
    generate_content_config=genai_types.GenerateContentConfig(
        automatic_function_calling=genai_types.AutomaticFunctionCallingConfig(
            disable=False,
            maximum_remote_calls=10,
        )
    ),
)


# Alternative names ADK may look for.
agent = root_agent
main_agent = root_agent
