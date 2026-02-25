import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from google import genai
from google.adk.agents.llm_agent import Agent
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
        self.pause_every_calls = max(1, int(pause_every_calls))
        self.pause_seconds = max(1, int(pause_seconds))
        self.total_calls = 0

    def register_search_call(self, apply_sleep: bool = True) -> None:
        self.total_calls += 1
        if apply_sleep and self.total_calls % self.pause_every_calls == 0:
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


global_quota_controller = QuotaPauseController(pause_every_calls=1, pause_seconds=30)
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


class BaseColumnSubagent:
    """Subagent contract for one spreadsheet column/field."""

    field_name: str = ""

    def __init__(self, ctx: ResearchContext):
        self.ctx = ctx

    def existing(self, *keys: str) -> str | None:
        for key in keys:
            v = str(self.ctx.church.get(key, "")).strip()
            if v and v.lower() != "unknown":
                return v
        return None

    @staticmethod
    def pick(text: str, pattern: str) -> str:
        m = re.search(pattern, text, flags=re.IGNORECASE)
        return m.group(1) if m else "Unknown"

    def run(self) -> str:
        raise NotImplementedError

    @staticmethod
    def is_found(value: str) -> bool:
        v = (value or "").strip().lower()
        return bool(v) and v not in {"unknown", "needs manual review"}

    def search_until_found(self, query_suffixes: list[str], extractor) -> str:
        """
        Literal subagent behavior:
        - Run primary queries first, then deeper fallback queries for this column.
        - Stop as soon as info is found.
        - Otherwise return Needs manual review.
        """
        max_queries = max(1, int(os.getenv("MAX_SEARCHES_PER_ATTRIBUTE", "2")))
        for suffix in query_suffixes[:max_queries]:
            text = self.ctx.search_query(self.ctx.base_query(suffix))
            if text.startswith("__RATE_LIMITED__") or text.startswith("__TRANSIENT_TIMEOUT__"):
                return "Needs manual review"
            value = extractor(text)
            if self.is_found(value):
                return value
        return "Needs manual review"


class AddressSubagent(BaseColumnSubagent):
    field_name = "address"

    def run(self) -> str:
        v = self.existing("address")
        if v and "not verified" not in v.lower():
            return v
        result = self.search_until_found(
            [
                "address location",
                "google maps address",
                "official website address",
                "contact address",
            ],
            lambda t: self.pick(
                t,
                r"(\d{1,6}\s+[a-z0-9.\-'\s]+(?:st|street|ave|avenue|rd|road|blvd|drive|dr|ln|lane|way|ct|court)\b[^,.]*)",
            ),
        )
        return "ADDRESS NOT VERIFIED - needs manual review" if result == "Needs manual review" else result


class PhoneSubagent(BaseColumnSubagent):
    field_name = "phone"

    def run(self) -> str:
        v = self.existing("phone")
        if v:
            return v
        return self.search_until_found(
            [
                "phone number",
                "contact phone",
                "google maps phone",
                "facebook phone",
            ],
            lambda t: self.pick(t, r"(\+?1?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})"),
        )


class WebsiteSubagent(BaseColumnSubagent):
    field_name = "website"

    def run(self) -> str:
        v = self.existing("website")
        if v:
            return v
        def extract(text: str) -> str:
            m = re.search(r"(https?://[^\s)]+)", text, flags=re.IGNORECASE)
            return m.group(1) if m else "Unknown"

        return self.search_until_found(
            [
                "official website",
                "website",
                "homepage",
                "contact site",
            ],
            extract,
        )


class AttendanceSubagent(BaseColumnSubagent):
    field_name = "explicit_attendance"

    def run(self) -> str:
        v = self.existing("explicit_attendance", "attendance", "members", "families")
        if v:
            return v
        def extract(text: str) -> str:
            m = re.search(
                r"(?:attendance|members?|families|congregation)\D{0,20}(\d{1,4})|(\d{1,4})\D{0,20}(?:attendance|members?|families|congregation)",
                text,
                flags=re.IGNORECASE,
            )
            if not m:
                return "Unknown"
            return (m.group(1) or m.group(2) or "Unknown").strip()

        return self.search_until_found(
            [
                "attendance members families",
                "about us congregation size",
                "membership count",
                "average sunday attendance",
            ],
            extract,
        )


class GoogleReviewsSubagent(BaseColumnSubagent):
    field_name = "google_reviews"

    def run(self) -> str:
        v = self.existing("google_reviews", "review_count", "reviews")
        if v:
            return v
        return self.search_until_found(
            [
                "google maps reviews",
                "google reviews",
                "maps rating reviews",
                "business profile reviews",
            ],
            lambda t: self.pick(t, r"(\d{1,5})\s+reviews?"),
        )


class YelpReviewsSubagent(BaseColumnSubagent):
    field_name = "yelp_reviews"

    def run(self) -> str:
        v = self.existing("yelp_reviews", "yelp_review_count")
        if v:
            return v
        return self.search_until_found(
            [
                "yelp reviews",
                "site:yelp.com reviews",
                "yelp rating",
                "yelp church listing",
            ],
            lambda t: self.pick(t, r"yelp[^0-9]{0,40}(\d{1,5})\s+reviews?"),
        )


class YahooLocalReviewsSubagent(BaseColumnSubagent):
    field_name = "yahoo_local_reviews"

    def run(self) -> str:
        v = self.existing("yahoo_local_reviews", "yahoo_reviews")
        if v:
            return v
        return self.search_until_found(
            [
                "yahoo local reviews",
                "site:local.yahoo.com reviews",
                "yahoo reviews",
                "yahoo church listing",
            ],
            lambda t: self.pick(t, r"yahoo(?:\s+local)?[^0-9]{0,40}(\d{1,5})\s+reviews?"),
        )


class FacebookFollowersSubagent(BaseColumnSubagent):
    field_name = "facebook_followers"

    def run(self) -> str:
        v = self.existing("facebook_followers", "fb_followers", "facebook_likes")
        if v:
            return v
        return self.search_until_found(
            [
                "site:facebook.com followers",
                "facebook page likes",
                "facebook church page",
                "facebook ministry profile",
            ],
            lambda t: self.pick(t, r"(\d+(?:\.\d+)?[km]?)\s+(?:followers?|likes?)"),
        )


class InstagramFollowersSubagent(BaseColumnSubagent):
    field_name = "instagram_followers"

    def run(self) -> str:
        v = self.existing("instagram_followers", "ig_followers")
        if v:
            return v
        return self.search_until_found(
            [
                "site:instagram.com followers",
                "instagram profile followers",
                "instagram church account",
                "instagram ministry profile",
            ],
            lambda t: self.pick(t, r"(\d+(?:\.\d+)?[km]?)\s+followers?"),
        )


class YouTubeSubscribersSubagent(BaseColumnSubagent):
    field_name = "youtube_subscribers"

    def run(self) -> str:
        v = self.existing("youtube_subscribers", "yt_subscribers")
        if v:
            return v
        return self.search_until_found(
            [
                "site:youtube.com subscribers",
                "youtube channel subscribers",
                "youtube sermons",
                "youtube livestream church",
            ],
            lambda t: self.pick(t, r"(\d+(?:\.\d+)?[km]?)\s+subscribers?"),
        )


class ActivitiesSubagent(BaseColumnSubagent):
    field_name = "activities"

    def run(self) -> str:
        v = self.existing("activities")
        if v:
            return v
        keywords = [
            "podcast",
            "livestream",
            "youtube",
            "community outreach",
            "food pantry",
            "youth ministry",
            "bible study",
            "school",
            "daycare",
            "mission",
            "retreat",
            "conference",
        ]

        def extract(text: str) -> str:
            lower = text.lower()
            found = [k for k in keywords if k in lower]
            return ", ".join(sorted(set(found))) if found else "Unknown"

        return self.search_until_found(
            [
                "podcast livestream",
                "outreach youth ministry events",
                "community programs ministries",
                "news events sermons",
            ],
            extract,
        )


class ConditionSubagent(BaseColumnSubagent):
    field_name = "condition"

    def run(self) -> str:
        v = self.existing("condition")
        if v:
            return v
        def extract(text: str) -> str:
            lower = text.lower()
            notes: list[str] = []
            if any(k in lower for k in ["historic", "older", "aging", "long-standing", "longstanding"]):
                notes.append("older/aging")
            if any(k in lower for k in ["new church", "new campus", "newly opened", "recently opened"]):
                notes.append("very new")
            if any(k in lower for k in ["multi-generational", "multigenerational", "served generations"]):
                notes.append("multi-generational")
            if any(k in lower for k in ["youth ministry", "next generation", "younger generation"]):
                notes.append("next-generation dependence signals")
            return "; ".join(notes) if notes else "Unknown"

        return self.search_until_found(
            [
                "historic aging new campus multi-generational community",
                "church history generations",
                "about us history mission",
                "youth ministry next generation",
            ],
            extract,
        )


class SignalsSubagent(BaseColumnSubagent):
    field_name = "signals"

    def run(self) -> str:
        v = self.existing("signals")
        if v:
            return v
        merged = " ".join(self.ctx.cache.values())
        return self.ctx.compact(merged) if merged.strip() else "Needs manual review"


class ChurchResearchWorker:
    """
    Class X: church-specific orchestrator that runs subagent tools per column.
    """

    def __init__(
        self,
        church: dict[str, Any],
        area: str,
        search: Any,
        persisted_cache: dict[str, str] | None = None,
        on_cache_update=None,
    ):
        self.ctx = ResearchContext(
            church=church,
            area=area,
            search=search,
            persisted_cache=persisted_cache,
            on_cache_update=on_cache_update,
        )
        self.subagents: list[BaseColumnSubagent] = [
            AddressSubagent(self.ctx),
            PhoneSubagent(self.ctx),
            WebsiteSubagent(self.ctx),
            AttendanceSubagent(self.ctx),
            GoogleReviewsSubagent(self.ctx),
            YelpReviewsSubagent(self.ctx),
            YahooLocalReviewsSubagent(self.ctx),
            FacebookFollowersSubagent(self.ctx),
            InstagramFollowersSubagent(self.ctx),
            YouTubeSubscribersSubagent(self.ctx),
            ActivitiesSubagent(self.ctx),
            ConditionSubagent(self.ctx),
            SignalsSubagent(self.ctx),
        ]

    def run_all_column_tools(self) -> dict[str, Any]:
        for subagent in self.subagents:
            try:
                self.ctx.church[subagent.field_name] = subagent.run()
            except Exception:
                self.ctx.church[subagent.field_name] = self.ctx.church.get(subagent.field_name, "Needs manual review") or "Needs manual review"

        if not str(self.ctx.church.get("followers", "")).strip():
            self.ctx.church["followers"] = self.ctx.church.get("explicit_attendance", "Needs manual review")
        return self.ctx.church


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
        max_calls_per_run = max(10, int(os.getenv("MAX_SEARCH_CALLS_PER_RUN", "60")))
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
            cached_row = church_results.get(key)
            if isinstance(cached_row, dict):
                enriched.append(cached_row)
                continue
            if isinstance(self.search, RateLimitedGoogleSearchTool) and time.time() < self.search.cooldown_until:
                # During cooldown, skip expensive enrichment and mark for manual review.
                row = (
                    {
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
                        "signals": f"Rate-limited: {self.search.last_error or 'RESOURCE_EXHAUSTED'}",
                        "followers": church.get("followers", "Needs manual review"),
                    }
                )
                enriched.append(row)
                church_results[key] = row
                self.store.save({"search_cache": search_cache, "church_results": church_results})
                continue
            worker = ChurchResearchWorker(
                church=church,
                area=area,
                search=self.search,
                persisted_cache=search_cache,
                on_cache_update=on_cache_update,
            )
            row = worker.run_all_column_tools()
            enriched.append(row)
            church_results[key] = row
            self.store.save({"search_cache": search_cache, "church_results": church_results})

        result = self.sheet_tool.write_churches(enriched)
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
    description="Find churches in a user-provided area and enrich all spreadsheet columns per church.",
    instruction=(
        """
You help find churches in a user-provided area.

Workflow (must follow):
1) Discovery:
- Build an array of churches for the requested area using search.
- Deduplicate by name + address.
- Keep at least name and address for each church.
- Search calls can run back-to-back.

2) Enrichment delegation:
- Call enrich_churches_and_write once with:
  - churches_json: JSON array from step 1
  - area: user location
- Do not call write_churches_to_sheet directly unless enrich_churches_and_write fails.
- Enrichment subagents each do 1-2 searches for their own column, then return "Needs manual review" if still missing.
- IMPORTANT: Do per-church, per-attribute lookups only.
- Never run one search intended to fetch addresses/reviews/followers for multiple churches at once.
- For each church row, resolve address/reviews/socials/activities independently.
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
