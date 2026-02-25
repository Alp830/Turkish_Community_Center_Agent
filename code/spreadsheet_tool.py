"""Google Sheets tool for writing church data to spreadsheet.

Upgrades:
- Factor-based deal scoring with auditability:
  * Attendance/community strength
  * Facility condition/capex signals
  * Lifecycle/sustainability signals
  * Sale-likelihood signals (for sale/lease/closing/merger, etc.)
- Weighted attendance estimation:
  * Explicit attendance/membership numbers (highest confidence)
  * Google reviews proxy (reviews * 5)
  * Facebook fallback (followers * 0.2)
  * Instagram/YouTube tertiary fallback
- Community Strength Index (CSI) column:
  * Based on attendance estimate + digital footprint strength
- Stores score breakdown + signals found + followers confidence/source.
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, Tuple
import time

import gspread


class GoogleSheetsSpreadsheetTool:
    """Tool to write church/mosque data to Google Sheets."""

    def __init__(self, spreadsheet_name: str = "Church Finder Results"):
        self.name = "google_sheets_spreadsheet_tool"
        self.description = "Write church/mosque data into a Google Sheets spreadsheet."
        self.spreadsheet_name = spreadsheet_name
        self._client = None  # Lazy load on first use

        # Expanded headers to include audit/debug info for grading.
        self.headers = [
            "Church Name",
            "Address",
            "Phone Number",
            "Website",
            "Followers (Estimated)",
            "Followers Estimate Source",
            "Followers Confidence",
            "Google Reviews",
            "Facebook Followers",
            "Instagram Followers",
            "YouTube Subscribers",
            "Digital Footprint Score (1-10)",
            "Engagement Strength",
            "Community Strength Index (1-10)",
            "Condition Notes",
            "Sale Signals Found",
            "Deal Score Breakdown",
            "Deal Score (1-10)",
        ]

    # -----------------------------
    # Parsing / extraction helpers
    # -----------------------------

    @staticmethod
    def _parse_followers(value: Any) -> int | None:
        """Parse follower count from numeric or text values (supports k/m suffixes)."""
        if value is None:
            return None
        if isinstance(value, int):
            return value
        text = str(value).strip()
        if not text:
            return None
        normalized = text.lower().replace(",", "")

        # Supports values like 1.2k, 3m, 850 followers.
        m = re.search(r"(\d+(?:\.\d+)?)\s*([km])?\b", normalized)
        if not m:
            digits = "".join(ch for ch in normalized if ch.isdigit())
            if not digits:
                return None
            try:
                return int(digits)
            except ValueError:
                return None

        number = float(m.group(1))
        suffix = m.group(2)
        if suffix == "k":
            number *= 1_000
        elif suffix == "m":
            number *= 1_000_000
        return int(number)

    @classmethod
    def _extract_explicit_attendance(cls, *texts: Any) -> int | None:
        """Extract attendance/membership number from text snippets."""
        blob = cls._norm_text(*texts)
        if not blob:
            return None

        patterns = [
            r"average (?:sunday )?attendance\s*(?:of|is|:)?\s*(\d{1,4})",
            r"(?:serve|serves|serving)\s*(\d{1,4})\s*(?:families|members|attendees|worshipers|worshippers|congregants)",
            r"congregation of\s*(\d{1,4})",
            r"(\d{1,4})\s*(?:families|members|attendees|worshipers|worshippers|congregants)",
        ]
        for pat in patterns:
            m = re.search(pat, blob)
            if not m:
                continue
            value = cls._parse_followers(m.group(1))
            if value is not None and 5 <= value <= 5000:
                return value
        return None

    @classmethod
    def _extract_review_count(cls, *texts: Any) -> int | None:
        """Extract Google-style review counts from snippets."""
        blob = cls._norm_text(*texts)
        if not blob:
            return None

        patterns = [
            r"\b(\d{1,5})\s*reviews?\b",
            r"rated\s*\d(?:\.\d)?\s*\(?\s*(\d{1,5})\s*reviews?\)?",
        ]
        for pat in patterns:
            m = re.search(pat, blob)
            if not m:
                continue
            value = cls._parse_followers(m.group(1))
            if value is not None and value >= 0:
                return value
        return None

    @classmethod
    def _extract_social_count(cls, platform: str, *texts: Any) -> int | None:
        """Extract social follower/subscriber counts for a platform from free text."""
        blob = cls._norm_text(*texts)
        if not blob:
            return None

        platform = platform.strip().lower()
        platform_aliases = {
            "facebook": ["facebook", "fb"],
            "instagram": ["instagram", "insta", "ig"],
            "youtube": ["youtube", "yt"],
        }
        aliases = platform_aliases.get(platform, [platform])
        alias_group = "(?:" + "|".join(re.escape(a) for a in aliases) + ")"

        patterns = [
            rf"{alias_group}[^0-9]{{0,40}}(\d+(?:\.\d+)?)\s*([km])?\s*(?:followers?|likes?|subscribers?)",
            rf"(\d+(?:\.\d+)?)\s*([km])?\s*(?:followers?|likes?|subscribers?)[^a-z0-9]{{0,30}}{alias_group}",
            rf"{alias_group}[^0-9]{{0,30}}(\d{{2,7}})",
        ]

        for pat in patterns:
            m = re.search(pat, blob)
            if not m:
                continue
            number = m.group(1)
            suffix = m.group(2) if len(m.groups()) >= 2 else None
            text_value = f"{number}{suffix or ''}"
            value = cls._parse_followers(text_value)
            if value is not None and value >= 0:
                return value
        return None

    @staticmethod
    def _clip_score_1_10(value: float) -> int:
        """Clip float score to 1-10 integer."""
        return max(1, min(10, int(round(value))))

    @classmethod
    def _digital_footprint(
        cls,
        reviews: int | None,
        facebook: int | None,
        instagram: int | None,
        youtube: int | None,
    ) -> Tuple[int, str]:
        """
        Compute digital footprint score and engagement label.
        Higher means stronger online presence.
        """
        raw = 1.0
        if reviews is not None:
            raw += min(4.0, reviews / 40.0)
        if facebook is not None:
            raw += min(2.5, facebook / 500.0)
        if instagram is not None:
            raw += min(1.5, instagram / 1000.0)
        if youtube is not None:
            raw += min(1.0, youtube / 500.0)

        score = cls._clip_score_1_10(raw)
        if score <= 3:
            label = "Weak"
        elif score <= 6:
            label = "Moderate"
        else:
            label = "Strong"
        return score, label

    @classmethod
    def _community_strength_index(
        cls,
        followers_estimated: int | None,
        digital_footprint_score: int,
    ) -> int:
        """
        Community Strength Index (CSI), where higher means stronger congregation/community.
        Combines estimated attendance with digital footprint.
        """
        if followers_estimated is None:
            attendance_component = 1.0
        elif followers_estimated < 50:
            attendance_component = 2.0
        elif followers_estimated < 150:
            attendance_component = 4.0
        elif followers_estimated < 300:
            attendance_component = 6.0
        elif followers_estimated < 600:
            attendance_component = 7.5
        elif followers_estimated < 1000:
            attendance_component = 8.5
        else:
            attendance_component = 9.5

        # Attendance is the stronger signal for this workflow.
        raw = (attendance_component * 0.75) + (digital_footprint_score * 0.25)
        return cls._clip_score_1_10(raw)

    @classmethod
    def _estimate_followers(
        cls,
        church: dict[str, Any],
        condition: str,
        signals_text: str,
    ) -> Tuple[int | None, str, str, int | None, int | None, int | None, int | None]:
        """
        Weighted attendance estimate strategy.
        Returns:
        (estimated_followers, estimate_source, confidence, reviews, facebook, instagram, youtube)
        """
        explicit_attendance = cls._parse_followers(church.get("explicit_attendance"))
        if explicit_attendance is None:
            explicit_attendance = cls._extract_explicit_attendance(
                church.get("followers"),
                condition,
                signals_text,
            )

        reviews = cls._parse_followers(church.get("google_reviews"))
        if reviews is None:
            reviews = cls._parse_followers(church.get("review_count"))
        if reviews is None:
            reviews = cls._extract_review_count(
                signals_text,
                church.get("followers"),
                condition,
            )

        facebook = cls._parse_followers(church.get("facebook_followers"))
        if facebook is None:
            facebook = cls._extract_social_count(
                "facebook",
                signals_text,
                church.get("followers"),
                condition,
            )

        instagram = cls._parse_followers(church.get("instagram_followers"))
        if instagram is None:
            instagram = cls._extract_social_count(
                "instagram",
                signals_text,
                church.get("followers"),
                condition,
            )

        youtube = cls._parse_followers(church.get("youtube_subscribers"))
        if youtube is None:
            youtube = cls._extract_social_count(
                "youtube",
                signals_text,
                church.get("followers"),
                condition,
            )

        if explicit_attendance is not None:
            return explicit_attendance, "Explicit attendance/membership", "High", reviews, facebook, instagram, youtube
        if reviews is not None:
            return int(reviews * 5), "Google reviews x5 proxy", "Medium", reviews, facebook, instagram, youtube
        if facebook is not None:
            return int(facebook * 0.2), "Facebook followers x0.2", "Low", reviews, facebook, instagram, youtube
        if instagram is not None:
            return int(instagram * 0.12), "Instagram followers x0.12", "Low", reviews, facebook, instagram, youtube
        if youtube is not None:
            return int(youtube * 0.2), "YouTube subscribers x0.2", "Low", reviews, facebook, instagram, youtube

        # Backward compatibility if upstream agent only provides followers.
        fallback = cls._parse_followers(church.get("followers"))
        if fallback is not None:
            return fallback, "Provided followers field", "Low", reviews, facebook, instagram, youtube
        return None, "Unknown", "Low", reviews, facebook, instagram, youtube

    @staticmethod
    def _followers_confidence(followers_raw: Any) -> str:
        """
        Heuristic confidence:
        - High: explicit numeric value (int) or text containing clear digits.
        - Medium: has fuzzy words suggesting estimate ("about", "approx", "~") + digits.
        - Low: no digits / unknown / purely qualitative.
        """
        if followers_raw is None:
            return "Low"
        if isinstance(followers_raw, int):
            return "High"

        text = str(followers_raw).strip().lower()
        if not text:
            return "Low"

        has_digits = any(ch.isdigit() for ch in text)
        if not has_digits:
            return "Low"

        if any(k in text for k in ["about", "approx", "approximately", "~", "est", "estimate", "estimated"]):
            return "Medium"
        return "High"

    @staticmethod
    def _norm_text(*parts: Any) -> str:
        """Normalize multiple text parts into one searchable blob."""
        joined = " ".join(str(p) for p in parts if p is not None).strip().lower()
        joined = re.sub(r"\s+", " ", joined)
        return joined

    @staticmethod
    def _find_keywords(text: str, keywords: Iterable[str]) -> list[str]:
        """Return list of matched keywords (unique, stable order by input)."""
        matches: list[str] = []
        for k in keywords:
            if k in text and k not in matches:
                matches.append(k)
        return matches

    # -----------------------------
    # Scoring (factor-based)
    # -----------------------------

    @staticmethod
    def _attendance_points(followers: int | None) -> int:
        """
        Attendance/community strength points.
        Higher = more likely deal (weaker community).
        """
        if followers is None:
            return 0
        if followers <= 50:
            return 3
        if followers <= 150:
            return 2
        if followers <= 300:
            return 1
        if followers <= 600:
            return 0
        if followers <= 1000:
            return -1
        return -2

    @staticmethod
    def _condition_points(text: str) -> Tuple[int, list[str]]:
        """
        Facility condition/capex points.
        Higher = more likely deal (needs work / deferred maintenance).
        Returns (points, matched_signals).
        """
        poor = ["falling apart", "dilapidated", "deferred maintenance", "needs renovation", "needs repairs", "poor condition"]
        fair = ["fair", "dated", "older building", "needs updates", "outdated"]
        good = ["good", "well maintained", "well-maintained", "good condition"]
        excellent = ["excellent", "newly renovated", "recently renovated", "fully renovated", "strong community"]

        matched: list[str] = []
        if any(k in text for k in poor):
            matched += [k for k in poor if k in text]
            return 2, matched
        if any(k in text for k in fair):
            matched += [k for k in fair if k in text]
            return 1, matched
        if any(k in text for k in good):
            matched += [k for k in good if k in text]
            return -1, matched
        if any(k in text for k in excellent):
            matched += [k for k in excellent if k in text]
            return -2, matched
        return 0, matched

    @staticmethod
    def _lifecycle_points(text: str, followers: int | None) -> Tuple[int, list[str]]:
        """
        Lifecycle/sustainability points.
        Higher = more likely deal (aging + struggling generational handoff).
        Lower = less likely deal (multi-generational stability).
        """
        is_very_new = any(k in text for k in ["brand new", "really new", "newly opened", "recently opened", "new church", "new campus"])
        is_old = any(k in text for k in ["old", "historic", "aging", "older building", "long-standing", "longstanding"])
        teen_handoff = any(k in text for k in ["teens", "youth-led", "youth led", "next generation", "younger generation", "youth ministry"])
        generation_stable = any(
            k in text
            for k in ["multi-generational", "multigenerational", "survived generations", "served generations", "for generations"]
        )

        matched: list[str] = []
        points = 0

        if is_very_new:
            matched += [k for k in ["brand new", "newly opened", "recently opened", "new church", "new campus"] if k in text]
            return 0, matched  # neutral

        if generation_stable:
            matched += [k for k in ["multi-generational", "multigenerational", "survived generations", "served generations", "for generations"] if k in text]
            points -= 2

        # Reward the "aging + handoff + smallish" pattern as higher deal chance.
        if is_old and teen_handoff and (followers is None or followers <= 300):
            matched += [k for k in ["old", "historic", "aging", "older building", "long-standing", "teens", "youth-led", "next generation"] if k in text]
            points += 2
        elif (not is_old) and (not generation_stable) and (not is_very_new):
            # Slightly reduce if it's neither old nor stable (often indicates "normal" active church)
            points -= 1

        return points, matched

    @staticmethod
    def _sale_signal_points(text: str) -> Tuple[int, list[str]]:
        """
        Sale-likelihood signals.
        Returns (points, matched_signals).
        """
        # Strong signals
        sale_kw = ["for sale", "listed for sale", "property for sale", "real estate listing", "loopnet", "crexi", "commercial listing"]
        closing_kw = ["closing", "final service", "last service", "ceasing operations", "shutting down", "closing its doors"]
        merger_kw = ["merged with", "merger", "consolidating", "consolidation", "joining with", "merged into"]
        lease_kw = ["for lease", "space available", "available for rent", "tenant", "lease opportunity", "rent our sanctuary", "rent our hall", "facility rental"]

        # Weak signals
        weak_kw = ["declining membership", "small congregation", "struggling", "financial challenges", "budget shortfall", "unable to maintain"]

        matched: list[str] = []

        if any(k in text for k in sale_kw):
            matched += [k for k in sale_kw if k in text]
            return 4, matched
        if any(k in text for k in closing_kw):
            matched += [k for k in closing_kw if k in text]
            return 4, matched
        if any(k in text for k in merger_kw):
            matched += [k for k in merger_kw if k in text]
            return 3, matched
        if any(k in text for k in lease_kw):
            matched += [k for k in lease_kw if k in text]
            return 2, matched
        if any(k in text for k in weak_kw):
            matched += [k for k in weak_kw if k in text]
            return 1, matched

        return 0, matched

    @classmethod
    def _compute_deal_score_v2(
        cls,
        followers_count: int | None,
        condition_text: str,
        extra_signals_text: str = "",
    ) -> Tuple[int, str, list[str]]:
        """
        Compute deal score and return:
        - deal_score (1-10)
        - score_breakdown string
        - signals_found list

        Uses factor-based scoring:
          score = 5 + attendance + condition + lifecycle + sale_signals
        """
        base = 5
        blob = cls._norm_text(condition_text, extra_signals_text)

        attendance = cls._attendance_points(followers_count)
        cond_pts, cond_matches = cls._condition_points(blob)
        life_pts, life_matches = cls._lifecycle_points(blob, followers_count)
        sale_pts, sale_matches = cls._sale_signal_points(blob)

        score = base + attendance + cond_pts + life_pts + sale_pts
        score = max(1, min(10, score))

        signals_found = []
        # Keep signals readable and non-spammy.
        signals_found.extend(sale_matches)
        signals_found.extend([m for m in cond_matches if m not in signals_found])
        signals_found.extend([m for m in life_matches if m not in signals_found])

        breakdown = f"Base:{base} Attendance:{attendance} Condition:{cond_pts} Lifecycle:{life_pts} SaleSignals:{sale_pts}"
        return score, breakdown, signals_found

    # -----------------------------
    # Auth + sheet I/O
    # -----------------------------

    @property
    def client(self) -> gspread.Client:
        """Lazily authenticate and return client on first access."""
        if self._client is None:
            self._client = self._authenticate()
        return self._client

    @staticmethod
    def _service_account_email_hint() -> str:
        """Return service account email from credentials file, if available."""
        raw_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or os.getenv("GOOGLE_SHEETS_CREDENTIALS_PATH")
        if not raw_path:
            return ""
        path = Path(raw_path).expanduser()
        if not path.exists():
            return ""
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return str(data.get("client_email", "")).strip()
        except Exception:
            return ""

    @staticmethod
    def _authenticate() -> gspread.Client:
        """
        Authenticate with Google Sheets API.

        Returns:
            Authenticated gspread client
        """
        # 1) Service account via explicit env var paths.
        env_paths = [
            os.getenv("GOOGLE_SHEETS_CREDENTIALS_PATH"),
            os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
        ]
        for raw_path in env_paths:
            if not raw_path:
                continue
            candidate = Path(raw_path).expanduser()
            if candidate.exists():
                try:
                    return gspread.service_account(filename=str(candidate))
                except Exception as e:
                    raise ValueError(f"Failed to authenticate with service account file {candidate}: {e}")

        # 2) Service account from common local paths.
        service_account_paths = [
            Path("credentials.json"),
            Path(__file__).parent / "credentials.json",
            Path(__file__).parent.parent / "credentials.json",
            Path.home() / ".config" / "gspread" / "credentials.json",
        ]
        for candidate in service_account_paths:
            if not candidate.exists():
                continue
            try:
                return gspread.service_account(filename=str(candidate))
            except Exception as e:
                raise ValueError(f"Found credentials file at {candidate}, but authentication failed: {e}")

        # 3) OAuth fallback for local/manual usage.
        oauth_credentials = Path(__file__).parent.parent / "oauth_credentials.json"
        oauth_authorized_user = Path(__file__).parent.parent / "authorized_user.json"
        if oauth_credentials.exists():
            try:
                return gspread.oauth(
                    credentials_filename=str(oauth_credentials),
                    authorized_user_filename=str(oauth_authorized_user),
                )
            except Exception as e:
                raise ValueError(f"Failed OAuth authentication using {oauth_credentials}: {e}")

        checked_paths = [str(Path(p).expanduser()) for p in env_paths if p] + [str(p) for p in service_account_paths] + [
            str(oauth_credentials)
        ]
        raise FileNotFoundError(
            "Google Sheets credentials not found. Checked: "
            + ", ".join(checked_paths)
            + ". Set GOOGLE_APPLICATION_CREDENTIALS or GOOGLE_SHEETS_CREDENTIALS_PATH "
            "in code/.env, or run: python setup_google_auth.py"
        )

    def create_or_get_spreadsheet(self) -> gspread.Spreadsheet:
        """Create a new spreadsheet or get existing one."""
        spreadsheet_id = os.getenv("GOOGLE_SHEETS_SPREADSHEET_ID", "").strip()
        if spreadsheet_id:
            try:
                spreadsheet = self.client.open_by_key(spreadsheet_id)
                print(f"Opened spreadsheet by ID: {spreadsheet_id}")
                return spreadsheet
            except Exception as e:
                raw = str(e)
                if "PERMISSION_DENIED" in raw or "403" in raw:
                    email = self._service_account_email_hint()
                    share_hint = f" Share the sheet with service account: {email}." if email else " Share the sheet with your service account email."
                    raise PermissionError("Permission denied opening spreadsheet by ID." + share_hint) from e
                raise

        try:
            spreadsheet = self.client.open(self.spreadsheet_name)
            print(f"Opened existing spreadsheet: {self.spreadsheet_name}")
            return spreadsheet
        except gspread.SpreadsheetNotFound:
            spreadsheet = self.client.create(self.spreadsheet_name)
            print(f"Created new spreadsheet: {self.spreadsheet_name}")
            return spreadsheet

    # -----------------------------
    # Main write
    # -----------------------------

    def write_churches(self, churches_data: list[dict[str, Any]]) -> str:
        """
        Write church data to Google Sheets.

        Expected keys:
          Required: name, address, phone, website
          Optional:
            - explicit_attendance: int/text from website or listing
            - google_reviews/review_count: int/text
            - facebook_followers: int/text
            - instagram_followers: int/text
            - youtube_subscribers: int/text
            - followers: int or text (legacy fallback)
            - condition: free text notes (building/community)
            - deal_score: int override (1-10)
            - signals: extra text blob from search snippets/listings (recommended)
        """
        if not churches_data:
            return "No church data to write."

        try:
            spreadsheet = self.create_or_get_spreadsheet()
            worksheet = spreadsheet.sheet1

            worksheet.clear()
            worksheet.append_row(self.headers)

            for church in churches_data:
                name = str(church.get("name", "")).strip()
                address = str(church.get("address", "")).strip() or "ADDRESS NOT VERIFIED - needs manual review"
                phone = str(church.get("phone", "")).strip()
                website = str(church.get("website", "")).strip()

                condition = str(church.get("condition", "")).strip()
                # Optional: pass snippets/listing text here to improve sale-signal detection
                # without changing your agent's required output shape.
                signals_text = str(church.get("signals", "")).strip()

                (
                    followers_count,
                    estimate_source,
                    confidence,
                    reviews,
                    facebook,
                    instagram,
                    youtube,
                ) = self._estimate_followers(church=church, condition=condition, signals_text=signals_text)
                followers_text = str(followers_count) if followers_count is not None else "Unknown"
                digital_score, engagement = self._digital_footprint(
                    reviews=reviews,
                    facebook=facebook,
                    instagram=instagram,
                    youtube=youtube,
                )
                csi = self._community_strength_index(
                    followers_estimated=followers_count,
                    digital_footprint_score=digital_score,
                )

                provided_score = church.get("deal_score")
                if isinstance(provided_score, int):
                    deal_score = max(1, min(10, provided_score))
                    breakdown = "ProvidedByAgent"
                    signals_found: list[str] = []
                else:
                    deal_score, breakdown, signals_found = self._compute_deal_score_v2(
                        followers_count=followers_count,
                        condition_text=condition,
                        extra_signals_text=signals_text,
                    )

                row = [
                    name,
                    address,
                    phone,
                    website,
                    followers_text,
                    estimate_source,
                    confidence,
                    str(reviews) if reviews is not None else "",
                    str(facebook) if facebook is not None else "",
                    str(instagram) if instagram is not None else "",
                    str(youtube) if youtube is not None else "",
                    digital_score,
                    engagement,
                    csi,
                    condition,
                    ", ".join(signals_found) if signals_found else "",
                    breakdown,
                    deal_score,
                ]
                worksheet.append_row(row)

            # Format header row (optional): some gspread installs do not expose gspread.formatting.
            try:
                header_format = gspread.formatting.CellFormat(
                    text_format=gspread.formatting.TextFormat(bold=True),
                    horizontal_alignment="CENTER",
                )
                end_col = chr(ord("A") + len(self.headers) - 1)
                worksheet.format(f"A1:{end_col}1", [{"format": header_format}] * len(self.headers))
            except AttributeError:
                pass

            url = spreadsheet.url
            message = f"Successfully wrote {len(churches_data)} churches to spreadsheet. URL: {url}"
            print(message)
            return message

        except Exception as e:
            raw = str(e)
            if "ProxyError" in raw or "oauth2.googleapis.com" in raw or "127.0.0.1:9" in raw:
                return (
                    "Error writing to Google Sheets: proxy/network configuration is blocking "
                    "Google auth token requests (oauth2.googleapis.com). Clear HTTP_PROXY/HTTPS_PROXY/"
                    "ALL_PROXY (currently often set to 127.0.0.1:9), restart adk web, and try again."
                )
            if "Sheets API has not been used" in raw or "sheets.googleapis.com" in raw or "SERVICE_DISABLED" in raw:
                return (
                    "Error writing to Google Sheets: Google Sheets API is disabled for the "
                    "service account's Google Cloud project. Enable Sheets API in that project "
                    "and retry after a few minutes."
                )
            if "Google Drive API has not been used" in raw or "Google Drive API" in raw:
                return (
                    "Error writing to Google Sheets: Google Drive API is disabled. "
                    "Either enable Google Drive API in your Google Cloud project, or set "
                    "GOOGLE_SHEETS_SPREADSHEET_ID in code/.env so the tool opens a known sheet by ID."
                )
            if isinstance(e, PermissionError) or "PERMISSION_DENIED" in raw or "403" in raw:
                email = self._service_account_email_hint()
                share_hint = f" Share sheet with: {email} (Editor)." if email else " Share sheet with your service account email (Editor)."
                return "Error writing to Google Sheets: permission denied." + share_hint

            details = str(e).strip() or repr(e)
            error_msg = f"Error writing to Google Sheets: {type(e).__name__}: {details}"
            print(error_msg)
            return error_msg

    def __call__(self, churches_json: str) -> str:
        """Make the tool callable."""
        try:
            churches_data = json.loads(churches_json)
            return self.write_churches(churches_data)
        except json.JSONDecodeError as e:
            return f"Invalid JSON format: {e}"
