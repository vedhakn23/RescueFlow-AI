
import os
import re
import sqlite3
from datetime import datetime
from difflib import SequenceMatcher

import streamlit as st
from dotenv import load_dotenv
from google import genai

load_dotenv()

# ---------------- CONFIGURATION ----------------

DB_PATH = "rescueflow.db"
MODEL_NAME = "models/gemma-4-26b-a4b-it"

STATUSES = [
    "Reported",
    "Needs clarification",
    "Related report found",
    "Corroborated",
    "Action checklist prepared",
    "Action recorded",
]

st.set_page_config(
    page_title="RescueFlow AI",
    page_icon="🚨",
    layout="wide",
)

# ---------------- DATABASE ----------------

def connect_db():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with connect_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS incidents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT,
                report TEXT,
                location TEXT,
                status TEXT DEFAULT 'Reported',
                incident_type TEXT,
                ai_analysis TEXT
            )
        """)

        # Safely migrate databases created by the earlier version.
        columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(incidents)")
        }

        if "priority" not in columns:
            conn.execute(
                "ALTER TABLE incidents ADD COLUMN priority TEXT DEFAULT 'Medium'"
            )

        conn.execute("""
            CREATE TABLE IF NOT EXISTS evidence_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                case_id INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                details TEXT NOT NULL,
                FOREIGN KEY(case_id) REFERENCES incidents(id)
            )
        """)


def add_history(case_id, event_type, details):
    with connect_db() as conn:
        conn.execute("""
            INSERT INTO evidence_history
            (case_id, timestamp, event_type, details)
            VALUES (?, ?, ?, ?)
        """, (
            case_id,
            datetime.now().astimezone().isoformat(timespec="seconds"),
            event_type,
            details,
        ))


def save_incident(report, location, incident_type, priority):
    with connect_db() as conn:
        cursor = conn.execute("""
            INSERT INTO incidents
            (created_at, report, location, status, incident_type, priority)
            VALUES (?, ?, ?, 'Reported', ?, ?)
        """, (
            datetime.now().astimezone().isoformat(timespec="seconds"),
            report.strip(),
            location.strip(),
            incident_type,
            priority,
        ))
        case_id = cursor.lastrowid

    add_history(
        case_id,
        "Report received",
        f"Initial report submitted. Category: {incident_type}. "
        f"Rule-based preliminary priority: {priority}.",
    )
    return case_id


def save_analysis(case_id, analysis):
    with connect_db() as conn:
        conn.execute(
            "UPDATE incidents SET ai_analysis = ? WHERE id = ?",
            (analysis, case_id),
        )

    add_history(
        case_id,
        "AI analysis",
        "Gemma analysis saved. This is AI-generated decision support, "
        "not independent verification.",
    )


def get_incidents():
    with connect_db() as conn:
        return [
            dict(row)
            for row in conn.execute(
                "SELECT * FROM incidents ORDER BY id DESC"
            ).fetchall()
        ]


def get_history(case_id):
    with connect_db() as conn:
        return [
            dict(row)
            for row in conn.execute("""
                SELECT timestamp, event_type, details
                FROM evidence_history
                WHERE case_id = ?
                ORDER BY id DESC
            """, (case_id,)).fetchall()
        ]


def update_status(case_id, status):
    with connect_db() as conn:
        conn.execute(
            "UPDATE incidents SET status = ? WHERE id = ?",
            (status, case_id),
        )

    add_history(
        case_id,
        "Status updated",
        f"Case status changed to '{status}' by the dashboard user.",
    )


# ---------------- TRANSPARENT PRIORITY RULES ----------------

def calculate_priority(report, incident_type):
    """
    A simple, explainable screening rule.
    It does not verify an incident or replace emergency services.
    """
    text = f"{incident_type} {report}".lower()

    high_signals = [
        "trapped person",
        "person trapped",
        "people trapped",
        "injured",
        "injury",
        "electrocution",
        "live wire",
        "building collapse",
        "house collapsed",
        "people in danger",
        "person in danger",
        "life threatening",
    ]

    medium_signals = [
        "flood",
        "flooding",
        "waterlogging",
        "water logged",
        "blocked drain",
        "fallen tree",
        "road blocked",
        "road obstruction",
        "overflow",
        "dangerous",
        "damaged",
    ]

    if any(signal in text for signal in high_signals):
        return "High"
    if any(signal in text for signal in medium_signals):
        return "Medium"
    return "Low"


# ---------------- EVIDENCELOOP MATCHING ----------------

def normalize_text(text):
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())


def similarity(first, second):
    first = normalize_text(first)
    second = normalize_text(second)

    if not first or not second:
        return 0.0

    return SequenceMatcher(None, first, second).ratio()


def compare_reports(report, location, incident_type, existing_cases):
    matches = []

    for case in existing_cases:
        text_similarity = similarity(report, case.get("report", ""))
        location_similarity = similarity(
            location, case.get("location", "")
        )
        same_type = (
            (incident_type or "").lower()
            == (case.get("incident_type") or "").lower()
        )

        related = (
            (location_similarity >= 0.75 and text_similarity >= 0.35)
            or (text_similarity >= 0.65 and same_type)
        )

        if related:
            matches.append({
                "id": case["id"],
                "location": case.get("location", ""),
                "incident_type": case.get("incident_type", ""),
                "status": case.get("status", "Reported"),
                "text_similarity": round(text_similarity * 100),
                "location_similarity": round(location_similarity * 100),
            })

    return sorted(
        matches,
        key=lambda item: (
            item["text_similarity"] + item["location_similarity"]
        ),
        reverse=True,
    )


# ---------------- GEMMA LLM ----------------

def analyze_with_gemma(report, location, incident_type):
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return (
            "Gemma analysis unavailable: GEMINI_API_KEY was not found. "
            "Check your local .env file. No API key is displayed here."
        )

    prompt = f"""
You are the incident-report analysis assistant for RescueFlow AI.

Analyze this community-submitted report carefully.

Incident category selected by reporter: {incident_type}
Reported location: {location}
Report text: {report}

Provide these sections:
1. Summary of the report
2. Facts explicitly stated by the reporter
3. Important missing information or uncertainty
4. One useful clarification question
5. Suggested next steps for a human reviewer
6. Possible safety concerns

Rules:
- Do not invent facts, locations, casualties, times, or evidence.
- Clearly label assumptions and uncertainty.
- Do not claim the incident has been independently verified.
- Do not claim authorities have been contacted or action completed.
- Do not assign the official case status.
- Do not give a definitive safety clearance.
- Keep the answer concise and practical.
- The Python application calculates preliminary priority separately.
"""

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
        )
        result = getattr(response, "text", None)

        if result and result.strip():
            return result.strip()

        return "Gemma returned no text. A human reviewer should inspect this case."

    except Exception as exc:
        # Do not display credentials or detailed environment information.
        print(f"Gemma API error: {type(exc).__name__}: {exc}")
        return (
            "Gemma analysis is temporarily unavailable. "
            "The report is still saved. Check internet connectivity, "
            "API access, model availability, and quota."
        )


# ---------------- INTERFACE ----------------

init_db()

st.title("🚨 RescueFlow AI")
st.caption(
    "EvidenceLoop — evidence-linked incident reporting and investigation support"
)

st.info(
    "Decision-support prototype. Reports and AI-generated findings are not "
    "independently verified. This dashboard does not dispatch emergency services."
)

cases = get_incidents()
total_cases = len(cases)
high_cases = sum(c.get("priority") == "High" for c in cases)
open_cases = sum(
    c.get("status") not in ("Action recorded", "Corroborated")
    for c in cases
)

m1, m2, m3 = st.columns(3)
m1.metric("Total reports", total_cases)
m2.metric("Preliminary high priority", high_cases)
m3.metric("Cases needing review", open_cases)

report_tab, cases_tab, history_tab = st.tabs([
    "📝 Report an Incident",
    "📂 Incident Cases",
    "🔎 Evidence History",
])

# ---------------- NEW REPORT ----------------

with report_tab:
    st.subheader("Submit a community incident report")
    st.write(
        "Enter the information available. Incomplete reports are allowed "
        "so the system can identify what needs clarification."
    )

    with st.form("incident_form", clear_on_submit=True):
        incident_type = st.selectbox(
            "Incident category",
            [
                "Flooding / Waterlogging",
                "Blocked Drain",
                "Fallen Tree",
                "Road Obstruction",
                "Infrastructure Damage",
                "Public Safety Concern",
                "Other",
            ],
        )

        location = st.text_input(
            "Reported location",
            placeholder="e.g. Near the main bus stop, Ward 5",
        )

        report = st.text_area(
            "Incident description",
            placeholder=(
                "Describe what was reported, what was observed, "
                "and any known time or impact."
            ),
            height=130,
        )

        submitted = st.form_submit_button(
            "Analyze and Save Report",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        if not report.strip() or not location.strip():
            st.error("Please enter both a description and a reported location.")
        else:
            # Compare against previously saved cases before saving this one.
            previous_cases = get_incidents()
            priority = calculate_priority(report, incident_type)

            with st.spinner("Saving the report..."):
                case_id = save_incident(
                    report, location, incident_type, priority
                )

            st.success(f"Report saved as Case #{case_id}.")
            st.metric("Rule-based preliminary priority", priority)

            st.caption(
                "Priority is based on transparent keyword rules, not "
                "independent confirmation of the incident."
            )

            matches = compare_reports(
                report, location, incident_type, previous_cases
            )

            st.markdown("### 🔎 EvidenceLoop: Related Report Check")

            if matches:
                st.warning(
                    f"{len(matches)} potentially related case(s) found. "
                    "These are candidate links for human review, not proof "
                    "that the reports describe the same incident."
                )

                for match in matches:
                    st.markdown(
                        f"**Case #{match['id']}** — "
                        f"{match['incident_type']} at {match['location']}"
                    )
                    st.write(
                        f"Description similarity: "
                        f"{match['text_similarity']}% · "
                        f"Location similarity: "
                        f"{match['location_similarity']}% · "
                        f"Current status: {match['status']}"
                    )

                    add_history(
                        case_id,
                        "Possible related report",
                        f"Candidate Case #{match['id']} flagged for human review. "
                        f"Description similarity: {match['text_similarity']}%; "
                        f"location similarity: {match['location_similarity']}%. "
                        "Similarity does not establish that the incidents match.",
                    )
            else:
                st.write(
                    "No potentially related reports were found in the "
                    "current local database."
                )

            st.markdown("### 🤖 Gemma Report Analysis")

            with st.spinner("Asking Gemma to analyze the report..."):
                analysis = analyze_with_gemma(
                    report, location, incident_type
                )

            save_analysis(case_id, analysis)
            st.markdown(analysis)

            st.caption(
                "The AI output is advisory. A human must assess the report "
                "and any proposed actions."
            )


# ---------------- CASE LIST AND STATUS UPDATES ----------------

with cases_tab:
    st.subheader("Saved incident cases")

    if not cases:
        st.info("No reports have been saved yet.")
    else:
        search = st.text_input(
            "Filter cases by ID, location, category, or description"
        ).strip().lower()

        filtered_cases = [
            c for c in cases
            if not search
            or search in str(c["id"]).lower()
            or search in (c.get("location") or "").lower()
            or search in (c.get("incident_type") or "").lower()
            or search in (c.get("report") or "").lower()
        ]

        for case in filtered_cases:
            with st.expander(
                f"Case #{case['id']} · {case.get('incident_type', 'Unknown')} "
                f"· {case.get('priority') or 'Medium'} priority "
                f"· {case.get('status') or 'Reported'}"
            ):
                st.write("**Reported location:**", case.get("location", ""))
                st.write("**Submitted:**", case.get("created_at", ""))
                st.write("**Description:**", case.get("report", ""))
                st.write("**Current status:**", case.get("status", "Reported"))
                st.write(
                    "**Rule-based preliminary priority:**",
                    case.get("priority") or "Medium",
                )

                if case.get("ai_analysis"):
                    st.markdown("**Saved Gemma analysis**")
                    st.markdown(case["ai_analysis"])
                else:
                    st.write("No saved AI analysis.")

                st.markdown("**Update case status**")
                status_key = f"status_{case['id']}"
                selected_status = st.selectbox(
                    "Status",
                    STATUSES,
                    index=(
                        STATUSES.index(case["status"])
                        if case.get("status") in STATUSES
                        else 0
                    ),
                    key=status_key,
                )

                if st.button(
                    "Save status",
                    key=f"save_status_{case['id']}",
                ):
                    if selected_status != case.get("status"):
                        update_status(case["id"], selected_status)
                        st.success("Status updated and recorded in history.")
                        st.rerun()
                    else:
                        st.info("No status change to save.")


# ---------------- EVIDENCE HISTORY / NEW EVIDENCE ----------------

with history_tab:
    st.subheader("EvidenceLoop: Case history")
    st.write(
        "Record new information against an existing case. Entries are "
        "timestamped so reviewers can inspect how the case evolved."
    )

    current_cases = get_incidents()

    if not current_cases:
        st.info("Save a report first to start an evidence history.")
    else:
        case_options = {
            f"Case #{c['id']} — {c.get('incident_type', 'Unknown')} "
            f"— {c.get('location', '')}": c["id"]
            for c in current_cases
        }

        with st.form("add_evidence_form"):
            selected_case_label = st.selectbox(
                "Choose a case",
                list(case_options.keys()),
            )

            evidence_source = st.selectbox(
                "Information source",
                [
                    "Community follow-up",
                    "Human reviewer observation",
                    "Official source supplied by reviewer",
                    "Other / unverified",
                ],
            )

            evidence_text = st.text_area(
                "New information",
                placeholder=(
                    "Enter the new information. Do not mark it verified "
                    "unless you have actually checked the source."
                ),
                height=110,
            )

            evidence_submitted = st.form_submit_button(
                "Add to Case History",
                type="primary",
            )

        if evidence_submitted:
            if not evidence_text.strip():
                st.error("Enter the new information first.")
            else:
                selected_case_id = case_options[selected_case_label]

                add_history(
                    selected_case_id,
                    "New information recorded",
                    f"Source label: {evidence_source}. "
                    f"Content: {evidence_text.strip()} "
                    "This entry is recorded, not independently verified.",
                )

                st.success(
                    f"Information added to Case #{selected_case_id}'s history."
                )

        st.divider()
        st.markdown("### View case history")

        history_case_label = st.selectbox(
            "Select a case to inspect",
            list(case_options.keys()),
            key="history_case_selector",
        )
        history_case_id = case_options[history_case_label]

        history_items = get_history(history_case_id)

        if not history_items:
            st.info("No history entries for this case yet.")
        else:
            for item in history_items:
                st.markdown(
                    f"**{item['event_type']}** · {item['timestamp']}"
                )
                st.write(item["details"])
                st.divider()

st.caption(
    "RescueFlow AI prototype · Gemma LLM + Python similarity matching + "
    "SQLite evidence history. Similarity, AI analysis, and recorded reports "
    "do not independently verify an incident."
)