
import os
import re
import json
import time
import sqlite3
import difflib
import base64
import urllib.parse
import urllib.request
import urllib.error
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

try:
    from google import genai
except ImportError:
    genai = None


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

APP_TITLE = "RescueFlow AI"
DB_PATH = os.getenv("RESCUEFLOW_DB_PATH", "rescueflow.db")
MODEL_NAME = os.getenv(
    "GEMMA_MODEL",
    "models/gemma-4-26b-a4b-it"
)

MAX_INITIAL_QUESTIONS = 4
MAX_TOTAL_QUESTIONS = 8

INCIDENT_TYPES = [
    "Flooding / Waterlogging",
    "Blocked Drain",
    "Fallen Tree",
    "Road Obstruction",
    "Infrastructure Damage",
    "Public Safety Concern",
    "Other",
]

DEPARTMENTS = {
    "Flooding / Waterlogging": {
        "name": "BBMP / Local Municipal Authority",
        "phone_env": "RESCUEFLOW_BBMP_PHONE",
    },
    "Blocked Drain": {
        "name": "BBMP / Local Municipal Authority",
        "phone_env": "RESCUEFLOW_BBMP_PHONE",
    },
    "Fallen Tree": {
        "name": "BBMP / Local Municipal Authority",
        "phone_env": "RESCUEFLOW_BBMP_PHONE",
    },
    "Road Obstruction": {
        "name": "BBMP / Local Municipal Authority",
        "phone_env": "RESCUEFLOW_BBMP_PHONE",
    },
    "Infrastructure Damage": {
        "name": "BBMP / Relevant Infrastructure Authority",
        "phone_env": "RESCUEFLOW_BBMP_PHONE",
    },
    "Public Safety Concern": {
        "name": "Relevant Public Safety Authority",
        "phone_env": "RESCUEFLOW_SAFETY_PHONE",
    },
    "Other": {
        "name": "Relevant Local Authority",
        "phone_env": "RESCUEFLOW_OTHER_DEPARTMENT_PHONE",
    },
}


# ============================================================
# DATABASE AND MIGRATIONS
# ============================================================

def get_connection():
    conn = sqlite3.connect(DB_PATH, timeout=15)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 15000")
    return conn


def table_columns(conn, table):
    rows = conn.execute(
        f'PRAGMA table_info("{table}")'
    ).fetchall()
    return {row["name"] for row in rows}


def add_column_if_missing(conn, table, column, definition):
    columns = table_columns(conn, table)
    if column not in columns:
        conn.execute(
            f'ALTER TABLE "{table}" ADD COLUMN "{column}" {definition}'
        )


def init_db():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS incidents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT,
                report TEXT,
                location TEXT,
                status TEXT DEFAULT 'New',
                incident_type TEXT DEFAULT 'Other',
                ai_analysis TEXT
            )
        """)
        
        add_column_if_missing(
             conn,
            "followup_questions",
            "status",
         "TEXT NOT NULL DEFAULT 'pending'"
        )

        # Preserve older versions of the incidents table.
        migrations = [
            ("priority", "TEXT DEFAULT 'Medium'"),
            ("workflow_state", "TEXT DEFAULT 'clarifying'"),
            ("followup_question", "TEXT"),
            ("final_report", "TEXT"),
            ("extracted_details", "TEXT"),
            ("department", "TEXT"),
            ("notification_status", "TEXT DEFAULT 'Not sent'"),
        ]

        for column, definition in migrations:
            add_column_if_missing(
                conn, "incidents", column, definition
            )

        conn.execute("""
            CREATE TABLE IF NOT EXISTS followup_answers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                incident_id INTEGER NOT NULL,
                question_number INTEGER NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                created_at TEXT,
                UNIQUE (incident_id, question_number)
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS followup_questions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                incident_id INTEGER NOT NULL,
                question_number INTEGER NOT NULL,
                question TEXT NOT NULL,
                status TEXT DEFAULT 'pending',
                created_at TEXT,
                UNIQUE (incident_id, question_number)
            )
        """)

        # A separate audit table avoids assuming the schema of any
        # evidence_history table created by an earlier version.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS investigation_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                incident_id INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                details TEXT,
                created_at TEXT NOT NULL
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS notification_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                incident_id INTEGER NOT NULL,
                department TEXT,
                recipient TEXT,
                status TEXT,
                details TEXT,
                created_at TEXT NOT NULL
            )
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_incidents_location
            ON incidents(location)
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_followup_incident
            ON followup_answers(incident_id)
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_history_incident
            ON investigation_history(incident_id)
        """)

        conn.commit()


def log_event(incident_id, event_type, details):
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO investigation_history
                (incident_id, event_type, details, created_at)
            VALUES (?, ?, ?, ?)
        """, (
            incident_id,
            event_type,
            str(details),
            datetime.now().isoformat(timespec="seconds"),
        ))


def get_incident(incident_id):
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM incidents WHERE id = ?",
            (incident_id,),
        ).fetchone()
    return dict(row) if row else None


def get_answers(incident_id):
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT question_number, question, answer, created_at
            FROM followup_answers
            WHERE incident_id = ?
            ORDER BY question_number
        """, (incident_id,)).fetchall()
    return [dict(row) for row in rows]


def get_questions(incident_id):
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT question_number, question, status
            FROM followup_questions
            WHERE incident_id = ?
            ORDER BY question_number
        """, (incident_id,)).fetchall()
    return [dict(row) for row in rows]


def get_next_question(incident_id):
    with get_connection() as conn:
        row = conn.execute("""
            SELECT question_number, question
            FROM followup_questions
            WHERE incident_id = ? AND status = 'pending'
            ORDER BY question_number
            LIMIT 1
        """, (incident_id,)).fetchone()
    return dict(row) if row else None


# ============================================================
# GEMMA: RETRIES, TIMEOUT HANDLING AND SAFE FAILURE
# ============================================================

def call_gemma(prompt, retries=1):
    """
    Returns generated text or raises a readable exception.
    Callers must handle failure and use their fallback behavior.
    """
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is missing from your .env file."
        )

    if genai is None:
        raise RuntimeError(
            "Google GenAI SDK is missing. Install google-genai."
        )

    last_error = None

    for attempt in range(retries + 1):
        try:
            client = genai.Client(api_key=api_key)

            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
            )

            text = getattr(response, "text", None)

            if text and text.strip():
                return text.strip()

            raise RuntimeError(
                "Gemma returned an empty response."
            )

        except Exception as exc:
            last_error = exc

            if attempt < retries:
                time.sleep(0.8)

    # Do not expose API credentials or the full request.
    raise RuntimeError(
        f"Gemma request failed after retry: "
        f"{type(last_error).__name__}"
    )


# ============================================================
# INCIDENT PRIORITY
# ============================================================

def calculate_priority(report, incident_type):
    text = f"{report} {incident_type}".lower()

    high_keywords = [
        "person trapped", "trapped person", "injured",
        "injury", "electrocution", "live wire", "collapsed",
        "building collapse", "life threatening", "life-threatening",
        "major flooding", "house flooded", "people trapped",
        "fire", "immediate danger", "emergency",
    ]

    medium_keywords = [
        "blocked road", "road blocked", "waterlogging",
        "overflowing", "overflow", "fallen tree",
        "drain blocked", "traffic blocked", "crack",
        "damaged", "unsafe", "hazard",
    ]

    if any(word in text for word in high_keywords):
        return "High"

    if any(word in text for word in medium_keywords):
        return "Medium"

    return "Low"


# ============================================================
# EVIDENCELOOP: SIMILARITY, LOCATION AND CASE COMPARISON
# ============================================================

def normalize_text(value):
    value = (value or "").lower()
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def text_similarity(text_a, text_b):
    a = normalize_text(text_a)
    b = normalize_text(text_b)

    if not a or not b:
        return 0.0

    return difflib.SequenceMatcher(None, a, b).ratio()


def location_similarity(location_a, location_b):
    a = normalize_text(location_a)
    b = normalize_text(location_b)

    if not a or not b:
        return 0.0

    if a == b:
        return 1.0

    return difflib.SequenceMatcher(None, a, b).ratio()


def compare_reports(report, location, incident_type, exclude_id=None):
    """
    Local EvidenceLoop pass.
    A possible match is not proof that two reports describe the same event.
    """
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT id, created_at, report, location, status,
                   incident_type, priority
            FROM incidents
            ORDER BY id DESC
            LIMIT 500
        """).fetchall()

    matches = []

    for row in rows:
        case = dict(row)

        if exclude_id is not None and case["id"] == exclude_id:
            continue

        text_score = text_similarity(
            report, case.get("report", "")
        )
        place_score = location_similarity(
            location, case.get("location", "")
        )

        same_type = (
            normalize_text(incident_type)
            == normalize_text(case.get("incident_type", ""))
        )

        # Weighted heuristic, not a trained probability.
        score = (
            0.60 * text_score
            + 0.30 * place_score
            + 0.10 * float(same_type)
        )

        # Same-place cases may be relevant even if the wording differs.
        if score >= 0.38 or (
            place_score >= 0.82 and same_type
        ):
            matches.append({
                "id": case["id"],
                "created_at": case.get("created_at"),
                "report": case.get("report", ""),
                "location": case.get("location", ""),
                "status": case.get("status", "Unknown"),
                "incident_type": case.get("incident_type", "Other"),
                "priority": case.get("priority", "Medium"),
                "text_similarity": round(text_score * 100),
                "location_similarity": round(place_score * 100),
                "match_score": round(score * 100),
            })

    matches.sort(
        key=lambda item: item["match_score"],
        reverse=True,
    )

    return matches[:10]


def evidence_gaps(report, location, incident_type, answers):
    text = f"{report} " + " ".join(
        item["answer"] for item in answers
    )
    text = text.lower()

    gaps = []

    if not location or len(location.strip()) < 3:
        gaps.append("A specific location is missing.")

    if not any(word in text for word in [
        "today", "yesterday", "am", "pm", "morning",
        "afternoon", "evening", "night", "hour", "minute",
        "just now", "currently",
    ]):
        gaps.append(
            "The incident time or when it was last observed is unclear."
        )

    if not any(word in text for word in [
        "still", "stopped", "resolved", "ongoing",
        "continues", "currently", "now", "no longer",
    ]):
        gaps.append(
            "The current status of the incident may need confirmation."
        )

    if not any(word in text for word in [
        "person", "people", "resident", "vehicle",
        "traffic", "house", "school", "shop",
        "road", "building", "property", "injured",
    ]):
        gaps.append(
            "The impact on people, traffic or property is not clear."
        )

    return gaps


def detect_conflicts(answers, matches):
    conflicts = []

    current_text = " ".join(
        item["answer"] for item in answers
    ).lower()

    current_resolved = any(word in current_text for word in [
        "resolved", "stopped", "cleared", "returned to normal",
        "no longer happening", "fixed",
    ])

    current_ongoing = any(word in current_text for word in [
        "still happening", "still blocked", "still overflowing",
        "continues", "ongoing", "still flooded",
    ])

    if current_resolved and current_ongoing:
        conflicts.append(
            "The answers may contain conflicting descriptions "
            "of whether the incident is ongoing."
        )

    for match in matches:
        old_text = (match.get("report") or "").lower()
        old_status = (match.get("status") or "").lower()

        old_resolved = (
            old_status in {"resolved", "closed"}
            or any(word in old_text for word in [
                "resolved", "cleared", "fixed",
            ])
        )

        if current_ongoing and old_resolved:
            conflicts.append(
                f"Possible conflict with case #{match['id']}: "
                "this report appears ongoing, while the older case "
                "may have been resolved. Verify whether this is a "
                "new incident or a recurrence."
            )
            break

    return conflicts


# ============================================================
# CATEGORY-SPECIFIC QUESTION FALLBACKS
# ============================================================

FALLBACK_QUESTIONS = {
    "Flooding / Waterlogging": [
        "Is the waterlogging still happening, and how deep is the water approximately?",
        "Which exact road, junction or nearby landmark is affected?",
        "Are homes, shops, pedestrians or vehicles currently at risk?",
        "When did the flooding start, and has the water level changed?",
        "Is water entering any building or blocking an emergency access route?",
        "Has the local authority already been informed?",
        "Can you safely provide a recent photo or another observation?",
        "Has the water level increased, decreased or remained the same?",
    ],
    "Blocked Drain": [
        "Is the drain currently blocked or overflowing?",
        "Where exactly is the drain, using a street name or landmark?",
        "Is wastewater spilling onto the road or near any homes?",
        "When was the blockage first noticed?",
        "Is there a visible obstruction, such as waste, mud or fallen material?",
        "Has the issue been reported previously?",
        "Is the blockage affecting pedestrians or traffic?",
        "Has anyone confirmed whether the drain has been inspected or cleared?",
    ],
    "Fallen Tree": [
        "Is the tree still blocking the road, footpath or property?",
        "Where exactly did the tree fall, including a nearby landmark?",
        "Are any people injured, trapped or in immediate danger?",
        "Are electrical wires touching the tree or lying nearby?",
        "Is any vehicle or emergency access route blocked?",
        "When did the tree fall, and is there any sign of further instability?",
        "Has the area been cordoned off or reported to the authorities?",
        "Can you safely provide a photo from a safe distance?",
    ],
    "Road Obstruction": [
        "What is obstructing the road, and is the obstruction still present?",
        "Which road, direction or nearby landmark identifies the location?",
        "Is the entire road blocked or is one lane still usable?",
        "Are traffic congestion or safety risks increasing?",
        "When was the obstruction first noticed?",
        "Are emergency vehicles unable to pass?",
        "Has the obstruction moved or changed since it was reported?",
        "Has any authority or response team already arrived?",
    ],
    "Infrastructure Damage": [
        "What infrastructure is damaged: a road, bridge, building, pole or something else?",
        "Where exactly is the damaged structure?",
        "Is there a visible risk of collapse, electric shock or injury?",
        "When was the damage first noticed?",
        "Is the damage getting worse or affecting public access?",
        "Has the area been restricted or marked as unsafe?",
        "Has an authority inspected the damage?",
        "Can you safely provide a photo without approaching the hazard?",
    ],
    "Public Safety Concern": [
        "What is the immediate safety concern, and is the danger ongoing?",
        "Where exactly is the situation occurring?",
        "Is anyone injured, trapped or facing an immediate threat?",
        "When did the situation begin or when was it last observed?",
        "Who or what is currently at risk?",
        "Have emergency services or local authorities been contacted?",
        "Has the situation changed since the initial report?",
        "What additional verified information could help responders assess the situation?",
    ],
    "Other": [
        "Can you describe the main problem and its current impact?",
        "Where exactly is the incident occurring?",
        "When did the issue begin, and is it still happening?",
        "Are people, traffic or property at immediate risk?",
        "Has the issue been reported to any authority before?",
        "What evidence or observation supports the report?",
        "Has anything changed since the incident was first noticed?",
        "What outcome or assistance is needed?",
    ],
}


def clean_question(question):
    question = re.sub(
        r"^\s*(?:Q(?:uestion)?\s*)?\d+\s*[\.\):\-]\s*",
        "",
        question,
        flags=re.IGNORECASE,
    )
    question = question.strip().strip('"').strip("'")
    return question


def parse_questions_from_gemma(text):
    questions = []

    for line in text.splitlines():
        line = line.strip()

        if not line:
            continue

        if re.match(
            r"^(?:Q(?:uestion)?\s*)?\d+\s*[\.\):\-]",
            line,
            flags=re.IGNORECASE,
        ):
            question = clean_question(line)

            if question.endswith("?") and len(question) > 12:
                questions.append(question)

    # Remove duplicates within the generated batch.
    unique = []

    for question in questions:
        if not any(
            text_similarity(question, previous) >= 0.78
            for previous in unique
        ):
            unique.append(question)

    return unique


def make_initial_analysis_and_questions(
    report, location, incident_type, matches
):
    """
    Uses one Gemma call for initial analysis and question generation.
    Falls back to local rules and category-specific questions.
    """
    related_text = "\n".join(
        f"- Case #{m['id']}: {m['incident_type']}; "
        f"location={m['location']}; status={m['status']}; "
        f"report={m['report'][:220]}"
        for m in matches[:5]
    ) or "No likely related cases found."

    prompt = f"""
You are the incident intake assistant for RescueFlow AI.

Analyze this citizen incident report. Do not invent facts.
Treat historical matches as possible matches, not proof.

Incident category: {incident_type}
Citizen location: {location}
Citizen report: {report}

Possible related historical cases:
{related_text}

Return:
SUMMARY: A concise summary using only known facts.
MISSING: Important missing details or uncertainties.
PRIORITY: High, Medium, or Low, with a short reason.
QUESTIONS:
1. A short, relevant question?
2. A different relevant question?
3. Another distinct question?
4. Another distinct question?

Requirements:
- Questions must be specific to this incident and its category.
- Ask about useful missing information, safety, location, time,
  impact, current status or evidence.
- Do not repeat the report or ask for information already supplied.
- Do not claim that a possible historical match is confirmed.
- Do not provide a diagnosis or fabricate evidence.
- Keep the response concise.
"""

    analysis = ""
    questions = []

    try:
        result = call_gemma(prompt, retries=1)
        questions = parse_questions_from_gemma(result)
        analysis = result.strip()
    except Exception:
        analysis = (
            "Gemma was unavailable during intake. The incident has "
            "been saved, and local EvidenceLoop checks will continue. "
            "This is a provisional assessment, not an AI-verified result."
        )

    # Use category-aware local questions when Gemma fails or returns
    # too few parseable questions.
    fallback = FALLBACK_QUESTIONS.get(
        incident_type, FALLBACK_QUESTIONS["Other"]
    )

    for question in fallback:
        if len(questions) >= MAX_INITIAL_QUESTIONS:
            break

        if not any(
            text_similarity(question, existing) >= 0.78
            for existing in questions
        ):
            questions.append(question)

    questions = questions[:MAX_INITIAL_QUESTIONS]

    priority = calculate_priority(report, incident_type)

    match_text = (
        f"\n\nEvidenceLoop found {len(matches)} possible related "
        "case(s). Matches are preliminary and require verification."
        if matches
        else "\n\nEvidenceLoop did not find a strong related-case match."
    )

    return (
        analysis + match_text,
        questions,
        priority,
    )


def generate_additional_questions(
    incident, answers, existing_questions, matches
):
    if len(existing_questions) >= MAX_TOTAL_QUESTIONS:
        return []

    already_asked = "\n".join(
        f"- {q['question']}" for q in existing_questions
    )

    answer_text = "\n".join(
        f"Q: {a['question']}\nA: {a['answer']}"
        for a in answers
    ) or "No answers yet."

    match_text = "\n".join(
        f"- Case #{m['id']}: {m['report'][:180]}"
        for m in matches[:5]
    ) or "No likely related cases."

    prompt = f"""
You are EvidenceLoop, the evidence-based investigation component
of RescueFlow AI.

Incident category: {incident.get('incident_type')}
Location: {incident.get('location')}
Original report: {incident.get('report')}

Saved answers:
{answer_text}

Possible related cases:
{match_text}

Questions already asked:
{already_asked}

Suggest up to 4 additional short questions only if they would
materially reduce important uncertainty or help verify a conflict.

Do not repeat earlier questions. Do not ask for information already
provided. Do not invent facts. Return one numbered question per line,
with each question ending in a question mark. If no additional
questions are useful, return: NO_MORE_QUESTIONS
"""

    try:
        result = call_gemma(prompt, retries=1)
        if "NO_MORE_QUESTIONS" in result.upper():
            return []

        candidates = parse_questions_from_gemma(result)
    except Exception:
        candidates = []

    incident_type = incident.get("incident_type", "Other")
    fallback = FALLBACK_QUESTIONS.get(
        incident_type, FALLBACK_QUESTIONS["Other"]
    )

    all_old = [
        q["question"] for q in existing_questions
    ]

    for question in fallback:
        if len(candidates) >= 4:
            break

        if not any(
            text_similarity(question, old) >= 0.75
            for old in all_old + candidates
        ):
            candidates.append(question)

    unique = []

    for question in candidates:
        if any(
            text_similarity(question, old) >= 0.75
            for old in all_old + unique
        ):
            continue

        unique.append(question)

        if len(unique) >= 4:
            break

    remaining = MAX_TOTAL_QUESTIONS - len(existing_questions)
    return unique[:remaining]


def insert_questions(incident_id, questions, start_number):
    with get_connection() as conn:
        for offset, question in enumerate(questions):
            conn.execute("""
                INSERT OR IGNORE INTO followup_questions
                    (incident_id, question_number, question,
                     status, created_at)
                VALUES (?, ?, ?, 'pending', ?)
            """, (
                incident_id,
                start_number + offset,
                question,
                datetime.now().isoformat(timespec="seconds"),
            ))

        conn.commit()


# ============================================================
# ANSWER SAVING
# ============================================================

def save_followup_answer(incident_id, question_number, answer):
    answer = answer.strip()

    if not answer:
        raise ValueError("Please enter an answer.")

    with get_connection() as conn:
        question_row = conn.execute("""
            SELECT question, status
            FROM followup_questions
            WHERE incident_id = ? AND question_number = ?
        """, (incident_id, question_number)).fetchone()

        if not question_row:
            raise ValueError(
                "This question could not be found. Refresh the page."
            )

        if question_row["status"] == "answered":
            # Idempotent behavior: do not create a duplicate answer.
            existing = conn.execute("""
                SELECT answer FROM followup_answers
                WHERE incident_id = ? AND question_number = ?
            """, (incident_id, question_number)).fetchone()

            if existing:
                return

        conn.execute("""
            INSERT INTO followup_answers
                (incident_id, question_number, question, answer, created_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(incident_id, question_number)
            DO UPDATE SET
                question = excluded.question,
                answer = excluded.answer,
                created_at = excluded.created_at
        """, (
            incident_id,
            question_number,
            question_row["question"],
            answer,
            datetime.now().isoformat(timespec="seconds"),
        ))

        conn.execute("""
            UPDATE followup_questions
            SET status = 'answered'
            WHERE incident_id = ? AND question_number = ?
        """, (incident_id, question_number))

        conn.execute("""
            UPDATE incidents
            SET followup_question = NULL,
                workflow_state = 'clarifying'
            WHERE id = ?
        """, (incident_id,))

        conn.commit()

    log_event(
        incident_id,
        "Evidence added",
        f"Citizen answered question {question_number}.",
    )


# ============================================================
# FINAL REPORT: GEMMA PLUS RULE-BASED FALLBACK
# ============================================================

def build_fallback_report(incident, answers, matches, gaps, conflicts):
    """
    A deterministic fallback that works without Gemma.
    It is deliberately labelled as rule-based.
    """
    lines = [
        "RESCUEFLOW AI — PROVISIONAL FALLBACK REPORT",
        "",
        "Generation method: Rule-based fallback (Gemma unavailable).",
        "This report has not been independently verified.",
        "",
        f"Incident ID: {incident['id']}",
        f"Incident category: {incident.get('incident_type', 'Other')}",
        f"Location: {incident.get('location') or 'Not provided'}",
        f"Reported at: {incident.get('created_at') or 'Unknown'}",
        f"Priority: {incident.get('priority') or 'Medium'}",
        f"Current case status: {incident.get('status') or 'New'}",
        "",
        "1. CITIZEN'S ORIGINAL REPORT",
        incident.get("report") or "No report available.",
        "",
        "2. FOLLOW-UP EVIDENCE",
    ]

    if answers:
        for item in answers:
            lines.append(f"Q{item['question_number']}: {item['question']}")
            lines.append(f"Answer: {item['answer']}")
            lines.append("")
    else:
        lines.append("No follow-up answers recorded.")
        lines.append("")

    lines.extend([
        "3. EVIDENCELOOP RELATED-CASE CHECK",
        f"Possible related cases found: {len(matches)}",
    ])

    if matches:
        for match in matches[:5]:
            lines.append(
                f"- Case #{match['id']} | "
                f"category: {match['incident_type']} | "
                f"location: {match['location'] or 'Unknown'} | "
                f"status: {match['status']} | "
                f"heuristic match score: {match['match_score']}%"
            )
            lines.append(
                "  This is a possible match, not a confirmed duplicate."
            )
    else:
        lines.append("No likely related cases found by the local check.")

    lines.extend(["", "4. UNRESOLVED EVIDENCE GAPS"])

    if gaps:
        lines.extend(f"- {gap}" for gap in gaps)
    else:
        lines.append(
            "No gaps were detected by the basic local rules. "
            "This does not guarantee that the evidence is complete."
        )

    lines.extend(["", "5. POSSIBLE CONFLICTS"])

    if conflicts:
        lines.extend(f"- {conflict}" for conflict in conflicts)
    else:
        lines.append(
            "No conflicts were detected by the basic local rules."
        )

    lines.extend([
        "",
        "6. RECOMMENDED NEXT ACTION",
        "A human operator should verify the location, current incident "
        "status, priority and any possible related cases before taking "
        "consequential action. Request further evidence if important "
        "uncertainties remain.",
        "",
        "7. LIMITATIONS",
        "This report is a provisional summary of the submitted report, "
        "saved answers and local comparison rules. It does not establish "
        "that an incident is verified, and similarity scores are not "
        "probabilities.",
    ])

    return "\n".join(lines)


def generate_final_report(incident_id):
    """
    Tries Gemma once, with a retry. On failure, creates a fallback.
    Always stores the resulting report and preserves citizen answers.
    """
    incident = get_incident(incident_id)

    if not incident:
        raise ValueError("Incident not found.")

    answers = get_answers(incident_id)

    matches = compare_reports(
        incident.get("report", ""),
        incident.get("location", ""),
        incident.get("incident_type", "Other"),
        exclude_id=incident_id,
    )

    gaps = evidence_gaps(
        incident.get("report", ""),
        incident.get("location", ""),
        incident.get("incident_type", "Other"),
        answers,
    )

    conflicts = detect_conflicts(answers, matches)

    answer_text = "\n".join(
        f"Question: {a['question']}\nAnswer: {a['answer']}"
        for a in answers
    ) or "No follow-up answers supplied."

    matches_text = "\n".join(
        f"Case #{m['id']}: category={m['incident_type']}; "
        f"location={m['location']}; status={m['status']}; "
        f"similarity heuristic={m['match_score']}%; "
        f"report={m['report'][:250]}"
        for m in matches[:5]
    ) or "No likely related cases found."

    prompt = f"""
Create a concise incident investigation report for RescueFlow AI.

Use only the facts supplied below. Distinguish citizen statements,
historical records, automated inferences and unverified claims.

INCIDENT ID: {incident['id']}
CATEGORY: {incident.get('incident_type')}
LOCATION: {incident.get('location')}
REPORTED AT: {incident.get('created_at')}
PRIORITY RULE RESULT: {incident.get('priority')}
ORIGINAL REPORT:
{incident.get('report')}

SAVED FOLLOW-UP QUESTIONS AND ANSWERS:
{answer_text}

POSSIBLE RELATED CASES:
{matches_text}

RULE-BASED EVIDENCE GAPS:
{json.dumps(gaps, ensure_ascii=False)}

POSSIBLE CONFLICTS:
{json.dumps(conflicts, ensure_ascii=False)}

Use these headings:
1. Incident Summary
2. Facts Reported by the Citizen
3. Follow-up Evidence
4. EvidenceLoop Comparison
5. Missing Information and Uncertainty
6. Possible Conflicts Requiring Verification
7. Priority and Reason
8. Recommended Next Actions
9. Human Review Requirements

Requirements:
- Do not invent information, timestamps, addresses, or evidence.
- A historical similarity score is not a probability or proof.
- Do not claim a department has been contacted.
- Explicitly identify unknown or conflicting information.
- Do not claim an incident is verified unless verified evidence exists.
- Be concise and practical.
"""

    used_fallback = False

    try:
        report = call_gemma(prompt, retries=1)
        if not report or len(report.strip()) < 40:
            raise RuntimeError("Generated report was too short.")
    except Exception:
        report = build_fallback_report(
            incident, answers, matches, gaps, conflicts
        )
        used_fallback = True

    now = datetime.now().isoformat(timespec="seconds")
    status = "Awaiting Review"

    with get_connection() as conn:
        conn.execute("""
            UPDATE incidents
            SET final_report = ?,
                ai_analysis = ?,
                workflow_state = 'ready_for_review',
                status = ?
            WHERE id = ?
        """, (report, report, status, incident_id))
        conn.commit()

    log_event(
        incident_id,
        "Final report generated",
        "Rule-based fallback used."
        if used_fallback
        else "Gemma report generated successfully.",
    )

    return report, used_fallback


# ============================================================
# DEPARTMENT ROUTING AND OPTIONAL REAL SMS
# ============================================================

def get_department(incident_type):
    return DEPARTMENTS.get(
        incident_type,
        DEPARTMENTS["Other"],
    )


def send_department_sms(incident, recipient):
    """
    Sends a real SMS using Twilio if configured.
    No SMS is sent when credentials or recipient are missing.
    """
    sid = os.getenv("TWILIO_ACCOUNT_SID")
    token = os.getenv("TWILIO_AUTH_TOKEN")
    sender = os.getenv("TWILIO_FROM_NUMBER")

    if not sid or not token or not sender:
        return (
            False,
            "SMS not configured. Set TWILIO_ACCOUNT_SID, "
            "TWILIO_AUTH_TOKEN and TWILIO_FROM_NUMBER in .env.",
        )

    if not recipient:
        return False, "No department recipient number is configured."

    report = incident.get("final_report") or incident.get("ai_analysis") or ""
    excerpt = re.sub(r"\s+", " ", report).strip()[:250]

    message = (
        f"RescueFlow AI incident #{incident['id']}. "
        f"Category: {incident.get('incident_type')}. "
        f"Priority: {incident.get('priority')}. "
        f"Location: {incident.get('location') or 'Not provided'}. "
        f"Status: {incident.get('status')}. "
        f"Summary: {excerpt}"
    )

    url = (
        f"https://api.twilio.com/2010-04-01/Accounts/"
        f"{sid}/Messages.json"
    )

    data = urllib.parse.urlencode({
        "To": recipient,
        "From": sender,
        "Body": message,
    }).encode("utf-8")

    credentials = base64.b64encode(
        f"{sid}:{token}".encode("utf-8")
    ).decode("ascii")

    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Basic {credentials}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            payload = json.loads(
                response.read().decode("utf-8")
            )

        provider_status = payload.get("status", "accepted")
        provider_sid = payload.get("sid", "unknown")

        # Provider acceptance is not proof of handset delivery.
        return True, (
            f"SMS accepted by provider. Message ID: {provider_sid}. "
            f"Provider status: {provider_status}. "
            "Check provider delivery status for confirmation."
        )

    except urllib.error.HTTPError as exc:
        # Do not print credentials or raw authorization headers.
        return False, (
            f"SMS provider rejected the request (HTTP {exc.code}). "
            "Check recipient format, credentials and provider settings."
        )
    except Exception as exc:
        return False, (
            f"SMS request failed: {type(exc).__name__}. "
            "The notification can be retried."
        )


def record_notification(incident_id, department, recipient, status, details):
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO notification_history
                (incident_id, department, recipient, status,
                 details, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            incident_id,
            department,
            recipient,
            status,
            details,
            datetime.now().isoformat(timespec="seconds"),
        ))

        conn.execute("""
            UPDATE incidents
            SET notification_status = ?
            WHERE id = ?
        """, (status, incident_id))

        conn.commit()

    log_event(
        incident_id,
        "Department notification",
        f"{status}: {details}",
    )


# ============================================================
# STREAMLIT UI HELPERS
# ============================================================

def show_incident_summary(incident):
    cols = st.columns(4)
    cols[0].metric("Incident ID", f"#{incident['id']}")
    cols[1].metric(
        "Priority",
        incident.get("priority") or "Medium",
    )
    cols[2].metric(
        "Status",
        incident.get("status") or "New",
    )
    cols[3].metric(
        "Questions answered",
        len(get_answers(incident["id"])),
    )

    st.write("**Category:**", incident.get("incident_type", "Other"))
    st.write("**Location:**", incident.get("location") or "Not provided")
    st.write("**Submitted:**", incident.get("created_at") or "Unknown")
    st.write("**Citizen report:**")
    st.write(incident.get("report") or "No report available.")


def show_evidence_loop(incident):
    incident_id = incident["id"]

    answers = get_answers(incident_id)

    matches = compare_reports(
        incident.get("report", ""),
        incident.get("location", ""),
        incident.get("incident_type", "Other"),
        exclude_id=incident_id,
    )

    gaps = evidence_gaps(
        incident.get("report", ""),
        incident.get("location", ""),
        incident.get("incident_type", "Other"),
        answers,
    )

    conflicts = detect_conflicts(answers, matches)

    st.subheader("EvidenceLoop assessment")
    st.caption(
        "Local heuristic assessment. Similarity is not proof of a duplicate."
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Possible related cases", len(matches))
    col2.metric("Evidence gaps", len(gaps))
    col3.metric("Possible conflicts", len(conflicts))

    if matches:
        st.markdown("**Possible related cases**")

        for match in matches:
            with st.expander(
                f"Case #{match['id']} — "
                f"{match['match_score']}% heuristic match"
            ):
                st.write("**Category:**", match["incident_type"])
                st.write("**Location:**", match["location"] or "Unknown")
                st.write("**Status:**", match["status"])
                st.write(
                    "**Text similarity:**",
                    f"{match['text_similarity']}%",
                )
                st.write(
                    "**Location similarity:**",
                    f"{match['location_similarity']}%",
                )
                st.write("**Original report:**", match["report"])

    else:
        st.info("No likely related cases found by the local comparison.")

    if gaps:
        st.markdown("**Potential evidence gaps**")
        for gap in gaps:
            st.write(f"- {gap}")
    else:
        st.write("No gaps detected by the basic local rules.")

    if conflicts:
        st.markdown("**Possible conflicts requiring verification**")
        for conflict in conflicts:
            st.warning(conflict)
    else:
        st.write("No conflicts detected by the basic local rules.")

    st.caption(
        "These checks are heuristic. Human review is required before "
        "treating a match or conflict as established fact."
    )


def show_followup_workflow(incident):
    incident_id = incident["id"]

    questions = get_questions(incident_id)
    answers = get_answers(incident_id)
    next_question = get_next_question(incident_id)

    st.subheader("Follow-up investigation")
    st.write(
        f"Answered **{len(answers)}** of **{len(questions)}** "
        "currently queued questions."
    )

    if next_question:
        st.markdown(
            f"**Question {next_question['question_number']}:** "
            f"{next_question['question']}"
        )

        with st.form(
            f"answer_form_{incident_id}_{next_question['question_number']}"
        ):
            answer = st.text_area(
                "Your answer",
                key=(
                    f"answer_{incident_id}_"
                    f"{next_question['question_number']}"
                ),
                placeholder="Enter the details you know...",
            )

            submitted = st.form_submit_button(
                "Save answer and continue",
                type="primary",
            )

            if submitted:
                if not answer.strip():
                    st.error("Please enter an answer.")
                else:
                    try:
                        save_followup_answer(
                            incident_id,
                            next_question["question_number"],
                            answer,
                        )
                        st.success("Answer saved.")
                        st.rerun()
                    except Exception as exc:
                        st.error(
                            f"Could not save the answer: {exc}"
                        )

    else:
        st.success(
            "All queued questions have been answered."
        )

    if answers:
        with st.expander("View saved answers"):
            for item in answers:
                st.markdown(
                    f"**Q{item['question_number']}:** "
                    f"{item['question']}"
                )
                st.write(item["answer"])

    if len(questions) < MAX_TOTAL_QUESTIONS:
        if st.button(
            "Generate more targeted questions",
            key=f"more_questions_{incident_id}",
        ):
            with st.spinner(
                "EvidenceLoop is checking what information is missing..."
            ):
                current_incident = get_incident(incident_id)
                current_answers = get_answers(incident_id)
                current_questions = get_questions(incident_id)

                matches = compare_reports(
                    current_incident.get("report", ""),
                    current_incident.get("location", ""),
                    current_incident.get("incident_type", "Other"),
                    exclude_id=incident_id,
                )

                new_questions = generate_additional_questions(
                    current_incident,
                    current_answers,
                    current_questions,
                    matches,
                )

                if new_questions:
                    insert_questions(
                        incident_id,
                        new_questions,
                        len(current_questions) + 1,
                    )
                    log_event(
                        incident_id,
                        "More questions generated",
                        f"Added {len(new_questions)} question(s).",
                    )
                    st.success(
                        f"Added {len(new_questions)} question(s)."
                    )
                    st.rerun()
                else:
                    st.info(
                        "No new distinct questions were generated. "
                        "You can generate the report with the evidence "
                        "already collected."
                    )

    st.caption(
        "Answers are saved immediately. Report generation does not "
        "depend on generating additional questions successfully."
    )


def show_report_and_notification(incident):
    incident_id = incident["id"]

    st.subheader("Final incident report")

    incident = get_incident(incident_id)

    report = (
        incident.get("final_report")
        or incident.get("ai_analysis")
        or ""
    )

    if report:
        st.text_area(
            "Saved report",
            value=report,
            height=350,
            key=f"report_view_{incident_id}",
        )

        st.download_button(
            "Download report",
            data=report,
            file_name=f"rescueflow_case_{incident_id}.txt",
            mime="text/plain",
            key=f"download_report_{incident_id}",
        )

    if st.button(
        "Generate / Retry final report",
        type="primary",
        key=f"generate_report_{incident_id}",
    ):
        with st.spinner(
            "Generating the report. Saved answers will be preserved..."
        ):
            try:
                new_report, used_fallback = generate_final_report(
                    incident_id
                )

                if used_fallback:
                    st.warning(
                        "Gemma was unavailable. A provisional "
                        "rule-based fallback report has been saved. "
                        "You can retry AI report generation later."
                    )
                else:
                    st.success("Gemma report generated and saved.")

                st.rerun()

            except Exception as exc:
                st.error(
                    "Report generation could not be completed: "
                    f"{type(exc).__name__}. Your saved answers remain "
                    "in the database. Please retry."
                )

    st.divider()
    st.subheader("Department notification")

    incident = get_incident(incident_id)
    department_info = get_department(
        incident.get("incident_type", "Other")
    )

    st.write(
        "**Suggested department:**",
        department_info["name"],
    )

    recipient_env = department_info["phone_env"]
    configured_recipient = os.getenv(recipient_env, "")

    recipient = st.text_input(
        "Department SMS recipient number",
        value=configured_recipient,
        key=f"recipient_{incident_id}",
        help=(
            f"Set {recipient_env} in .env for a default number. "
            "Verify the number before sending."
        ),
    )

    st.warning(
        "Review the report and recipient before sending. SMS delivery "
        "does not itself register an official government complaint."
    )

    can_notify = bool(
        (incident.get("final_report") or incident.get("ai_analysis"))
        and recipient.strip()
    )

    confirm = st.checkbox(
        "I have reviewed this report and confirmed the recipient.",
        key=f"confirm_notify_{incident_id}",
    )

    if st.button(
        "Send department SMS",
        disabled=not (can_notify and confirm),
        key=f"send_sms_{incident_id}",
    ):
        current_incident = get_incident(incident_id)

        with st.spinner("Sending notification through the SMS provider..."):
            success, details = send_department_sms(
                current_incident,
                recipient.strip(),
            )

        if success:
            record_notification(
                incident_id,
                department_info["name"],
                recipient.strip(),
                "Accepted by provider",
                details,
            )
            st.success(details)
        else:
            record_notification(
                incident_id,
                department_info["name"],
                recipient.strip(),
                "Failed",
                details,
            )
            st.error(details)

    st.caption(
        "Configure Twilio credentials in .env to enable real SMS. "
        "Provider acceptance is not the same as confirmed delivery."
    )


# ============================================================
# MAIN APPLICATION
# ============================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🚨",
    layout="wide",
)

init_db()

st.title("🚨 RescueFlow AI")
st.caption(
    "Evidence-based incident investigation, clarification and reporting"
)

st.info(
    "Gemma provides language-model assistance. EvidenceLoop uses "
    "local heuristics and saved case history. All findings are "
    "provisional until verified."
)

tab_report, tab_cases, tab_history = st.tabs([
    "Report an Incident",
    "Incident Cases",
    "Investigation History",
])


# ============================================================
# TAB 1: SUBMIT AND INVESTIGATE
# ============================================================

with tab_report:
    st.subheader("Submit a new incident")

    with st.form("new_incident_form", clear_on_submit=False):
        report = st.text_area(
            "Describe the incident",
            placeholder=(
                "Example: A tree has fallen across the road near "
                "the bus stop, blocking traffic..."
            ),
            height=130,
        )

        location = st.text_input(
            "Incident location",
            placeholder="Street, area, junction or nearby landmark",
        )

        incident_type = st.selectbox(
            "Incident category",
            INCIDENT_TYPES,
        )

        submitted = st.form_submit_button(
            "Submit report and start EvidenceLoop",
            type="primary",
        )

    if submitted:
        if not report.strip():
            st.error("Please describe the incident.")
        elif not location.strip():
            st.error("Please provide the incident location.")
        else:
            created_at = datetime.now().isoformat(timespec="seconds")
            priority = calculate_priority(report, incident_type)
            department = get_department(incident_type)["name"]

            # Save the report first so a Gemma outage cannot lose it.
            with get_connection() as conn:
                cursor = conn.execute("""
                    INSERT INTO incidents (
                        created_at, report, location, status,
                        incident_type, ai_analysis, priority,
                        workflow_state, department, notification_status
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    created_at,
                    report.strip(),
                    location.strip(),
                    "New",
                    incident_type,
                    "",
                    priority,
                    "clarifying",
                    department,
                    "Not sent",
                ))

                incident_id = cursor.lastrowid
                conn.commit()

            log_event(
                incident_id,
                "Incident submitted",
                f"Category={incident_type}; location={location.strip()}",
            )

            with st.spinner(
                "Saving the report and checking related cases..."
            ):
                matches = compare_reports(
                    report,
                    location,
                    incident_type,
                    exclude_id=incident_id,
                )

                analysis, questions, priority = (
                    make_initial_analysis_and_questions(
                        report,
                        location,
                        incident_type,
                        matches,
                    )
                )

                insert_questions(
                    incident_id,
                    questions,
                    1,
                )

                with get_connection() as conn:
                    conn.execute("""
                        UPDATE incidents
                        SET ai_analysis = ?,
                            priority = ?,
                            followup_question = ?,
                            workflow_state = 'clarifying'
                        WHERE id = ?
                    """, (
                        analysis,
                        priority,
                        questions[0] if questions else None,
                        incident_id,
                    ))
                    conn.commit()

                log_event(
                    incident_id,
                    "Initial EvidenceLoop assessment",
                    f"Found {len(matches)} possible related case(s); "
                    f"queued {len(questions)} question(s).",
                )

            st.session_state["selected_incident_id"] = incident_id

            st.success(
                f"Incident #{incident_id} saved. "
                f"Priority: {priority}. "
                f"{len(questions)} follow-up question(s) prepared."
            )

            st.rerun()

    st.divider()

    selected_id = st.session_state.get("selected_incident_id")

    if selected_id:
        incident = get_incident(selected_id)

        if incident:
            st.header(f"Incident #{selected_id}")
            show_incident_summary(incident)

            st.divider()
            show_evidence_loop(incident)

            st.divider()
            show_followup_workflow(incident)

            st.divider()
            show_report_and_notification(incident)

    else:
        st.caption(
            "Submit an incident or select an existing case in "
            "the Incident Cases tab to continue an investigation."
        )


# ============================================================
# TAB 2: CASE MANAGEMENT
# ============================================================

with tab_cases:
    st.subheader("Incident case management")

    with get_connection() as conn:
        rows = conn.execute("""
            SELECT id, created_at, incident_type, location,
                   status, priority, workflow_state,
                   notification_status, report
            FROM incidents
            ORDER BY id DESC
        """).fetchall()

    cases = [dict(row) for row in rows]

    if not cases:
        st.info("No incident cases have been submitted yet.")
    else:
        search = st.text_input(
            "Search cases by report, location or category"
        ).strip().lower()

        status_options = sorted({
            item.get("status") or "New"
            for item in cases
        })

        status_filter = st.selectbox(
            "Filter by status",
            ["All"] + status_options,
        )

        filtered = []

        for case in cases:
            searchable = " ".join([
                str(case.get("id", "")),
                case.get("report") or "",
                case.get("location") or "",
                case.get("incident_type") or "",
            ]).lower()

            if search and search not in searchable:
                continue

            if (
                status_filter != "All"
                and (case.get("status") or "New") != status_filter
            ):
                continue

            filtered.append(case)

        st.write(f"Showing **{len(filtered)}** case(s).")

        if filtered:
            st.dataframe(
                [{
                    "ID": case["id"],
                    "Created": case.get("created_at"),
                    "Category": case.get("incident_type"),
                    "Location": case.get("location"),
                    "Priority": case.get("priority"),
                    "Status": case.get("status"),
                    "Notification": case.get("notification_status"),
                    "Report": (case.get("report") or "")[:100],
                } for case in filtered],
                use_container_width=True,
                hide_index=True,
            )

            case_ids = [case["id"] for case in filtered]

            current_selected = st.session_state.get(
                "selected_incident_id"
            )

            default_index = (
                case_ids.index(current_selected)
                if current_selected in case_ids
                else 0
            )

            chosen_id = st.selectbox(
                "Choose a case to investigate",
                case_ids,
                index=default_index,
                format_func=lambda value: (
                    f"#{value} — "
                    f"{next(
                        (c.get('incident_type', 'Other')
                         for c in filtered if c['id'] == value),
                        'Other'
                    )}"
                ),
            )

            if st.button(
                "Open selected case",
                type="primary",
            ):
                st.session_state["selected_incident_id"] = chosen_id
                st.rerun()

            selected_case = get_incident(chosen_id)

            if selected_case:
                st.divider()
                st.markdown(f"### Case #{chosen_id}")
                show_incident_summary(selected_case)

                current_status = selected_case.get("status") or "New"

                valid_statuses = [
                    "New",
                    "Under Investigation",
                    "Awaiting Evidence",
                    "Awaiting Review",
                    "Verified",
                    "Resolved",
                    "Closed",
                ]

                status_index = (
                    valid_statuses.index(current_status)
                    if current_status in valid_statuses
                    else 0
                )

                new_status = st.selectbox(
                    "Update case status",
                    valid_statuses,
                    index=status_index,
                    key=f"case_status_{chosen_id}",
                )

                if st.button(
                    "Save status change",
                    key=f"save_status_{chosen_id}",
                ):
                    with get_connection() as conn:
                        conn.execute("""
                            UPDATE incidents
                            SET status = ?
                            WHERE id = ?
                        """, (new_status, chosen_id))
                        conn.commit()

                    log_event(
                        chosen_id,
                        "Status updated",
                        f"{current_status} -> {new_status}",
                    )

                    st.success("Case status updated.")
                    st.rerun()

                if st.button(
                    "Continue this investigation",
                    key=f"continue_case_{chosen_id}",
                ):
                    st.session_state["selected_incident_id"] = chosen_id
                    st.rerun()

        else:
            st.info("No cases match your filters.")


# ============================================================
# TAB 3: INVESTIGATION HISTORY AND NOTIFICATIONS
# ============================================================

with tab_history:
    st.subheader("Investigation audit trail")

    with get_connection() as conn:
        events = conn.execute("""
            SELECT incident_id, event_type, details, created_at
            FROM investigation_history
            ORDER BY id DESC
            LIMIT 500
        """).fetchall()

        notifications = conn.execute("""
            SELECT incident_id, department, recipient,
                   status, details, created_at
            FROM notification_history
            ORDER BY id DESC
            LIMIT 200
        """).fetchall()

    if events:
        st.dataframe(
            [dict(row) for row in events],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No investigation events recorded yet.")

    st.divider()
    st.subheader("Department notification history")

    if notifications:
        st.dataframe(
            [dict(row) for row in notifications],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No department notifications have been attempted.")

    st.divider()
    st.subheader("Existing evidence history")

    # Display an older evidence_history table if it exists, without
    # assuming a particular column layout or modifying its records.
    with get_connection() as conn:
        existing_tables = {
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }

        if "evidence_history" in existing_tables:
            evidence_columns = table_columns(conn, "evidence_history")

            if evidence_columns:
                safe_columns = ", ".join(
                    f'"{column}"' for column in evidence_columns
                )

                evidence_rows = conn.execute(
                    f"SELECT {safe_columns} FROM evidence_history "
                    "LIMIT 500"
                ).fetchall()

                if evidence_rows:
                    st.dataframe(
                        [dict(row) for row in evidence_rows],
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.caption("The existing evidence history is empty.")
            else:
                st.caption("The evidence history table has no columns.")
        else:
            st.caption(
                "No legacy evidence_history table was found. "
                "New investigation events are stored in "
                "investigation_history."
            )


st.divider()
st.caption(
    "RescueFlow AI is a decision-support prototype. Similarity scores, "
    "priority rules and AI-generated reports are not independently "
    "verified facts. Verify important findings before taking action."
)