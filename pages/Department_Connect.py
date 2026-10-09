
import sqlite3
from datetime import datetime
from pathlib import Path

import streamlit as st

# Uses the same SQLite database as the main RescueFlow app.
DB_PATH = Path(__file__).resolve().parent.parent / "rescueflow.db"

st.set_page_config(
    page_title="Department Connect | RescueFlow AI",
    page_icon="🏛️",
    layout="wide",
)

# --------------------------------------------------
# OFFICIAL CONTACT DIRECTORY
# Verify contact details periodically before relying on them.
# --------------------------------------------------

DEPARTMENTS = {
    "Karnataka Janaspandana": {
        "purpose": "Karnataka government public grievances",
        "phone": "1902",
        "website": "https://ipgrs.karnataka.gov.in/",
        "website_label": "Open Janaspandana portal",
    },
    "BBMP Sahaaya": {
        "purpose": "Civic complaints within BBMP jurisdiction",
        "phone": "9480685700",
        "alternate_phone": "080-22975550",
        "website": "https://site.bbmp.gov.in/contact.html",
        "website_label": "Open official BBMP contact page",
    },
    "BESCOM": {
        "purpose": "Electricity-related complaints in BESCOM service areas",
        "phone": "1912",
        "website": "https://www.bescom.co.in/",
        "website_label": "Open official BESCOM website",
    },
}

# --------------------------------------------------
# DATABASE
# --------------------------------------------------

def connect_db():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def init_contact_log():
    with connect_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS department_contact_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                case_id INTEGER NOT NULL,
                department TEXT NOT NULL,
                contact_channel TEXT NOT NULL,
                contact_status TEXT NOT NULL,
                reference_number TEXT DEFAULT '',
                notes TEXT DEFAULT '',
                created_at TEXT NOT NULL
            )
        """)


def get_cases():
    with connect_db() as conn:
        return [
            dict(row)
            for row in conn.execute(
                "SELECT * FROM incidents ORDER BY id DESC"
            ).fetchall()
        ]


def get_case_history(case_id):
    """Read follow-up answers if the current app has saved them."""
    with connect_db() as conn:
        table = conn.execute("""
            SELECT name
            FROM sqlite_master
            WHERE type = 'table' AND name = 'followup_answers'
        """).fetchone()

        if not table:
            return []

        columns = {
            row["name"]
            for row in conn.execute(
                "PRAGMA table_info(followup_answers)"
            ).fetchall()
        }

        if not {"case_id", "answer"}.issubset(columns):
            return []

        question_column = (
            "question" if "question" in columns else None
        )
        order_column = (
            "question_number"
            if "question_number" in columns
            else "rowid"
        )

        if question_column:
            query = f"""
                SELECT {question_column} AS question, answer
                FROM followup_answers
                WHERE case_id = ?
                ORDER BY {order_column}
            """
        else:
            query = f"""
                SELECT answer
                FROM followup_answers
                WHERE case_id = ?
                ORDER BY {order_column}
            """

        return [
            dict(row)
            for row in conn.execute(query, (case_id,)).fetchall()
        ]


def save_contact_record(
    case_id,
    department,
    channel,
    status,
    reference_number,
    notes,
):
    with connect_db() as conn:
        conn.execute("""
            INSERT INTO department_contact_log (
                case_id,
                department,
                contact_channel,
                contact_status,
                reference_number,
                notes,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            case_id,
            department,
            channel,
            status,
            reference_number.strip(),
            notes.strip(),
            datetime.now().astimezone().isoformat(timespec="seconds"),
        ))


def get_contact_records(case_id):
    with connect_db() as conn:
        return [
            dict(row)
            for row in conn.execute("""
                SELECT *
                FROM department_contact_log
                WHERE case_id = ?
                ORDER BY id DESC
            """, (case_id,)).fetchall()
        ]


# --------------------------------------------------
# DEPARTMENT RECOMMENDATIONS
# Recommendations depend on the incident and jurisdiction.
# --------------------------------------------------

def recommend_departments(case):
    incident_type = (
        case.get("incident_type") or ""
    ).lower()

    report = (case.get("report") or "").lower()
    combined = f"{incident_type} {report}"

    recommendations = []

    electrical_hazard = any(
        term in combined
        for term in [
            "electric wire",
            "electrical wire",
            "live wire",
            "hanging wire",
            "power line",
            "electricity wire",
            "electrocution",
        ]
    )

    if electrical_hazard:
        recommendations.append((
            "BESCOM",
            "Possible electrical hazard. Use this channel if the location "
            "is within BESCOM's service area. Do not approach the wires."
        ))

    if any(term in combined for term in [
        "fallen tree", "tree fallen", "road obstruction",
        "road blocked", "pothole", "blocked drain",
        "waterlogging", "flood", "flooding",
    ]):
        recommendations.append((
            "BBMP Sahaaya",
            "Potential civic issue, if the incident is within BBMP jurisdiction."
        ))

    recommendations.append((
        "Karnataka Janaspandana",
        "Government grievance route. Choose the appropriate department "
        "and jurisdiction when registering the grievance."
    ))

    # Remove duplicate department recommendations.
    unique = {}
    for name, reason in recommendations:
        unique[name] = reason

    return [
        (name, reason, DEPARTMENTS[name])
        for name, reason in unique.items()
    ]


# --------------------------------------------------
# INTERFACE
# --------------------------------------------------

init_contact_log()

st.title("🏛️ Department Connect")
st.caption(
    "RescueFlow AI — connect incident reports to relevant public authorities"
)

st.info(
    "This page helps residents find contact channels and prepare complaints. "
    "It does not automatically contact a department, dispatch responders, "
    "or confirm that a complaint has been submitted."
)

try:
    cases = get_cases()
except sqlite3.Error as exc:
    st.error(
        "Could not read the RescueFlow database. Start the main app from "
        "the project folder and check that rescueflow.db exists."
    )
    st.stop()

if not cases:
    st.warning(
        "No incident cases were found. Submit a report in RescueFlow AI first."
    )
    st.stop()

case_options = {
    f"Case #{case['id']} · "
    f"{case.get('incident_type') or 'Incident'} · "
    f"{case.get('location') or 'Location not provided'}": case
    for case in cases
}

selected_label = st.selectbox(
    "Choose an incident",
    list(case_options.keys()),
)

case = case_options[selected_label]
case_id = case["id"]

st.subheader(f"Incident Case #{case_id}")

col1, col2 = st.columns(2)
with col1:
    st.write("**Category:**", case.get("incident_type") or "Not specified")
    st.write("**Location:**", case.get("location") or "Not specified")

with col2:
    st.write("**Priority:**", case.get("priority") or "Not assigned")
    st.write("**Case status:**", case.get("status") or "Reported")

st.write("**Original report**")
st.write(case.get("report") or "No report text available.")

recommendations = recommend_departments(case)

st.divider()
st.subheader("Recommended contact options")
st.caption(
    "Recommendations are based on report keywords and may not identify "
    "the legally responsible authority. Check the location and jurisdiction."
)

for department_name, reason, details in recommendations:
    with st.container(border=True):
        st.markdown(f"### {department_name}")
        st.write(reason)
        st.write("**Purpose:**", details["purpose"])
        st.write("**Phone:**", details["phone"])

        if details.get("alternate_phone"):
            st.write(
                "**Additional number:**",
                details["alternate_phone"],
            )

        st.markdown(
            f"[{details['website_label']}]({details['website']})"
        )

        if department_name == "BESCOM":
            st.warning(
                "If wires may be live, keep away and warn others from a safe "
                "distance. Do not touch the tree, wires, or nearby objects. "
                "Use the appropriate emergency channel if there is immediate danger."
            )

st.divider()
st.subheader("Prepare a complaint")

followup_answers = get_case_history(case_id)

complaint_lines = [
    "RESCUEFLOW AI — INCIDENT COMPLAINT DRAFT",
    f"Case ID: {case_id}",
    f"Category: {case.get('incident_type') or 'Not specified'}",
    f"Reported location: {case.get('location') or 'Not specified'}",
    "",
    "Original report:",
    case.get("report") or "Not provided",
]

if followup_answers:
    complaint_lines.extend(["", "Follow-up information:"])
    for index, item in enumerate(followup_answers, start=1):
        question = item.get("question")
        answer = item.get("answer", "")

        if question:
            complaint_lines.append(f"Question: {question}")
        complaint_lines.append(f"Answer {index}: {answer}")

complaint_lines.extend([
    "",
    "Note: This complaint draft contains resident-provided information. "
    "The incident has not been independently verified by RescueFlow AI.",
])

complaint_draft = "\n".join(complaint_lines)

edited_draft = st.text_area(
    "Review and edit before submitting",
    value=complaint_draft,
    height=260,
    key=f"complaint_draft_{case_id}",
)

st.download_button(
    "Download complaint draft (.txt)",
    data=edited_draft,
    file_name=f"RescueFlow_Case_{case_id}_complaint.txt",
    mime="text/plain",
    use_container_width=True,
)

st.caption(
    "Download the draft, then use the relevant official contact channel. "
    "Submitting through an external portal is a separate action."
)

st.divider()
st.subheader("Record department contact")

st.write(
    "After contacting a department, record what actually happened. "
    "Do not mark a complaint as submitted unless you submitted it."
)

with st.form(f"contact_record_form_{case_id}"):
    department = st.selectbox(
        "Department",
        list(DEPARTMENTS.keys()),
    )

    channel = st.selectbox(
        "Contact method",
        [
            "Phone call",
            "Official grievance portal",
            "In-person visit",
            "Email",
            "Other",
        ],
    )

    contact_status = st.selectbox(
        "Contact outcome",
        [
            "Not contacted",
            "Contact attempted",
            "Complaint submitted by resident",
            "Acknowledgement received",
            "Follow-up required",
            "Resolved — reported by resident",
        ],
    )

    reference_number = st.text_input(
        "Complaint / acknowledgement number (optional)"
    )

    notes = st.text_area(
        "Notes",
        placeholder=(
            "For example: Called the department; asked to submit "
            "the complaint through the official portal."
        ),
    )

    save_contact = st.form_submit_button(
        "Save contact record",
        type="primary",
        use_container_width=True,
    )

if save_contact:
    save_contact_record(
        case_id,
        department,
        channel,
        contact_status,
        reference_number,
        notes,
    )

    st.success(
        "Contact record saved in the local database. "
        "This records your entry; it does not verify the department's response."
    )

st.divider()
st.subheader("Contact history for this case")

records = get_contact_records(case_id)

if not records:
    st.info("No department contact records have been added for this case.")
else:
    for record in records:
        with st.container(border=True):
            st.write(
                f"**{record['department']}** · {record['contact_status']}"
            )
            st.caption(
                f"{record['created_at']} · {record['contact_channel']}"
            )

            if record.get("reference_number"):
                st.write(
                    "**Reference number:**",
                    record["reference_number"],
                )

            if record.get("notes"):
                st.write(record["notes"])

st.caption(
    "RescueFlow AI prototype · Department suggestions are advisory. "
    "Contact details and jurisdiction should be checked before use."
)