
import sqlite3
from datetime import datetime
import streamlit as st

DB_PATH = "rescueflow.db"

st.set_page_config(
    page_title="RescueFlow AI",
    page_icon="🚨",
    layout="wide",
)

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS incidents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                report TEXT NOT NULL,
                location TEXT NOT NULL,
                status TEXT NOT NULL,
                incident_type TEXT NOT NULL
            )
        """)

def save_incident(report, location, incident_type):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """INSERT INTO incidents
               (created_at, report, location, status, incident_type)
               VALUES (?, ?, ?, ?, ?)""",
            (
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                report,
                location,
                "Reported",
                incident_type,
            ),
        )

def get_incidents():
    with sqlite3.connect(DB_PATH) as conn:
        return conn.execute(
            """SELECT id, created_at, report, location, status,
                      incident_type
               FROM incidents ORDER BY id DESC"""
        ).fetchall()

init_db()

st.title("🚨 RescueFlow AI")
st.caption(
    "EvidenceLoop · Evidence-linked incident investigation "
    "and decision support"
)

st.info(
    "Demonstration mode: submitted reports are user-provided "
    "and are not independently verified."
)

tab_submit, tab_cases = st.tabs(
    ["➕ Report an Incident", "📋 Incident Cases"]
)

with tab_submit:
    st.subheader("Submit a community incident")

    with st.form("incident_form", clear_on_submit=True):
        report = st.text_area(
            "Describe what happened",
            placeholder="Example: Water is collecting near the main gate.",
        )
        location = st.text_input(
            "Location",
            placeholder="Example: College main gate",
        )
        incident_type = st.selectbox(
            "Incident category",
            [
                "Waterlogging / Flooding",
                "Blocked Drain",
                "Fallen Tree",
                "Road Obstruction",
                "Other Infrastructure Issue",
                "Unknown / Needs Classification",
            ],
        )

        submitted = st.form_submit_button(
            "Save Incident",
            type="primary",
        )

    if submitted:
        if not report.strip() or not location.strip():
            st.error("Enter both a report description and location.")
        else:
            save_incident(
                report.strip(),
                location.strip(),
                incident_type,
            )
            st.success(
                "Incident saved successfully. "
                "This confirms database storage, not incident verification."
            )

with tab_cases:
    st.subheader("Incident Register")
    incidents = get_incidents()

    if not incidents:
        st.write("No incidents yet. Submit a report to get started.")
    else:
        st.metric("Total recorded reports", len(incidents))

        for case in incidents:
            (
                case_id,
                created_at,
                report,
                location,
                status,
                incident_type,
            ) = case

            with st.expander(
                f"Case #{case_id} · {incident_type} · {status}"
            ):
                st.write("**Description:**", report)
                st.write("**Location:**", location)
                st.write("**Recorded at:**", created_at)
                st.write("**Status:**", status)
                st.caption(
                    "Status reflects the workflow, not confirmation "
                    "that the reported event occurred."
                )