import streamlit as st
import requests
import pandas as pd
import time

API = "http://127.0.0.1:8000"

st.set_page_config(page_title="DocGuard AI", page_icon="🛡️", layout="wide")

st.sidebar.title("🛡️ DocGuard AI")
st.sidebar.markdown("Enterprise Document Intelligence & Governance")
st.sidebar.divider()
page = st.sidebar.radio("Navigation", [
    "📄 Document Upload",
    "📊 Evaluation Dashboard",
    "👁️ Review Queue",
    "⚠️ Drift Alerts"
])
st.sidebar.divider()
st.sidebar.caption("Built with Claude API · LangChain · ChromaDB")

def risk_color(level):
    return {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(level, "⚪")

def norm(val):
    if val is None:
        return 0.0
    v = float(val)
    return round(v / 10.0 if v > 1.0 else v, 3)

def fmt(val):
    return f"{norm(val):.2f}" if val is not None else "N/A"

if page == "📄 Document Upload":
    st.title("📄 Document Processing")
    st.markdown("Upload a document to run it through the full pipeline: classify → verify → evaluate")
    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded = st.file_uploader("Choose a document", type=["pdf", "docx", "txt"])
        mode = st.radio("Inference mode", ["realtime", "batch"], horizontal=True)
        if uploaded and st.button("🚀 Process Document", type="primary"):
            with st.spinner("Running through pipeline..."):
                try:
                    resp = requests.post(
                        f"{API}/documents/process?mode={mode}",
                        files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
                    )
                    st.session_state["last_result"] = resp.json()
                except Exception as e:
                    st.error(f"API error: {e}")
    with col2:
        st.markdown("**Supported formats**")
        st.markdown("- PDF\n- DOCX\n- TXT")
        st.markdown("**Pipeline layers**")
        st.markdown("1. Claude classification\n2. RAG fraud check\n3. LLM judge scoring")
    if "last_result" in st.session_state:
        r = st.session_state["last_result"]
        st.divider()
        st.subheader("Results")
        ev = r.get("eval") or {}
        overall = norm(ev.get("overall"))
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Category", r.get("category", "N/A").replace("_", " ").title())
        c2.metric("Confidence", f"{r.get('confidence', 0):.0%}")
        c3.metric("Risk Level", f"{risk_color(r.get('verification', {}).get('risk_level', 'low'))} {r.get('verification', {}).get('risk_level', 'N/A').title()}")
        c4.metric("Eval Score", f"{overall:.2f}" if ev else "N/A")
        if r.get("flagged"):
            st.warning(f"⚠️ Flagged for human review: {r.get('flag_reason', '')}")
        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("**Extracted entities**")
            for e in r.get("output", {}).get("key_entities", []):
                st.markdown(f"- {e}")
            st.markdown("**Summary**")
            st.caption(r.get("output", {}).get("summary", "N/A"))
        with col_r:
            st.markdown("**Evaluation scores**")
            if ev:
                s1, s2 = st.columns(2)
                s1.metric("Correctness", fmt(ev.get("correctness")))
                s2.metric("Faithfulness", fmt(ev.get("faithfulness")))
                s3, s4 = st.columns(2)
                s3.metric("Completeness", fmt(ev.get("completeness")))
                s4.metric("Hallucination Risk", fmt(ev.get("hallucination_risk")))
            st.markdown("**Verification**")
            report = r.get("verification", {}).get("report", "N/A")
            st.caption(str(report)[:300] + "..." if len(str(report)) > 300 else report)
        st.markdown("**Document ID**")
        st.code(r.get("document_id", "N/A"))

elif page == "📊 Evaluation Dashboard":
    st.title("📊 Evaluation Dashboard")
    col1, col2 = st.columns([3, 1])
    with col2:
        days = st.selectbox("Time window", [7, 14, 30])
        category = st.text_input("Filter by category", placeholder="e.g. invoice")
    try:
        params = {"days": days}
        if category:
            params["category"] = category
        stats = requests.get(f"{API}/governance/stats", params=params).json()
        if stats.get("count", 0) == 0:
            st.info("No evaluations yet. Upload some documents first.")
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Documents Evaluated", stats.get("count", 0))
            c2.metric("Avg Quality Score", fmt(stats.get("avg", 0)))
            c3.metric("Flagged for Review", stats.get("flagged", 0))
            c4.metric("Drift Alerts", stats.get("drift_alerts", 0))
            st.divider()
            col_l, col_r = st.columns(2)
            with col_l:
                st.markdown("**Score range**")
                st.bar_chart(pd.DataFrame({
                    "Metric": ["Min", "Avg", "Max"],
                    "Value": [norm(stats.get("min")), norm(stats.get("avg")), norm(stats.get("max"))]
                }).set_index("Metric"))
            with col_r:
                st.markdown("**Quality distribution**")
                total = stats.get("count", 1)
                flagged = stats.get("flagged", 0)
                st.bar_chart(pd.DataFrame({
                    "Status": ["Passed", "Flagged"],
                    "Count": [total - flagged, flagged]
                }).set_index("Status"))
    except Exception as e:
        st.error(f"Could not load stats: {e}")

elif page == "👁️ Review Queue":
    st.title("👁️ Human Review Queue")
    try:
        data = requests.get(f"{API}/review-queue?limit=20").json()
        if data.get("count", 0) == 0:
            st.success("✅ Review queue is empty.")
        else:
            st.warning(f"{data['count']} documents pending review")
            for item in data.get("items", []):
                score = item.get("score") or 0
                with st.expander(f"📄 {str(item.get('doc_id',''))[:16]}... | Score: {norm(score):.2f}"):
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.json(item.get("output", {}))
                    with col2:
                        st.metric("Confidence", f"{norm(score):.2f}")
                        st.caption(item.get("reason", "N/A"))
                        action = st.radio("Decision", ["accepted", "rejected"], key=f"a_{item['id']}", horizontal=True)
                        notes = st.text_input("Notes", key=f"n_{item['id']}")
                        if st.button("Submit", key=f"s_{item['id']}"):
                            res = requests.post(f"{API}/review-queue/{item['id']}/decide", json={"action": action, "notes": notes}).json()
                            if res.get("ok"):
                                st.success("✅ Recorded")
                                time.sleep(1)
                                st.rerun()
    except Exception as e:
        st.error(f"Could not load queue: {e}")

elif page == "⚠️ Drift Alerts":
    st.title("⚠️ Model Drift Alerts")
    show_resolved = st.toggle("Show resolved", value=False)
    try:
        data = requests.get(f"{API}/governance/drift-alerts?resolved={str(show_resolved).lower()}").json()
        if data.get("count", 0) == 0:
            st.success("✅ No active drift alerts.")
        else:
            st.error(f"{data['count']} active alerts")
            for alert in data.get("alerts", []):
                drop = alert.get("drop") or 0
                with st.expander(f"🔴 {alert.get('category','N/A')} | Drop: {drop:.3f}"):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Baseline", f"{norm(alert.get('baseline')):.3f}")
                    c2.metric("Current", f"{norm(alert.get('current')):.3f}", delta=f"{-drop:.3f}")
                    c3.metric("Metric", alert.get("metric", "N/A"))
                    st.caption(f"Review recent {alert.get('category','')} documents and consider retraining.")
    except Exception as e:
        st.error(f"Could not load alerts: {e}")