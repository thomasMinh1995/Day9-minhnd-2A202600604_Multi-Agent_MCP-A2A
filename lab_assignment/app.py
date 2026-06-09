"""Streamlit app for the Supervisor-Workers legal lab assignment."""

from __future__ import annotations

import streamlit as st

from .supervisor import LegalSupervisorSystem


st.set_page_config(page_title="Legal Supervisor-Workers", page_icon="SW", layout="wide")


@st.cache_resource
def get_system() -> LegalSupervisorSystem:
    return LegalSupervisorSystem()


st.title("Legal Supervisor-Workers")
st.caption("Supervisor plans the workflow, then delegates to retrieval, analysis, drafting, and compliance workers.")

with st.sidebar:
    st.subheader("Workers")
    for card in get_system().discover_workers():
        st.markdown(f"**{card['name']}**")
        st.caption(card["role"])
    top_k = st.slider("Top K evidence", min_value=1, max_value=10, value=5)

query = st.chat_input("Ask a Vietnamese drug-law question...")
if query:
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Supervisor is delegating work..."):
            result = get_system().ask(query, top_k=top_k)
        st.markdown(result["answer"])

        with st.expander("Supervisor plan and trace"):
            st.json(result["supervisor"])
        with st.expander("Analysis"):
            st.json(result["analysis"])
        with st.expander("Audit"):
            st.json(result["audit"])
