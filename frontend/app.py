from __future__ import annotations

import html
import os
import re
import threading
import time
from urllib.parse import quote

import requests
import streamlit as st

API_URL = os.environ.get("F1_API_URL", "http://localhost:8000").rstrip("/")
REQUEST_TIMEOUT = float(os.environ.get("F1_API_TIMEOUT", "240"))
FIA_URL = "https://www.fia.com/regulation/category/110"
EXAMPLES = [
    ("What is the pit lane speed limit?", "speed"),
    ("How many ECU applications are allowed in 2029?", "calendar"),
    ("What does the driver licence note say?", "doc"),
]
ICONS = {
    "speed": "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='1.4' stroke-linecap='round' stroke-linejoin='round'><path d='M4 17a8.5 8.5 0 1 1 16 0'/><path d='M12 17l4-5'/><circle cx='12' cy='17' r='0.6'/></svg>",
    "calendar": "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='1.4' stroke-linecap='round' stroke-linejoin='round'><rect x='4' y='5.5' width='16' height='14.5' rx='2.5'/><path d='M4 10h16M9 3.5v4M15 3.5v4'/></svg>",
    "doc": "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='1.4' stroke-linecap='round' stroke-linejoin='round'><path d='M7 3.5h7l4 4V20a.5.5 0 0 1-.5.5h-10.5a.5.5 0 0 1-.5-.5V4a.5.5 0 0 1 .5-.5z'/><path d='M14 3.5V8h4M9 12.5h6M9 16h6'/></svg>",
}
ABSTAIN_TEXT = {
    "no_evidence": "No relevant regulation text was found for this question.",
    "model_declined": "The regulations retrieved do not contain the information asked for.",
    "no_verified_claims": "The generated answer could not be verified against the cited regulation text, so it is not shown.",
    "no_quantity_in_evidence": "The question asks for a figure or limit that the retrieved regulation text does not state.",
    "requested_information_missing": "The specific information asked for is not stated in the retrieved regulation text.",
    "generation_failed": "The language model could not be reached. Start Ollama with `ollama serve` and check that the configured model is installed.",
}


def build_css(answer_view: bool) -> str:
    icon_rules = "".join(
        f'.st-key-ex{i} button p::before {{mask-image: url("data:image/svg+xml;utf8,{quote(ICONS[icon])}"); '
        f'-webkit-mask-image: url("data:image/svg+xml;utf8,{quote(ICONS[icon])}");}}\n'
        for i, (_, icon) in enumerate(EXAMPLES)
    )
    top = "3.5rem" if answer_view else "max(26vh, 5rem)"
    form = "100%" if answer_view else "415px"
    return CSS_BASE.replace("__FORM__", form).replace("__TOP__", top).replace("__WIDTH__", "660px" if answer_view else "455px") + "<style>" + icon_rules + "</style>"


CSS_BASE = """
<style>
html, body, .stApp {background-color: #FBFBFA !important;}
.stApp, .stApp button, .stApp input, .stApp p, .stApp h1, .stApp summary {font-family: "Google Sans", "Inter", -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;}
.stApp {background-image:
  radial-gradient(60vw 50vh at 12% 8%, rgba(214, 216, 222, 0.38), transparent 70%),
  radial-gradient(55vw 55vh at 88% 22%, rgba(222, 220, 226, 0.34), transparent 70%),
  radial-gradient(70vw 50vh at 45% 105%, rgba(212, 215, 220, 0.36), transparent 70%);
  background-attachment: fixed;}
header[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"], footer, #MainMenu, [data-testid="stSidebar"] {display: none !important;}
.block-container {max-width: __WIDTH__ !important; padding: __TOP__ 20px 4rem 20px !important; animation: f1fade .45s ease;}
@keyframes f1fade {from {opacity: 0; transform: translateY(6px);} to {opacity: 1; transform: none;}}
h1.f1-title {text-align: center; font-size: 2rem; font-weight: 400; letter-spacing: -0.012em; color: #1F1F1F; margin: 0 0 1.5rem 0; padding: 0; line-height: 1.2;}
[data-testid="stForm"] {border: 0 !important; padding: 0 !important; max-width: __FORM__; margin: 0 auto;}
[data-testid="stForm"] [data-testid="stVerticalBlock"] {gap: 0;}
[data-testid="stTextInput"] label {display: none;}
[data-testid="stTextInput"] > div {width: 100%;}
[data-testid="stTextInput"] div[data-baseweb="input"], [data-testid="stTextInput"] div[data-baseweb="base-input"] {background: #fff !important; border: 0 !important; border-radius: 999px !important;}
div[data-baseweb="input"] {box-shadow: 0 1px 2px rgba(60, 64, 67, 0.08), 0 4px 18px rgba(60, 64, 67, 0.10); height: 52px; transition: box-shadow .2s ease;}
div[data-baseweb="input"]:focus-within {box-shadow: 0 1px 3px rgba(60, 64, 67, 0.12), 0 6px 22px rgba(60, 64, 67, 0.16);}
[data-testid="stTextInput"] input {height: 52px; padding: 0 24px !important; font-size: 0.95rem; color: #1F1F1F; background: transparent !important; border-radius: 999px;}
[data-testid="stTextInput"] input::placeholder {color: #9A9DA3; opacity: 1;}
[data-testid="InputInstructions"] {display: none !important;}
[data-testid="stFormSubmitButton"] {position: absolute; width: 1px; height: 1px; overflow: hidden; opacity: 0; pointer-events: none;}
[data-testid="stForm"] {margin-bottom: 0.9rem !important;}
[class*="st-key-ex"] {width: 100%; max-width: 415px; margin: 0 auto;}
[class*="st-key-ex"] + [class*="st-key-ex"] {margin-top: -0.5rem;}
[class*="st-key-ex"] button {background: transparent !important; border: 0 !important; box-shadow: none !important; color: #4A4D52 !important; justify-content: flex-start; padding: 0.42rem 0.9rem !important; min-height: 0 !important; border-radius: 999px; width: 100%;}
[class*="st-key-ex"] button:hover {background: rgba(60, 64, 67, 0.05) !important; color: #1F1F1F !important;}
[class*="st-key-ex"] button p {font-size: 0.875rem; font-weight: 400; display: flex; align-items: center; gap: 0.75rem; margin: 0; text-align: left;}
[class*="st-key-ex"] button p::before {content: ""; width: 17px; height: 17px; flex: none; background: #6B6E74; mask-size: contain; mask-repeat: no-repeat; mask-position: center; -webkit-mask-size: contain; -webkit-mask-repeat: no-repeat; -webkit-mask-position: center;}
.f1-wait {color: #6B6E74; font-size: 0.9rem; text-align: center; margin: 1.6rem 0; animation: f1pulse 1.6s ease-in-out infinite;}
@keyframes f1pulse {0%, 100% {opacity: .55;} 50% {opacity: 1;}}
.f1-q {color: #6B6E74; font-size: 0.85rem; margin: 2rem 0 0.2rem 0;}
.f1-line {font-size: 1.02rem; line-height: 1.65; color: #1F1F1F; margin: 0 0 0.85rem 0;}
.f1-tag {display: inline-block; font-size: 0.7rem; color: #5A5D63; background: rgba(60, 64, 67, 0.08); border-radius: 999px; padding: 1px 9px; margin-right: 0.5rem; vertical-align: 1px;}
.f1-ref {color: #6B6E74; font-size: 0.78rem; margin-left: 0.15rem; vertical-align: super;}
.f1-note {color: #6B6E74; font-size: 0.88rem; line-height: 1.55; margin: 0.2rem 0 0.7rem 0;}
.f1-err {color: #8A3B3B; font-size: 0.95rem; margin-top: 2rem; text-align: center;}
.f1-cite-h {color: #6B6E74; font-size: 0.8rem; margin: 1.8rem 0 0.3rem 0;}
[data-testid="stExpander"] {border: 0 !important; border-bottom: 1px solid rgba(60, 64, 67, 0.12) !important; border-radius: 0 !important; background: transparent !important;}
[data-testid="stExpander"] details {border: 0 !important;}
[data-testid="stExpander"] summary {padding: 0.55rem 0 !important; font-size: 0.85rem; color: #4A4D52;}
[data-testid="stExpander"] summary:hover {color: #1F1F1F;}
[data-testid="stExpander"] [data-testid="stMarkdownContainer"] p {font-size: 0.85rem; color: #4A4D52;}
a {color: #3B5BA5;}
@media (max-width: 560px) {
  h1.f1-title {font-size: 1.6rem;}
  .block-container {padding-left: 16px !important; padding-right: 16px !important;}
  [data-testid="stForm"], [class*="st-key-ex"] {max-width: 100%;}
}
</style>
"""
REF_RE = re.compile(r"\[C(\d+)\]")
NB_PREFIX = "Non-binding comment:"
NOT_COVERED_MARK = "Not covered by the available evidence:"


def call_api(question: str, box: dict) -> None:
    try:
        r = requests.post(f"{API_URL}/ask", json={"question": question}, timeout=REQUEST_TIMEOUT)
        if r.ok:
            box["data"] = r.json()
        else:
            try:
                detail = r.json().get("detail")
            except ValueError:
                detail = None
            box["error"] = detail if isinstance(detail, str) and r.status_code == 503 else f"The API returned an error (HTTP {r.status_code})."
    except requests.ConnectionError:
        box["error"] = f"Cannot reach the API at {API_URL}. Start the backend first (see the instructions below)."
    except requests.Timeout:
        box["error"] = "The request timed out. The language model may be overloaded; try again."
    except requests.RequestException as e:
        box["error"] = f"Request failed: {type(e).__name__}"


def ask_with_progress(question: str) -> dict:
    box: dict = {}
    worker = threading.Thread(target=call_api, args=(question, box), daemon=True)
    worker.start()
    start = time.perf_counter()
    ticker = st.empty()
    while worker.is_alive():
        elapsed = time.perf_counter() - start
        stage = "Searching the regulations" if elapsed < 2 else "Writing a cited answer"
        ticker.markdown(f'<div class="f1-wait">{stage}…</div>', unsafe_allow_html=True)
        time.sleep(0.4)
    worker.join()
    ticker.empty()
    return box


def pdf_link(url: str, page: int) -> str:
    return f"{url}#page={page}" if url and page else url


def render_line(raw: str) -> str:
    text = raw.strip()
    tags = ""
    if text.startswith(NB_PREFIX):
        text = text[len(NB_PREFIX):].strip()
        tags += '<span class="f1-tag">Non-binding comment</span>'
    if "(approved future change" in text:
        tags += '<span class="f1-tag">Future change · not in force</span>'
        text = re.sub(r"\s*\(approved future change[^)]*\)", "", text)
    body = html.escape(text)
    body = REF_RE.sub(lambda m: f'<span class="f1-ref">[C{m.group(1)}]</span>', body)
    return f'<div class="f1-line">{tags}{body}</div>'


def note(text: str) -> None:
    st.markdown(f'<div class="f1-note">{html.escape(text)}</div>', unsafe_allow_html=True)


def render_answer(data: dict) -> None:
    answer = data["answer"].replace(f" {NOT_COVERED_MARK}", f"\n{NOT_COVERED_MARK}")
    lines = [ln for ln in answer.split("\n") if ln.strip()]
    answer_lines = [ln for ln in lines if not ln.startswith(("Note:", NOT_COVERED_MARK))]
    missing = [ln[len(NOT_COVERED_MARK):].strip() for ln in lines if ln.startswith(NOT_COVERED_MARK)]
    if data["abstained"]:
        reason = ABSTAIN_TEXT.get(data.get("abstain_reason") or "", "The regulations do not support an answer to this question.")
        st.markdown(f'<div class="f1-line">No supported answer. {html.escape(reason)}</div>', unsafe_allow_html=True)
    else:
        if missing or any(ln.startswith("Note:") for ln in lines):
            note("Partial answer: only the parts supported by the regulations are shown.")
        for ln in answer_lines:
            st.markdown(render_line(ln), unsafe_allow_html=True)
    if data.get("abstain_reason") != "generation_failed":
        for m in missing:
            note(f"Not covered by the available evidence: {m}")


def render_notices(data: dict) -> None:
    info = data.get("year_interpretation") or {}
    if info.get("mode") in ("future", "ambiguous_both", "past"):
        note(f"Year interpretation: {info.get('reason', '')}")
    if data.get("context_truncated"):
        note("Context truncated: the requested article is longer than the evidence provided to the model, so the answer may be incomplete. Ask about a specific clause for a complete answer.")
    omitted = [w for w in data.get("warnings", []) if "removed" in w or "failed verification" in w]
    if omitted and not data["abstained"]:
        note("Some generated statements were removed because they could not be verified against the cited regulation text.")


def citation_label(c: dict) -> str:
    article = f"Art. {c['article_id']}" if c.get("article_id") else "Article n/a"
    pages = f"p. {c['page_start']}" if c["page_start"] == c["page_end"] else f"pp. {c['page_start']}–{c['page_end']}"
    return f"{c.get('ref') or '[C?]'} · Section {c['section']} · {article} · Issue {c['issue']} · {pages}"


def render_citations(data: dict) -> None:
    citations = data.get("citations") or []
    if not citations:
        return
    st.markdown('<div class="f1-cite-h">Citations</div>', unsafe_allow_html=True)
    for c in citations:
        with st.expander(citation_label(c)):
            tags = []
            if c.get("is_binding") is False:
                tags.append("Non-binding comment")
            if c.get("is_current") is False:
                year = c.get("effective_year")
                tags.append(f"Future change effective {year}, not in force" if year else "Future change, not in force")
            if tags:
                st.markdown(" · ".join(f"**{t}**" for t in tags))
            st.markdown(
                f"**Section:** {c['section']} ({c.get('section_name') or 'n/a'})  \n"
                f"**Article:** {c.get('article_id') or 'n/a'}"
                + (f" — {c['article_title']}" if c.get("article_title") else "")
                + f"  \n**Issue:** {c['issue']}  \n**PDF page:** {c['page_start']}"
                + (f"–{c['page_end']}" if c["page_end"] != c["page_start"] else "")
                + f"  \n**Season:** {c['season']}"
            )
            if c.get("source_url"):
                st.markdown(f"[Open the official FIA PDF at this page]({pdf_link(c['source_url'], c['page_start'])})")


def set_question(text: str) -> None:
    st.session_state["question"] = text
    st.session_state["run"] = True


def main() -> None:
    st.set_page_config(page_title="F1 Regulations AI", page_icon="🏁", layout="centered")
    st.session_state.setdefault("question", "")
    st.session_state.setdefault("run", False)
    st.session_state.setdefault("result", None)
    answer_view = st.session_state["result"] is not None or st.session_state["run"]
    st.markdown(build_css(answer_view), unsafe_allow_html=True)

    if not answer_view:
        st.markdown('<h1 class="f1-title">Where should we start?</h1>', unsafe_allow_html=True)

    with st.form("ask", border=False):
        st.text_input("Question", key="question", placeholder="Ask about the 2026 regulations", label_visibility="collapsed")
        submitted = st.form_submit_button("Ask")

    if not answer_view:
        for i, (text, _) in enumerate(EXAMPLES):
            st.button(text, key=f"ex{i}", on_click=set_question, args=(text,))

    question = st.session_state["question"].strip()
    if (submitted or st.session_state["run"]) and question:
        st.session_state["run"] = False
        st.session_state["result"] = {"question": question, **ask_with_progress(question)}
        st.rerun()

    result = st.session_state["result"]
    if result:
        if "error" in result:
            st.markdown(f'<div class="f1-err">{html.escape(result["error"])}</div>', unsafe_allow_html=True)
        else:
            data = result["data"]
            render_answer(data)
            render_notices(data)
            render_citations(data)


main()
