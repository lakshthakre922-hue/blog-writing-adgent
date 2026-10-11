from __future__ import annotations

import base64
import os
import re
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

# Import LangGraph App and Schemas from backend
from bwa_backend import (
    app,
    Plan,
    Task,
    EvidenceItem,
    RouterDecision,
    ImageSpec,
    GlobalImagePlan,
    run,
)


# ============================================================
# Helpers & Data Processing
# ============================================================

def resolve_images_for_display(md_text: str) -> str:
    """
    Replaces relative images/ paths in Markdown with base64 data URIs so that
    Streamlit's st.markdown() renders the local images directly in the browser
    without 404 broken image icons.
    """
    if not md_text:
        return ""

    def replacer(match):
        alt = match.group(1)
        rel_path = match.group(2).replace("\\", "/").strip()
        p = Path(rel_path)
        if p.exists() and p.is_file():
            mime = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
            try:
                b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
                return f"![{alt}](data:{mime};base64,{b64})"
            except Exception:
                pass
        return match.group(0)

    # Match ![alt](images/filename.png)
    return re.sub(r'!\[(.*?)\]\(((?:images[\\/][^\)]+))\)', replacer, md_text)


def extract_images_from_md(md_text: str) -> List[Dict[str, str]]:
    """
    Extracts image specifications from Markdown if loaded from a saved file.
    """
    if not md_text:
        return []
    matches = re.findall(r'!\[(.*?)\]\(((?:images[\\/][^\)]+))\)(?:\n\*(.*?)\*)?', md_text)
    specs = []
    for idx, (alt, rel_path, caption) in enumerate(matches, 1):
        filename = Path(rel_path).name
        specs.append({
            "placeholder": f"[[IMAGE_{idx}]]",
            "filename": filename,
            "alt": alt,
            "caption": caption or alt,
            "prompt": alt,
        })
    return specs


def safe_slug(title: str) -> str:
    """
    Format blog title to match backend file naming convention:
    re.sub(r'[\\/:*?"<>|]', "", title).strip().lower().replace(" ", "_")
    """
    safe = re.sub(r'[\\/:*?"<>|]', "", title).strip().lower().replace(" ", "_")
    safe = re.sub(r"_+", "_", safe).strip("_")
    return safe or "blog_post"


def read_md_file(path: Path) -> str:
    """Read markdown file content with UTF-8 encoding."""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def extract_title_from_md(md: str, fallback: str = "Technical Blog") -> str:
    """Extract blog title from the first level-1 heading (# Title)."""
    for line in md.splitlines():
        line_clean = line.strip()
        if line_clean.startswith("# "):
            title = line_clean[2:].strip()
            if title:
                return title
    return fallback


def extract_sections_from_md(md_text: str) -> List[str]:
    """
    Extract individual sections starting with '## ' from a markdown string.
    Works for both live results and loaded past blogs.
    """
    chunks = re.split(r"(?=\n##\s+)", md_text)
    sections = []
    for chunk in chunks:
        c = chunk.strip()
        if c.startswith("## "):
            sections.append(c)
    return sections


def normalize_sections(raw_sections: Any) -> List[str]:
    """
    Normalizes sections into a sorted list of strings.
    Handles [(1, "## Sec 1..."), (2, "...")] or ["## Sec 1...", ...]
    Deduplicates by task_id and sorts numerically.
    """
    if not raw_sections:
        return []
    if isinstance(raw_sections, list):
        if len(raw_sections) > 0 and isinstance(raw_sections[0], tuple):
            dedup_dict = {}
            for item in raw_sections:
                if isinstance(item, tuple) and len(item) >= 2:
                    dedup_dict[item[0]] = item[1]
            return [dedup_dict[k] for k in sorted(dedup_dict.keys())]
        seen = set()
        out = []
        for s in raw_sections:
            s_str = str(s)
            if s_str not in seen:
                seen.add(s_str)
                out.append(s_str)
        return out
    return [str(raw_sections)]


def list_past_blogs() -> List[Dict[str, Any]]:
    """
    Find generated markdown blogs in the workspace directory.
    Sorted with newest first.
    """
    cwd = Path(".")
    md_files = [
        p for p in cwd.glob("*.md")
        if p.is_file() and not p.name.startswith(".")
    ]

    results = []
    for p in sorted(md_files, key=lambda f: f.stat().st_mtime, reverse=True):
        try:
            mtime = datetime.fromtimestamp(p.stat().st_mtime).strftime("%b %d, %H:%M")
            content = read_md_file(p)
            title = extract_title_from_md(content, fallback=p.stem.replace("_", " ").title())
            word_count = len(content.split())
            results.append({
                "path": p,
                "name": p.name,
                "title": title,
                "mtime": mtime,
                "words": word_count,
                "size_kb": round(p.stat().st_size / 1024, 1),
            })
        except Exception:
            continue
    return results


def parse_plan(plan_obj: Any) -> Dict[str, Any]:
    """
    Safely convert Plan Pydantic instance or dictionary into a clean dict.
    """
    if plan_obj is None:
        return {}
    if hasattr(plan_obj, "model_dump"):
        return plan_obj.model_dump()
    if isinstance(plan_obj, dict):
        return plan_obj
    try:
        return json.loads(json.dumps(plan_obj, default=str))
    except Exception:
        return {}


def parse_task(task_obj: Any) -> Dict[str, Any]:
    """
    Safely extract Task attributes whether it is a Pydantic object or dict.
    """
    if isinstance(task_obj, dict):
        return {
            "id": task_obj.get("id", ""),
            "title": task_obj.get("title", "Untitled Section"),
            "goal": task_obj.get("goal", ""),
            "bullets": task_obj.get("bullets", []),
            "target_words": task_obj.get("target_words", 0),
            "tags": task_obj.get("tags", []),
            "requires_research": task_obj.get("requires_research", False),
            "requires_citations": task_obj.get("requires_citations", False),
            "requires_code": task_obj.get("requires_code", False),
            "section_type": task_obj.get("section_type", "core"),
        }
    return {
        "id": getattr(task_obj, "id", ""),
        "title": getattr(task_obj, "title", "Untitled Section"),
        "goal": getattr(task_obj, "goal", ""),
        "bullets": getattr(task_obj, "bullets", []),
        "target_words": getattr(task_obj, "target_words", 0),
        "tags": getattr(task_obj, "tags", []),
        "requires_research": getattr(task_obj, "requires_research", False),
        "requires_citations": getattr(task_obj, "requires_citations", False),
        "requires_code": getattr(task_obj, "requires_code", False),
        "section_type": getattr(task_obj, "section_type", "core"),
    }


def get_section_badge(sec_type: str) -> Tuple[str, str]:
    """Return label and styling color for section types."""
    badges = {
        "intro": ("🚀 Introduction", "#2563eb"),
        "core": ("⚙️ Core Concept", "#7c3aed"),
        "examples": ("💻 Code & Examples", "#059669"),
        "checklist": ("📋 Checklist", "#d97706"),
        "common_mistakes": ("⚠️ Common Mistakes", "#dc2626"),
        "conclusion": ("🏁 Conclusion", "#4b5563"),
    }
    return badges.get(sec_type, (f"📌 {sec_type.title()}", "#6b7280"))


def get_mode_badge(mode: str) -> Tuple[str, str]:
    """Return label and color for router modes."""
    modes = {
        "closed_book": ("📘 Closed Book (Evergreen)", "#2563eb"),
        "hybrid": ("📗 Hybrid (Concepts + Fresh Tools)", "#059669"),
        "open_book": ("📙 Open Book (Live Web News/Trends)", "#d97706"),
    }
    return modes.get(mode, (f"📖 {mode}", "#6b7280"))


# ============================================================
# Streamlit Configuration & Custom Styling
# ============================================================

st.set_page_config(
    page_title="Technical Blog Generator | LangGraph Multi-Agent",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
        background: linear-gradient(90deg, #4f46e5, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: rgba(99, 102, 241, 0.06);
        border: 1px solid rgba(99, 102, 241, 0.15);
        border-radius: 10px;
        padding: 14px 18px;
        text-align: center;
    }
    .metric-val {
        font-size: 1.5rem;
        font-weight: 700;
        color: #4338ca;
    }
    .metric-lbl {
        font-size: 0.8rem;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .section-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        color: white;
        margin-right: 6px;
        margin-bottom: 6px;
    }
    .feature-pill {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        background-color: #f1f5f9;
        color: #334155;
        border: 1px solid #cbd5e1;
        margin-right: 4px;
        margin-bottom: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# Session State Initialization
# ============================================================

if "last_out" not in st.session_state:
    st.session_state["last_out"] = None

if "logs" not in st.session_state:
    st.session_state["logs"] = []

if "active_topic" not in st.session_state:
    st.session_state["active_topic"] = ""


def log_event(message: str):
    """Add timestamped event to logs."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state["logs"].append(f"[{timestamp}] {message}")


# ============================================================
# Header
# ============================================================

st.markdown('<div class="main-title">✍️ AI Technical Blog Generator</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Advanced Multi-Agent System powered by <b>LangGraph</b>: Router ➔ Tavily Web Research ➔ Orchestrator ➔ Parallel Workers ➔ Reducer Subgraph (with Image Generation)</div>',
    unsafe_allow_html=True
)


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.header("⚡ Generate New Blog")

    sample_topics = [
        "Self Attention in Transformer Architecture",
        "State of Multimodal LLMs in 2026",
        "Building Production RAG with Hybrid Search & Reranking",
        "Zero-Downtime Database Migrations in Kubernetes",
        "High-Throughput Async Microservices with Python & FastAPI",
        "Observability with OpenTelemetry, Prometheus & Grafana"
    ]

    selected_sample = st.selectbox(
        "💡 Quick Topic Ideas",
        options=["Select a sample idea..."] + sample_topics,
        index=0
    )

    default_prompt = ""
    if selected_sample and selected_sample != "Select a sample idea...":
        default_prompt = selected_sample

    topic_input = st.text_area(
        "Blog Topic / Prompt",
        value=default_prompt or st.session_state.get("active_topic", ""),
        height=120,
        placeholder="Enter technical topic (e.g. Self Attention in Transformer Architecture)..."
    )

    generate_clicked = st.button(
        "🚀 Generate Blog Post",
        type="primary",
        use_container_width=True
    )

    # --------------------------------------------------------
    # Past Blogs Explorer
    # --------------------------------------------------------
    st.divider()
    st.subheader("📚 Saved Blogs Library")

    past_blogs = list_past_blogs()

    if not past_blogs:
        st.caption("No generated markdown files found in the workspace.")
    else:
        st.caption(f"{len(past_blogs)} blog(s) available on disk")

        blog_labels = [
            f"{b['title'][:38]}... ({b['words']}w · {b['mtime']})"
            for b in past_blogs
        ]

        selected_idx = st.selectbox(
            "Select past blog",
            options=range(len(past_blogs)),
            format_func=lambda i: blog_labels[i],
            label_visibility="collapsed"
        )

        col_load, col_del = st.columns([3, 1])

        with col_load:
            if st.button("📂 Load Selected", use_container_width=True):
                selected_item = past_blogs[selected_idx]
                md_content = read_md_file(selected_item["path"])
                sections_list = extract_sections_from_md(md_content)
                extracted_specs = extract_images_from_md(md_content)

                st.session_state["last_out"] = {
                    "topic": selected_item["title"],
                    "mode": "closed_book",
                    "needs_research": False,
                    "queries": [],
                    "evidence": [],
                    "plan": None,
                    "sections": sections_list,
                    "image_specs": extracted_specs,
                    "final": md_content,
                    "file_path": selected_item["path"].name,
                }
                log_event(f"Loaded saved blog from '{selected_item['name']}'")
                st.rerun()

        with col_del:
            if st.button("🗑️", help="Delete this file from disk"):
                try:
                    past_blogs[selected_idx]["path"].unlink()
                    log_event(f"Deleted file {past_blogs[selected_idx]['name']}")
                    st.success("Deleted!")
                    time.sleep(0.5)
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to delete: {e}")


# ============================================================
# Main Execution: Blog Generation
# ============================================================

if generate_clicked:
    clean_topic = topic_input.strip()
    if not clean_topic:
        st.warning("⚠️ Please provide a blog topic before clicking generate.")
        st.stop()

    st.session_state["active_topic"] = clean_topic
    inputs: Dict[str, Any] = {
        "topic": clean_topic,
        "mode": "",
        "needs_research": False,
        "queries": [],
        "evidence": [],
        "plan": None,
        "as_of": datetime.now().strftime("%Y-%m-%d"),
        "recency_days": 7,
        "sections": [],
        "merged_md": "",
        "md_with_placeholders": "",
        "image_specs": [],
        "final": "",
    }

    # Tracking state
    current_state: Dict[str, Any] = {
        "topic": clean_topic,
        "mode": "closed_book",
        "needs_research": False,
        "queries": [],
        "evidence": [],
        "plan": None,
        "sections": [],
        "merged_md": "",
        "md_with_placeholders": "",
        "image_specs": [],
        "final": "",
        "file_path": "",
    }

    status_box = st.status("🚀 Starting multi-agent blog generation workflow...", expanded=True)
    progress_bar = st.progress(0, text="Initializing workflow graph...")

    log_event(f"Workflow triggered for topic: '{clean_topic}'")

    planned_tasks: List[Dict[str, Any]] = []
    total_sections_expected = 5

    try:
        status_box.write("🧭 **Router Node**: Evaluating topic volatility & web research requirements...")
        progress_bar.progress(10, text="🧭 Router evaluating topic requirements...")

        for step in app.stream(inputs, stream_mode="updates"):
            if not isinstance(step, dict):
                continue

            for node_name, node_output in step.items():
                if not isinstance(node_output, dict):
                    continue

                # 1. Router Step
                if node_name == "router":
                    current_state["mode"] = node_output.get("mode", "closed_book")
                    current_state["needs_research"] = node_output.get("needs_research", False)
                    current_state["queries"] = node_output.get("queries", [])

                    mode_label = current_state["mode"].upper().replace("_", " ")
                    status_box.write(
                        f"🧭 **Router Decision**: Mode = **`{mode_label}`** "
                        f"(Needs web research: `{'YES' if current_state['needs_research'] else 'NO'}`). "
                        f"{len(current_state['queries'])} search queries generated."
                    )
                    log_event(f"Router completed: mode={current_state['mode']}, queries={len(current_state['queries'])}")
                    progress_bar.progress(20, text="🧭 Routing step completed...")

                # 2. Research Step (Tavily Search)
                elif node_name == "research":
                    evidence_list = node_output.get("evidence", [])
                    current_state["evidence"] = [e.model_dump() if hasattr(e, "model_dump") else e for e in evidence_list]
                    status_box.write(
                        f"🌐 **Research Node**: Gathered **{len(evidence_list)}** authoritative sources via Tavily search."
                    )
                    log_event(f"Tavily research retrieved {len(evidence_list)} evidence items.")
                    progress_bar.progress(35, text="🌐 Tavily research completed & deduplicated...")

                # 3. Orchestrator Step
                elif node_name == "orchestrator":
                    plan_obj = node_output.get("plan")
                    current_state["plan"] = plan_obj
                    plan_dict = parse_plan(plan_obj)
                    tasks_raw = plan_dict.get("tasks", [])
                    planned_tasks = [parse_task(t) for t in tasks_raw]
                    total_sections_expected = max(len(planned_tasks), 1)

                    blog_title = plan_dict.get("blog_title", clean_topic)
                    kind_label = plan_dict.get("blog_kind", "explainer").title()
                    status_box.write(
                        f"📋 **Plan Formulated:** *{blog_title}* "
                        f"({kind_label} · {len(planned_tasks)} sections, audience: {plan_dict.get('audience', 'Developers')})"
                    )
                    log_event(f"Orchestrator planned {len(planned_tasks)} sections for '{blog_title}'")
                    progress_bar.progress(
                        45,
                        text=f"✍️ Dispatching {len(planned_tasks)} sections to parallel worker agents..."
                    )

                # 4. Worker Step
                elif node_name == "worker":
                    new_sections = node_output.get("sections", [])
                    if isinstance(new_sections, list):
                        current_state["sections"].extend(new_sections)
                    else:
                        current_state["sections"].append(new_sections)

                    count = len(current_state["sections"])
                    sec_title = f"Section {count}"
                    if count <= len(planned_tasks):
                        sec_title = planned_tasks[count - 1].get("title", sec_title)

                    status_box.write(f"✍️ **Section completed:** {sec_title} ({count}/{total_sections_expected})")
                    log_event(f"Worker generated section {count}/{total_sections_expected}: {sec_title}")

                    worker_progress = min(45 + int((count / total_sections_expected) * 40), 85)
                    progress_bar.progress(
                        worker_progress,
                        text=f"✍️ Generating sections ({count}/{total_sections_expected} complete)..."
                    )

                # 5. Reducer Step (Subgraph)
                elif node_name == "reducer":
                    if "final" in node_output:
                        current_state["final"] = node_output.get("final", "")
                    if "image_specs" in node_output:
                        current_state["image_specs"] = node_output.get("image_specs", [])
                    if "md_with_placeholders" in node_output:
                        current_state["md_with_placeholders"] = node_output.get("md_with_placeholders", "")
                    if "merged_md" in node_output:
                        current_state["merged_md"] = node_output.get("merged_md", "")

                    status_box.write("🎨 **Reducer Subgraph**: Merged markdown content & generated technical visuals.")
                    log_event("Reducer subgraph finished execution.")
                    progress_bar.progress(95, text="📝 Finalizing blog post...")

        # If final is empty, fallback to normalizing sections
        if not current_state.get("final"):
            plan_dict = parse_plan(current_state.get("plan"))
            b_title = plan_dict.get("blog_title", clean_topic)
            norm_secs = normalize_sections(current_state.get("sections", []))
            final_text = f"# {b_title}\n\n" + "\n\n".join(norm_secs).strip() + "\n"
            current_state["final"] = final_text

        # Determine file name matching backend
        plan_dict = parse_plan(current_state.get("plan"))
        b_title = plan_dict.get("blog_title", clean_topic)
        expected_filename = safe_slug(b_title) + ".md"
        current_state["file_path"] = expected_filename

        # Store in session state
        st.session_state["last_out"] = current_state

        progress_bar.progress(100, text="✅ Blog post generated successfully!")
        status_box.update(
            label="🎉 Blog post generated and saved successfully!",
            state="complete",
            expanded=False
        )
        log_event(f"Blog generation complete! Saved to {expected_filename}")
        time.sleep(0.5)
        st.rerun()

    except Exception as exc:
        log_event(f"Streaming failed: {exc}. Attempting direct backend run() fallback...")
        status_box.write("⚠️ Streaming update interrupted. Attempting direct fallback via `app.invoke()`...")

        try:
            out = app.invoke(inputs)
            plan_dict = parse_plan(out.get("plan"))
            b_title = plan_dict.get("blog_title", clean_topic)
            expected_filename = safe_slug(b_title) + ".md"

            st.session_state["last_out"] = {
                "topic": clean_topic,
                "mode": out.get("mode", "closed_book"),
                "needs_research": out.get("needs_research", False),
                "queries": out.get("queries", []),
                "evidence": out.get("evidence", []),
                "plan": out.get("plan"),
                "sections": out.get("sections", []),
                "merged_md": out.get("merged_md", ""),
                "md_with_placeholders": out.get("md_with_placeholders", ""),
                "image_specs": out.get("image_specs", []),
                "final": out.get("final", ""),
                "file_path": expected_filename,
            }
            progress_bar.progress(100, text="✅ Blog generated via fallback!")
            status_box.update(label="✅ Blog generation completed!", state="complete", expanded=False)
            log_event("Direct invoke fallback succeeded!")
            time.sleep(0.5)
            st.rerun()

        except Exception as invoke_err:
            status_box.update(label="❌ Generation failed", state="error", expanded=True)
            log_event(f"Error during execution: {invoke_err}")
            st.error(f"**Generation Error:** {invoke_err}")
            st.exception(invoke_err)


# ============================================================
# Blog Output Rendering
# ============================================================

out = st.session_state.get("last_out")

if out:
    plan_obj = out.get("plan")
    plan_dict = parse_plan(plan_obj)
    raw_sections = out.get("sections") or []
    final_md = out.get("final") or ""
    mode_str = out.get("mode", "closed_book")
    evidence_list = out.get("evidence") or []
    queries_list = out.get("queries") or []
    image_specs = out.get("image_specs") or []

    # Normalized clean section strings
    sections = normalize_sections(raw_sections)
    if not sections and final_md:
        sections = extract_sections_from_md(final_md)

    # Extract or infer title
    if plan_dict and plan_dict.get("blog_title"):
        blog_title = plan_dict["blog_title"]
    else:
        blog_title = extract_title_from_md(final_md, fallback=out.get("topic", "Technical Blog"))

    # Compute key stats
    word_count = len(final_md.split())
    reading_time = max(round(word_count / 200), 1)
    char_count = len(final_md)
    num_sections = len(sections)

    # --------------------------------------------------------
    # Stats Bar
    # --------------------------------------------------------
    col_m1, col_m2, col_m3, col_m4, col_m5, col_m6 = st.columns(6)
    with col_m1:
        st.markdown(
            f'<div class="metric-card"><div class="metric-val">{word_count:,}</div><div class="metric-lbl">Total Words</div></div>',
            unsafe_allow_html=True
        )
    with col_m2:
        st.markdown(
            f'<div class="metric-card"><div class="metric-val">~{reading_time} min</div><div class="metric-lbl">Read Time</div></div>',
            unsafe_allow_html=True
        )
    with col_m3:
        st.markdown(
            f'<div class="metric-card"><div class="metric-val">{num_sections}</div><div class="metric-lbl">Sections</div></div>',
            unsafe_allow_html=True
        )
    with col_m4:
        mode_label, _ = get_mode_badge(mode_str)
        st.markdown(
            f'<div class="metric-card"><div class="metric-val" style="font-size:1.1rem; padding-top:4px;">{mode_str.replace("_"," ").title()}</div><div class="metric-lbl">Mode</div></div>',
            unsafe_allow_html=True
        )
    with col_m5:
        st.markdown(
            f'<div class="metric-card"><div class="metric-val">{len(evidence_list)}</div><div class="metric-lbl">Web Sources</div></div>',
            unsafe_allow_html=True
        )
    with col_m6:
        st.markdown(
            f'<div class="metric-card"><div class="metric-val">{len(image_specs)}</div><div class="metric-lbl">Visuals</div></div>',
            unsafe_allow_html=True
        )

    st.write("")

    # --------------------------------------------------------
    # Tabs Interface
    # --------------------------------------------------------
    tab_preview, tab_plan, tab_research, tab_images, tab_sections, tab_source, tab_export, tab_logs = st.tabs([
        "📖 Read Article",
        "🧩 Plan & Outline",
        "🌐 Research & Sources",
        "🎨 Diagrams & Visuals",
        "✍️ Sections View",
        "💻 Raw Markdown",
        "⬇️ Export & Details",
        "🧾 Execution Logs"
    ])

    # ========================================================
    # TAB 1: READ ARTICLE (Rendered Markdown)
    # ========================================================
    with tab_preview:
        if final_md:
            st.markdown(resolve_images_for_display(final_md), unsafe_allow_html=True)
        else:
            st.info("No content available to preview.")

    # ========================================================
    # TAB 2: PLAN & OUTLINE
    # ========================================================
    with tab_plan:
        if plan_dict:
            st.markdown(f"### 📋 {plan_dict.get('blog_title', blog_title)}")

            c_aud, c_tone, c_kind = st.columns(3)
            with c_aud:
                st.markdown(f"**🎯 Target Audience:** {plan_dict.get('audience', 'Software Engineers')}")
            with c_tone:
                st.markdown(f"**🎙️ Tone & Style:** {plan_dict.get('tone', 'Technical & Instructive')}")
            with c_kind:
                st.markdown(f"**📑 Blog Kind:** `{plan_dict.get('blog_kind', 'explainer')}`")

            constraints = plan_dict.get("constraints", [])
            if constraints:
                st.markdown("**⚡ Editorial Constraints:**")
                for c in constraints:
                    st.markdown(f"- *{c}*")

            st.divider()

            tasks = plan_dict.get("tasks", [])
            st.subheader(f"Detailed Sections Blueprint ({len(tasks)} Sections)")

            for t_raw in tasks:
                t = parse_task(t_raw)
                sec_type = t.get("section_type", "core")
                badge_label, badge_color = get_section_badge(sec_type)

                with st.expander(f"Section {t['id']}: {t['title']} ({t['target_words']} words)", expanded=True):
                    st.markdown(
                        f'<span class="section-badge" style="background-color: {badge_color};">{badge_label}</span>',
                        unsafe_allow_html=True
                    )

                    # Feature tags
                    tags_html = ""
                    if t.get("requires_code"):
                        tags_html += '<span class="feature-pill">💻 Minimal Code (MWE)</span>'
                    if t.get("requires_citations"):
                        tags_html += '<span class="feature-pill">🔗 Citations Required</span>'
                    if t.get("requires_research"):
                        tags_html += '<span class="feature-pill">🌐 Fresh Research</span>'
                    for tag in t.get("tags", []):
                        tags_html += f'<span class="feature-pill">🏷️ {tag}</span>'

                    if tags_html:
                        st.markdown(tags_html, unsafe_allow_html=True)
                        st.write("")

                    st.markdown(f"**🎯 Section Goal:** {t['goal']}")

                    bullets = t.get("bullets", [])
                    if bullets:
                        st.markdown("**Key Subpoints to Cover:**")
                        for b in bullets:
                            st.markdown(f"- {b}")
        else:
            st.info(
                "💡 This blog was loaded from a saved Markdown file. The structured plan metadata is created during a live generation session."
            )
            if sections:
                st.markdown("### 📑 Detected Outline")
                for i, sec in enumerate(sections, 1):
                    first_line = sec.splitlines()[0].replace("##", "").strip() if sec.splitlines() else f"Section {i}"
                    st.markdown(f"{i}. **{first_line}**")

    # ========================================================
    # TAB 3: RESEARCH & SOURCES (Tavily Evidence)
    # ========================================================
    with tab_research:
        st.subheader("🌐 Web Research & Grounding Evidence")
        mode_label, mode_color = get_mode_badge(mode_str)
        st.markdown(
            f'**Routing Mode:** <span class="section-badge" style="background-color:{mode_color};">{mode_label}</span>',
            unsafe_allow_html=True
        )

        if queries_list:
            st.markdown("#### 🔍 Queries Generated by Router Agent:")
            for q in queries_list:
                st.markdown(f"- `{q}`")
            st.write("")

        if evidence_list:
            st.markdown(f"#### 📚 Verified Evidence Items ({len(evidence_list)} Sources):")
            for idx, e in enumerate(evidence_list, 1):
                if isinstance(e, dict):
                    e_title = e.get("title", "Source")
                    e_url = e.get("url", "#")
                    e_date = e.get("published_at") or "Date: N/A"
                    e_snippet = e.get("snippet", "")
                else:
                    e_title = getattr(e, "title", "Source")
                    e_url = getattr(e, "url", "#")
                    e_date = getattr(e, "published_at", None) or "Date: N/A"
                    e_snippet = getattr(e, "snippet", "")

                with st.expander(f"{idx}. {e_title} ({e_date})", expanded=(idx <= 3)):
                    st.markdown(f"**🔗 Link:** [{e_url}]({e_url})")
                    if e_snippet:
                        st.markdown(f"**Summary Snippet:**\n> {e_snippet}")
        else:
            if mode_str == "closed_book":
                st.success("📘 **Closed Book Topic:** Evaluated as an evergreen fundamental topic — relies on pure verified knowledge without external web retrieval.")
            else:
                st.info("No external web sources were required for this run.")

    # ========================================================
    # TAB 4: DIAGRAMS & VISUALS
    # ========================================================
    with tab_images:
        st.subheader("🎨 Technical Diagrams & Visual Visualizations")
        if image_specs:
            st.caption(f"Found {len(image_specs)} technical visual planned by the Reducer Agent:")
            for idx, spec in enumerate(image_specs, 1):
                fname = spec.get("filename", f"image_{idx}.png")
                img_path = Path("images") / fname
                col_i1, col_i2 = st.columns([1, 1])

                with col_i1:
                    st.markdown(f"### Visual {idx}: `{fname}`")
                    st.markdown(f"**Placeholder:** `{spec.get('placeholder')}`")
                    st.markdown(f"**Alt:** *{spec.get('alt')}*")
                    st.markdown(f"**Caption:** {spec.get('caption')}")
                    st.markdown(f"**Prompt:**\n```\n{spec.get('prompt')}\n```")

                with col_i2:
                    if img_path.exists():
                        st.image(str(img_path), caption=spec.get("caption"), use_container_width=True)
                    else:
                        st.info(f"Image placeholder placed in Markdown (`images/{fname}`).")
                st.divider()
        else:
            st.info("No diagrams or images were requested for this blog post.")

    # ========================================================
    # TAB 5: SECTIONS VIEW
    # ========================================================
    with tab_sections:
        if sections:
            st.caption(f"Explore and copy each of the {len(sections)} sections individually.")
            for i, sec in enumerate(sections, 1):
                first_line = sec.splitlines()[0] if sec.splitlines() else f"Section {i}"
                clean_sec_title = first_line.replace("##", "").strip()
                sec_words = len(sec.split())

                with st.expander(f"Section {i}: {clean_sec_title}  ·  ({sec_words} words)", expanded=(i == 1)):
                    st.markdown(resolve_images_for_display(sec), unsafe_allow_html=True)
        else:
            st.info("No sections available.")

    # ========================================================
    # TAB 6: RAW MARKDOWN SOURCE
    # ========================================================
    with tab_source:
        st.caption("View and copy the clean raw Markdown text:")
        st.code(final_md, language="markdown")

    # ========================================================
    # TAB 7: EXPORT & DETAILS
    # ========================================================
    with tab_export:
        st.subheader("💾 Export & File Information")

        file_slug = safe_slug(blog_title)
        md_filename = f"{file_slug}.md"
        saved_on_disk = Path(md_filename).exists()

        col_d1, col_d2 = st.columns([1, 1])
        with col_d1:
            st.download_button(
                label=f"⬇️ Download '{md_filename}'",
                data=final_md.encode("utf-8"),
                file_name=md_filename,
                mime="text/markdown",
                type="primary",
                use_container_width=True
            )

        with col_d2:
            if saved_on_disk:
                st.success(f"✅ Saved on disk: `{md_filename}`")
            else:
                st.info(f"📁 Target filename: `{md_filename}`")

        st.divider()
        st.markdown("#### 📊 Blog Statistics")
        st.markdown(f"- **Blog Title:** {blog_title}")
        st.markdown(f"- **Routing Mode:** {mode_str.upper().replace('_', ' ')}")
        st.markdown(f"- **Total Words:** {word_count:,}")
        st.markdown(f"- **Estimated Reading Time:** ~{reading_time} minutes")
        st.markdown(f"- **Number of Sections:** {len(sections)}")
        st.markdown(f"- **Web Sources:** {len(evidence_list)}")
        st.markdown(f"- **Diagrams & Visuals:** {len(image_specs)}")

    # ========================================================
    # TAB 8: EXECUTION LOGS
    # ========================================================
    with tab_logs:
        st.subheader("🧾 Workflow Execution Logs")
        logs = st.session_state.get("logs", [])
        if logs:
            st.text_area(
                "Event Log Trace",
                value="\n".join(logs[-150:]),
                height=450,
                disabled=True
            )
            if st.button("Clear Logs"):
                st.session_state["logs"] = []
                st.rerun()
        else:
            st.info("No logs recorded yet.")

else:
    # ========================================================
    # Empty State / Landing View
    # ========================================================
    st.info("👈 Enter a topic in the sidebar and click **'Generate Blog Post'** to start.")

    col_h1, col_h2 = st.columns(2)

    with col_h1:
        st.markdown(
            """
            ### 🤖 How the Multi-Agent System Operates

            1. **🧭 Router Node**
               - Analyzes topic volatility (closed_book, hybrid, open_book).
               - Decides if web research is required and formulates targeted search queries.

            2. **🌐 Tavily Research Node**
               - Executes search queries and synthesizes authoritative evidence with source URLs and dates.

            3. **📋 Orchestrator Node**
               - Formulates a 5–9 section blueprint with goals, actionable subpoints, and technical constraints.

            4. **🔀 LangGraph Fanout & Workers**
               - Dispatches sections in parallel. Enforces code MWEs, citations, and developer tone.

            5. **🎨 Reducer Subgraph with Visuals**
               - Merges sections, decides technical diagrams, and places generated visuals or flowcharts.
            """
        )

    with col_h2:
        st.markdown(
            """
            ### 💡 Supported Modes & Capabilities

            - **📘 Closed Book:** Evergreen concepts (e.g., *Self Attention*, *Binary Search Trees*).
            - **📗 Hybrid:** Core concepts requiring modern stack examples (e.g., *RAG with LangChain & Qdrant in 2026*).
            - **📙 Open Book:** Fast-moving news and rankings (e.g., *Latest AI Models released in 2026*).
            - **💻 Code Sketches:** Automatically detects sections that need runnable Python/PyTorch code.
            - **🖼️ Technical Visuals:** Generates diagrams and flowcharts automatically into the `images/` directory.
            """
        )