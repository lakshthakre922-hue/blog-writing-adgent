import base64
import io
import operator
import os
import re
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path
from typing import TypedDict, List, Optional, Literal, Annotated

from PIL import Image, ImageDraw, ImageFont
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults

# -----------------------------
# 0) Environment & LLM Setup
# -----------------------------
load_dotenv()

# Select LLM dynamically based on available API keys
openai_key = os.getenv("OPENAI_API_KEY")
groq_key = os.getenv("GROQ_API_KEY")

if openai_key:
    llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"))
elif groq_key:
    llm = ChatGroq(model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"), max_retries=10, max_tokens=8000)
else:
    # Default fallback
    llm = ChatGroq(model="openai/gpt-oss-120b", max_retries=10, max_tokens=8000)


# -----------------------------
# 1) Schemas
# -----------------------------
class Task(BaseModel):
    id: int
    title: str

    goal: str = Field(
        ...,
        description="One sentence describing what the reader should be able to do/understand after this section.",
    )
    bullets: List[str] = Field(
        ...,
        min_length=3,
        max_length=6,
        description="3–6 concrete, non-overlapping subpoints to cover in this section.",
    )
    target_words: int = Field(..., description="Target word count for this section (300–750) for comprehensive, high-value depth.")

    tags: List[str] = Field(default_factory=list)
    requires_research: bool = False
    requires_citations: bool = False
    requires_code: bool = False
    section_type: Optional[str] = "core"


class Plan(BaseModel):
    blog_title: str
    audience: str
    tone: str
    blog_kind: Literal["explainer", "tutorial", "news_roundup", "comparison", "system_design"] = "explainer"
    constraints: List[str] = Field(default_factory=list)
    tasks: List[Task]


class EvidenceItem(BaseModel):
    title: str
    url: str
    published_at: Optional[str] = None  # keep if Tavily provides; DO NOT rely on it
    snippet: Optional[str] = None
    source: Optional[str] = None


class RouterDecision(BaseModel):
    needs_research: bool
    mode: Literal["closed_book", "hybrid", "open_book"]
    queries: List[str] = Field(default_factory=list)


class EvidencePack(BaseModel):
    evidence: List[EvidenceItem] = Field(default_factory=list)


class ImageSpec(BaseModel):
    placeholder: str = Field(default="[[IMAGE]]", description="e.g. [[IMAGE_1]]")
    section_title: Optional[str] = Field(default="", description="The section title where this diagram belongs.")
    filename: str = Field(..., description="Save under images/, e.g. qkv_flow.png")
    alt: str = Field(..., description="Short descriptive alt text for accessibility.")
    caption: str = Field(..., description="Informative figure caption.")
    prompt: str = Field(..., description="Detailed description of the technical diagram components, layout, and connections.")
    mermaid_code: Optional[str] = Field(
        default=None,
        description="Clean, valid Mermaid diagram syntax (graph TD ... or sequenceDiagram ...) accurately depicting the technical flow.",
    )
    size: Literal["1024x1024", "1024x1536", "1536x1024"] = "1024x1024"
    quality: Literal["low", "medium", "high"] = "medium"


class GlobalImagePlan(BaseModel):
    images: List[ImageSpec] = Field(default_factory=list, description="List of 2 to 3 high-impact technical diagrams.")
    md_with_placeholders: Optional[str] = Field(default="", description="Optional markdown text with placeholders.")


class State(TypedDict):
    topic: str

    # routing / research
    mode: str
    needs_research: bool
    queries: List[str]
    evidence: List[EvidenceItem]
    plan: Optional[Plan]
    as_of: Optional[str]
    recency_days: Optional[int]

    # workers
    sections: Annotated[List[tuple[int, str]], operator.add]  # (task_id, section_md)

    # reducer/image
    merged_md: str
    md_with_placeholders: str
    image_specs: List[dict]

    final: str


# -----------------------------
# 2) Router (decide upfront)
# -----------------------------
ROUTER_SYSTEM = """You are a routing module for a technical blog planner.

Decide whether web research is needed BEFORE planning.

Modes:
- closed_book (needs_research=false):
  Evergreen topics where correctness does not depend on recent facts (concepts, fundamentals).
- hybrid (needs_research=true):
  Mostly evergreen but needs up-to-date examples/tools/models to be useful.
- open_book (needs_research=true):
  Mostly volatile: weekly roundups, "this week", "latest", rankings, pricing, policy/regulation.

If needs_research=true:
- Output 3–10 high-signal queries.
- Queries should be scoped and specific (avoid generic queries like just "AI" or "LLM").
- If user asked for "last week/this week/latest", reflect that constraint IN THE QUERIES.
"""

def router_node(state: State) -> dict:
    topic = state["topic"]
    decider = llm.with_structured_output(RouterDecision)
    decision = decider.invoke(
        [
            SystemMessage(content=ROUTER_SYSTEM),
            HumanMessage(content=f"Topic: {topic}"),
        ]
    )

    return {
        "needs_research": decision.needs_research,
        "mode": decision.mode,
        "queries": decision.queries,
    }


def route_next(state: State) -> str:
    return "research" if state.get("needs_research") else "orchestrator"


# -----------------------------
# 3) Research (Tavily)
# -----------------------------
def _tavily_search(query: str, max_results: int = 5) -> List[dict]:
    try:
        tool = TavilySearchResults(max_results=max_results)
        results = tool.invoke({"query": query})

        normalized: List[dict] = []
        for r in results or []:
            if isinstance(r, dict):
                normalized.append(
                    {
                        "title": r.get("title") or "",
                        "url": r.get("url") or "",
                        "snippet": r.get("content") or r.get("snippet") or "",
                        "published_at": r.get("published_date") or r.get("published_at"),
                        "source": r.get("source"),
                    }
                )
        return normalized
    except Exception as e:
        print(f"Tavily search error for '{query}': {e}")
        return []


RESEARCH_SYSTEM = """You are a research synthesizer for technical writing.

Given raw web search results, produce a deduplicated list of EvidenceItem objects.

Rules:
- Only include items with a non-empty url.
- Prefer relevant + authoritative sources (company blogs, docs, reputable outlets).
- If a published date is explicitly present in the result payload, keep it as YYYY-MM-DD.
  If missing or unclear, set published_at=null. Do NOT guess.
- Keep snippets short.
- Deduplicate by URL.
"""

def research_node(state: State) -> dict:
    queries = state.get("queries", []) or []
    max_results = 4

    raw_results: List[dict] = []
    for q in queries[:4]:
        raw_results.extend(_tavily_search(q, max_results=max_results))

    if not raw_results:
        return {"evidence": []}

    # Truncate snippets for safety with token limits
    truncated_results = []
    for r in raw_results:
        truncated_results.append({
            "title": str(r.get("title", ""))[:120],
            "url": r.get("url", ""),
            "snippet": str(r.get("snippet", ""))[:300],
            "published_at": r.get("published_at"),
            "source": r.get("source"),
        })

    extractor = llm.with_structured_output(EvidencePack)
    try:
        pack = extractor.invoke(
            [
                SystemMessage(content=RESEARCH_SYSTEM),
                HumanMessage(content=f"Raw results:\n{truncated_results[:10]}"),
            ]
        )
        evidence_items = pack.evidence
    except Exception as e:
        print(f"Research synthesis fallback: {e}")
        evidence_items = [
            EvidenceItem(
                title=r["title"],
                url=r["url"],
                published_at=r.get("published_at"),
                snippet=r["snippet"],
                source=r.get("source"),
            )
            for r in truncated_results if r.get("url")
        ]

    # Deduplicate by URL
    dedup = {}
    for e in evidence_items:
        if e.url:
            dedup[e.url] = e

    return {"evidence": list(dedup.values())}


# -----------------------------
# 4) Orchestrator (Plan)
# -----------------------------
ORCH_SYSTEM = """You are a senior technical writer, architect, and developer advocate.
Your job is to produce a highly actionable, comprehensive outline for an authoritative technical blog post.

Hard requirements:
- Create 5–9 sections (tasks) suitable for the topic and audience.
- Each task must include:
  1) goal (1 sentence)
  2) 3–6 bullets that are concrete, specific, and non-overlapping
  3) target word count (300–750) to ensure thorough, production-ready depth (overall blog 2,000–4,000+ words).

Quality bar:
- Assume the reader is a developer or engineer; use correct terminology.
- Bullets must be actionable: build/compare/measure/verify/debug.
- Ensure the overall plan includes at least 2 of these somewhere:
  * minimal code sketch / MWE (set requires_code=True for that section)
  * edge cases / failure modes / trade-offs
  * performance/cost/latency considerations
  * security/privacy considerations (if relevant)
  * debugging/observability/production tips

Grounding rules:
- Mode closed_book: keep it evergreen; do not depend on evidence.
- Mode hybrid:
  - Use evidence for up-to-date examples (models/tools/releases) in bullets.
  - Mark sections using fresh info as requires_research=True and requires_citations=True.
- Mode open_book:
  - Set blog_kind = "news_roundup".
  - Every section is about summarizing events + implications.
  - DO NOT include tutorial/how-to sections unless user explicitly asked for that.
  - If evidence is empty or insufficient, create a plan that transparently says "insufficient sources"
    and includes only what can be supported.

Output must strictly match the Plan schema.
"""

def orchestrator_node(state: State) -> dict:
    planner = llm.with_structured_output(Plan)

    evidence = state.get("evidence", [])
    mode = state.get("mode", "closed_book")

    evidence_dumps = [e.model_dump() if hasattr(e, "model_dump") else e for e in evidence][:16]

    plan = planner.invoke(
        [
            SystemMessage(content=ORCH_SYSTEM),
            HumanMessage(
                content=(
                    f"Topic: {state['topic']}\n"
                    f"Mode: {mode}\n\n"
                    f"Evidence (ONLY use for fresh claims; may be empty):\n"
                    f"{evidence_dumps}"
                )
            ),
        ]
    )

    return {"plan": plan}


# -----------------------------
# 5) Fanout
# -----------------------------
def fanout(state: State):
    plan = state["plan"]
    tasks = plan.tasks if hasattr(plan, "tasks") else plan.get("tasks", [])
    plan_dict = plan.model_dump() if hasattr(plan, "model_dump") else plan
    evidence_dumps = [e.model_dump() if hasattr(e, "model_dump") else e for e in state.get("evidence", [])]

    return [
        Send(
            "worker",
            {
                "task": task.model_dump() if hasattr(task, "model_dump") else task,
                "topic": state["topic"],
                "mode": state.get("mode", "closed_book"),
                "plan": plan_dict,
                "evidence": evidence_dumps,
            },
        )
        for task in tasks
    ]


# -----------------------------
# 6) Worker (write one section)
# -----------------------------
WORKER_SYSTEM = """You are a senior technical writer and developer advocate.
Write ONE section of a technical blog post in Markdown.

Hard constraints:
- Follow the provided Goal and cover ALL Bullets thoroughly in order (do not skip or merge bullets).
- Aim for high technical depth and completeness. Match or exceed the Target words (typically 350–750 words).
- Output ONLY the section content in Markdown (no blog title H1, no extra commentary).
- Start with a '## <Section Title>' heading.

Writing Guidelines:
- Treat this as an authoritative, production-ready engineering guide.
- Do NOT write shallow 1-2 sentence summaries for bullets. Expand each bullet point into rich paragraphs explaining the "why", "how", architectural mechanisms, edge cases, and real-world trade-offs.
- Avoid being brief or superficial; provide concrete explanations, industry best practices, and actionable insights.

Scope guard:
- If blog_kind == "news_roundup": focus on summarizing events and implications.

Grounding policy:
- If mode == open_book:
  - Do NOT introduce any specific event/company/model/funding/policy claim unless it is supported by provided Evidence URLs.
  - For each event claim, attach a source as a Markdown link: ([Source](URL)).
  - Only use URLs provided in Evidence. If not supported, write: "Not found in provided sources."
- If requires_citations == true:
  - For outside-world claims, cite Evidence URLs the same way.
- Evergreen reasoning is OK without citations unless requires_citations is true.

Code:
- If requires_code == true, include at least one complete, realistic, syntactically correct code snippet relevant to the bullets with inline explanations.

Style:
- Short clear paragraphs, bold lead-ins for readability, tables where comparison is helpful, and code fences for code.
- Avoid fluff/marketing. Be precise, deep, and implementation-oriented.
"""

def worker_node(payload: dict) -> dict:
    raw_task = payload["task"]
    task = Task(**raw_task) if isinstance(raw_task, dict) else raw_task

    raw_plan = payload["plan"]
    plan = Plan(**raw_plan) if isinstance(raw_plan, dict) else raw_plan

    raw_evidence = payload.get("evidence", [])
    evidence = [EvidenceItem(**e) if isinstance(e, dict) else e for e in raw_evidence]

    topic = payload["topic"]
    mode = payload.get("mode", "closed_book")

    bullets_text = "\n- " + "\n- ".join(task.bullets)

    evidence_text = ""
    if evidence:
        evidence_text = "\n".join(
            f"- {e.title} | {e.url} | {e.published_at or 'date:unknown'}".strip()
            for e in evidence[:20]
        )

    section_md = llm.invoke(
        [
            SystemMessage(content=WORKER_SYSTEM),
            HumanMessage(
                content=(
                    f"Blog title: {plan.blog_title}\n"
                    f"Audience: {plan.audience}\n"
                    f"Tone: {plan.tone}\n"
                    f"Blog kind: {plan.blog_kind}\n"
                    f"Constraints: {plan.constraints}\n"
                    f"Topic: {topic}\n"
                    f"Mode: {mode}\n\n"
                    f"Section title: {task.title}\n"
                    f"Goal: {task.goal}\n"
                    f"Target words: {task.target_words}\n"
                    f"Tags: {task.tags}\n"
                    f"requires_research: {task.requires_research}\n"
                    f"requires_citations: {task.requires_citations}\n"
                    f"requires_code: {task.requires_code}\n"
                    f"Bullets:{bullets_text}\n\n"
                    f"Evidence (ONLY use these URLs when citing):\n{evidence_text}\n"
                )
            ),
        ]
    ).content.strip()

    return {"sections": [(task.id, section_md)]}


# ============================================================
# 7) ReducerWithImages (subgraph)
#    merge_content -> decide_images -> generate_and_place_images
# ============================================================
def _sanitize_filename(title: str) -> str:
    """Sanitize title for safe Windows/POSIX file paths."""
    safe = re.sub(r'[\\/*?:"<>|]', "", title).strip().lower().replace(" ", "_")
    safe = re.sub(r"_+", "_", safe).strip("_")
    return safe or "blog_post"


def merge_content(state: State) -> dict:
    plan = state.get("plan")
    if hasattr(plan, "blog_title"):
        blog_title = plan.blog_title
    elif isinstance(plan, dict):
        blog_title = plan.get("blog_title", state.get("topic", "blog_post"))
    else:
        blog_title = state.get("topic", "blog_post")

    raw_sections = state.get("sections", [])
    # Deduplicate by task_id taking the latest generated content
    if raw_sections and isinstance(raw_sections[0], tuple):
        dedup_dict = {}
        for item in raw_sections:
            if isinstance(item, tuple) and len(item) >= 2:
                dedup_dict[item[0]] = item[1]
        ordered_sections = [dedup_dict[k] for k in sorted(dedup_dict.keys())]
    else:
        seen = set()
        ordered_sections = []
        for s in raw_sections:
            s_str = str(s)
            if s_str not in seen:
                seen.add(s_str)
                ordered_sections.append(s_str)

    body = "\n\n".join(ordered_sections).strip()

    # Prepend H1 blog title if the body does not already start with a level-1 heading (# Title)
    if not re.match(r"^#\s+[^\n#]", body):
        merged_md = f"# {blog_title}\n\n{body}\n"
    else:
        merged_md = f"{body}\n"

    return {"merged_md": merged_md}


DECIDE_IMAGES_SYSTEM = """You are an expert technical editor and visual diagram architect.
Select 2–3 high-impact technical diagrams or flowcharts for this blog post.

For each diagram:
1. Identify the section it belongs to (section_title matching one of the available sections).
2. Write a clean filename ending in .png (e.g. k8s_migration_flow.png under images/).
3. Provide alt text and an informative caption.
4. Provide a detailed image prompt describing the technical visual components, layout, and connections.
5. Provide clean, valid Mermaid.js diagram code (mermaid_code) depicting the architecture, flowchart, or state machine (e.g. graph TD\n    A[Client] --> B[Service]\n...). Use simple alphanumeric node IDs and clean text labels. Avoid special characters in node labels.

Rules:
- 2 to 3 images max.
- Always provide valid mermaid_code for technical topics (flowcharts, sequence diagrams, architectures).
- Prefer technical flowcharts, architecture diagrams, and state transitions over generic decorative pictures.
- Return strictly GlobalImagePlan.
"""

def decide_images(state: State) -> dict:
    planner = llm.with_structured_output(GlobalImagePlan)
    merged_md = state.get("merged_md", "")
    plan = state.get("plan")

    if not merged_md:
        return {
            "md_with_placeholders": "",
            "image_specs": [],
        }

    blog_kind = getattr(plan, "blog_kind", "explainer") if plan else "explainer"
    if isinstance(plan, dict):
        blog_kind = plan.get("blog_kind", "explainer")

    # Extract all section headings from merged_md so the LLM has complete context without massive token bloat
    headings = re.findall(r"^##\s+(.+)$", merged_md, re.MULTILINE)
    outline_context = "\n".join(f"- {h.strip()}" for h in headings) if headings else "Available sections"

    try:
        image_plan = planner.invoke(
            [
                SystemMessage(content=DECIDE_IMAGES_SYSTEM),
                HumanMessage(
                    content=(
                        f"Blog Title: {getattr(plan, 'blog_title', state.get('topic', ''))}\n"
                        f"Blog Kind: {blog_kind}\n"
                        f"Topic: {state.get('topic', '')}\n\n"
                        f"Article Sections:\n{outline_context}\n\n"
                        "Propose 2-3 technical diagrams for these sections."
                    )
                ),
            ]
        )
        images = [img.model_dump() for img in image_plan.images]
    except Exception as e:
        print(f"Warning: decide_images error ({e}). Proceeding without images.")
        images = []

    # Insert placeholders deterministically into merged_md without losing ANY content
    md_with_placeholders = merged_md
    for idx, spec in enumerate(images, 1):
        placeholder = f"[[IMAGE_{idx}]]"
        spec["placeholder"] = placeholder
        target_sec = spec.get("section_title", "").strip()

        inserted = False
        if target_sec:
            pattern = re.compile(rf"(^##\s+.*?{re.escape(target_sec[:20])}.*?$)", re.MULTILINE | re.IGNORECASE)
            match = pattern.search(md_with_placeholders)
            if match:
                pos = match.end()
                md_with_placeholders = md_with_placeholders[:pos] + f"\n\n{placeholder}\n" + md_with_placeholders[pos:]
                inserted = True

        if not inserted:
            all_headings = list(re.finditer(r"^##\s+.*?$", md_with_placeholders, re.MULTILINE))
            if all_headings:
                target_idx = min(idx - 1, len(all_headings) - 1)
                pos = all_headings[target_idx].end()
                md_with_placeholders = md_with_placeholders[:pos] + f"\n\n{placeholder}\n" + md_with_placeholders[pos:]
            else:
                md_with_placeholders += f"\n\n{placeholder}\n"

    return {
        "md_with_placeholders": md_with_placeholders,
        "image_specs": images,
    }


def _generate_diagram_from_mermaid(mermaid_code: str) -> Optional[bytes]:
    """Render Mermaid syntax to high-res PNG via mermaid.ink."""
    try:
        clean = mermaid_code.strip()
        if clean.startswith("```"):
            clean = re.sub(r"^```(?:mermaid)?\n?", "", clean)
            clean = re.sub(r"\n?```$", "", clean)
        clean = clean.strip()
        if not clean:
            return None
        encoded = base64.b64encode(clean.encode("utf-8")).decode("ascii")
        url = f"https://mermaid.ink/img/{encoded}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        with urllib.request.urlopen(req, timeout=12) as resp:
            data = resp.read()
            if len(data) > 1000:
                return data
    except Exception as e:
        print(f"Mermaid rendering note: {e}")
    return None


def _generate_ai_image_pollinations(prompt: str) -> Optional[bytes]:
    """Attempt Pollinations endpoint with sanitized concise prompt."""
    try:
        clean_words = re.sub(r"[^\w\s-]", "", prompt).split()
        short_prompt = " ".join(clean_words[:10])
        url_prompt = urllib.parse.quote(short_prompt)
        url = f"https://image.pollinations.ai/prompt/{url_prompt}?width=1024&height=768&nologo=true"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = resp.read()
            if len(data) > 1000:
                return data
    except Exception as e:
        print(f"Pollinations note: {e}")
    return None


def _generate_pil_diagram(title: str, caption: str, prompt: str, mermaid_code: Optional[str] = None) -> bytes:
    """
    Offline fallback: generates a sleek, modern technical architecture diagram
    card using PIL so image generation is 100% resilient and never fails.
    """
    w, h = 1200, 675
    img = Image.new("RGB", (w, h), color=(10, 15, 29))
    draw = ImageDraw.Draw(img)

    # Subtle grid lines
    for x in range(0, w, 60):
        draw.line([(x, 0), (x, h)], fill=(18, 25, 43), width=1)
    for y in range(0, h, 60):
        draw.line([(0, y), (w, y)], fill=(18, 25, 43), width=1)

    # Outer border
    draw.rounded_rectangle([15, 15, w - 15, h - 15], radius=16, outline=(56, 189, 248), width=2)
    # Header bar
    draw.rounded_rectangle([35, 30, w - 35, 105], radius=12, fill=(20, 30, 55), outline=(37, 99, 235), width=1)

    try:
        font_badge = ImageFont.truetype("arialbd.ttf", 14)
        font_title = ImageFont.truetype("arialbd.ttf", 22)
        font_box_title = ImageFont.truetype("arialbd.ttf", 17)
        font_box_body = ImageFont.truetype("arial.ttf", 14)
        font_caption = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font_badge = ImageFont.load_default()
        font_title = ImageFont.load_default()
        font_box_title = ImageFont.load_default()
        font_box_body = ImageFont.load_default()
        font_caption = ImageFont.load_default()

    # Header badge & title
    draw.rounded_rectangle([50, 42, 230, 68], radius=6, fill=(37, 99, 235))
    draw.text((60, 47), "SYSTEM ARCHITECTURE", fill=(255, 255, 255), font=font_badge)

    clean_title = re.sub(r"[^\w\s\-,.:]", "", title).strip()
    if len(clean_title) > 65:
        clean_title = clean_title[:62] + "..."
    draw.text((50, 75), clean_title or "Technical Architecture Diagram", fill=(241, 245, 249), font=font_title)

    # Steps / Boxes
    raw_text = f"{title} {caption} {prompt}"
    keywords = re.findall(r"\b[A-Z][a-zA-Z0-9_-]+\b", raw_text)
    unique_kw = []
    for k in keywords:
        if k not in unique_kw and len(k) > 2 and k.lower() not in {"draw", "show", "keep", "with", "from", "each", "step"}:
            unique_kw.append(k)

    steps = [
        ("Step 1: Ingress & Routing", "API Gateway / Traffic Splitter\nCanary Routing & Service Proxies", (37, 99, 235)),
        ("Step 2: Service Layer", "Application Pods (v1 / v2)\nFeature Flags & Dual-Write Pipeline", (14, 165, 233)),
        ("Step 3: Storage & State", "Database Cluster & Replicas\nSchema Migrations & WAL Sync", (16, 185, 129)),
    ]

    if len(unique_kw) >= 3:
        steps[0] = (f"1. {unique_kw[0]} Layer", "Client traffic & ingress filter\nInitial validation & routing", (37, 99, 235))
        steps[1] = (f"2. {unique_kw[1]} Processing", "Core processing & business logic\nDual-read / shadow verification", (14, 165, 233))
        steps[2] = (f"3. {unique_kw[2]} Persistence", "Managed storage & replication\nState synchronization & checkpoint", (16, 185, 129))

    box_w = 330
    box_h = 240
    y_top = 180
    gap = (w - 70 - (3 * box_w)) // 2

    for i, (b_hdr, b_desc, accent_color) in enumerate(steps):
        x1 = 35 + i * (box_w + gap)
        x2 = x1 + box_w
        y1 = y_top
        y2 = y1 + box_h

        draw.rounded_rectangle([x1, y1, x2, y2], radius=14, fill=(18, 27, 48), outline=accent_color, width=2)
        draw.rounded_rectangle([x1, y1, x2, y1 + 45], radius=14, fill=(24, 36, 64))
        draw.rectangle([x1, y1 + 30, x2, y1 + 45], fill=(24, 36, 64))
        draw.text((x1 + 18, y1 + 14), b_hdr, fill=(255, 255, 255), font=font_box_title)

        lines = b_desc.split("\n")
        curr_y = y1 + 65
        for line in lines:
            draw.text((x1 + 18, curr_y), f"• {line}", fill=(203, 213, 225), font=font_box_body)
            curr_y += 30

        draw.rounded_rectangle([x1 + 18, y2 - 40, x1 + 140, y2 - 15], radius=6, fill=(15, 23, 42), outline=accent_color, width=1)
        draw.text((x1 + 30, y2 - 34), "STATUS: ACTIVE", fill=accent_color, font=font_badge)

        if i < 2:
            arrow_x1 = x2 + 5
            arrow_x2 = arrow_x1 + gap - 10
            arrow_y = y1 + (box_h // 2)
            draw.line([(arrow_x1, arrow_y), (arrow_x2, arrow_y)], fill=(56, 189, 248), width=3)
            draw.polygon(
                [(arrow_x2, arrow_y), (arrow_x2 - 12, arrow_y - 7), (arrow_x2 - 12, arrow_y + 7)],
                fill=(56, 189, 248),
            )

    draw.rounded_rectangle([35, h - 90, w - 35, h - 30], radius=10, fill=(15, 23, 42), outline=(30, 41, 59), width=1)
    footer_text = f"Figure: {caption.strip() or title.strip()}"
    if len(footer_text) > 130:
        footer_text = footer_text[:127] + "..."
    draw.text((50, h - 70), footer_text, fill=(148, 163, 184), font=font_caption)
    draw.text((w - 240, h - 70), "Generated by Multi-Agent Engine", fill=(100, 116, 139), font=font_caption)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _generate_image_bytes(spec: dict) -> bytes:
    """
    Multi-tier resilient image generator:
    1. Gemini API (if key available and has quota)
    2. Mermaid diagram via mermaid.ink (crisp vector-like flowchart PNG)
    3. Pollinations endpoint (AI image fallback)
    4. PIL Synthesizer (offline guarantee)
    Never throws an exception; always returns valid image bytes.
    """
    prompt = spec.get("prompt", "")
    mermaid_code = spec.get("mermaid_code")
    title = spec.get("alt", "Technical Architecture")
    caption = spec.get("caption", "")

    # 1. Try Gemini
    api_key = os.environ.get("GOOGLE_API_KEY")
    if api_key:
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=api_key)
            resp = client.models.generate_content(
                model="gemini-3.1-flash-lite-image",
                contents=prompt,
                config=types.GenerateContentConfig(response_modalities=["IMAGE"]),
            )
            parts = getattr(resp, "parts", None)
            if not parts and getattr(resp, "candidates", None):
                parts = resp.candidates[0].content.parts
            if parts:
                for part in parts:
                    inline = getattr(part, "inline_data", None)
                    if inline and getattr(inline, "data", None):
                        return inline.data
        except Exception as e:
            print(f"Gemini image attempt note: {e}")

    # 2. Try Mermaid diagram if provided
    if mermaid_code:
        m_bytes = _generate_diagram_from_mermaid(mermaid_code)
        if m_bytes:
            return m_bytes

    # 3. Try Pollinations
    p_bytes = _generate_ai_image_pollinations(prompt)
    if p_bytes:
        return p_bytes

    # 4. Fallback PIL synthesizer (guaranteed success)
    return _generate_pil_diagram(title=title, caption=caption, prompt=prompt, mermaid_code=mermaid_code)


def generate_and_place_images(state: State) -> dict:
    plan = state.get("plan")
    if hasattr(plan, "blog_title"):
        title = plan.blog_title
    elif isinstance(plan, dict):
        title = plan.get("blog_title", state.get("topic", "blog_post"))
    else:
        title = state.get("topic", "blog_post")

    safe_title = _sanitize_filename(title)
    filename = f"{safe_title}.md"

    md = state.get("md_with_placeholders") or state.get("merged_md", "")
    image_specs = state.get("image_specs", []) or []

    # If no images requested, write merged markdown directly
    if not image_specs:
        Path(filename).write_text(md, encoding="utf-8")
        return {"final": md}

    images_dir = Path("images")
    images_dir.mkdir(exist_ok=True)

    for spec in image_specs:
        placeholder = spec.get("placeholder", "")
        raw_fname = spec.get("filename", "diagram.png")
        img_filename = re.sub(r'[\\/*?:"<>|]', "", raw_fname).strip()
        if not img_filename.endswith((".png", ".jpg", ".jpeg")):
            img_filename += ".png"

        out_path = images_dir / img_filename

        # Generate image if not yet existing on disk
        if not out_path.exists():
            img_bytes = _generate_image_bytes(spec)
            out_path.write_bytes(img_bytes)

        img_md = f"\n\n![{spec.get('alt', 'Architecture Diagram')}](images/{img_filename})\n*{spec.get('caption', '')}*\n\n"
        if placeholder and placeholder in md:
            md = md.replace(placeholder, img_md)
        else:
            md = md + img_md

    # Clean up any leftover unreplaced placeholders
    md = re.sub(r"\[\[IMAGE_\d+\]\]", "", md)

    Path(filename).write_text(md, encoding="utf-8")
    return {"final": md}


# Build reducer subgraph
reducer_graph = StateGraph(State)
reducer_graph.add_node("merge_content", merge_content)
reducer_graph.add_node("decide_images", decide_images)
reducer_graph.add_node("generate_and_place_images", generate_and_place_images)
reducer_graph.add_edge(START, "merge_content")
reducer_graph.add_edge("merge_content", "decide_images")
reducer_graph.add_edge("decide_images", "generate_and_place_images")
reducer_graph.add_edge("generate_and_place_images", END)
reducer_subgraph = reducer_graph.compile()


# -----------------------------
# 8) Build main graph
# -----------------------------
g = StateGraph(State)
g.add_node("router", router_node)
g.add_node("research", research_node)
g.add_node("orchestrator", orchestrator_node)
g.add_node("worker", worker_node)
g.add_node("reducer", reducer_subgraph)

g.add_edge(START, "router")
g.add_conditional_edges("router", route_next, {"research": "research", "orchestrator": "orchestrator"})
g.add_edge("research", "orchestrator")

g.add_conditional_edges("orchestrator", fanout, ["worker"])
g.add_edge("worker", "reducer")
g.add_edge("reducer", END)

app = g.compile()

# Compatibility Aliases
orchestrator = orchestrator_node
worker = worker_node
reducer = reducer_subgraph


# -----------------------------
# 9) Runner Function
# -----------------------------
def run(topic: str, as_of: Optional[str] = None) -> dict:
    if as_of is None:
        as_of = date.today().isoformat()

    out = app.invoke(
        {
            "topic": topic,
            "mode": "",
            "needs_research": False,
            "queries": [],
            "evidence": [],
            "plan": None,
            "as_of": as_of,
            "recency_days": 7,
            "sections": [],
            "merged_md": "",
            "md_with_placeholders": "",
            "image_specs": [],
            "final": "",
        }
    )

    return out


if __name__ == "__main__":
    import sys
    test_topic = sys.argv[1] if len(sys.argv) > 1 else "Self Attention in Transformer Architecture"
    print(f"Starting blog generation for: '{test_topic}'...")
    res = run(test_topic)
    print("\n[SUCCESS] Completed run!")
    print("Keys in output:", list(res.keys()))