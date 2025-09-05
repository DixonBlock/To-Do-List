# app.py
# Streamlit Brain Dump → Sticky Notes → Matrix → Energy-aware Priority List
# Author: you + your AI teammate

import uuid
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
import math
import time

import pandas as pd
import numpy as np
import streamlit as st

# --- Optional components (graceful fallbacks if not installed)
try:
    from streamlit_sortables import sort_items as _sort_items  # API wrapper: returns lists after drag
    HAS_SORTABLES = True
except Exception:
    HAS_SORTABLES = False

try:
    from streamlit_elements import elements, mui, dashboard
    HAS_ELEMENTS = True
except Exception:
    HAS_ELEMENTS = False

try:
    from notion_client import Client as NotionClient
    HAS_NOTION = True
except Exception:
    HAS_NOTION = False


st.set_page_config(
    page_title="Brain Dump Whiteboard",
    page_icon="🧠",
    layout="wide",
    menu_items={"about": "Brain dump → Sort → Prioritize, with Notion export."},
)

# ======= THEME / CSS ==========================================================
# Corkboard/whiteboard background + sticky note look
st.markdown("""
<style>
/* page background */
.main {
  background: #f6f3e7; /* corkboard-ish */
}

/* sticky "post-it" card */
.sticky {
  background: #fff59d; /* soft yellow */
  border-radius: 12px;
  padding: 10px 12px;
  box-shadow: 0 6px 18px rgba(0,0,0,0.08);
  border: 1px solid rgba(0,0,0,0.05);
  font-size: 0.95rem;
}

/* quadrant headings */
.quad-title {
  font-weight: 700;
  margin-bottom: 6px;
}

/* faint tags */
.tag {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 999px;
  background: rgba(0,0,0,0.06);
  font-size: 0.75rem;
  margin-right: 6px;
}

/* urgency highlight */
.urgent {
  outline: 2px solid #ef5350;
}

/* smaller help text alignment */
.small {
  font-size: 0.85rem;
  color: #555;
}
</style>
""", unsafe_allow_html=True)


# ======= DATA MODELS ==========================================================

@dataclass
class Task:
    id: str
    text: str
    urgent: bool = False
    importance: float = 0.5   # 0..1
    effort: float = 0.5       # 0..1
    energy: str = "Medium"    # Low | Medium | High
    quadrant: str = "Q2"      # default heuristic later
    created_at: float = 0.0

    def score(self) -> float:
        """
        ### [SECTION: PRIORITIZATION]
        Composite score blending:
        - Pareto: lean toward higher importance (impact)
        - Eisenhower: urgency bump (but important > urgent)
        - Effort: prefer lower effort when impact is high (quick wins)
        - Energy flow: match user-selected mode
        """
        # Base weights (tweak in-place)
        w_importance = 0.60
        w_urgency    = 0.20
        w_effort     = 0.15  # subtractive
        w_quickwin   = 0.10  # bonus if high imp & low effort

        s = self.importance * w_importance
        s += (1.0 if self.urgent else 0.0) * w_urgency
        s -= self.effort * w_effort

        # Quick win bonus
        if self.importance >= 0.7 and self.effort <= 0.3:
            s += w_quickwin

        return s


# ======= STATE INIT ===========================================================

def _init_state():
    if "tasks" not in st.session_state:
        st.session_state.tasks: Dict[str, Task] = {}
    if "lists" not in st.session_state:
        # Four quadrants: Q1(High Imp, Low Eff), Q2(High, High), Q3(Low, Low), Q4(Low, High)
        st.session_state.lists: Dict[str, List[str]] = {
            "Q1": [],
            "Q2": [],
            "Q3": [],
            "Q4": [],
        }
    if "cork_positions" not in st.session_state:
        st.session_state.cork_positions: Dict[str, Tuple[int, int, int, int]] = {}  # x,y,w,h for elements
    if "ONE_THING" not in st.session_state:
        st.session_state.ONE_THING: Optional[str] = None

_init_state()


# ======= HEURISTICS ===========================================================

def rough_estimate(text: str) -> Tuple[float, float]:
    """
    Light heuristic for importance/effort from keywords.
    You can tune this later if it misclassifies for you.
    """
    t = text.lower()
    imp = 0.5
    eff = 0.5

    high_imp_keys = ["deadline", "deliver", "today", "tonight", "tomorrow",
                     "payment", "invoice", "contract", "exam", "grant", "submission"]
    low_eff_keys  = ["email", "call", "text", "book", "post", "schedule", "file", "clean"]

    if any(k in t for k in high_imp_keys): imp = min(1.0, imp + 0.35)
    if "idea" in t or "brainstorm" in t:   imp = min(1.0, imp + 0.10)
    if any(k in t for k in low_eff_keys):  eff = max(0.0, eff - 0.25)
    if "write" in t or "design" in t or "record" in t: eff = min(1.0, eff + 0.15)

    return round(imp, 2), round(eff, 2)


def compute_quadrant(importance: float, effort: float) -> str:
    return (
        "Q1" if importance >= 0.5 and effort < 0.5 else
        "Q2" if importance >= 0.5 and effort >= 0.5 else
        "Q3" if importance <  0.5 and effort <  0.5 else
        "Q4"
    )


# ======= HELPERS ==============================================================

def add_tasks_from_text(raw: str):
    for line in [ln.strip() for ln in raw.splitlines() if ln.strip()]:
        tid = str(uuid.uuid4())
        imp, eff = rough_estimate(line)
        quad = compute_quadrant(imp, eff)
        t = Task(id=tid, text=line, importance=imp, effort=eff, quadrant=quad, created_at=time.time())
        st.session_state.tasks[tid] = t
        st.session_state.lists[quad].append(tid)


def sticky_html(task: Task) -> str:
    urgent_class = " urgent" if task.urgent else ""
    return f"""
    <div class="sticky{urgent_class}">
      <div><span class="tag">{task.energy}</span>
           <span class="tag">Imp {task.importance:.2f}</span>
           <span class="tag">Eff {task.effort:.2f}</span>
           {"<span class='tag'>URGENT</span>" if task.urgent else ""}
      </div>
      <div style="margin-top:6px;">{task.text}</div>
    </div>
    """


def export_markdown(tasks: List[Task]) -> str:
    lines = ["# Prioritized Tasks", ""]
    for i, t in enumerate(tasks, 1):
        lines.append(f"{i}. {'**[URGENT]** ' if t.urgent else ''}{t.text} "
                     f"(Imp {t.importance:.2f}, Eff {t.effort:.2f}, Energy {t.energy}, {t.quadrant})")
    return "\n".join(lines)


def to_dataframe(tasks: List[Task]) -> pd.DataFrame:
    return pd.DataFrame([{
        "Task": t.text,
        "Urgent": t.urgent,
        "Importance": t.importance,
        "Effort": t.effort,
        "Energy": t.energy,
        "Quadrant": t.quadrant,
        "PriorityScore": round(t.score(), 4),
        "ONE_Thing": (t.id == st.session_state.ONE_THING)
    } for t in tasks])


# ======= SIDEBAR ==============================================================
with st.sidebar:
    st.header("⚙️ Settings")

    st.subheader("Input")
    demo = st.toggle("Preload demo tasks", value=False)
    if st.button("Clear all tasks", type="secondary"):
        st.session_state.tasks.clear()
        for k in st.session_state.lists: st.session_state.lists[k].clear()
        st.session_state.cork_positions.clear()
        st.session_state.ONE_THING = None
        st.success("Cleared.")

    st.subheader("Energy flow")
    energy_mode = st.selectbox("Mode", ["Cluster similar", "Alternate heavy/light"], index=0,
                               help="How to arrange tasks to keep momentum.")

    st.subheader("Notion (optional)")
    notion_token = st.text_input("Integration Token (secret_...)", type="password")
    notion_db_id = st.text_input("Database ID (xxxxxxxxxxxxxxxxxxxxxx)")
    notion_ready = HAS_NOTION and notion_token and notion_db_id

    st.markdown(
        "<p class='small'>Database should have properties: "
        "<b>Title</b> (Name), <b>Urgent</b> (checkbox), <b>Importance</b> (number), "
        "<b>Effort</b> (number), <b>Energy</b> (select), <b>Quadrant</b> (select).</p>",
        unsafe_allow_html=True
    )


# ======= MAIN LAYOUT ==========================================================
st.title("🧠 Brain Dump → Sticky Notes → Priority List")

colL, colR = st.columns([1.1, 1.4])

with colL:
    st.markdown("### 1) Brain Dump")
    raw = st.text_area("Paste or type. Each line becomes a sticky note:",
                       height=180,
                       placeholder="Example:\nEmail Anna contract update\nRecord tutorial intro\nBook venue for demo day\nPay VAT invoice\nSketch app icon ideas",
                       key="dumpbox")
    if st.button("➕ Add to board", type="primary"):
        add_tasks_from_text(raw)
        st.success("Added!")

    if demo and not st.session_state.tasks:
        add_tasks_from_text("""Email Anna contract update
Record tutorial intro
Book venue for demo day
Pay VAT invoice
Sketch app icon ideas
Prepare SPIEL booth checklist
Call bank about bridge mortgage
Draft Kickstarter update""")
        st.info("Demo tasks loaded. Toggle off to stop auto-loading.")

    st.markdown("### 2) Tweak tasks")
    # Quick adjustors
    for tid, t in list(st.session_state.tasks.items()):
        with st.expander(f"✏️ {t.text[:72]}{'...' if len(t.text)>72 else ''}", expanded=False):
            new_text = st.text_input("Text", value=t.text, key=f"text_{tid}")
            urg = st.toggle("Urgent", value=t.urgent, key=f"urg_{tid}")
            imp = st.slider("Importance", 0.0, 1.0, t.importance, 0.05, key=f"imp_{tid}")
            eff = st.slider("Effort", 0.0, 1.0, t.effort, 0.05, key=f"eff_{tid}")
            eng = st.selectbox("Energy", ["Low", "Medium", "High"], index=["Low","Medium","High"].index(t.energy), key=f"eng_{tid}")
            # Apply
            t.text, t.urgent, t.importance, t.effort, t.energy = new_text, urg, imp, eff, eng
            new_quad = compute_quadrant(t.importance, t.effort)
            if new_quad != t.quadrant:
                # Move between lists
                if tid in st.session_state.lists[t.quadrant]:
                    st.session_state.lists[t.quadrant].remove(tid)
                st.session_state.lists[new_quad].append(tid)
                t.quadrant = new_quad

            # Mark ONE Thing
            if st.radio("ONE Thing (pick at most one)", ["No", "Yes"], index=1 if st.session_state.ONE_THING==tid else 0, key=f"one_{tid}") == "Yes":
                st.session_state.ONE_THING = tid
            elif st.session_state.ONE_THING == tid:
                st.session_state.ONE_THING = None

            if st.button("🗑️ Delete", key=f"del_{tid}", type="secondary"):
                # Remove from lists and tasks
                try:
                    st.session_state.lists[t.quadrant].remove(tid)
                except ValueError:
                    pass
                del st.session_state.tasks[tid]
                st.warning("Deleted.")
                st.stop()

with colR:
    st.markdown("### 3) Board")

    tabs = st.tabs(["🔲 Matrix (Importance × Effort)", "📌 Corkboard (freeform drag)", "📜 Priority List & Export"])

    # ---------------- MATRIX TAB ----------------
    with tabs[0]:
        st.write("Drag notes between quadrants. Urgency toggles and sliders are on the left.")

        quad_labels = {
            "Q1": "Q1 • High Importance, Low Effort (Quick Wins)",
            "Q2": "Q2 • High Importance, High Effort (Projects)",
            "Q3": "Q3 • Low Importance, Low Effort (Fill-ins)",
            "Q4": "Q4 • Low Importance, High Effort (Avoid/Delegate)",
        }

        # Build containers for sortables: list[dict] with 'header' and 'items'
        containers = []
        quad_order = ["Q1", "Q2", "Q3", "Q4"]
        quad_labels = {
            "Q1": "Q1 • High Importance, Low Effort (Quick Wins)",
            "Q2": "Q2 • High Importance, High Effort (Projects)",
            "Q3": "Q3 • Low Importance, Low Effort (Fill-ins)",
            "Q4": "Q4 • Low Importance, High Effort (Avoid/Delegate)",
        }

        for q in quad_order:
            display_items = []
            for tid in st.session_state.lists[q]:
                t = st.session_state.tasks[tid]
                # Keep the task id in the string so we can map back after drag
                label = f"{tid}:: {t.text[:70]}{'...' if len(t.text) > 70 else ''}"
                display_items.append(label)
            containers.append({"header": quad_labels[q], "items": display_items})

        if HAS_SORTABLES and any(st.session_state.tasks):
            st.caption("Tip: drag items between boxes. This updates the task's quadrant.")
            new_containers = _sort_items(
                containers,
                multi_containers=True,
                direction="vertical",
                key="matrix_sortables"
            )

            # Write back new membership & ordering
            for idx, q in enumerate(quad_order):
                st.session_state.lists[q].clear()
                for item in new_containers[idx]["items"]:
                    tid = item.split("::", 1)[0]
                    st.session_state.lists[q].append(tid)
                    st.session_state.tasks[tid].quadrant = q
        else:
            st.info("Install `streamlit-sortables` to enable drag between quadrants. "
                    "If Cloud couldn't install it, you'll still see current assignments below.")
            c1, c2 = st.columns(2)
            c3, c4 = st.columns(2)
            spots = [c1, c2, c3, c4]
            for idx, q in enumerate(quad_order):
                with spots[idx]:
                    st.markdown(f"**{quad_labels[q]}**")
                    for tid in st.session_state.lists[q]:
                        t = st.session_state.tasks[tid]
                        st.markdown(sticky_html(t), unsafe_allow_html=True)

                with spots[idx]:
                    st.markdown(f"**{quad_labels[q]}**")
                    for tid in st.session_state.lists[q]:
                        t = st.session_state.tasks[tid]
                        st.markdown(sticky_html(t), unsafe_allow_html=True)

    # ---------------- CORKBOARD TAB ----------------
    with tabs[1]:
        if HAS_ELEMENTS and any(st.session_state.tasks):
            st.caption("Drag and resize sticky notes freely. (Powered by `streamlit-elements`)")
            # Simple responsive grid
            with elements("corkboard"):
                layout = []
                # If no saved position, place roughly by quadrant
                default_positions = {
                    "Q1": (0, 0), "Q2": (6, 0), "Q3": (0, 8), "Q4": (6, 8)
                }
                for tid, t in st.session_state.tasks.items():
                    if tid not in st.session_state.cork_positions:
                        x, y = default_positions.get(t.quadrant, (0,0))
                        st.session_state.cork_positions[tid] = (x, y, 6, 4)
                    x,y,w,h = st.session_state.cork_positions[tid]
                    layout.append(dashboard.Item(tid, x, y, w, h, isResizable=True, isDraggable=True))

                with dashboard.Grid(layout=layout, draggableHandle=None):
                    for tid, t in st.session_state.tasks.items():
                        x,y,w,h = st.session_state.cork_positions[tid]
                        with mui.Paper(key=tid, elevation=3, sx={"padding":"10px","background":"#fff59d","borderRadius":"12px","border":"1px solid rgba(0,0,0,0.05)"}):
                            mui.Typography(t.text)
                            mui.Chip(label=f"Imp {t.importance:.2f}", size="small", sx={"mr":0.5})
                            mui.Chip(label=f"Eff {t.effort:.2f}", size="small", sx={"mr":0.5})
                            if t.urgent:
                                mui.Chip(label="URGENT", color="error", size="small")

                # Read back positions so movements persist
                for item in dashboard.draggable("corkboard"):
                    st.session_state.cork_positions[item["i"]] = (item["x"], item["y"], item["w"], item["h"])
        else:
            st.info("Install `streamlit-elements` for a freeform, draggable corkboard. "
                    "Meanwhile, use the Matrix tab.")

    # ---------------- PRIORITY TAB ----------------
    with tabs[2]:
        st.write("Ordered list blends Pareto (impact), Eisenhower (urgent/important), quick-wins, and your energy mode.")
        all_tasks = list(st.session_state.tasks.values())

        # Energy flow shaping
        def energy_rank(e: str) -> int:
            return {"Low":0, "Medium":1, "High":2}.get(e, 1)

        # Base ordering by score
        base_sorted = sorted(all_tasks, key=lambda t: (t.score(), -t.effort), reverse=True)

        # ONE Thing goes first if set
        if st.session_state.ONE_THING:
            one = st.session_state.tasks.get(st.session_state.ONE_THING)
            if one and one in base_sorted:
                base_sorted.remove(one)
                base_sorted.insert(0, one)

        # Shape by energy mode
        if energy_mode == "Cluster similar":
            # stable sort by energy buckets while respecting score
            groups = {"High": [], "Medium": [], "Low": []}
            for t in base_sorted:
                groups.setdefault(t.energy, []).append(t)
            prioritized = groups["High"] + groups["Medium"] + groups["Low"]
        else:
            # Alternate heavy/light (High ↔ Low), then Mediums sprinkled
            highs = [t for t in base_sorted if t.energy == "High"]
            lows  = [t for t in base_sorted if t.energy == "Low"]
            meds  = [t for t in base_sorted if t.energy == "Medium"]
            prioritized = []
            while highs or lows:
                if highs: prioritized.append(highs.pop(0))
                if lows:  prioritized.append(lows.pop(0))
            # Insert mediums every 2 steps
            i = 2
            for m in meds:
                prioritized.insert(min(i, len(prioritized)), m)
                i += 3

        # Display table + downloads
        df = to_dataframe(prioritized)
       st.dataframe(df, width="stretch")

        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download CSV", data=csv_bytes, file_name="prioritized_tasks.csv", mime="text/csv")

        md = export_markdown(prioritized)
        st.download_button("⬇️ Download Markdown", data=md.encode("utf-8"), file_name="prioritized_tasks.md", mime="text/markdown")
        st.code(md, language="markdown")

        # Notion export
        if st.button("📤 Export to Notion", type="primary", disabled=not notion_ready):
            if not HAS_NOTION:
                st.error("Install `notion-client` first: pip install notion-client")
            else:
                try:
                    client = NotionClient(auth=notion_token)
                    created = 0
                    for t in prioritized:
                        client.pages.create(
                            parent={"database_id": notion_db_id},
                            properties={
                                "Name": {"title": [{"text": {"content": t.text}}]},
                                "Urgent": {"checkbox": t.urgent},
                                "Importance": {"number": float(t.importance)},
                                "Effort": {"number": float(t.effort)},
                                "Energy": {"select": {"name": t.energy}},
                                "Quadrant": {"select": {"name": t.quadrant}},
                            }
                        )
                        created += 1
                    st.success(f"Exported {created} tasks to Notion.")
                except Exception as e:
                    st.error(f"Notion export failed: {e}")
