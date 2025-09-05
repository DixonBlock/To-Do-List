# app.py
# Streamlit Brain Dump → Sticky Notes → Matrix → Energy-aware Priority List
# GitHub + Streamlit Cloud ready

import uuid
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import time
import re

import pandas as pd
import streamlit as st

# --- Optional components (graceful fallbacks if not installed)
try:
    from streamlit_sortables import sort_items as _sort_items
    HAS_SORTABLES = True
except Exception:
    HAS_SORTABLES = False

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
st.markdown(
    """
<style>
/* page background */
.main { background: #f6f3e7; }

/* sticky "post-it" card */
.sticky {
  background: #fff59d;
  border-radius: 12px;
  padding: 10px 12px;
  box-shadow: 0 6px 18px rgba(0,0,0,0.08);
  border: 1px solid rgba(0,0,0,0.05);
  font-size: 0.95rem;
  line-height: 1.25rem;      /* keeps consistent height */
  margin-bottom: 10px;       /* spacing between stickies */
}

/* quadrant headings */
.quad-title { font-weight: 700; margin-bottom: 6px; }

/* faint tags */
.tag {
  display: inline-block; padding: 2px 8px; border-radius: 999px;
  background: rgba(0,0,0,0.06); font-size: 0.75rem; margin-right: 6px;
}

/* urgency highlight */
.urgent { outline: 2px solid #ef5350; }

/* smaller help text alignment */
.small { font-size: 0.85rem; color: #555; }
</style>
""",
    unsafe_allow_html=True,
)

# ======= DATA MODEL ===========================================================

@dataclass
class Task:
    id: str
    text: str
    urgent: bool = False
    importance: float = 0.5   # 0..1
    effort: float = 0.5       # 0..1
    energy: str = "Medium"    # Low | Medium | High
    quadrant: str = "Q2"
    created_at: float = 0.0

    def score(self) -> float:
        """Composite score blending Pareto, Eisenhower, quick-wins, and effort penalty."""
        w_importance = 0.60
        w_urgency    = 0.20
        w_effort     = 0.15  # subtractive
        w_quickwin   = 0.10  # bonus if high imp & low effort

        s = self.importance * w_importance
        s += (1.0 if self.urgent else 0.0) * w_urgency
        s -= self.effort * w_effort
        if self.importance >= 0.7 and self.effort <= 0.3:
            s += w_quickwin
        return s


# ======= STATE INIT ===========================================================

def _init_state():
    if "tasks" not in st.session_state:
        st.session_state.tasks: Dict[str, Task] = {}
    if "lists" not in st.session_state:
        st.session_state.lists: Dict[str, List[str]] = {"Q1": [], "Q2": [], "Q3": [], "Q4": []}
    if "ONE_THING" not in st.session_state:
        st.session_state.ONE_THING: Optional[str] = None

_init_state()


# ======= HEURISTICS ===========================================================

def rough_estimate(text: str) -> Tuple[float, float]:
    t = text.lower()
    imp, eff = 0.5, 0.5
    high_imp_keys = ["deadline","deliver","today","tonight","tomorrow","payment","invoice","contract","exam","grant","submission"]
    low_eff_keys  = ["email","call","text","book","post","schedule","file","clean"]
    if any(k in t for k in high_imp_keys):
        imp = min(1.0, imp + 0.35)
    if "idea" in t or "brainstorm" in t:
        imp = min(1.0, imp + 0.10)
    if any(k in t for k in low_eff_keys):
        eff = max(0.0, eff - 0.25)
    if any(k in t for k in ["write","design","record"]):
        eff = min(1.0, eff + 0.15)
    return round(imp, 2), round(eff, 2)


def compute_quadrant(importance: float, effort: float) -> str:
    return (
        "Q1" if importance >= 0.5 and effort < 0.5 else
        "Q2" if importance >= 0.5 and effort >= 0.5 else
        "Q3" if importance <  0.5 and effort <  0.5 else
        "Q4"
    )

def quadrant_to_scores(q: str) -> Tuple[float, float]:
    if q == "Q1":  # High Imp, Low Eff
        return 0.85, 0.25
    if q == "Q2":  # High Imp, High Eff
        return 0.85, 0.80
    if q == "Q3":  # Low Imp, Low Eff
        return 0.30, 0.25
    return 0.30, 0.80   # Q4: Low Imp, High Eff


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
        lines.append(
            f"{i}. {'**[URGENT]** ' if t.urgent else ''}{t.text} "
            f"(Imp {t.importance:.2f}, Eff {t.effort:.2f}, Energy {t.energy}, {t.quadrant})"
        )
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
        "ONE_Thing": (t.id == st.session_state.ONE_THING),
    } for t in tasks])


# ======= SIDEBAR ==============================================================

with st.sidebar:
    st.header("⚙️ Settings")

    st.subheader("Input")
    demo = st.toggle("Preload demo tasks", value=False)
    if st.button("Clear all tasks", type="secondary"):
        st.session_state.tasks.clear()
        for k in st.session_state.lists:
            st.session_state.lists[k].clear()
        st.session_state.ONE_THING = None
        st.success("Cleared.")

    st.subheader("Energy flow")
    energy_mode = st.selectbox("Mode", ["Cluster similar", "Alternate heavy/light"], index=0)

    # Prefer Streamlit Cloud secrets if provided; otherwise let user type manually.
    cloud_token = st.secrets.get("NOTION_TOKEN") if "NOTION_TOKEN" in st.secrets else ""
    cloud_db    = st.secrets.get("NOTION_DB_ID") if "NOTION_DB_ID" in st.secrets else ""
    st.subheader("Notion (optional)")
    notion_token = st.text_input("Integration Token (secret_...)", type="password", value=cloud_token)
    notion_db_id = st.text_input("Database ID (xxxxxxxx...)", value=cloud_db)


# ======= MAIN LAYOUT ==========================================================

st.title("🧠 Brain Dump → Sticky Notes → Priority List")

colL, colR = st.columns([1.1, 1.4])

with colL:
    st.markdown("### 1) Brain Dump")
    raw = st.text_area(
        "Paste or type. Each line becomes a sticky note:",
        height=180,
        placeholder="Type tasks here.\nOne task per line.",
        key="dumpbox",
    )
    if st.button("➕ Add to board", type="primary"):
        add_tasks_from_text(raw)
        st.success("Added!")

    if demo and not st.session_state.tasks:
        add_tasks_from_text(
            """Email Anna contract update
Record tutorial intro
Book venue for demo day
Pay VAT invoice
Sketch app icon ideas
Prepare SPIEL booth checklist
Call bank about bridge mortgage
Draft Kickstarter update"""
        )
        st.info("Demo tasks loaded.")

with colR:
    st.markdown("### 2) Board")

    tabs = st.tabs(["🔲 Matrix (drag here)", "📜 Priority List & Export"])

    # ---- Shared style for containers (frame + min height for ~3 stickies)
    QUAD_STYLE = {
        "container": {
            "border": "3px solid #1f4e63",
            "borderRadius": "16px",
            "background": "#0c2f3b0d",
            "padding": "10px",
            "margin": "6px",
            "minHeight": "270px",  # adjust if you change sticky height
            "boxSizing": "border-box",
        },
        "header": {"fontWeight": 700, "padding": "6px 8px 10px 8px", "color": "#0c2f3b"},
        "item": {"marginBottom": "10px"},
    }

    # ---------------- MATRIX TAB ----------------
    with tabs[0]:
        st.write(
            "Drag sticky notes between boxes. Each quadrant has its **own** urgent corner:\n"
            "• Q3 urgent (top-left inside) • Q4 urgent (top-right inside) • Q1 urgent (bottom-left inside) • Q2 urgent (bottom-right inside).\n"
            "Drop into a quadrant to **unflag** urgent; drop into its adjacent urgent box to **flag** urgent for that quadrant."
        )

        quad_labels = {
            "Q1": "High Importance • Low Effort",
            "Q2": "High Importance • High Effort",
            "Q3": "Low Importance • Low Effort",
            "Q4": "Low Importance • High Effort",
        }

        def _lbl(tid: str) -> str:
            t = st.session_state.tasks[tid]
            clean_text = t.text[:70] + ("..." if len(t.text) > 70 else "")
            # embed UUID in a hidden span so we can recover it after DnD
            return f"<span style='display:none'>{tid}</span>{clean_text}"

        def _extract_id(item: str) -> str:
            match = re.search(r"<span style='display:none'>(.*?)</span>", item)
            return match.group(1) if match else item

        # Build containers
        top_containers = [
            {"header": f"Q3 • {quad_labels['Q3']}", "items": [_lbl(tid) for tid in st.session_state.lists["Q3"] if not st.session_state.tasks[tid].urgent]},
            {"header": "🔥 Urgent • Q3 corner", "items": [_lbl(tid) for tid, t in st.session_state.tasks.items() if t.urgent and t.quadrant == "Q3"]},
            {"header": "🔥 Urgent • Q4 corner", "items": [_lbl(tid) for tid, t in st.session_state.tasks.items() if t.urgent and t.quadrant == "Q4"]},
            {"header": f"Q4 • {quad_labels['Q4']}", "items": [_lbl(tid) for tid in st.session_state.lists["Q4"] if not st.session_state.tasks[tid].urgent]},
        ]
        bot_containers = [
            {"header": f"Q1 • {quad_labels['Q1']}", "items": [_lbl(tid) for tid in st.session_state.lists["Q1"] if not st.session_state.tasks[tid].urgent]},
            {"header": "🔥 Urgent • Q1 corner", "items": [_lbl(tid) for tid, t in st.session_state.tasks.items() if t.urgent and t.quadrant == "Q1"]},
            {"header": "🔥 Urgent • Q2 corner", "items": [_lbl(tid) for tid, t in st.session_state.tasks.items() if t.urgent and t.quadrant == "Q2"]},
            {"header": f"Q2 • {quad_labels['Q2']}", "items": [_lbl(tid) for tid in st.session_state.lists["Q2"] if not st.session_state.tasks[tid].urgent]},
        ]

        if HAS_SORTABLES and any(st.session_state.tasks):
            st.caption("Tip: you can drag across rows by dropping into an urgent box first, then into the target row.")

            top_res = _sort_items(
                top_containers,
                multi_containers=True,
                direction="horizontal",
                key="matrix_top",
                styles=QUAD_STYLE,
            )
            bot_res = _sort_items(
                bot_containers,
                multi_containers=True,
                direction="horizontal",
                key="matrix_bot",
                styles=QUAD_STYLE,
            )

            new_Q3_ids   = [_extract_id(s) for s in top_res[0]["items"]]
            new_UQ3_ids  = [_extract_id(s) for s in top_res[1]["items"]]
            new_UQ4_ids  = [_extract_id(s) for s in top_res[2]["items"]]
            new_Q4_ids   = [_extract_id(s) for s in top_res[3]["items"]]

            new_Q1_ids   = [_extract_id(s) for s in bot_res[0]["items"]]
            new_UQ1_ids  = [_extract_id(s) for s in bot_res[1]["items"]]
            new_UQ2_ids  = [_extract_id(s) for s in bot_res[2]["items"]]
            new_Q2_ids   = [_extract_id(s) for s in bot_res[3]["items"]]

            # reset lists
            for q in ["Q1","Q2","Q3","Q4"]:
                st.session_state.lists[q].clear()

            # assign non-urgent
            for tid in new_Q1_ids:
                st.session_state.lists["Q1"].append(tid)
                t = st.session_state.tasks[tid]; t.quadrant, t.urgent = "Q1", False
                t.importance, t.effort = quadrant_to_scores("Q1")
            for tid in new_Q2_ids:
                st.session_state.lists["Q2"].append(tid)
                t = st.session_state.tasks[tid]; t.quadrant, t.urgent = "Q2", False
                t.importance, t.effort = quadrant_to_scores("Q2")
            for tid in new_Q3_ids:
                st.session_state.lists["Q3"].append(tid)
                t = st.session_state.tasks[tid]; t.quadrant, t.urgent = "Q3", False
                t.importance, t.effort = quadrant_to_scores("Q3")
            for tid in new_Q4_ids:
                st.session_state.lists["Q4"].append(tid)
                t = st.session_state.tasks[tid]; t.quadrant, t.urgent = "Q4", False
                t.importance, t.effort = quadrant_to_scores("Q4")

            # urgent flags
            for tid in new_UQ1_ids:
                t = st.session_state.tasks[tid]; t.quadrant, t.urgent = "Q1", True
                t.importance, t.effort = quadrant_to_scores("Q1")
            for tid in new_UQ2_ids:
                t = st.session_state.tasks[tid]; t.quadrant, t.urgent = "Q2", True
                t.importance, t.effort = quadrant_to_scores("Q2")
            for tid in new_UQ3_ids:
                t = st.session_state.tasks[tid]; t.quadrant, t.urgent = "Q3", True
                t.importance, t.effort = quadrant_to_scores("Q3")
            for tid in new_UQ4_ids:
                t = st.session_state.tasks[tid]; t.quadrant, t.urgent = "Q4", True
                t.importance, t.effort = quadrant_to_scores("Q4")

        else:
            st.info("Install `streamlit-sortables` to enable drag.")

    # ---------------- PRIORITY TAB ----------------
    with tabs[1]:
        st.write("Ordered list blends Pareto (impact), Eisenhower (urgent/important), quick-wins, and your energy mode.")
        all_tasks = list(st.session_state.tasks.values())

        # Base ordering by score
        base_sorted = sorted(all_tasks, key=lambda t: (t.score(), -t.effort), reverse=True)

        # ONE Thing goes first if set
        if st.session_state.ONE_THING:
            one = st.session_state.tasks.get(st.session_state.ONE_THING)
            if one and one in base_sorted:
                base_sorted.remove(one)
                base_sorted.insert(0, one)

        # Shape by energy mode (from sidebar variable `energy_mode`)
        if energy_mode == "Cluster similar":
            groups = {"High": [], "Medium": [], "Low": []}
            for t in base_sorted:
                groups.setdefault(t.energy, []).append(t)
            prioritized = groups["High"] + groups["Medium"] + groups["Low"]
        else:
            highs = [t for t in base_sorted if t.energy == "High"]
            lows  = [t for t in base_sorted if t.energy == "Low"]
            meds  = [t for t in base_sorted if t.energy == "Medium"]
            prioritized = []
            while highs or lows:
                if highs:
                    prioritized.append(highs.pop(0))
                if lows:
                    prioritized.append(lows.pop(0))
            i = 2
            for m in meds:
                prioritized.insert(min(i, len(prioritized)), m)
                i += 3

        df = to_dataframe(prioritized)
        st.data_editor(df, width="stretch", disabled=True)

        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download CSV", data=csv_bytes, file_name="prioritized_tasks.csv", mime="text/csv")

        md = export_markdown(prioritized)
        st.download_button("⬇️ Download Markdown", data=md.encode("utf-8"), file_name="prioritized_tasks.md", mime="text/markdown")
        st.code(md, language="markdown")

        # Notion export button
        if HAS_NOTION:
            if st.button("📤 Export to Notion", type="primary", disabled=False):
                # Prefer Cloud secrets, fall back to sidebar fields
                token = st.secrets.get("NOTION_TOKEN", "") or notion_token
                dbid  = st.secrets.get("NOTION_DB_ID", "") or notion_db_id
                if not (token and dbid):
                    st.error("Missing Notion token or DB ID. Add them in the sidebar or in Streamlit Secrets.")
                else:
                    try:
                        client = NotionClient(auth=token)
                        created = 0
                        for t in prioritized:
                            client.pages.create(
                                parent={"database_id": dbid},
                                properties={
                                    "Name": {"title": [{"text": {"content": t.text}}]},
                                    "Urgent": {"checkbox": t.urgent},
                                    "Importance": {"number": float(t.importance)},
                                    "Effort": {"number": float(t.effort)},
                                    "Energy": {"select": {"name": t.energy}},
                                    "Quadrant": {"select": {"name": t.quadrant}},
                                },
                            )
                            created += 1
                        st.success(f"Exported {created} tasks to Notion.")
                    except Exception as e:
                        st.error(f"Notion export failed: {e}")
        else:
            st.info("Install `notion-client` (already in requirements) to enable Notion export.")
