# streamlit_app.py
# -------------------------------------------------------------
# Academic lineage tree builder — Streamlit version (pre-loaded data)
# Ships with a local ./db_lineages folder of CSVs so users don't
# need to upload anything. Still provides interactive HTML (PyVis),
# a PNG export, CSV samples, and an optional outline .md.
# -------------------------------------------------------------

from __future__ import annotations

import io
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import streamlit as st
from pyvis.network import Network

# ---------------------- Streamlit page config -----------------------------
st.set_page_config(page_title="Конструктор деревьев академической генеалогии", layout="wide")
st.title("📚 Конструктор деревьев академической генеалогии")
st.caption(
    "Data are loaded from the app repository — no uploads required. Adjust settings in the sidebar and generate lineages."
)

# ---------------------- Defaults (edit via UI) ----------------------------
DEFAULT_DATA_DIR = "db_lineages"     # folder inside the repo
DEFAULT_CSV_GLOB = "*.csv"           # which files to include
DEFAULT_AUTHOR_COL = "candidate_name"
DEFAULT_SUPERVISOR_COLS = [f"supervisors_{i}.name" for i in (1, 2)]

# ---------------------- Helpers (ported/adapted) --------------------------

def _norm(s: str) -> str:  # simple normalisation
    return re.sub(r"\s+", " ", s.replace(".", " ").strip().lower())


def _split(full: str):
    p = full.split()
    p += ["", "", ""]
    return (p[0], p[1] if len(p) > 1 else "", p[2] if len(p) > 2 else "")


def variants(full: str) -> Set[str]:
    last, first, mid = _split(full.strip())
    fi, mi = first[:1], mid[:1]
    init = fi + mi
    init_dots = ".".join(init) + "." if init else ""
    return {
        v.strip()
        for v in {
            full,
            f"{last} {first} {mid}".strip(),
            f"{last} {init}",
            f"{last} {init_dots}",
            f"{init} {last}",
            f"{init_dots} {last}",
        }
        if v
    }


def build_index(df: pd.DataFrame, supervisor_cols: List[str]) -> Dict[str, Set[int]]:
    idx: Dict[str, Set[int]] = {}
    for col in supervisor_cols:
        if col not in df.columns:
            continue
        for i, raw in df[col].dropna().items():
            for v in variants(str(raw)):
                idx.setdefault(_norm(v), set()).add(i)
    return idx


def rows_for(df: pd.DataFrame, index: Dict[str, Set[int]], name: str) -> pd.DataFrame:
    hits: Set[int] = set()
    for v in variants(name):
        hits.update(index.get(_norm(v), set()))
    return df.loc[list(hits)] if hits else df.iloc[0:0]


def lineage(df: pd.DataFrame, index: Dict[str, Set[int]], root: str, author_col: str) -> tuple[nx.DiGraph, pd.DataFrame]:
    G = nx.DiGraph()
    collected = []
    Q, seen = [root], set()
    while Q:
        cur = Q.pop(0)
        if cur in seen:
            continue
        seen.add(cur)
        rows = rows_for(df, index, cur)
        if not rows.empty:
            collected.append(rows)
        for _, r in rows.iterrows():
            child = str(r.get(author_col, "")).strip()
            if child:
                G.add_edge(cur, child)
                Q.append(child)
    subset = pd.concat(collected, ignore_index=True).drop_duplicates() if collected else df.iloc[0:0]
    return G, subset


def multiline(name: str) -> str:
    return "\n".join(str(name).split())


def slug(s: str) -> str:
    return re.sub(r"[^A-Za-zА-Яа-я0-9]+", "_", s).strip("_")


# ---- Layout helpers (Graphviz optional; fallback to simple tree layout) --

def _hierarchy_pos(G: nx.DiGraph, root: str):
    from collections import deque

    levels: Dict[int, List[str]] = {}
    q = deque([(root, 0)])
    seen = set()
    while q:
        n, d = q.popleft()
        if n in seen:
            continue
        seen.add(n)
        levels.setdefault(d, []).append(n)
        for c in G.successors(n):
            q.append((c, d + 1))

    pos: Dict[str, tuple[float, float]] = {}
    for depth, nodes in levels.items():
        width = len(nodes)
        for i, n in enumerate(nodes):
            x = (i + 1) / (width + 1)
            y = -depth
            pos[n] = (x, y)
    return pos


def draw_matplotlib(G: nx.DiGraph, root: str) -> plt.Figure:
    if G.number_of_nodes() == 0:
        fig = plt.figure(figsize=(8, 4))
        plt.axis("off")
        plt.text(0.5, 0.5, "No descendants found", ha="center", va="center")
        return fig

    try:
        import networkx.drawing.nx_pydot as nx_pydot  # type: ignore
        pos = nx_pydot.graphviz_layout(G, prog="dot")
    except Exception:
        pos = _hierarchy_pos(G, root)

    fig = plt.figure(figsize=(max(8, len(G)) * 0.6, 8))
    nx.draw(
        G,
        pos,
        with_labels=True,
        labels={n: multiline(n) for n in G.nodes},
        node_color="#ADD8E6",
        node_size=3200,
        font_size=9,
        arrows=True,
    )
    plt.title(f"Academic lineage – {root}")
    plt.tight_layout()
    return fig


def build_pyvis_html(G: nx.DiGraph, root: str) -> str:
    net = Network(height="850px", width="100%", directed=True, bgcolor="#ffffff")
    net.toggle_physics(True)

    for n in G.nodes:
        net.add_node(n, label=multiline(n), title=str(n), shape="box", color="#ADD8E6")
    for u, v in G.edges:
        net.add_edge(u, v, arrows="to")

    vis_opts = {
        "layout": {"hierarchical": {"direction": "UD", "sortMethod": "directed"}},
        "interaction": {"hover": True},
        "physics": {
            "hierarchicalRepulsion": {
                "nodeDistance": 180,
                "springLength": 200,
                "springConstant": 0.01,
            },
            "solver": "hierarchicalRepulsion",
            "stabilization": {"iterations": 200},
            "minVelocity": 0.1,
        },
    }
    net.set_options(json.dumps(vis_opts))

    try:
        html = net.generate_html()  # type: ignore[attr-defined]
    except Exception:
        tmp = Path("_tmp.html")
        net.save_graph(str(tmp))
        html = tmp.read_text(encoding="utf-8")
        try:
            tmp.unlink()
        except Exception:
            pass
    return html


# ---------------------- Data loading (pre-loaded) -------------------------
@st.cache_data(show_spinner=False)
def load_data(data_dir: str, csv_glob: str, sep: str | None) -> pd.DataFrame:
    base = Path(data_dir).expanduser().resolve()
    files = sorted(base.glob(csv_glob))
    if not files:
        raise FileNotFoundError(f"No CSVs matching '{csv_glob}' found in {base}")

    guess = sep
    if guess is None:
        # try to auto-detect using first file
        sample = pd.read_csv(files[0], nrows=5, dtype=str)
        # crude heuristic: if there's only 1 column but semicolons present, switch
        if sample.shape[1] == 1:
            guess = ";"
        else:
            guess = ","

    frames = [pd.read_csv(f, dtype=str, keep_default_na=False, sep=guess) for f in files]
    return pd.concat(frames, ignore_index=True)


# ---------------------- UI: inputs ---------------------------------------
with st.sidebar:
    st.header("Data source")
    data_dir = st.text_input("Folder with CSVs", value=DEFAULT_DATA_DIR, help="Path inside the repo or absolute path.")
    csv_glob = st.text_input("File pattern", value=DEFAULT_CSV_GLOB)
    sep_choice = st.selectbox("Delimiter", ["auto", ",", ";", "\t"], index=0)
    sep_val = None if sep_choice == "auto" else ("\t" if sep_choice == "\t" else sep_choice)

    st.markdown("**Columns**")
    author_col = st.text_input("Author column (candidate)", value=DEFAULT_AUTHOR_COL)
    sup_cols_str = st.text_input(
        "Supervisor columns (comma-separated)",
        value=", ".join(DEFAULT_SUPERVISOR_COLS),
        help="Columns that contain supervisor names",
    )

    export_md_outline = st.checkbox("Also export outline .md (XMind-like)", value=False)

    run = st.button("Построить деревья", type="primary")

# ---------------------- Main logic ---------------------------------------

try:
    df = load_data(data_dir, csv_glob, sep_val)
except Exception as e:
    st.error(str(e))
    st.stop()

# Validate columns
supervisor_cols = [c.strip() for c in sup_cols_str.split(",") if c.strip()]
missing_cols = [c for c in [author_col, *supervisor_cols] if c not in df.columns]
if missing_cols:
    st.error("Missing required column(s): " + ", ".join(f"`{c}`" for c in missing_cols))
    st.stop()

# Build index once
idx = build_index(df, supervisor_cols)

# Let users pick roots from known supervisor names (plus manual add)
all_supervisor_names: Set[str] = set()
for col in supervisor_cols:
    all_supervisor_names.update({v for v in df[col].dropna().astype(str).unique() if v})

st.subheader("Choose initial supervisors")
col_a, col_b = st.columns([2, 1])
with col_a:
    preselect = sorted(list(all_supervisor_names))[:20]  # small starter set
    roots = st.multiselect(
        "Pick from known names",
        options=sorted(all_supervisor_names),
        default=preselect,
        help="These come from the supervisor columns in your data.",
    )
with col_b:
    manual = st.text_area("Or add names (one per line)", height=120)
    manual_list = [r.strip() for r in manual.splitlines() if r.strip()]

roots = list(dict.fromkeys([*roots, *manual_list]))  # dedupe, keep order

if run:
    if not roots:
        st.warning("Please select or enter at least one supervisor.")
        st.stop()

    # prepare a ZIP for all outputs
    all_zip_buf = io.BytesIO()
    zf = None
    try:
        import zipfile
        zf = zipfile.ZipFile(all_zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED)
    except Exception:
        pass

    for root in roots:
        st.markdown("---")
        st.subheader(f"▶ {root}")
        G, subset = lineage(df, idx, root, author_col)

        if G.number_of_edges() == 0:
            st.info("No descendants found for this root.")
            continue

        # PNG via matplotlib
        fig = draw_matplotlib(G, root)
        st.pyplot(fig, use_container_width=True)
        png_buf = io.BytesIO()
        fig.savefig(png_buf, format="png", dpi=300, bbox_inches="tight")
        png_bytes = png_buf.getvalue()

        # Interactive HTML via pyvis
        html = build_pyvis_html(G, root)
        st.components.v1.html(html, height=850, scrolling=True)
        html_bytes = html.encode("utf-8")

        # CSV subset
        csv_bytes = subset.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")

        # Optional outline MD (XMind-like)
        md_bytes = None
        if export_md_outline:
            out_lines: List[str] = []
            def walk(n: str, d: int = 0):
                out_lines.append(f"{'  ' * d}- {n}")
                for c in G.successors(n):
                    walk(c, d + 1)
            walk(root)
            md_bytes = ("\n".join(out_lines)).encode("utf-8")

        s = slug(root)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.download_button("Download PNG", data=png_bytes, file_name=f"{s}.png", mime="image/png")
        with c2:
            st.download_button("Download HTML", data=html_bytes, file_name=f"{s}.html", mime="text/html")
        with c3:
            st.download_button("Download CSV sample", data=csv_bytes, file_name=f"{s}.sampling.csv", mime="text/csv")
        with c4:
            if export_md_outline and md_bytes is not None:
                st.download_button("Download outline .md", data=md_bytes, file_name=f"{s}.xmind.md", mime="text/markdown")
            else:
                st.empty()

        # add to ZIP
        if zf is not None:
            zf.writestr(f"{s}.png", png_bytes)
            zf.writestr(f"{s}.html", html_bytes)
            zf.writestr(f"{s}.sampling.csv", csv_bytes)
            if export_md_outline and md_bytes is not None:
                zf.writestr(f"{s}.xmind.md", md_bytes)

    # Close and expose combined ZIP
    if zf is not None:
        try:
            zf.close()
            if all_zip_buf.getbuffer().nbytes > 0:
                st.download_button(
                    label="⬇️ Download all results as ZIP",
                    data=all_zip_buf.getvalue(),
                    file_name="lineages_export.zip",
                    mime="application/zip",
                )
        except Exception:
            pass
else:
    st.info("👈 Выберите или введите имя научного руководителя в основной панели, затем нажмите **Построить деревья**.")

