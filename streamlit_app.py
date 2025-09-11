# streamlit_app.py (RU, preloaded, simplified UI)
# -------------------------------------------------------------
# Академический конструктор родословных (без загрузки файлов)
# Данные берутся из локальной папки ./db_lineages (в репозитории).
# Интерфейс на русском, без технических настроек в сайдбаре.
# -------------------------------------------------------------

from __future__ import annotations

import io
import json
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import streamlit as st
from urllib.parse import urlencode
import zipfile
from pyvis.network import Network

# ---------------------- Константы -----------------------------------------
DATA_DIR = "db_lineages"      # папка с CSV внутри репозитория
CSV_GLOB = "*.csv"            # какие файлы брать
AUTHOR_COLUMN = "candidate_name"
SUPERVISOR_COLUMNS = [f"supervisors_{i}.name" for i in (1, 2)]

# ---------------------- Оформление страницы -------------------------------
st.set_page_config(page_title="Академические родословные", layout="wide")

# Полноширинный (full-bleed) контейнер для компонентов
st.markdown("""
<style>
  iframe {
        width: 100%;
  }
</style>
""", unsafe_allow_html=True)

st.title("📚 Конструктор академических родословных")
st.caption(
    "Данные заранее загружены в репозиторий (папка db_lineages). Выберите начальных руководителей и создайте деревья."
)

# ---------------------- Хелперы -------------------------------------------

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace(".", " ").strip().lower())


def _split(full: str) -> Tuple[str, str, str]:
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


def lineage(df: pd.DataFrame, index: Dict[str, Set[int]], root: str) -> tuple[nx.DiGraph, pd.DataFrame]:
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
            child = str(r.get(AUTHOR_COLUMN, "")).strip()
            if child:
                G.add_edge(cur, child)
                Q.append(child)
    subset = pd.concat(collected, ignore_index=True).drop_duplicates() if collected else df.iloc[0:0]
    return G, subset


def multiline(name: str) -> str:
    return "\n".join(str(name).split())


def slug(s: str) -> str:
    return re.sub(r"[^A-Za-zА-Яа-я0-9]+", "_", s).strip("_")


def build_share_url(names: List[str]) -> str:
    params = urlencode([("root", n) for n in names])
    try:
        addr = st.get_option("browser.serverAddress")
        port = st.get_option("browser.serverPort")
        base_path = st.get_option("server.baseUrlPath") or ""
        base_path = base_path.rstrip("/")
        proto = "https" if str(port) == "443" else "http"
        return f"{proto}://{addr}:{port}{base_path}?{params}" if params else f"{proto}://{addr}:{port}{base_path}"
    except Exception:
        return f"?{params}" if params else ""


def share_button(names: List[str], key: str) -> None:
    if st.button("🔗 Поделиться", key=key):
        try:
            st.query_params.clear()
            st.query_params["root"] = names
        except Exception:
            try:
                st.experimental_set_query_params(root=names)
            except Exception:
                pass
        url = build_share_url(names)
        with st.modal("Ссылка для доступа"):
            st.text_input("URL", url, key=f"share_url_{key}")


# --------- Рисование PNG (уменьшаем шрифты и узлы) -----------------------

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
        fig = plt.figure(figsize=(6, 3.5))
        plt.axis("off")
        plt.text(0.5, 0.5, "Потомки не найдены", ha="center", va="center")
        return fig
    try:
        import networkx.drawing.nx_pydot as nx_pydot  # type: ignore
        pos = nx_pydot.graphviz_layout(G, prog="dot")
    except Exception:
        pos = _hierarchy_pos(G, root)
    fig = plt.figure(figsize=(max(6, len(G) * 0.45), 6))
    nx.draw(
        G,
        pos,
        with_labels=True,
        labels={n: multiline(n) for n in G.nodes},
        node_color="#ADD8E6",
        node_size=2000,   # было 3200 → немного меньше
        font_size=7,      # заметно меньше шрифт
        arrows=True,
    )
    plt.title(f"Академическая родословная – {root}", fontsize=10)
    plt.tight_layout()
    return fig


# --------- Интерактивная HTML-визуализация (уменьшаем шрифты) -----------

def build_pyvis_html(G: nx.DiGraph, root: str) -> str:
    net = Network(height="1000px", width="100%", directed=True, bgcolor="#ffffff")
    net.toggle_physics(True)

    for n in G.nodes:
        net.add_node(n, label=multiline(n), title=str(n), shape="box", color="#ADD8E6")
    for u, v in G.edges:
        net.add_edge(u, v, arrows="to")

    vis_opts = {
        "nodes": {"font": {"size": 12}},  # шрифт поменьше
        "layout": {"hierarchical": {"direction": "UD", "sortMethod": "directed"}},
        "interaction": {"hover": True},
        "physics": {
            "hierarchicalRepulsion": {
                "nodeDistance": 140,
                "springLength": 160,
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


# ---------------------- Загрузка данных ----------------------------------
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    base = Path(DATA_DIR).expanduser().resolve()
    files = sorted(base.glob(CSV_GLOB))
    if not files:
        raise FileNotFoundError(f"В {base} не найдено CSV по маске '{CSV_GLOB}'")

    # простая авто‑детекция разделителя по первому файлу
    try:
        sample = pd.read_csv(files[0], nrows=5, dtype=str)
        sep = ";" if sample.shape[1] == 1 else ","
    except Exception:
        sep = ","

    frames = [pd.read_csv(f, dtype=str, keep_default_na=False, sep=sep) for f in files]
    return pd.concat(frames, ignore_index=True)


# ====================== ИНТЕРФЕЙС (без технического сайдбара) ============
try:
    df = load_data()
except Exception as e:
    st.error(f"Ошибка при загрузке данных: {e}")
    st.stop()

# Проверяем обязательные колонки
missing = [c for c in [AUTHOR_COLUMN, *SUPERVISOR_COLUMNS] if c not in df.columns]
if missing:
    st.error("Отсутствуют нужные колонки: " + ", ".join(f"`{c}`" for c in missing))
    st.stop()

# Индекс для поиска по руководителям
idx = build_index(df, SUPERVISOR_COLUMNS)

# Список доступных руководителей для выбора
all_supervisor_names: Set[str] = set()
for col in SUPERVISOR_COLUMNS:
    all_supervisor_names.update({v for v in df[col].dropna().astype(str).unique() if v})

# Параметры из адресной строки (?root=...)
shared_roots = st.query_params.get_all("root")

st.subheader("Выбор научных руководителей для построения деревьев")
roots = st.multiselect(
    "Выберите имена из базы",
    options=sorted(all_supervisor_names),
    default=shared_roots,  # если пришли по ссылке, подставляем имена
    help="Список формируется из столбцов с руководителями",
)
manual = st.text_area(
    "Или добавьте имена вручную в формате: Фамилия Имя Отчество (по одному на строку)",
    height=120,
)
manual_list = [r.strip() for r in manual.splitlines() if r.strip()]
roots = list(dict.fromkeys([*roots, *manual_list]))  # убрать дубликаты, сохранить порядок

# Если есть имена в параметрах, сразу строим деревья
build = st.button("Построить деревья", type="primary") or bool(shared_roots)
export_md_outline = st.checkbox("Также сохранить оглавление (.md)", value=False)

if build:
    if not roots:
        st.warning("Пожалуйста, выберите или введите хотя бы одно имя руководителя.")
        st.stop()

    all_zip_buf = io.BytesIO()
    zf = zipfile.ZipFile(all_zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED)

    for root in roots:
        st.markdown("---")
        st.subheader(f"▶ {root}")
        G, subset = lineage(df, idx, root)

        if G.number_of_edges() == 0:
            st.info("Потомки для этого имени не найдены.")
            continue

        # PNG (миниатюра) + HTML (широкий)
        fig = draw_matplotlib(G, root)
        png_buf = io.BytesIO()
        fig.savefig(png_buf, format="png", dpi=300, bbox_inches="tight")
        png_bytes = png_buf.getvalue()

        st.image(png_bytes, caption="Миниатюра PNG", width=220)

        html = build_pyvis_html(G, root)
        st.components.v1.html(html, height=800, width=2000, scrolling=True)
        html_bytes = html.encode("utf-8")

        # CSV с выборкой
        csv_bytes = subset.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")

        # Markdown‑оглавление (по желанию)
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
            st.download_button("Скачать PNG", data=png_bytes, file_name=f"{s}.png", mime="image/png")
        with c2:
            st.download_button("Скачать HTML", data=html_bytes, file_name=f"{s}.html", mime="text/html")
        with c3:
            st.download_button("Скачать выборку CSV", data=csv_bytes, file_name=f"{s}.sampling.csv", mime="text/csv")
        with c4:
            if md_bytes is not None:
                st.download_button("Скачать оглавление .md", data=md_bytes, file_name=f"{s}.xmind.md", mime="text/markdown")
            else:
                st.empty()

        # ZIP для текущего руководителя
        person_zip_buf = io.BytesIO()
        try:
            with zipfile.ZipFile(person_zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED) as z_person:
                z_person.writestr(f"{s}.png", png_bytes)
                z_person.writestr(f"{s}.html", html_bytes)
                z_person.writestr(f"{s}.sampling.csv", csv_bytes)
                if md_bytes is not None:
                    z_person.writestr(f"{s}.xmind.md", md_bytes)
            person_zip = person_zip_buf.getvalue()
        except Exception:
            person_zip = None

        col_zip_person, col_share_person = st.columns([3, 1])
        with col_zip_person:
            if person_zip is not None:
                st.download_button(
                    label="⬇️ Скачать всё архивом (ZIP)",
                    data=person_zip,
                    file_name=f"{s}.zip",
                    mime="application/zip",
                    key=f"zip_{s}",
                )
        with col_share_person:
            share_button([root], key=f"share_{s}")

        zf.writestr(f"{s}.png", png_bytes)
        zf.writestr(f"{s}.html", html_bytes)
        zf.writestr(f"{s}.sampling.csv", csv_bytes)
        if md_bytes is not None:
            zf.writestr(f"{s}.xmind.md", md_bytes)

    zf.close()
    if all_zip_buf.getbuffer().nbytes > 0:
        col_zip, col_share = st.columns([3, 1])
        with col_zip:
            st.download_button(
                label="⬇️ Скачать всё архивом (ZIP)",
                data=all_zip_buf.getvalue(),
                file_name="lineages_export.zip",
                mime="application/zip",
            )
        with col_share:
            share_button(roots, key="share_all")
else:
    st.info("Выберите или добавьте имена руководителей и нажмите ‘Построить деревья’.")

