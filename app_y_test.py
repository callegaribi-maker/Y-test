"""
app_y_test.py
─────────────
Visualizador de Sinais — Y-Balance Test (Y-BT)

Versão dedicada ao teste Y (arquivos "*_y_d{n}_..."), derivada da rotina
original de Step-Down/Salto. Diferenças principais em relação ao app
original:

  • Apenas 2 grupos anatômicos: L5 (lombar) e Joelho (celular fixado
    próximo ao côndilo do joelho) — o Y-BT não usa Coxa/Tornozelo.
  • Kinem traz apenas os marcadores "L 5" e "Côndilo lateral dir.".
  • Sem cálculo de ângulo do joelho — fluxo termina na seleção de janela
    e no check de qualidade dos sinais.
  • Sincronização continua pelo pico de impacto do salto/calibração
    inicial (mesma lógica de find_highest_peak / find_sync_xcorr).
  • Exportação para Excel usa sempre os sinais BRUTOS (sem detrend nem
    filtro passa-baixa), recortados na janela selecionada — o
    pré-processamento (detrend/filtro) serve só para inspeção visual.

Reaproveita 100% o signal_utils.py original (funções genéricas,
parametrizadas por keywords — nada precisou mudar lá).
"""

import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy.signal import find_peaks

from signal_utils import (
    NONE_LABEL,
    apply_detrend,
    apply_lowpass,
    best_match,
    build_export_sheet,
    col_default,
    detect_time_axis,
    find_highest_peak,
    find_sync_xcorr,
    get_aligned_data,
    is_xyz_col,
    kinem_cols_for_body,
    load_file,
    numeric_cols,
    resample_to_regular,
    try_numeric,
)

st.set_page_config(page_title="Visualizador de Sinais — Y-Balance Test", layout="wide")
st.title("📊 Visualizador de Sinais — Y-Balance Test (Y-BT)")

NONE = NONE_LABEL

# Definição dos 2 grupos anatômicos do Y-BT: (chave, rótulo, cor, keywords
# p/ auto-match de arquivo de celular, keywords p/ colunas do Kinem)
GROUPS = {
    "l5": dict(label="L5", emoji="🟢", kinem_kw=("l5", "l 5"),
               file_kw=(("acel", "l5"), ("acc", "l5"))),
    "joelho": dict(label="Joelho", emoji="🟣", kinem_kw=("condilo", "joelho"),
                   file_kw=(("acel", "joelho"), ("acc", "joelho"))),
}
GYR_FILE_KW = {
    "l5": (("gyro", "l5"), ("gyr", "l5")),
    "joelho": (("gyro", "joelho"), ("gyr", "joelho")),
}

DEFAULT_SESSION_STATE = {
    "files_data": {},
    "raw_synced": {},
    "proc_data": {},
    "proc_data_nofilter": {},
    "proc_data_cycles": {},
    "offsets": {},
    "peak_ref": None,
    "target_fs": 100,
    "fs_info": {},
    "show_preview": False,
    "synced": False,
    "synced_kinem_cols": {},
    "wizard_step": 1,
}
for key, default in DEFAULT_SESSION_STATE.items():
    st.session_state.setdefault(key, default)

STEP_NAMES = {
    1: "1 · Sincronizar",
    2: "2 · Verificação de alinhamento",
    3: "3 · Visualização dos sinais",
    4: "4 · Check de qualidade",
    5: "5 · Janela e fases do teste",
    6: "6 · Ciclos separados",
    7: "7 · Exportar",
}


def _is_kinem_displacement_col(col):
    """True para colunas de deslocamento puro do Kinem (ex: 'L 5 X', 'Côndilo lateral
    dir. Z') — exclui velocidade v(X), aceleração a(X), Length, abs e #2D."""
    cn = col.strip().lower()
    if "(" in cn or "length" in cn or "abs" in cn or "#2d" in cn:
        return False
    return cn.endswith("x") or cn.endswith("y") or cn.endswith("z")


def _goto(n):
    st.session_state.wizard_step = n
    st.rerun()


def _step_nav(back_to=None, next_to=None, next_label="Avançar ▶", next_disabled=False, key_suffix=""):
    """Renderiza botões de navegação Voltar/Avançar entre etapas do assistente."""
    st.divider()
    if back_to is not None and next_to is not None:
        c1, c2 = st.columns(2)
        with c1:
            if st.button("◀ Voltar", use_container_width=True, key=f"back_{key_suffix}"):
                _goto(back_to)
        with c2:
            if st.button(next_label, type="primary", use_container_width=True,
                         disabled=next_disabled, key=f"next_{key_suffix}"):
                _goto(next_to)
    elif back_to is not None:
        if st.button("◀ Voltar", key=f"back_{key_suffix}"):
            _goto(back_to)
    elif next_to is not None:
        if st.button(next_label, type="primary", use_container_width=True,
                     disabled=next_disabled, key=f"next_{key_suffix}"):
            _goto(next_to)


def _detect_first_plateau_start(x, y, fs=100.0, min_dur=1.2, rel_thr=0.9,
                                 prominence_frac=0.3, ignore_before_t=0.5):
    """Estima onde termina o início 'de acomodação' do teste (salto + assentamento)
    e começa o primeiro platô estável do deslocamento vertical do joelho. Ignora
    ativamente uma pequena janela ao redor do pico do salto (ignore_before_t) para
    não confundir esse solavanco com o platô de fato. Retorna x[0] (sem corte) se
    não conseguir detectar nada confiável.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 20:
        return float(x[0]) if n else 0.0
    overall_range = np.nanmax(y) - np.nanmin(y)
    if overall_range <= 0:
        return float(x[0])

    search_mask = x >= ignore_before_t
    if not np.any(search_mask):
        search_mask = np.ones(n, dtype=bool)
    idx_offset = int(np.argmax(search_mask))
    y_search = y[search_mask]
    try:
        troughs, _ = find_peaks(-y_search, prominence=prominence_frac * overall_range)
    except Exception:
        return float(x[0])
    if len(troughs) == 0:
        return float(x[0])

    first_trough = troughs[0] + idx_offset
    seg_before = y[0:first_trough + 1]
    plateau_level = np.percentile(seg_before, 90)
    trough_val = y[first_trough]
    threshold = trough_val + rel_thr * (plateau_level - trough_val)

    win = max(int(fs * min_dur), 3)
    above = y >= threshold
    run_len = 0
    for i in range(first_trough + 1):
        if above[i]:
            run_len += 1
            if run_len >= win:
                return float(x[i - win + 1])
        else:
            run_len = 0
    return float(x[0])


def _auto_phase_boundaries(x, y, n_ciclos):
    """Estima automaticamente os limites de Preparação/Descida/Subida para N ciclos,
    usando os vales (picos inferiores) do sinal como referência: acha os N vales mais
    profundos, depois busca — para cada um — onde o sinal sai do platô (início da
    descida) e onde volta a se aproximar do platô (fim da subida). Retorna uma lista
    ordenada com 3*n_ciclos - 1 tempos, ou None se não for possível detectar.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 10 or n_ciclos < 1:
        return None

    span = x[-1] - x[0]
    fs = (n - 1) / span if span > 0 else 100.0
    min_dist = max(int(fs * span / (n_ciclos * 6)), 1)
    try:
        trough_idx, _ = find_peaks(-y, distance=min_dist)
    except Exception:
        return None
    if len(trough_idx) < n_ciclos:
        return None

    order = np.argsort(y[trough_idx])[:n_ciclos]
    trough_idx = np.sort(trough_idx[order])

    def edge_backward(i_start, i_ref, threshold):
        for k in range(i_ref, i_start - 1, -1):
            if y[k] >= threshold:
                return k
        return i_start

    def edge_forward(i_ref, i_end, threshold):
        for k in range(i_ref, i_end + 1):
            if y[k] >= threshold:
                return k
        return i_end

    bounds = []
    for ci, ti in enumerate(trough_idx):
        prev_bound = trough_idx[ci - 1] if ci > 0 else 0
        next_bound = trough_idx[ci + 1] if ci < len(trough_idx) - 1 else n - 1
        seg_before = y[prev_bound:ti + 1]
        seg_after = y[ti:next_bound + 1]
        if len(seg_before) == 0 or len(seg_after) == 0:
            return None
        plateau_before = np.percentile(seg_before, 90)
        plateau_after = np.percentile(seg_after, 90)
        trough_val = y[ti]
        thr_before = trough_val + 0.9 * (plateau_before - trough_val)
        thr_after = trough_val + 0.9 * (plateau_after - trough_val)
        a_i = edge_backward(prev_bound, ti, thr_before)
        c_i = edge_forward(ti, next_bound, thr_after)
        bounds.append(x[a_i])
        bounds.append(x[ti])
        bounds.append(x[c_i])
    bounds = sorted(bounds)
    return bounds if len(bounds) == 3 * n_ciclos else None


st.caption(f"**Etapa atual:** {STEP_NAMES.get(st.session_state.wizard_step, '')}")


# ══════════════════════════════════════════════
# 1 · Upload de arquivos
# ══════════════════════════════════════════════
with st.sidebar:
    st.header("1 · Carregar Arquivos")
    uploaded = st.file_uploader(
        "CSV ou TXT (até 5 arquivos: Kinem + ACC/GYR de L5 e Joelho)",
        type=["csv", "txt"], accept_multiple_files=True,
    )
    if uploaded:
        loaded, errors = {}, []
        for f in uploaded:
            df = load_file(f)
            if df is not None:
                loaded[f.name] = df
            else:
                errors.append(f.name)

        if set(loaded.keys()) != set(st.session_state.files_data.keys()):
            st.session_state.files_data = loaded
            st.session_state.proc_data = {}
            st.session_state.offsets = {}
            st.session_state.fs_info = {}

        if errors:
            st.error(f"Não carregou: {', '.join(errors)}")
        st.success(f"{len(loaded)} arquivo(s) ✔")

files_data = st.session_state.files_data
if not files_data:
    st.info("👈 Carregue os arquivos na barra lateral para começar.")
    st.stop()

file_names = list(files_data.keys())


# ══════════════════════════════════════════════
# 2 · Kinem (referência)
# ══════════════════════════════════════════════
with st.sidebar:
    st.header("2 · Kinem (referência)")
    kinem_idx = next((i for i, n in enumerate(file_names) if "kinem" in n.lower()), 0)
    kinem_ref = st.selectbox("Arquivo Kinem", file_names, index=kinem_idx)
    kinem_num = numeric_cols(files_data[kinem_ref])

    st.caption("As duas colunas vêm do mesmo arquivo — cada pico ocorre na mesma amostra do Kinem.")
    st.caption("⚠️ No Kinem: Vertical = Z, AP = Y, ML = X. Selecione a coluna Z (a) vertical de cada marcador.")

    kinem_sync_cols = {}
    kinem_sync_cols["l5"] = st.selectbox(
        "Coluna L5 vertical (referência sync)", kinem_num,
        index=col_default(kinem_num, ["l 5 a(z)", "l5 a(z)", "l5a(z)", "l 5 z", "l5_az", "l5"]),
        key="kinem_col_l5",
    )
    kinem_sync_cols["joelho"] = st.selectbox(
        "Coluna Joelho (Côndilo) vertical (referência sync)", kinem_num,
        index=col_default(kinem_num, [
            "condilo lateral dir. a(z)", "condilo a(z)", "condilo lateral dir.",
            "condilo", "joelho",
        ]),
        key="kinem_col_joelho",
    )

others = [n for n in file_names if n != kinem_ref]

# ══════════════════════════════════════════════
# 3/4 · Grupos de celular (L5, Joelho)
# ══════════════════════════════════════════════
phone_files = {}   # group_key -> {"acc": fname|NONE, "acc_col": colname|None, "gyr": fname|NONE}
section_titles = {"l5": "3 · Grupo L5 (celular)", "joelho": "4 · Grupo Joelho (celular)"}

with st.sidebar:
    for gkey, gdef in GROUPS.items():
        st.header(section_titles[gkey])
        if gkey == "l5":
            st.caption("ACC e GYR já saem sincronizados entre si pelo celular.")

        acc = st.selectbox(
            f"ACC {gdef['label']}", [NONE] + others,
            index=best_match(others, *gdef["file_kw"]), key=f"{gkey}_acc",
        )
        acc_col = None
        if acc != NONE:
            num = numeric_cols(files_data[acc])
            acc_col = st.selectbox(
                f"Coluna Y do ACC {gdef['label']}", num,
                index=col_default(num, ["y"]), key=f"{gkey}_acc_col",
            )
        gyr = st.selectbox(
            f"GYR {gdef['label']}  ← offset = ACC", [NONE] + others,
            index=best_match(others, *GYR_FILE_KW[gkey]), key=f"{gkey}_gyr",
        )
        phone_files[gkey] = {"acc": acc, "acc_col": acc_col, "gyr": gyr}


# ══════════════════════════════════════════════
# Configurações avançadas de sincronização
# ══════════════════════════════════════════════
with st.sidebar:
    with st.expander("⚙️ Configurações avançadas de sincronização", expanded=False):
        fs_target = st.number_input(
            "Frequência alvo após reamostragem (Hz)",
            min_value=1, max_value=10000, value=100, step=10,
            help="Todos os arquivos serão reamostrados para esta frequência comum.",
        )


# ══════════════════════════════════════════════
# Botões: Preview + Sincronizar
# ══════════════════════════════════════════════
btn_col1, btn_col2, btn_col3 = st.columns([2, 1, 2])

with btn_col1:
    if st.button("👁 Preview sinais brutos", use_container_width=True):
        st.session_state.show_preview = not st.session_state.show_preview

with btn_col2:
    janela_seg = st.number_input(
        "Pico nos primeiros (s)", min_value=0.1, max_value=300.0, value=16.0, step=0.5,
        help="Janela de busca do pico de sincronização (ex.: salto/batida de calibração no início do Y-BT).",
    )

with btn_col3:
    sincronizar = st.button("🔗 Sincronizar", type="primary", use_container_width=True)


def _sync_phone_group(kinem_col, peak_kinem, acc_file, acc_col, gyr_file,
                       raw_synced, fs, janela_samp, none_label=NONE):
    """Sincroniza um par ACC/GYR de celular contra uma coluna vertical do Kinem.

    Retorna (offsets_parciais, mensagem|None) — GYR herda o offset do ACC.
    """
    offsets = {}
    if acc_file == none_label or not acc_col:
        return offsets, None
    if acc_col not in raw_synced.get(acc_file, pd.DataFrame()).columns:
        return offsets, None

    p = find_sync_xcorr(raw_synced[kinem_ref][kinem_col], raw_synced[acc_file][acc_col],
                         peak_kinem, janela_samp, fs)
    offsets[acc_file] = peak_kinem - p
    msg = f"pico @ {p} ({p/fs:.2f} s) → offset {peak_kinem-p:+d}"
    if gyr_file != none_label:
        offsets[gyr_file] = peak_kinem - p
    return offsets, msg


def _find_secondary_peak(raw_synced, kinem_col, peak_l5, fs, win_seconds=1.0):
    """Pico do Kinem de um grupo secundário (joelho), buscado numa janela de
    ±win_seconds ao redor do pico de referência do L5."""
    win = int(win_seconds * fs)
    s = try_numeric(raw_synced[kinem_ref][kinem_col])
    k_start, k_end = max(0, peak_l5 - win), min(len(s), peak_l5 + win)
    return find_highest_peak(s.iloc[k_start:k_end].reset_index(drop=True), k_end - k_start, fs) + k_start


if sincronizar:
    with st.spinner("Reamostrando e detectando pico…"):
        raw_synced, fs_info, msgs_pre = {}, {}, []
        for fname, df in files_data.items():
            r, fs_orig, desc = resample_to_regular(df, fs_target)
            raw_synced[fname] = r
            fs_info[fname] = fs_orig
            msgs_pre.append(f"**{fname[:35]}**: {desc}")

        st.session_state.raw_synced = raw_synced
        st.session_state.target_fs = fs_target
        st.session_state.fs_info = fs_info
        st.session_state.proc_data = {}
        st.session_state.proc_data_nofilter = {}

        janela_samp = int(janela_seg * fs_target)
        offsets = {kinem_ref: 0}
        msgs_sync = []

        peak_l5 = find_highest_peak(
            try_numeric(raw_synced[kinem_ref][kinem_sync_cols["l5"]]), janela_samp, fs_target,
        )
        st.session_state.peak_ref = peak_l5
        st.session_state.synced = True
        st.session_state.show_preview = False
        msgs_sync.append(f"**Kinem L5** — pico @ {peak_l5} ({peak_l5/fs_target:.2f} s) → x=0")

        group_peaks = {"l5": peak_l5}
        pk_joelho = _find_secondary_peak(raw_synced, kinem_sync_cols["joelho"], peak_l5, fs_target)
        group_peaks["joelho"] = pk_joelho
        msgs_sync.append(
            f"**Kinem Joelho** — pico @ {pk_joelho} ({pk_joelho/fs_target:.2f} s) "
            f"→ Δ {(pk_joelho-peak_l5)/fs_target:+.3f} s"
        )

        for gkey, gdef in GROUPS.items():
            pf = phone_files[gkey]
            g_offs, g_msg = _sync_phone_group(
                kinem_sync_cols[gkey], group_peaks[gkey], pf["acc"], pf["acc_col"], pf["gyr"],
                raw_synced, fs_target, janela_samp,
            )
            offsets.update(g_offs)
            if g_msg:
                msgs_sync.append(f"**{gdef['label']} ACC** — {g_msg}")
                if pf["gyr"] != NONE and pf["gyr"] in g_offs:
                    msgs_sync.append(f"**{gdef['label']} GYR** — offset {g_offs[pf['gyr']]:+d} (= ACC {gdef['label']})")

        for fname in file_names:
            offsets.setdefault(fname, 0)
        st.session_state.offsets = offsets
        st.session_state.synced_kinem_cols = dict(kinem_sync_cols)

        with st.expander("📋 Detalhes da sincronização", expanded=False):
            st.markdown("**Frequências detectadas:**")
            for m in msgs_pre:
                st.write(m)
            st.markdown("**Offsets calculados:**")
            for m in msgs_sync:
                st.write(m)


# ══════════════════════════════════════════════
# Auto-resync quando alguma coluna de referência muda
# ══════════════════════════════════════════════
if st.session_state.synced and st.session_state.raw_synced and st.session_state.peak_ref is not None:
    prev_cols = st.session_state.synced_kinem_cols
    changed = any(prev_cols.get(k) != kinem_sync_cols[k] for k in kinem_sync_cols)

    if changed:
        raws = st.session_state.raw_synced
        tfs = st.session_state.target_fs or 100
        jsamp = int(janela_seg * tfs)
        offs = dict(st.session_state.offsets)

        if prev_cols.get("l5") != kinem_sync_cols["l5"] and kinem_sync_cols["l5"] in raws.get(kinem_ref, pd.DataFrame()).columns:
            pk_l5 = find_highest_peak(try_numeric(raws[kinem_ref][kinem_sync_cols["l5"]]), jsamp, tfs)
            st.session_state.peak_ref = pk_l5
            offs[kinem_ref] = 0
            pf = phone_files["l5"]
            g_offs, _ = _sync_phone_group(
                kinem_sync_cols["l5"], pk_l5, pf["acc"], pf["acc_col"], pf["gyr"], raws, tfs, jsamp,
            )
            offs.update(g_offs)

        pk_l5 = st.session_state.peak_ref
        if kinem_sync_cols["joelho"] in raws.get(kinem_ref, pd.DataFrame()).columns:
            pk_g = _find_secondary_peak(raws, kinem_sync_cols["joelho"], pk_l5, tfs)
            pf = phone_files["joelho"]
            g_offs, _ = _sync_phone_group(
                kinem_sync_cols["joelho"], pk_g, pf["acc"], pf["acc_col"], pf["gyr"], raws, tfs, jsamp,
            )
            offs.update(g_offs)

        st.session_state.offsets = offs
        st.session_state.synced_kinem_cols = dict(kinem_sync_cols)
        st.session_state.proc_data = {}  # força reprocessamento


# ══════════════════════════════════════════════
# Preview bruto
# ══════════════════════════════════════════════
if st.session_state.show_preview:
    st.subheader("👁 Sinais brutos — sem pré-processamento")

    sync_cols = [(kinem_ref, kinem_sync_cols["l5"])]
    col_joelho = kinem_sync_cols["joelho"]
    if col_joelho and col_joelho != kinem_sync_cols["l5"]:
        sync_cols.append((kinem_ref, col_joelho))
    for gkey, pf in phone_files.items():
        if pf["acc"] != NONE and pf["acc_col"]:
            sync_cols.append((pf["acc"], pf["acc_col"]))

    pc1, pc2 = st.columns(2)
    with pc1:
        prev_t_start = st.number_input("Ver a partir de (s)", min_value=0.0, value=0.0, step=1.0, key="prev_start")
    with pc2:
        prev_t_end = st.number_input("Até (s)  — 0 = fim do sinal", min_value=0.0, value=0.0, step=1.0, key="prev_end")

    n_prev = len(sync_cols)
    fig_p = make_subplots(
        rows=n_prev, cols=1, shared_xaxes=False,
        subplot_titles=[f"{fn} · {c}" for fn, c in sync_cols], vertical_spacing=0.06,
    )
    for row, (fname, col) in enumerate(sync_cols, start=1):
        t, _ = detect_time_axis(files_data[fname])
        x = t - t[0] if t is not None else np.arange(len(files_data[fname]))
        y = try_numeric(files_data[fname][col])
        mask = x >= prev_t_start
        if prev_t_end > prev_t_start:
            mask &= x <= prev_t_end
        fig_p.add_trace(go.Scatter(x=x[mask], y=y[mask], mode="lines", showlegend=False), row=row, col=1)

    fig_p.update_layout(
        height=240 * n_prev, template="plotly_white",
        title="Colunas de sync — tempo original de cada arquivo", hovermode="x unified",
    )
    st.plotly_chart(fig_p, use_container_width=True)
    st.divider()


# ══════════════════════════════════════════════
# Verificação de alinhamento
# ══════════════════════════════════════════════
def render_alignment_check(title, kinem_col, phone_file, phone_col, label_k, label_p,
                            vraw, vx, vfs):
    check = []
    df_k = vraw.get(kinem_ref, pd.DataFrame())
    if kinem_col in df_k.columns:
        check.append((df_k, kinem_col, label_k))
    if phone_file != NONE and phone_col and phone_file in vraw:
        df_p = vraw[phone_file]
        if phone_col in df_p.columns:
            check.append((df_p, phone_col, label_p))
    if len(check) < 2:
        return

    with st.expander(f"🔍 Verificação — alinhamento {title}", expanded=True):
        colors_v = ["blue", "red"]
        series, caps = [], []
        for df_s, col, lbl in check:
            s = try_numeric(df_s[col]).fillna(0).values.astype(float)
            pk = np.nanmax(np.abs(s))
            series.append((s / pk if pk > 0 else s, lbl))
            caps.append(f"`{col}`")

        cap = "  |  ".join(
            f"{'🔵' if i == 0 else '🔴'} **{series[i][1]}**: {caps[i]}" for i in range(len(series))
        )
        st.caption(cap + f"  ·  reamostrado a {vfs:.0f} Hz  ·  normalizado pelo pico  ·  sem filtro passa-baixa")

        mask_2 = (vx >= -2) & (vx <= 2)
        all_vals = np.concatenate([s[mask_2] for s, _ in series if len(s) == len(vx)])
        all_vals = all_vals[~np.isnan(all_vals)]
        y_lo, y_hi = (float(np.nanmin(all_vals)) - 0.5, float(np.nanmax(all_vals)) + 0.5) if len(all_vals) else (-1.5, 1.5)

        fig_v = go.Figure()
        for i, (s_n, lbl) in enumerate(series):
            fig_v.add_trace(go.Scatter(
                x=vx, y=s_n, mode="lines", line=dict(color=colors_v[i], width=2), name=lbl, opacity=0.85,
            ))
        if len(series) == 2:
            diff = series[0][0] - series[1][0]
            fig_v.add_trace(go.Scatter(
                x=vx, y=diff, mode="lines", line=dict(color="gray", width=1, dash="dot"), name="Diferença",
            ))
        fig_v.add_vline(x=0, line_dash="dash", line_color="black",
                         annotation_text="salto", annotation_position="top right")
        fig_v.update_layout(
            title=f"{title} — normalizado pelo pico (sem filtro)",
            xaxis=dict(title="Tempo (s)  —  0 = pico do salto", range=[-2, 2]),
            yaxis=dict(title="Amplitude norm.", range=[y_lo, y_hi]),
            hovermode="x unified", template="plotly_white", height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            margin=dict(t=50, b=50, l=60, r=20),
        )
        st.plotly_chart(fig_v, use_container_width=True, key=f"verif_{title}_{kinem_col}_{phone_col}")


if st.session_state.synced and st.session_state.raw_synced and st.session_state.peak_ref is not None:
    if st.session_state.wizard_step == 1:
        st.divider()
        st.success("✅ Sincronização concluída.")
        if st.button("Avançar para verificação de alinhamento ▶", type="primary",
                      use_container_width=True, key="next_1"):
            _goto(2)

    if st.session_state.wizard_step >= 2:
        vfs = st.session_state.target_fs or 100
        vraw, vx_samp, _ = get_aligned_data(
            st.session_state.raw_synced, st.session_state.offsets, st.session_state.peak_ref, ref_file=kinem_ref,
        )
        if vraw is None:
            vraw = {f: df.copy() for f, df in st.session_state.raw_synced.items()}
            vx_samp = np.arange(max(len(d) for d in vraw.values())) - st.session_state.peak_ref
        vx = vx_samp / vfs

        for gkey, gdef in GROUPS.items():
            pf = phone_files[gkey]
            render_alignment_check(
                gdef["label"], kinem_sync_cols[gkey], pf["acc"], pf["acc_col"],
                f"Kinem {gdef['label']}", f"ACC {gdef['label']}", vraw, vx, vfs,
            )

        # ══════════════════════════════════════
        # Processamento inline
        # ══════════════════════════════════════
        st.divider()
        proc_done = bool(st.session_state.proc_data)
        with st.expander(
            "⚙️ Processamento  ✔ Aplicado" if proc_done else "⚙️ Processamento  ← Configure e processe aqui",
            expanded=not proc_done,
        ):
            pc1, pc2 = st.columns(2)
            with pc1:
                do_detrend = st.checkbox("Detrend (remover tendência linear)", value=True, key="do_detrend")
            with pc2:
                do_lowpass = st.checkbox("Filtro passa-baixa (Butterworth)", value=True, key="do_lowpass")

            if do_lowpass:
                fl1, fl2 = st.columns(2)
                with fl1:
                    cutoff_hz = st.number_input(
                        "Frequência de corte (Hz)", min_value=0.1, max_value=float(fs_target // 2),
                        value=min(20.0, float(fs_target // 2 - 1)), step=0.5, key="cutoff_hz",
                    )
                with fl2:
                    filt_order = st.selectbox("Ordem do filtro", [2, 4, 6, 8], index=1, key="filt_order")
            else:
                cutoff_hz, filt_order = 20.0, 4

            if st.button("🔧 Processar", type="primary", use_container_width=True, key="btn_processar"):
                raw = st.session_state.raw_synced
                proc, proc_nofilter = {}, {}
                for fname, df in raw.items():
                    r = df.copy()
                    if do_detrend:
                        r = apply_detrend(r)
                    proc_nofilter[fname] = r.copy()
                    if do_lowpass:
                        r = apply_lowpass(r, fs_target, cutoff_hz, filt_order)
                    if fname == kinem_ref:
                        # O deslocamento (X/Y/Z) do Kinem nunca é alterado pelo
                        # detrend/filtro — só velocidade, aceleração e os demais
                        # arquivos (ACC/GYR do celular) recebem o processamento.
                        disp_cols = [c for c in df.columns if _is_kinem_displacement_col(c)]
                        for c in disp_cols:
                            if c in r.columns:
                                r[c] = df[c].values
                                proc_nofilter[fname][c] = df[c].values
                    proc[fname] = r
                st.session_state.proc_data = proc
                st.session_state.proc_data_nofilter = proc_nofilter
                st.rerun()

        _step_nav(back_to=1, next_to=3, next_label="Avançar para visualização ▶",
                  next_disabled=not bool(st.session_state.proc_data), key_suffix="2")


# ══════════════════════════════════════════════
# Auto-visualização — todos os eixos X, Y, Z
# ══════════════════════════════════════════════
if st.session_state.proc_data and st.session_state.synced:
    pfs = st.session_state.target_fs or 100

    aligned_data, x_samp, align_msg = get_aligned_data(
        st.session_state.proc_data, st.session_state.offsets, st.session_state.peak_ref, ref_file=kinem_ref,
    )
    if aligned_data is None:
        st.error(align_msg)
        st.stop()

    x_axis = x_samp / pfs
    x_min_data, x_max_data = float(x_axis.min()), float(x_axis.max())

    kdf = aligned_data.get(kinem_ref, pd.DataFrame())

    def get_phone_xyz(fname):
        if fname == NONE or fname not in aligned_data:
            return []
        return [c for c in aligned_data[fname].columns if is_xyz_col(c)]

    def make_auto_traces(gkey):
        gdef = GROUPS[gkey]
        pf = phone_files[gkey]
        k_cols = kinem_cols_for_body(kdf, *gdef["kinem_kw"])
        traces = [(kinem_ref, c, try_numeric(kdf[c])) for c in k_cols if c in kdf.columns]
        if pf["acc"] != NONE and pf["acc"] in aligned_data:
            for c in get_phone_xyz(pf["acc"]):
                traces.append((pf["acc"], c, try_numeric(aligned_data[pf["acc"]][c])))
        if pf["gyr"] != NONE and pf["gyr"] in aligned_data:
            for c in get_phone_xyz(pf["gyr"]):
                traces.append((pf["gyr"], c, try_numeric(aligned_data[pf["gyr"]][c])))
        return traces

    group_traces = {gkey: make_auto_traces(gkey) for gkey in GROUPS}

    if st.session_state.wizard_step >= 3:
        st.divider()
        st.subheader("📊 Sinais sincronizados — todos os eixos X, Y, Z")
        st.caption(align_msg)

        def render_auto_charts(traces):
            for fname, col, y in traces:
                fig_i = go.Figure()
                fig_i.add_trace(go.Scatter(x=x_axis, y=y, mode="lines", line=dict(width=1.5), showlegend=False))
                fig_i.add_vline(x=0, line_dash="dash", line_color="gray",
                                 annotation_text="salto", annotation_position="top right")
                fig_i.update_layout(
                    title=dict(text=f"<b>{fname[:26]}</b> · {col}", font_size=12),
                    xaxis=dict(title="Tempo (s)  —  0 = pico do salto", range=[x_min_data, x_max_data]),
                    yaxis_title="", height=220,
                    margin=dict(t=42, b=38, l=55, r=10), hovermode="x", template="plotly_white",
                )
                st.plotly_chart(fig_i, use_container_width=True)

        auto_cols = st.columns(2)
        for auto_col, gkey in zip(auto_cols, GROUPS):
            with auto_col:
                st.markdown(f"#### {GROUPS[gkey]['emoji']} {GROUPS[gkey]['label']}")
                render_auto_charts(group_traces[gkey])

        _step_nav(back_to=2, next_to=4, next_label="Avançar para check de qualidade ▶", key_suffix="3")

    if st.session_state.wizard_step >= 4:
        # ══════════════════════════════════════
        # Check de qualidade
        # ══════════════════════════════════════
        with st.expander("⚙️ Colunas para check de qualidade (1 por fonte)", expanded=False):
            st.caption("Escolha exatamente qual coluna usar de cada fonte. Os sinais serão plotados sobrepostos (z-score).")

            qa_kinem_keywords = {
                "l5": ["l 5 d(z)", "l5 d(z)", "l 5 v(z)", "l5 v(z)", "l 5 a(z)", "l5 a(z)", "l 5 z", "l5"],
                "joelho": ["condilo lateral dir. a(z)", "condilo a(z)", "condilo lateral dir.", "condilo", "joelho"],
            }

            qa_kinem_cols, qa_phone_cols = {}, {}
            qa_cols_ui = st.columns(2)
            for ui_col, gkey in zip(qa_cols_ui, GROUPS):
                with ui_col:
                    gdef = GROUPS[gkey]
                    qa_kinem_cols[gkey] = st.selectbox(
                        f"{gdef['emoji']} Kinem — {gdef['label']}", kinem_num, key=f"qa_kinem_{gkey}",
                        index=col_default(kinem_num, qa_kinem_keywords[gkey]),
                    )
                    pf = phone_files[gkey]
                    acc_num = numeric_cols(aligned_data.get(pf["acc"], pd.DataFrame())) if pf["acc"] != NONE else []
                    gyr_num = numeric_cols(aligned_data.get(pf["gyr"], pd.DataFrame())) if pf["gyr"] != NONE else []
                    qa_phone_cols[gkey] = {
                        "acc": st.selectbox(
                            f"{gdef['emoji']} ACC — {gdef['label']}", acc_num if acc_num else ["—"],
                            key=f"qa_acc_{gkey}", index=col_default(acc_num, ["z", "y", "x"]) if acc_num else 0,
                        ) if acc_num else None,
                        "gyr": st.selectbox(
                            f"{gdef['emoji']} GYR — {gdef['label']}", gyr_num if gyr_num else ["—"],
                            key=f"qa_gyr_{gkey}", index=col_default(gyr_num, ["z", "y", "x"]) if gyr_num else 0,
                        ) if gyr_num else None,
                    }

        qa_xmin, qa_xmax = x_min_data, x_max_data
        mask_qa = (x_axis >= qa_xmin) & (x_axis <= qa_xmax)
        x_view = x_axis[mask_qa]

        def get_qa_entry(fname, col_name):
            df_q = aligned_data.get(fname) if (fname and fname != NONE) else None
            if df_q is None or col_name is None or col_name not in df_q.columns:
                return None
            y = try_numeric(df_q[col_name]).values[mask_qa].astype(float)
            if np.all(np.isnan(y)):
                return None
            return (float(np.nanstd(y)), f"{fname[:20]} · {col_name}", y)

        qa_cols_out = st.columns(2)
        for ui_col, gkey in zip(qa_cols_out, GROUPS):
            gdef = GROUPS[gkey]
            pf = phone_files[gkey]
            group_entries = [e for e in [
                get_qa_entry(kinem_ref, qa_kinem_cols[gkey]),
                get_qa_entry(pf["acc"] if pf["acc"] != NONE else "", qa_phone_cols[gkey]["acc"]),
                get_qa_entry(pf["gyr"] if pf["gyr"] != NONE else "", qa_phone_cols[gkey]["gyr"]),
            ] if e]
            with ui_col:
                st.markdown(f"#### {gdef['emoji']} {gdef['label']} — Kinem vs Celular")
                if not group_entries:
                    st.info("Nenhum sinal classificado neste grupo.")
                    continue
                fig_qa = go.Figure()
                for std_val, lbl, y_raw in group_entries:
                    mn, sd = np.nanmean(y_raw), np.nanstd(y_raw)
                    y_norm = (y_raw - mn) / sd if sd > 0 else y_raw - mn
                    fig_qa.add_trace(go.Scatter(
                        x=x_view, y=y_norm, mode="lines", name=f"{lbl}  (σ_orig={std_val:.3f})",
                    ))
                fig_qa.add_vline(x=0, line_dash="dash", line_color="gray", annotation_text="salto")
                fig_qa.update_layout(
                    xaxis=dict(title="Tempo (s)  —  0 = pico do salto", range=[qa_xmin, qa_xmax]),
                    yaxis_title="z-score", height=360, template="plotly_white", hovermode="x unified",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0), margin=dict(t=30, b=40),
                )
                st.plotly_chart(fig_qa, use_container_width=True)

        _step_nav(back_to=3, next_to=5, next_label="Avançar para seleção das fases ▶", key_suffix="4")

    if st.session_state.wizard_step >= 5:
        # ══════════════════════════════════════
        # Seleção das fases do teste (inclui o corte de janela)
        # ══════════════════════════════════════
        st.subheader("🪟🦵 Janela e fases do teste")

        _default_knee_col_idx = col_default(kinem_num, [
            "côndilo lateral dir. z", "condilo lateral dir. z", "côndilo lateral dir. d(z)",
            "condilo lateral dir. d(z)", "condilo",
        ])
        _default_knee_col = kinem_num[_default_knee_col_idx] if kinem_num else None
        auto_t0 = x_min_data
        if _default_knee_col and _default_knee_col in kdf.columns:
            y_full_for_auto = try_numeric(kdf[_default_knee_col]).values[:len(x_axis)]
            auto_t0 = _detect_first_plateau_start(x_axis, y_full_for_auto)

        wc1, wc2 = st.columns(2)
        with wc1:
            view_start = st.number_input(
                "Início (s) relativo ao pico", value=float(auto_t0), step=0.5, key="view_start",
            )
        with wc2:
            view_end = st.number_input(
                "Fim (s) relativo ao pico", value=float(x_max_data), step=0.5, key="view_end",
            )
        st.caption(
            "💡 O início já pula automaticamente o começo do teste (salto + acomodação) até o "
            "primeiro platô do joelho. Ajuste se precisar incluir mais ou menos coisa."
        )

        st.divider()
        st.caption(
            "Cada vale do deslocamento vertical do joelho é um ciclo. Escolha quantos ciclos "
            "existem na janela e ajuste os marcadores para dividir cada um em 3 fases: "
            "**Preparação → Descida → Subida**."
        )

        knee_disp_col = st.selectbox(
            "Coluna de deslocamento vertical do joelho (Kinem)", kinem_num,
            index=_default_knee_col_idx,
            key="knee_disp_col",
        )

        mask_phase = (x_axis >= view_start) & (x_axis <= view_end)
        x_phase = x_axis[mask_phase]
        y_phase = (
            try_numeric(kdf[knee_disp_col]).values[:len(x_axis)][mask_phase]
            if knee_disp_col in kdf.columns else np.array([])
        )

        n_ciclos = st.number_input(
            "Número de ciclos nesta janela", min_value=1, max_value=10, value=3, step=1, key="n_ciclos",
        )
        PHASE_NAMES = ["Preparação", "Descida", "Subida"]
        PHASE_COLORS = {
            "Preparação": "rgba(148,163,184,0.18)",
            "Descida": "rgba(251,191,36,0.18)",
            "Subida": "rgba(52,211,153,0.18)",
        }
        n_bounds = 3 * n_ciclos
        span = max(view_end - view_start, 0.1)

        auto_bounds = (
            _auto_phase_boundaries(x_phase, y_phase, n_ciclos)
            if len(x_phase) > 10 and not np.all(np.isnan(y_phase)) else None
        )
        if auto_bounds is None:
            st.caption(
                "⚠️ Não consegui detectar os ciclos automaticamente nesta janela/coluna — "
                "marcadores começam espaçados igualmente. Ajuste manualmente."
            )
        else:
            st.caption("✅ Marcadores posicionados automaticamente a partir dos vales do sinal — ajuste fino se precisar.")

        boundary_vals = []
        n_cols = 3
        for row_start in range(0, n_bounds, n_cols):
            row_cols = st.columns(min(n_cols, n_bounds - row_start))
            for j, col in enumerate(row_cols):
                i = row_start + j
                seg_a, seg_b = i, i + 1
                cyc_a, ph_a = seg_a // 3 + 1, PHASE_NAMES[seg_a % 3]
                if seg_b < 3 * n_ciclos:
                    cyc_b, ph_b = seg_b // 3 + 1, PHASE_NAMES[seg_b % 3]
                    label = (
                        f"C{cyc_a} {ph_a} → C{cyc_b} {ph_b}" if cyc_a != cyc_b
                        else f"C{cyc_a}: {ph_a} → {ph_b}"
                    )
                else:
                    label = f"C{cyc_a}: {ph_a} → Fim (fecha o ciclo)"
                key = f"phase_b_{i}"
                default_val = auto_bounds[i] if auto_bounds is not None else view_start + (i + 1) * span / (n_bounds + 1)
                if key in st.session_state and not (view_start <= st.session_state[key] <= view_end):
                    st.session_state[key] = default_val
                with col:
                    val = st.slider(
                        label, min_value=float(view_start), max_value=float(view_end),
                        value=float(default_val), step=0.05, key=key,
                    )
                boundary_vals.append(val)

        boundary_vals = sorted(boundary_vals)
        boundaries_full = [view_start] + boundary_vals + [view_end]

        if len(x_phase) == 0 or np.all(np.isnan(y_phase)):
            st.info("Selecione uma coluna de deslocamento vertical válida para visualizar as fases.")
        else:
            fig_phase = go.Figure()
            fig_phase.add_trace(go.Scatter(
                x=x_phase, y=y_phase, mode="lines", line=dict(color="#7c3aed", width=2),
                name="Deslocamento vertical — Joelho",
            ))
            y_lo = float(np.nanmin(y_phase))
            y_hi = float(np.nanmax(y_phase))
            pad = (y_hi - y_lo) * 0.08 or 1.0
            for seg in range(3 * n_ciclos):
                x0, x1 = boundaries_full[seg], boundaries_full[seg + 1]
                phase = PHASE_NAMES[seg % 3]
                cyc = seg // 3 + 1
                fig_phase.add_vrect(
                    x0=x0, x1=x1, fillcolor=PHASE_COLORS[phase], line_width=0,
                    annotation_text=f"C{cyc}·{phase}" if n_ciclos > 1 else phase,
                    annotation_position="top left", annotation_font_size=10,
                )
            for b in boundary_vals:
                fig_phase.add_vline(x=b, line_dash="dash", line_color="rgba(80,80,80,0.6)")
            fig_phase.update_layout(
                xaxis=dict(title="Tempo (s)  —  0 = pico do salto", range=[view_start, view_end]),
                yaxis=dict(title="Deslocamento vertical (Z)", range=[y_lo - pad, y_hi + pad]),
                height=440, template="plotly_white", hovermode="x unified", margin=dict(t=40, b=40),
            )
            st.plotly_chart(fig_phase, use_container_width=True)

        with st.expander("Ver intervalos de cada fase/ciclo", expanded=False):
            for seg in range(3 * n_ciclos):
                x0, x1 = boundaries_full[seg], boundaries_full[seg + 1]
                phase = PHASE_NAMES[seg % 3]
                cyc = seg // 3 + 1
                st.write(f"**Ciclo {cyc} — {phase}:** {x0:+.2f} s → {x1:+.2f} s")
            tail_start = boundaries_full[3 * n_ciclos]
            if tail_start < view_end - 1e-6:
                st.caption(f"⚪ Fora de qualquer ciclo (não entra em nenhuma fase): {tail_start:+.2f} s → {view_end:+.2f} s")

        _step_nav(back_to=4, next_to=6, next_label="Avançar para ver ciclos separados ▶", key_suffix="5")

    if st.session_state.wizard_step >= 6:
        # ══════════════════════════════════════
        # Ciclos separados — todos os eixos (sensores + cinemática)
        # ══════════════════════════════════════
        st.subheader("🔀 Ciclos separados — todos os eixos")

        DIRECTION_NAMES = ["Anterior", "Posteromedial", "Posterolateral"]

        def _cycle_label(c):
            if n_ciclos == 3 and 1 <= c <= 3:
                return f"Ciclo {c} — {DIRECTION_NAMES[c - 1]}"
            return f"Ciclo {c}"

        if n_ciclos == 3:
            st.caption(
                "Ciclo 1 = alcance **Anterior**, Ciclo 2 = **Posteromedial**, "
                "Ciclo 3 = **Posterolateral** (ordem padrão do Y-Balance Test)."
            )
        else:
            st.caption(f"{n_ciclos} ciclo(s) definidos na etapa anterior, mostrados separadamente abaixo.")

        with st.expander("⚙️ Reprocessar antes de ver os ciclos (opcional)", expanded=False):
            st.caption(
                "Reaplica detrend/filtro só para esta visualização, com configurações "
                "independentes das usadas na etapa 2 — não afeta a exportação (que é sempre bruta)."
            )
            rpc1, rpc2 = st.columns(2)
            with rpc1:
                do_detrend_f = st.checkbox("Detrend", value=True, key="do_detrend_f7")
            with rpc2:
                do_lowpass_f = st.checkbox("Filtro passa-baixa", value=True, key="do_lowpass_f7")
            if do_lowpass_f:
                rfl1, rfl2 = st.columns(2)
                with rfl1:
                    cutoff_f = st.number_input(
                        "Frequência de corte (Hz)", min_value=0.1, max_value=float(fs_target // 2),
                        value=min(20.0, float(fs_target // 2 - 1)), step=0.5, key="cutoff_f7",
                    )
                with rfl2:
                    filt_order_f = st.selectbox("Ordem do filtro", [2, 4, 6, 8], index=1, key="filt_order_f7")
            else:
                cutoff_f, filt_order_f = 20.0, 4

            if st.button("🔧 Reprocessar para esta visualização", key="btn_reprocess_f7"):
                raw = st.session_state.raw_synced
                proc2 = {}
                for fname, df in raw.items():
                    r = df.copy()
                    if do_detrend_f:
                        r = apply_detrend(r)
                    if do_lowpass_f:
                        r = apply_lowpass(r, fs_target, cutoff_f, filt_order_f)
                    if fname == kinem_ref:
                        disp_cols = [c for c in df.columns if _is_kinem_displacement_col(c)]
                        for c in disp_cols:
                            if c in r.columns:
                                r[c] = df[c].values
                    proc2[fname] = r
                st.session_state.proc_data_cycles = proc2
                st.rerun()

            if st.session_state.get("proc_data_cycles"):
                if st.button("↩ Voltar a usar o processamento padrão (etapa 2)", key="btn_reset_reprocess_f7"):
                    st.session_state.proc_data_cycles = {}
                    st.rerun()

        def _rebuild_group_traces(aligned_src):
            kdf_src = aligned_src.get(kinem_ref, pd.DataFrame())

            def _phone_xyz(fname):
                if fname == NONE or fname not in aligned_src:
                    return []
                return [c for c in aligned_src[fname].columns if is_xyz_col(c)]

            out = {}
            for gk in GROUPS:
                gd = GROUPS[gk]
                pfk = phone_files[gk]
                k_cols = kinem_cols_for_body(kdf_src, *gd["kinem_kw"])
                traces = [(kinem_ref, c, try_numeric(kdf_src[c])) for c in k_cols if c in kdf_src.columns]
                if pfk["acc"] != NONE and pfk["acc"] in aligned_src:
                    for c in _phone_xyz(pfk["acc"]):
                        traces.append((pfk["acc"], c, try_numeric(aligned_src[pfk["acc"]][c])))
                if pfk["gyr"] != NONE and pfk["gyr"] in aligned_src:
                    for c in _phone_xyz(pfk["gyr"]):
                        traces.append((pfk["gyr"], c, try_numeric(aligned_src[pfk["gyr"]][c])))
                out[gk] = traces
            return out

        if st.session_state.get("proc_data_cycles"):
            aligned_cycles, _, _ = get_aligned_data(
                st.session_state.proc_data_cycles, st.session_state.offsets, st.session_state.peak_ref, ref_file=kinem_ref,
            )
            group_traces_cycles = _rebuild_group_traces(aligned_cycles) if aligned_cycles else group_traces
            st.info("🔁 Usando o reprocessamento definido acima (independente da etapa 2).")
        else:
            group_traces_cycles = group_traces

        def _traces_by_family(gkey):
            """Agrupa os traços de um grupo (L5/Joelho) em 5 famílias: Deslocamento,
            Velocidade e Aceleração (Kinem, 3 eixos cada) + ACC e GYR (celular, 3 eixos)."""
            pf = phone_files[gkey]
            fam = {"Deslocamento": [], "Velocidade": [], "Aceleração": [], "ACC (celular)": [], "GYR (celular)": []}
            for fname, colname, y in group_traces_cycles[gkey]:
                if fname == kinem_ref:
                    cn = colname.lower()
                    if "v(" in cn:
                        fam["Velocidade"].append((colname, y))
                    elif "a(" in cn:
                        fam["Aceleração"].append((colname, y))
                    else:
                        fam["Deslocamento"].append((colname, y))
                elif fname == pf["acc"]:
                    fam["ACC (celular)"].append((colname, y))
                elif fname == pf["gyr"]:
                    fam["GYR (celular)"].append((colname, y))
            return fam

        def _square_phase_chart(title, x, traces, phase_regions):
            fig = go.Figure()
            for label, y in traces:
                fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=label, line=dict(width=1.6)))
            for x0, x1, phname in phase_regions:
                if x1 > x0:
                    fig.add_vrect(x0=x0, x1=x1, fillcolor=PHASE_COLORS[phname], line_width=0)
            fig.update_layout(
                title=dict(text=title, font_size=12),
                xaxis=dict(title="Tempo (s)"), yaxis_title="",
                width=380, height=380, margin=dict(t=38, b=34, l=48, r=10),
                template="plotly_white", hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font_size=9),
            )
            return fig

        for c in range(1, n_ciclos + 1):
            seg_base = (c - 1) * 3
            c_start, b1, b2, c_end = (
                boundaries_full[seg_base], boundaries_full[seg_base + 1],
                boundaries_full[seg_base + 2], boundaries_full[seg_base + 3],
            )
            phase_regions = [(c_start, b1, "Preparação"), (b1, b2, "Descida"), (b2, c_end, "Subida")]
            mask_c = (x_axis >= c_start) & (x_axis <= c_end)
            x_c = x_axis[mask_c]

            with st.expander(
                f"{_cycle_label(c)}   ·   {c_start:+.2f} s → {c_end:+.2f} s", expanded=(c == 1),
            ):
                for gkey in GROUPS:
                    st.markdown(f"#### {GROUPS[gkey]['emoji']} {GROUPS[gkey]['label']}")
                    fam = _traces_by_family(gkey)
                    families_present = [(name, traces) for name, traces in fam.items() if traces]
                    fam_cols = st.columns(3)
                    for idx, (fam_name, traces) in enumerate(families_present):
                        sliced = [(label, y[mask_c]) for label, y in traces]
                        fig_f = _square_phase_chart(
                            f"{gkey.upper()} · {fam_name}", x_c, sliced, phase_regions,
                        )
                        with fam_cols[idx % 3]:
                            st.plotly_chart(fig_f, use_container_width=False)

        _step_nav(back_to=5, next_to=7, next_label="Avançar para exportação ▶", key_suffix="6")

    if st.session_state.wizard_step >= 7:
        # ══════════════════════════════════════
        # Exportar Excel — sinais BRUTOS, apenas janela selecionada
        # ══════════════════════════════════════
        st.subheader("📥 Exportar Excel")
        st.caption(
            f"Exporta todos os eixos X, Y, Z **sem detrend/filtro** (dados brutos reamostrados e "
            f"sincronizados), com colunas **Ciclo**, **Direção** e **Fase** (Preparação/Descida/Subida) • janela: "
            f"**{view_start:+.1f} s → {view_end:+.1f} s** relativo ao pico"
        )

        if st.button("Gerar arquivo Excel (L5 + Joelho — dados brutos)", use_container_width=True):
            # Realinha os dados BRUTOS (sem detrend/lowpass) com os mesmos offsets/pico já
            # calculados na sincronização — garante que a exportação nunca carrega filtragem,
            # independentemente do que estiver configurado na etapa de Processamento acima.
            aligned_raw_export, x_samp_raw, align_msg_raw = get_aligned_data(
                st.session_state.raw_synced, st.session_state.offsets, st.session_state.peak_ref, ref_file=kinem_ref,
            )

            if aligned_raw_export is None:
                st.error(align_msg_raw)
            else:
                x_axis_raw = x_samp_raw / pfs
                mask_exp = (x_axis_raw >= view_start) & (x_axis_raw <= view_end)
                win_idx = np.where(mask_exp)[0]

                if len(win_idx) == 0:
                    st.error("Janela vazia — ajuste os limites de início/fim.")
                else:
                    windowed = {fname: df.iloc[win_idx].reset_index(drop=True) for fname, df in aligned_raw_export.items()}
                    t_w = np.arange(len(win_idx)) / pfs

                    # Rotula cada amostra com Ciclo (1..N), Direção (quando N=3, no
                    # padrão do Y-Balance Test) e Fase (Preparação/Descida/Subida)
                    # definidos na etapa anterior. Amostras após o fechamento do
                    # último ciclo (fora de qualquer fase) ficam em branco.
                    x_win_raw = x_axis_raw[win_idx]
                    seg_idx_raw = np.searchsorted(boundaries_full, x_win_raw, side="right") - 1
                    in_cycle = (seg_idx_raw >= 0) & (seg_idx_raw < 3 * n_ciclos)
                    seg_idx_safe = np.clip(seg_idx_raw, 0, 3 * n_ciclos - 1)

                    fase_col = np.where(in_cycle, np.array(PHASE_NAMES)[seg_idx_safe % 3], "")
                    ciclo_num = seg_idx_safe // 3 + 1
                    ciclo_col = np.where(in_cycle, ciclo_num.astype(object), None)

                    DIRECTION_NAMES_EXP = ["Anterior", "Posteromedial", "Posterolateral"]
                    if n_ciclos == 3:
                        dir_col = np.where(
                            in_cycle, np.array(DIRECTION_NAMES_EXP)[np.clip(ciclo_num - 1, 0, 2)], "",
                        )
                    else:
                        dir_col = np.full(len(x_win_raw), "", dtype=object)

                    sheets = {}
                    for gkey, gdef in GROUPS.items():
                        pf = phone_files[gkey]
                        sheet = build_export_sheet(
                            windowed, kinem_ref, pf["acc"], pf["gyr"], gdef["kinem_kw"], t_w,
                        )
                        sheet.insert(1, "Ciclo", ciclo_col[:len(sheet)])
                        sheet.insert(2, "Direção", dir_col[:len(sheet)])
                        sheet.insert(3, "Fase", fase_col[:len(sheet)])
                        sheets[gdef["label"]] = sheet

                    buf = io.BytesIO()
                    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                        for sheet_name, df_sheet in sheets.items():
                            df_sheet.to_excel(writer, sheet_name=sheet_name, index=False)
                    buf.seek(0)

                    st.download_button(
                        "⬇ Baixar sinais_brutos_ytest.xlsx", buf, file_name="sinais_brutos_ytest.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                    )

        _step_nav(back_to=6, key_suffix="7")
