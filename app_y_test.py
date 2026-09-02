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
  • Sem ângulo de flexão do joelho via Kinem (exigiria 3 marcadores:
    quadril, joelho e tornozelo — não disponíveis neste protocolo).
    Em vez disso, oferece um ângulo relativo tronco(L5)–perna(Joelho)
    exploratório, via filtro complementar ACC+GYR de cada celular —
    útil como indicador de oscilação postural durante o alcance do Y-BT,
    não é um ângulo clínico de flexão de joelho.
  • Sincronização continua pelo pico de impacto do salto/calibração
    inicial (mesma lógica de find_highest_peak / find_sync_xcorr).

Reaproveita 100% o signal_utils.py original (funções genéricas,
parametrizadas por keywords — nada precisou mudar lá).
"""

import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from signal_utils import (
    NONE_LABEL,
    apply_detrend,
    apply_lowpass,
    best_match,
    build_export_sheet,
    col_default,
    complementary_angle,
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
    "offsets": {},
    "peak_ref": None,
    "target_fs": 100,
    "fs_info": {},
    "show_preview": False,
    "synced": False,
    "synced_kinem_cols": {},
}
for key, default in DEFAULT_SESSION_STATE.items():
    st.session_state.setdefault(key, default)


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
        cf_alpha = st.slider(
            "Filtro complementar (ângulo relativo) — peso do giroscópio", 0.80, 0.999,
            value=0.98, step=0.005,
            help="Mais próximo de 1 = confia mais no giroscópio (menos deriva do acelerômetro).",
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

    # ══════════════════════════════════════════
    # Processamento inline
    # ══════════════════════════════════════════
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
                proc[fname] = r
            st.session_state.proc_data = proc
            st.session_state.proc_data_nofilter = proc_nofilter
            st.rerun()


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

    st.divider()

    # ══════════════════════════════════════════
    # Seleção de janela
    # ══════════════════════════════════════════
    st.subheader("🪟 Seleção de janela")
    wc1, wc2 = st.columns(2)
    with wc1:
        view_start = st.number_input(
            "Início (s) relativo ao pico", value=float(max(x_min_data, -2.0)), step=0.5, key="view_start",
        )
    with wc2:
        view_end = st.number_input(
            "Fim (s) relativo ao pico", value=float(min(x_max_data, 8.0)), step=0.5, key="view_end",
        )

    st.divider()

    # ══════════════════════════════════════════
    # Ângulo relativo Tronco (L5) × Joelho — exploratório
    # ══════════════════════════════════════════
    st.subheader("🧍 Ângulo relativo Tronco (L5) × Joelho")
    st.caption(
        "Estimativa exploratória via filtro complementar (ACC+GYR) da diferença de "
        "inclinação sagital entre o segmento do L5 (tronco/lombar) e o segmento do "
        "Joelho. **Não é o ângulo clínico de flexão do joelho** — esse exigiria um "
        "terceiro marcador (quadril/tornozelo), ausente no protocolo do Y-BT. Serve "
        "como indicador relativo de oscilação/compensação postural durante o alcance."
    )

    aligned_raw, _, _ = get_aligned_data(
        st.session_state.raw_synced, st.session_state.offsets, st.session_state.peak_ref, ref_file=kinem_ref,
    )

    angle_rel = None
    pf_l5, pf_joelho = phone_files["l5"], phone_files["joelho"]
    if aligned_raw and all(pf_l5[k] != NONE for k in ("acc", "gyr")) and all(pf_joelho[k] != NONE for k in ("acc", "gyr")):
        needed = [pf_l5["acc"], pf_l5["gyr"], pf_joelho["acc"], pf_joelho["gyr"]]
        if all(f in aligned_raw for f in needed):
            ang_l5 = complementary_angle(
                aligned_raw[pf_l5["acc"]], aligned_raw[pf_l5["gyr"]], pfs, role="l5", alpha=cf_alpha,
            )
            ang_joelho = complementary_angle(
                aligned_raw[pf_joelho["acc"]], aligned_raw[pf_joelho["gyr"]], pfs, role="limb", alpha=cf_alpha,
            )
            if ang_l5 is not None and ang_joelho is not None:
                n = min(len(ang_l5), len(ang_joelho))
                angle_rel = ang_l5[:n] - ang_joelho[:n]

    if angle_rel is None:
        st.info("Selecione ACC + GYR de L5 e Joelho (celular) na barra lateral para calcular o ângulo relativo.")
    else:
        mask_ang = (x_axis >= view_start) & (x_axis <= view_end)
        n = min(len(angle_rel), len(x_axis))
        y_r = angle_rel[:n]
        m = mask_ang[:n]
        fig_ang = go.Figure()
        fig_ang.add_trace(go.Scatter(
            x=x_axis[:n][m], y=y_r[m], mode="lines",
            line=dict(color="purple", width=2), name="Ângulo relativo L5 − Joelho",
        ))
        fig_ang.add_vline(x=0, line_dash="dash", line_color="gray", annotation_text="salto")
        fig_ang.update_layout(
            xaxis=dict(title="Tempo (s)  —  0 = pico do salto", range=[view_start, view_end]),
            yaxis_title="Ângulo (graus)", height=420, template="plotly_white", hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0), margin=dict(t=30, b=40),
        )
        st.plotly_chart(fig_ang, use_container_width=True)

    st.divider()

    # ══════════════════════════════════════════
    # Check de qualidade
    # ══════════════════════════════════════════
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

    show_qa = st.checkbox("🔍 Checar qualidade dos dados", value=False)
    if show_qa:
        qa_xmin, qa_xmax = view_start, view_end
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

    st.divider()

    # ══════════════════════════════════════════
    # Exportar Excel — apenas janela selecionada
    # ══════════════════════════════════════════
    st.subheader("📥 Exportar Excel")
    st.caption(f"Exporta todos os eixos X, Y, Z + ângulo relativo • janela: **{view_start:+.1f} s → {view_end:+.1f} s** relativo ao pico")

    if st.button("Gerar arquivo Excel (L5 + Joelho + Ângulo relativo)", use_container_width=True):
        mask_exp = (x_axis >= view_start) & (x_axis <= view_end)
        win_idx = np.where(mask_exp)[0]

        if len(win_idx) == 0:
            st.error("Janela vazia — ajuste os limites de início/fim.")
        else:
            windowed = {fname: df.iloc[win_idx].reset_index(drop=True) for fname, df in aligned_data.items()}
            t_w = np.arange(len(win_idx)) / pfs

            sheets = {}
            for gkey, gdef in GROUPS.items():
                pf = phone_files[gkey]
                sheets[gdef["label"]] = build_export_sheet(
                    windowed, kinem_ref, pf["acc"], pf["gyr"], gdef["kinem_kw"], t_w,
                )

            df_angle = pd.DataFrame({"Tempo (s)": t_w})
            if angle_rel is not None:
                valid_idx = win_idx[win_idx < len(angle_rel)]
                y_r = np.full(len(win_idx), np.nan)
                y_r[:len(valid_idx)] = angle_rel[valid_idx]
                df_angle["Angulo_Relativo_L5_Joelho_graus"] = y_r

            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                for sheet_name, df_sheet in sheets.items():
                    df_sheet.to_excel(writer, sheet_name=sheet_name, index=False)
                df_angle.to_excel(writer, sheet_name="Angulo_Relativo", index=False)
            buf.seek(0)

            st.download_button(
                "⬇ Baixar sinais_sincronizados_ytest.xlsx", buf, file_name="sinais_sincronizados_ytest.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
