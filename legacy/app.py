"""
Electricity Theft Risk Dashboard (Streamlit)
============================================
Run:   streamlit run app.py
Needs: theft_core.py and train_compare.py in the same folder, plus the outputs of
       train_compare.py (results/) and, optionally, data_cleaning.py (cleaned/).

Tabs
  1. Overview        - how well the risk levels work on the TEST set (never used for training)
  2. Risk ranking    - which meters to inspect first (filter + download)
  3. Meter detail    - usage chart + WHY the meter got its score (XGBoost SHAP values)
  4. Upload new data - score new customers in the same format as SGCC data.csv
"""
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "ml"))
import theft_core as core  # noqa: E402

st.set_page_config(page_title="Electricity Theft Risk", page_icon="⚡", layout="wide")


@st.cache_resource(show_spinner="กำลังโหลดโมเดลและผลลัพธ์...")
def get_artifacts(results_dir):
    return core.load_artifacts(results_dir)


@st.cache_resource(show_spinner="กำลังโหลดข้อมูลการใช้ไฟ...")
def get_series(cleaned_dir):
    return core.load_clean_series(cleaned_dir)


def contribution_chart(reasons):
    d = pd.DataFrame({"factor": reasons["ปัจจัย"], "contribution": reasons["contribution"],
                      "direction": reasons["ผลต่อความเสี่ยง"]})
    return alt.Chart(d).mark_bar().encode(
        x=alt.X("contribution:Q", title="ผลต่อคะแนนความเสี่ยง (log-odds)  |  ขวา = เพิ่มความเสี่ยง, ซ้าย = ลดความเสี่ยง"),
        y=alt.Y("factor:N", sort=None, title=None),
        color=alt.Color("direction:N", legend=None,
                        scale=alt.Scale(domain=["เพิ่มขึ้น", "ลดลง"], range=["#d62728", "#1f77b4"])),
        tooltip=["factor", "contribution"]).properties(height=280)


def show_meter(meter_id, values, dates, feats, contrib, pop_median, note=""):
    """Usage chart + explanation for one meter."""
    left, right = st.columns([3, 2])
    with left:
        st.markdown("**การใช้ไฟรายวัน**")
        if values is not None:
            st.line_chart(core.series_frame(values, dates))
            st.caption("ข้อมูลหลังทำความสะอาด (ช่วงที่ขาดนานถูกเติมเป็น 0) " + note)
        else:
            st.info("ไม่พบไฟล์ข้อมูลที่ล้างแล้ว (cleaned/) จึงยังไม่แสดงกราฟการใช้ไฟ รัน data_cleaning.py ก่อน")
    with right:
        st.markdown("**ทำไมจึงได้คะแนนนี้**")
        if contrib is None:
            st.info("โมเดลนี้ไม่รองรับการอธิบายรายปัจจัย (ต้องใช้ XGBoost)")
        else:
            reasons = core.top_reasons(contrib.loc[meter_id], feats.loc[meter_id], pop_median, k=6)
            st.altair_chart(contribution_chart(reasons))
            st.dataframe(reasons.drop(columns="contribution").round(3), hide_index=True)


# ---------------------------------------------------------------------------- sidebar
def clear_upload():
    st.session_state.pop("upload", None)


st.sidebar.header("⚙️ ตั้งค่า")
results_dir = st.sidebar.text_input("โฟลเดอร์ผลลัพธ์โมเดล", "models/experiments/results", on_change=clear_upload)
cleaned_dir = st.sidebar.text_input("โฟลเดอร์ข้อมูลที่ล้างแล้ว", "data/processed/cleaned")
st.sidebar.markdown("**เกณฑ์ระดับความเสี่ยง** (เปอร์เซ็นไทล์ของคะแนน)")
high = st.sidebar.slider("ระดับ 'สูง' ตั้งแต่", 50, 100, 95)
medium = st.sidebar.slider("ระดับ 'ปานกลาง' ตั้งแต่", 0, high, min(80, high))
st.sidebar.caption("คะแนนความเสี่ยง 0-100 = ตำแหน่งของมิเตอร์เทียบกับลูกค้าในชุดทดสอบ "
                   "(100 = เสี่ยงที่สุด) ใช้เกณฑ์เดียวกันกับข้อมูลใหม่ทุกขนาด")

try:
    art = get_artifacts(str(BASE / results_dir))
except FileNotFoundError as e:
    st.error(f"{e}\n\nรัน `python train_compare.py --data data.csv --out results` ก่อน แล้วเปิดแอปใหม่")
    st.stop()
series = get_series(str(BASE / cleaned_dir))

preds = art["preds"].copy()
preds["risk_score"] = core.risk_percentile(preds["score_XGBoost"].to_numpy(), art["ref_scores"])
preds["risk_level"] = core.risk_level(preds["risk_score"].to_numpy(), high, medium)
y_test = preds["y_true"].to_numpy()
pop_median = art["feats"].median()

st.title("⚡ ระบบวิเคราะห์ความเสี่ยงการลักลอบใช้ไฟฟ้า")
st.caption("ผลลัพธ์เป็นเครื่องมือ **คัดกรอง** เพื่อจัดลำดับการตรวจสอบ ไม่ใช่การตัดสินว่าผู้ใช้ไฟกระทำผิด "
           "ต้องให้เจ้าหน้าที่ตรวจสอบยืนยันก่อนดำเนินการใดๆ")

tab1, tab2, tab3, tab4 = st.tabs(["📊 ภาพรวม", "📋 จัดอันดับความเสี่ยง", "🔎 รายละเอียดมิเตอร์", "📤 อัปโหลดข้อมูลใหม่"])

# ---------------------------------------------------------------------------- 1. overview
with tab1:
    lt = core.level_table(preds["risk_score"].to_numpy(), y_test, high, medium)
    base_rate = y_test.mean() * 100
    top = lt.iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("มิเตอร์ที่วิเคราะห์ (ชุดทดสอบ)", f"{len(preds):,}")
    c2.metric("กลุ่มเสี่ยงสูง", f"{int(top['จำนวนมิเตอร์']):,} ราย")
    lift = top["อัตราที่เจอจริง (%)"] / base_rate if base_rate else 0
    c3.metric("เจอผู้ลักลอบใช้ไฟจริงในกลุ่มเสี่ยงสูง", f"{top['อัตราที่เจอจริง (%)']}%",
              delta=f"{lift:.1f}× เทียบกับการสุ่มตรวจ ({base_rate:.2f}%)")
    c4.metric("สัดส่วนผู้ลักลอบใช้ไฟที่ถูกจับได้", f"{top['สัดส่วนผู้โกงที่จับได้ (%)']}%",
              help="ผู้ลักลอบใช้ไฟทั้งหมดในชุดทดสอบ ที่อยู่ในกลุ่มเสี่ยงสูง")
    st.subheader("ผลในชุดทดสอบ แยกตามระดับความเสี่ยง")
    st.caption("ชุดทดสอบคือลูกค้าที่โมเดลไม่เคยเห็นตอนฝึก คอลัมน์ 'ผลจริง' มาจากป้ายกำกับในข้อมูล "
               "(ในการใช้งานจริงจะไม่มี ต้องรอผลการตรวจภาคสนาม)")
    st.dataframe(lt, hide_index=True)
    if art["metrics"] is not None:
        st.subheader("เปรียบเทียบโมเดล (ชุดทดสอบ)")
        st.dataframe(art["metrics"].round(3), hide_index=True)
    imgs = [(BASE / results_dir / n) for n in ("pr_curves.png", "precision_at_k.png")]
    cols = st.columns(2)
    for col, p in zip(cols, imgs):
        if p.exists():
            col.image(str(p))

# ---------------------------------------------------------------------------- 2. ranking
with tab2:
    st.subheader("มิเตอร์ที่ควรตรวจสอบก่อน")
    f1, f2, f3 = st.columns(3)
    min_score = f1.slider("คะแนนความเสี่ยงขั้นต่ำ", 0, 100, int(high))
    levels = f2.multiselect("ระดับความเสี่ยง", ["สูง", "ปานกลาง", "ต่ำ"], default=["สูง", "ปานกลาง"])
    top_n = f3.number_input("แสดงสูงสุด (ราย)", 10, 5000, 300, step=50)
    view = preds[(preds["risk_score"] >= min_score) & (preds["risk_level"].isin(levels))]
    view = view.sort_values("score_XGBoost", ascending=False).head(int(top_n))
    table = pd.DataFrame({
        "อันดับ": np.arange(1, len(view) + 1), "รหัสผู้ใช้ไฟ": view["CONS_NO"].to_numpy(),
        "คะแนนเสี่ยง": view["risk_score"].round(1).to_numpy(), "ระดับ": view["risk_level"].to_numpy(),
        "ผลจริงในข้อมูล": np.where(view["y_true"].to_numpy() == 1, "ลักลอบใช้ไฟ", "ปกติ")})
    st.write(f"พบ {len(table):,} ราย")
    st.dataframe(table, hide_index=True,
                 column_config={"คะแนนเสี่ยง": st.column_config.ProgressColumn("คะแนนเสี่ยง", min_value=0, max_value=100,
                                                                              format="%.1f")})
    st.download_button("⬇️ ดาวน์โหลด CSV", table.to_csv(index=False).encode("utf-8-sig"), "risk_ranking.csv", "text/csv")

# ---------------------------------------------------------------------------- 3. detail
with tab3:
    ordered = preds.sort_values("score_XGBoost", ascending=False)
    score_of = dict(zip(ordered["CONS_NO"], ordered["risk_score"]))
    sel = st.selectbox("เลือกมิเตอร์ (เรียงจากเสี่ยงสูงสุด)", ordered["CONS_NO"].tolist(),
                       format_func=lambda i: f"{i[:14]}…  |  คะแนน {score_of[i]:.1f}")
    row = preds.set_index("CONS_NO").loc[sel]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("คะแนนความเสี่ยง", f"{row['risk_score']:.1f} / 100")
    m2.metric("ระดับ", row["risk_level"])
    m3.metric("ข้อมูลขาดหาย", f"{art['feats'].loc[sel, 'missing_ratio'] * 100:.0f}%")
    m4.metric("ผลจริงในข้อมูล", "ลักลอบใช้ไฟ" if row["y_true"] == 1 else "ปกติ")
    values, dates = None, None
    if series is not None and sel in series["index"]:
        values, dates = series["values"][series["index"][sel]], series["dates"]
    contrib = core.explain(art["model"], art["feats"].loc[[sel]])
    show_meter(sel, values, dates, art["feats"], contrib, pop_median)

# ---------------------------------------------------------------------------- 4. upload
with tab4:
    st.subheader("วิเคราะห์ข้อมูลใหม่")
    st.write("อัปโหลดไฟล์ CSV รูปแบบเดียวกับ SGCC: คอลัมน์ `CONS_NO` ตามด้วยคอลัมน์การใช้ไฟรายวัน "
             "(ชื่อคอลัมน์เป็นวันที่ เช่น 2016/1/1) อย่างน้อย 365 วัน คอลัมน์ `FLAG` ใส่หรือไม่ใส่ก็ได้")
    up = st.file_uploader("เลือกไฟล์ CSV", type=["csv"], on_change=clear_upload)
    if up is not None and st.button("🔍 วิเคราะห์ความเสี่ยง", type="primary"):
        try:
            with st.spinner("กำลังทำความสะอาดข้อมูล สร้าง feature และคำนวณคะแนน..."):
                raw_df = pd.read_csv(up, dtype={"CONS_NO": str})
                prep = core.prepare_upload(raw_df, art["features"])
                scored = core.score_features(art["model"], prep["feats"], art["ref_scores"], high, medium)
                st.session_state["upload"] = {"prep": prep, "scored": scored,
                                              "contrib": core.explain(art["model"], prep["feats"])}
        except ValueError as e:
            st.error(str(e))
            st.session_state.pop("upload", None)

    res = st.session_state.get("upload")
    if res:
        prep, scored, contrib = res["prep"], res["scored"], res["contrib"]
        scored = scored.copy()
        scored["risk_level"] = core.risk_level(scored["risk_score"].to_numpy(), high, medium)
        if prep["absent"]:
            st.warning("ไม่สามารถคำนวณ feature บางตัวได้ (ตั้งเป็น 0): " + ", ".join(prep["absent"]) +
                       "  (มักเกิดจากชื่อคอลัมน์ไม่ใช่วันที่)")
        counts = scored["risk_level"].value_counts()
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("มิเตอร์ทั้งหมด", f"{len(scored):,}")
        k2.metric("ความเสี่ยงสูง", f"{counts.get('สูง', 0):,}")
        k3.metric("ปานกลาง", f"{counts.get('ปานกลาง', 0):,}")
        k4.metric("ต่ำ", f"{counts.get('ต่ำ', 0):,}")
        ranked = scored.sort_values("probability", ascending=False).reset_index(drop=True)
        out = pd.DataFrame({"อันดับ": np.arange(1, len(ranked) + 1), "รหัสผู้ใช้ไฟ": ranked["CONS_NO"],
                            "คะแนนเสี่ยง": ranked["risk_score"].round(1), "ระดับ": ranked["risk_level"]})
        if prep["y"] is not None:
            truth = pd.Series(prep["y"], index=prep["feats"].index)
            out["ผลจริงในไฟล์"] = np.where(truth.loc[ranked["CONS_NO"]].to_numpy() == 1, "ลักลอบใช้ไฟ", "ปกติ")
        st.dataframe(out, hide_index=True,
                     column_config={"คะแนนเสี่ยง": st.column_config.ProgressColumn("คะแนนเสี่ยง", min_value=0,
                                                                                  max_value=100, format="%.1f")})
        st.download_button("⬇️ ดาวน์โหลดผลการวิเคราะห์ (CSV)", out.to_csv(index=False).encode("utf-8-sig"),
                           "upload_risk_ranking.csv", "text/csv")
        st.divider()
        pick = st.selectbox("ดูรายละเอียดมิเตอร์", ranked["CONS_NO"].tolist(), key="upload_pick",
                            format_func=lambda i: f"{i[:14]}…")
        pos = prep["feats"].index.get_loc(pick)
        show_meter(pick, prep["clean"][pos], prep["dates"], prep["feats"], contrib, pop_median,
                   note="(ข้อมูลที่อัปโหลด)")
