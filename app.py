'''
import streamlit as st
from PIL import Image

from inference import predict_latex_from_pil


st.set_page_config(
    page_title="Handwritten Math Expression Recognition",
    layout="centered",
)

st.title("✏️ 手写数学公式识别 Demo")
st.write("上传一张手写数学公式图片，我会帮你识别成 **LaTeX 代码** 并渲染成直观公式。")

uploaded = st.file_uploader("请选择一张图片文件", type=["png", "jpg", "jpeg"])

col1, col2 = st.columns(2)
with col1:
    decode_method = st.radio(
        "解码方式",
        options=["beam", "greedy"],
        index=0,
        help="Beam Search 一般更准确，但会稍慢一些。",
    )
with col2:
    beam_size = st.slider(
        "Beam size（仅在 Beam 模式下生效）",
        min_value=2,
        max_value=7,
        value=3,
        step=1,
    )

max_len = st.number_input(
    "最大解码长度 max_len",
    min_value=32,
    max_value=512,
    value=128,
    step=16,
    help="可以用来控制生成公式的最长长度，过长时可以适当减小。",
)

if uploaded is not None:
    img = Image.open(uploaded)
    st.image(img, caption="上传的图片", use_column_width=True)

    if st.button("开始识别"):
        with st.spinner("识别中，请稍候..."):
            latex = predict_latex_from_pil(
                img,
                decode_method=decode_method,
                beam_size=beam_size,
                max_len=max_len,
            )

        if not latex.strip():
            st.error("识别结果为空，可能是模型、图片或权重有问题。")
        else:
            st.success("识别完成！")

            st.subheader("LaTeX 代码：")
            st.code(latex, language="latex")

            st.subheader("渲染后的公式：")
            # ✅ 这里就是“把 LaTeX 转成直观公式”的关键：
            st.latex(latex)
'''

import streamlit as st
from PIL import Image
from datetime import datetime

from inference import predict_latex_from_pil

st.set_page_config(
    page_title="Handwritten Math Expression Recognition",
    layout="centered",
)

# ---------------------- 新增：网页最上端署名 ----------------------
st.markdown("""
<div style='text-align: center; font-size: 24px; font-weight: bold; margin-bottom: 10px;'>
    制作者：Allen
</div>
""", unsafe_allow_html=True)
# ---------------------------------------------------------------

st.title("✏️ 手写数学公式识别 Demo")
st.write("上传一张手写数学公式图片，我会帮你识别成 **LaTeX 代码** 并渲染成直观公式。")

# ---------------------- 初始化历史记录 ----------------------
if "history" not in st.session_state:
    st.session_state["history"] = []


uploaded = st.file_uploader("请选择一张图片文件", type=["png", "jpg", "jpeg"])

col1, col2 = st.columns(2)
with col1:
    decode_method = st.radio(
        "解码方式",
        options=["beam", "greedy"],
        index=0,
        help="Beam Search 一般更准确，但会稍慢一些。",
    )
with col2:
    beam_size = st.slider(
        "Beam size（仅在 Beam 模式下生效）",
        min_value=2,
        max_value=7,
        value=3,
        step=1,
    )

max_len = st.number_input(
    "最大解码长度 max_len",
    min_value=32,
    max_value=512,
    value=128,
    step=16,
    help="用于控制生成公式的最大长度（防止无限生成）。",
)

current_result = None


# ---------------------- 主识别逻辑 ----------------------
if uploaded is not None:
    img = Image.open(uploaded)
    st.image(img, caption="上传的图片", use_column_width=True)

    if st.button("开始识别"):
        with st.spinner("识别中，请稍候..."):
            latex = predict_latex_from_pil(
                img,
                decode_method=decode_method,
                beam_size=beam_size,
                max_len=max_len,
            )

        if not latex.strip():
            st.error("识别结果为空，可能是模型、图片或权重有问题。")
        else:
            st.success("识别完成！")

            # -------------- 本次结果区域 --------------
            st.subheader("本次识别结果")

            c1, c2 = st.columns([1, 1])

            with c1:
                st.markdown("**原图：**")
                st.image(img, use_column_width=True)

            with c2:
                st.markdown("**渲染后的公式：**")
                st.latex(latex)

                st.markdown("**LaTeX 代码：**")
                st.code(latex, language="latex")

                st.download_button(
                    label="💾 下载 LaTeX 代码（.tex）",
                    data=latex,
                    file_name="formula.tex",
                    mime="text/plain",
                    key="download_current_latex",
                )

            # 添加到历史记录
            rec = {
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "image": img.copy(),
                "latex": latex,
                "decode_method": decode_method,
                "beam_size": beam_size,
                "max_len": max_len,
            }
            st.session_state.history.append(rec)
            current_result = rec


# ---------------------- 历史识别记录 ----------------------
if st.session_state.history:
    st.markdown("---")
    st.subheader("📜 历史识别记录")

    for idx, rec in enumerate(reversed(st.session_state.history)):
        hist_index = len(st.session_state.history) - 1 - idx

        with st.expander(f"[{rec['time']}] 记录 #{hist_index + 1}"):
            h1, h2 = st.columns([1, 1])

            with h1:
                st.markdown("**原图：**")
                st.image(rec["image"], use_column_width=True)

            with h2:
                st.markdown(
                    f"**解码方式：** {rec['decode_method']}  "
                    f"(beam_size={rec['beam_size']}, max_len={rec['max_len']})"
                )

                st.markdown("**渲染后的公式：**")
                st.latex(rec["latex"])

                st.markdown("**LaTeX 代码：**")
                st.code(rec["latex"], language="latex")

                st.download_button(
                    label="💾 下载该条 LaTeX（.tex）",
                    data=rec["latex"],
                    file_name=f"formula_{hist_index + 1}.tex",
                    mime="text/plain",
                    key=f"download_hist_{hist_index}",
                )

    if st.button("🧹 清空历史记录"):
        st.session_state.history = []
        st.experimental_rerun()
