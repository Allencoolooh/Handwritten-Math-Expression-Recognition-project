# app.py
import streamlit as st
from PIL import Image

from inference import predict_latex_from_pil

st.set_page_config(page_title="Handwritten Math Expression Recognition", layout="centered")

st.title("✏️ 手写数学公式识别 Demo")
st.write("上传一张手写数学公式图片，我帮你识别并渲染成直观公式。")

uploaded = st.file_uploader("请选择一张图片文件", type=["png", "jpg", "jpeg"])

use_beam = st.checkbox("使用 Beam Search 解码（更准确但更慢）", value=False)

if uploaded is not None:
    img = Image.open(uploaded)
    st.image(img, caption="上传的图片", use_column_width=True)

    if st.button("开始识别"):
        with st.spinner("识别中，请稍候..."):
            latex = predict_latex_from_pil(img, use_beam=use_beam)

        if latex.strip() == "":
            st.error("没有识别出内容，可能是模型或图片有问题。")
        else:
            st.success("识别完成！")

            st.subheader("LaTeX 代码：")
            st.code(latex, language="latex")

            st.subheader("渲染后的公式：")
            # ✅ 这里就是把 LaTeX 代码转换成直观公式的关键
            st.latex(latex)
