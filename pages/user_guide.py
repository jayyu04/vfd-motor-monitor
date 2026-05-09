import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="使用說明 — VFD Motor Monitoring", layout="wide")

# 隱藏 Streamlit 預設的頁面元素
st.markdown("""
<style>
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }
.block-container { padding: 0 !important; }
</style>
""", unsafe_allow_html=True)

with open("static/user_guide.html", "r", encoding="utf-8") as f:
    html = f.read()

components.html(html, height=4000, scrolling=True)

