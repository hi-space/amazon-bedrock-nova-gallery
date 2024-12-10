import streamlit as st

def load_styles():
    st.markdown("""
        <style>
            /* 이미지 스타일링 */
            .stImage {
                border-radius: 8px !important;
                transition: transform 0.2s !important;
                cursor: pointer !important;
            }
            .stImage:hover {
                transform: scale(1.03) !important;
            }
            
            /* 이미지 캡션 스타일링 */
            .stImage img {
                border-radius: 8px !important;
            }
                
            video {
                border-radius: 8px !important;
            }
            
            /* 히스토리 카드 스타일링 */
            div[data-testid="stExpander"] {
                background-color: #f8f9fa;
                border-radius: 8px;
                margin-bottom: 1rem;
                box-shadow: 0 1px 3px rgba(0,0,0,0.12);
            }
                
            /* Code 블록 스타일링 */
            div[data-testid="stCode"] > pre {
                background-color: white !important;
                padding: 1rem !important;
                margin: 0 !important;
                border-radius: 1rem;
            }
        </style>
    """, unsafe_allow_html=True)