import json
import streamlit as st
from typing import List, Dict, Any
from enums import MediaType
from utils import format_datetime


def show_gallery(media_items: List[dict] = [], cols_per_row: int = 3, show_details: bool = False):
    st.title("🖼️ GenAI Gallery")

    if media_items and len(media_items) > 0:
        display_media_grid(media_items, cols_per_row, show_details)
    else:
        st.info("표시할 미디어가 없습니다.")

def display_media_grid(media_items: List[Dict[str, Any]], cols_per_row: int, show_details: bool):
    cols = st.columns(cols_per_row)
    
    for idx, item in enumerate(media_items):
        col_idx = idx % cols_per_row
        
        with cols[col_idx]:
            display_media_item(item, show_details)

def display_media_item(item: Dict[str, Any], show_details: bool):
    container = st.container()
    
    media_type = item.get('media_type', '')
    url = item.get('url', '')
    
    if url and media_type == MediaType.IMAGE.value:
        container.image(url)
    elif url and media_type == MediaType.VIDEO.value:
        container.video(url)
    
    if show_details:
        with container.expander(f"**{item.get('id', '')}** - {format_datetime(item.get('created_at', ''), seconds=False)}", expanded=False):
            st.code(item.get('prompt', ''), wrap_lines=True, language='txt')
            st.markdown(f"**ID:** {item.get('id', '')}")
            st.markdown(f"**모델:** {item.get('model_type', '')}")
            if details := item.get('details'):
                st.markdown("**상세 정보:**")
                st.json(json.loads(json.dumps(details, default=float)), expanded=False)
    else:
        container.caption(f"_{item.get('prompt', '')}_")
