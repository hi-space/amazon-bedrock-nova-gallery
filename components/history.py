import json
import streamlit as st
from typing import List
from genai_kit.utils.images import base64_to_image
from enums import MediaType
from utils import format_datetime


def show_history(media_items: List[dict] = [], cols_per_row: int = 1, show_details: bool = False):
    st.title("📋 Request History")

    if media_items and len(media_items) > 0:
        cols = st.columns(cols_per_row)
        
        for idx, item in enumerate(media_items):
            with cols[idx % cols_per_row]:
                icon = _get_emoji(item['media_type'])
                with st.expander(
                    f"**{item['media_type']}** - **{item.get('id', '')}** - {format_datetime(item['created_at'])}",
                    expanded=show_details,
                    icon=icon,
                ):
                    display_history_item(item)
    else:
        st.info("아직 요청 기록이 없습니다.")


def display_history_item(item):
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown("**ID:**")
        st.markdown("**요청 시간:**")
        st.markdown("**요청 유형:**")
        st.markdown("**모델 타입:**")
        st.markdown("**프롬프트:**")
    
    with col2:
        media_type = item['media_type']
        st.text(item['id'])
        st.text(format_datetime(item['created_at'], seconds=True))
        st.text(media_type)
        st.text(item['model_type'])

        ref_image = item.get('ref_image', None)
        if ref_image:            
            st.image(base64_to_image(ref_image),width=400)

        st.code(item.get('prompt', ''), wrap_lines=True, language='txt')

        st.divider()
        
        url = item.get('url', '')
        if url and media_type == MediaType.IMAGE.value:
            st.image(url)
        elif url and media_type == MediaType.VIDEO.value:
            st.video(url)
                
        st.json(json.loads(json.dumps(item['details'], default=float)))


def _get_emoji(media_type: str):
    if media_type == MediaType.IMAGE.value:
        return '🎨'
    elif media_type == MediaType.VIDEO.value:
        return '🎥'
    return '❓'
