import streamlit as st
from PIL import Image
from genai_kit.aws.bedrock import BedrockModel
from genai_kit.utils.images import encode_image_base64, resize_image
from services.bedrock_service import (
    gen_english,
    gen_mm_video_prompt,
    gen_video,
    get_video_job,
)
from session import SessionManager
from enums import MediaType


def show_video_generator(session_manager: SessionManager):
    st.title("🎥 Video Generator")
    initialize_video_session_state()
    
    vcol1, vcol2, vcol3 = st.columns(3)
    
    with vcol1:
        show_video_prompt_input_section()
    
    with vcol2:
        show_video_prompt_edit_section()
    
    with vcol3:
        generate_clicked = show_video_model_section()

    if generate_clicked:
        generate_video(session_manager)

def initialize_video_session_state():
    if 'video_generation_prompt' not in st.session_state:
        st.session_state.video_generation_prompt = ""
    if 'video_generation_image' not in st.session_state:
        st.session_state.video_generation_image = None
    if 'video_model_type' not in st.session_state:
        st.session_state.video_model_type = ""
    if 'video_generation_configs' not in st.session_state:
        st.session_state.video_generation_configs = None

def show_video_prompt_input_section():
    st.subheader("Generate a Video Prompt")
    video_prompt_type = st.selectbox(
        "Choose an option:",
        ["Basic Video Prompt", "Augmented Video Prompt"],
        key="video_prompt_type_select"
    )

    with st.expander("LLM Configuration"):
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7, key="video_temperature")
        top_p = st.slider("Top P", 0.0, 1.0, 0.9, key="video_top_p")
        top_k = st.slider("Top K", 0, 500, 250, 5, key="video_top_k")
    
    if video_prompt_type == "Basic Video Prompt":
        prompt_text = st.text_area("Enter your prompt:", height=150, key="video_prompt_text")
        if st.button("Generate Video Prompt", icon='📝', type="primary", use_container_width=True, key="video_gen_prompt_btn"):
            with st.spinner("Generating video prompt..."):
                st.session_state.video_generation_prompt = gen_english(
                    request=prompt_text,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k
                )
    else:  # Augmented Video Prompt
        multimodal_keyword_text = st.text_area("Enter the keyword:", height=100, key="video_keyword_text")
        reference_image = st.file_uploader("Upload a reference image:", type=['png', 'jpg', 'jpeg'], key="video_reference_upload")
        
        if reference_image:
            st.image(reference_image)
            
        if st.button("Generate Video Prompt", icon='📝', type="primary", use_container_width=True, key="video_gen_mm_prompt_btn"):
            with st.spinner("Generating video prompt..."):
                image = None
                if reference_image:
                    img = Image.open(reference_image)
                    resized_img = resize_image(img, width=1280, height=720)

                    with st.expander("Resized Image", expanded=False):
                        st.image(resized_img)

                    image = encode_image_base64(resized_img)
                    st.session_state.video_generation_image = image
                st.session_state.video_generation_prompt = gen_mm_video_prompt(
                    keyword=multimodal_keyword_text,
                    image=image,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k
                )

def show_video_prompt_edit_section():
    st.subheader("Video Prompt")
    prompt_value = st.text_area(
        label="Edit or modify the prompt",
        value=st.session_state.video_generation_prompt,
        height=400,
        key="video_prompt_edit"
    )

    if prompt_value != st.session_state.video_generation_prompt:
        st.session_state.video_generation_prompt = prompt_value

def show_video_model_section():
    st.subheader("Select Video Model")
    model_type = st.selectbox(
        "Choose a model:",
        [BedrockModel.NOVA_REAL.value],
        key="video_model_select"
    )

    st.session_state.video_model_type = model_type

    with st.expander("Video Configuration", expanded=True):
        configs = _get_video_model_configurations()
        st.session_state.video_generation_configs = configs
    
    return st.button("Generate Videos", icon='🎥', type="primary", use_container_width=True, key="video_generate_btn")

def generate_video(session_manager: SessionManager):
    st.divider()
    st.subheader("Generated Videos")
    
    with st.status("Generating videos...", expanded=True) as status:
        try:
            configs = st.session_state.video_generation_configs
                        
            invocation_arn = gen_video(
                text=st.session_state.video_generation_prompt,
                image=st.session_state.video_generation_image,
                params=configs
            )
            
            with st.spinner("Generating video..."):
                invocation = get_video_job(invocation_arn)
                st.json(invocation)
                        
                # Add to history
                session_manager.add_to_history(
                    media_type=MediaType.VIDEO,
                    prompt=st.session_state.video_generation_prompt,
                    model_type=BedrockModel(st.session_state.video_model_type),
                    details=invocation,
                    ref_image=st.session_state.video_generation_image,
                )
            
                status.update(label="비디오 생성이 시작되었습니다. 작업은 약 5분 소요됩니다.", state="complete")
            
        except Exception as e:
            status.update(label=f"Error: {str(e)}", state="error")
            st.error(f"비디오 생성 중 오류가 발생했습니다: {str(e)}")
        
def _get_video_model_configurations():
    duration = st.slider("Duration", 1, 30, 6, 1, key="video_duration_slider", disabled=True)
    fps = st.number_input("FPS", 24, key="video_fps_input", disabled=True)
    seed = st.number_input("Seed", 0, 2147483646, 0, key="video_seed_input")
    dimension_options = {
        "1280x720"
    }

    selected_resolution = st.selectbox("Video Resolution", 
                                     options=list(dimension_options),
                                     key="video_resolution_select")
    
    return {
        'durationSeconds': duration,
        'fps': fps,
        'dimension': selected_resolution,
        'seed': seed,
    }
