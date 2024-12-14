import streamlit as st
from typing import List
from genai_kit.aws.amazon_image import ImageParams, TitanImageSize, NovaImageSize
from genai_kit.aws.sd_image import SDImageSize
from genai_kit.aws.bedrock import BedrockModel
from genai_kit.utils.images import encode_image_base64, base64_to_bytes
from services.bedrock_service import edit_image
from session import SessionManager
from constants import EditingMode, MediaType


def show_image_editor(session_manager: SessionManager):
    st.title("🖌️ Image Editor")
    initialize_session_state()
    
    col1, col2, col3 = st.columns(3)

    with col1:
        show_input_image_section()
    
    with col2:
        show_editing_params_secion()
    
    with col3:
        generate_clicked = show_model_section()

    if generate_clicked:
        generate_image(session_manager)

def initialize_session_state():
    if 'editing_mode' not in st.session_state:
        st.session_state.editing_mode = ""
    if 'mask_prompt' not in st.session_state:
        st.session_state.mask_prompt = None
    if 'editing_text' not in st.session_state:
        st.session_state.editing_text = None
    if 'ref_image' not in st.session_state:
        st.session_state.ref_image = None
    if 'model_type' not in st.session_state:
        st.session_state.model_type = ""


def show_input_image_section():
    st.subheader("Reference Image")

    reference_image = st.file_uploader(
        "Upload a reference image:",
        type=['png', 'jpg', 'jpeg'],
        key="edit_ref_image_uploader"
    )
    if reference_image:
        st.image(reference_image, caption="Reference Image")
        st.session_state.ref_image = encode_image_base64(reference_image)
        

def show_editing_params_secion():
    st.subheader("Editing Mode")

    editing_mode = st.selectbox(
        "Editing Mode",
        options=list(EditingMode),
        key="editing_mode_select",
        format_func=lambda x: x.name
    )
    st.session_state.editing_mode = editing_mode
    
    if editing_mode in [
        EditingMode.INPAINTING,
        EditingMode.OUTPAINTING,
    ]:
        mask_prompt = st.text_input("Mask Prompt:")
        st.session_state.mask_prompt = mask_prompt

    if editing_mode in [
        EditingMode.IMAGE_VARIATION,
        EditingMode.INPAINTING,
        EditingMode.OUTPAINTING,
        EditingMode.IMAGE_CONDITIONING,
    ]:
        editing_text = st.text_input("Text:")
        st.session_state.editing_text = editing_text


def show_model_section():
    st.subheader("Select a Model")
    model_type = st.selectbox(
        "Choose a model:",
        [BedrockModel.NOVA_CANVAS, BedrockModel.TITAN_IMAGE],
         format_func=lambda x: x.name
    )

    st.session_state.model_type = model_type

    if st.session_state.editing_mode != EditingMode.BACKGROUND_REMOVAL:
        with st.expander("Image Configuration", expanded=True):
            configs = _get_model_configurations(model_type)
            st.session_state.generation_configs = configs
    
    return st.button("Editing Images", icon='🎨', type="primary", use_container_width=True)


def generate_image(session_manager: SessionManager):
    st.divider()
    st.subheader("Generated Images")
    
    with st.status("Generating images...", expanded=True) as status:
        try:
            configs = st.session_state.generation_configs

            model_type = BedrockModel(st.session_state.model_type)
            imgs, configuration = edit_image(
                model_type=model_type,
                editing_mode=st.session_state.editing_mode,
                text=st.session_state.editing_text,
                mask_prompt=st.session_state.mask_prompt,
                ref_image=st.session_state.ref_image,
                size=configs['size'],
                count=configs['num_images'],
                seed=configs['seed'],
                cfg=configs['cfg_scale'],
            )

            print(configuration)

            # Display prompt and generated image
            if st.session_state.editing_text and len(st.session_state.editing_text) > 0:
                st.info(st.session_state.editing_text)\

            cols = st.columns(len(imgs))
            for idx, img in enumerate(imgs):
                with cols[idx]:
                    image_data = base64_to_bytes(img)
                    st.image(image_data, use_container_width=True)
            
                # Add to history
                session_manager.add_to_history(
                    prompt=st.session_state.editing_text,
                    media_type = MediaType.IMAGE,
                    model_type = model_type,
                    media_file=image_data,
                    details = configuration,
                    ref_image = st.session_state.ref_image,
                )
            
            status.update(label="Generation completed!", state="complete")
            
        except Exception as e:
            status.update(label=f"Error: {str(e)}", state="error")
            st.error(f"이미지 생성 중 오류가 발생했습니다: {str(e)}")
        finally:
            st.session_state.is_generating_image = False

def _get_model_configurations(model_type: str):
    num_images = st.slider("Number of Images", 1, 5, 1, key="editing_num_image")
    cfg_scale = st.slider("CFG Scale", 1.0, 10.0, 8.0, 0.5, key="editing_cfg_scale")
    seed = st.number_input("Seed", 0, 2147483646, 0, key="editing_seed")
    
    # size
    if model_type == BedrockModel.TITAN_IMAGE:
        size_enum = TitanImageSize
        size_options = {f"{size.width} X {size.height}": size for size in size_enum}
    elif model_type == BedrockModel.NOVA_CANVAS:
        size_enum = NovaImageSize
        size_options = {f"{size.width} X {size.height}": size for size in size_enum}
    
    selected_size = st.selectbox("Image Size", options=list(size_options.keys()), key="editing_size")
    
    return {
        'num_images': num_images,
        'cfg_scale': cfg_scale,
        'seed': seed,
        'size': size_options[selected_size],
    }
