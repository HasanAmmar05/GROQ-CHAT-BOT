import streamlit as st
from typing import Generator
from groq import Groq
from streamlit_extras.colored_header import colored_header
from streamlit_extras.app_logo import add_logo
from streamlit_extras.stylable_container import stylable_container

# Set page config
st.set_page_config(page_icon="💬", layout="wide", page_title="Groq Goes Brrrrrrrr...")

# Custom CSS for gradients and advanced styling
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    .css-1d391kg {
        background: linear-gradient(135deg, #ffffff 0%, #f1f5f9 100%);
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton>button {
        background: linear-gradient(90deg, #4e54c8 0%, #8f94fb 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    .stSelectbox {
        border-radius: 5px;
    }
    .stSlider {
        color: #4e54c8;
    }
</style>
""", unsafe_allow_html=True)

# Add logo (replace URL with your actual logo URL)
add_logo("https://example.com/your_logo.png")

# Colored header
colored_header(
    label="Groq Chat Streamlit App",
    description="Powered by advanced language models",
    color_name="blue-70"
)

client = Groq(
    api_key=st.secrets["GROQ_API_KEY"],
)

# Initialize chat history and selected model
if "messages" not in st.session_state:
    st.session_state.messages = []

if "selected_model" not in st.session_state:
    st.session_state.selected_model = None

# Define model details
models = {
    "gemma-7b-it": {"name": "Gemma-7b-it", "tokens": 8192, "developer": "Google"},
    "llama2-70b-4096": {"name": "LLaMA2-70b-chat", "tokens": 4096, "developer": "Meta"},
    "llama3-70b-8192": {"name": "LLaMA3-70b-8192", "tokens": 8192, "developer": "Meta"},
    "llama3-8b-8192": {"name": "LLaMA3-8b-8192", "tokens": 8192, "developer": "Meta"},
    "mixtral-8x7b-32768": {"name": "Mixtral-8x7b-Instruct-v0.1", "tokens": 32768, "developer": "Mistral"},
}

# Layout for model selection and max_tokens slider
with stylable_container(
    key="model_selection",
    css_styles="""
        {
            background: linear-gradient(90deg, #e0eafc 0%, #cfdef3 100%);
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
    """
):
    col1, col2 = st.columns(2)

    with col1:
        model_option = st.selectbox(
            "Choose a model:",
            options=list(models.keys()),
            format_func=lambda x: f"{models[x]['name']} ({models[x]['developer']})",
            index=4  # Default to mixtral
        )

    # Detect model change and clear chat history if model has changed
    if st.session_state.selected_model != model_option:
        st.session_state.messages = []
        st.session_state.selected_model = model_option

    max_tokens_range = models[model_option]["tokens"]

    with col2:
        # Adjust max_tokens slider dynamically based on the selected model
        max_tokens = st.slider(
            "Max Tokens:",
            min_value=512,
            max_value=max_tokens_range,
            value=min(32768, max_tokens_range),
            step=512,
            help=f"Adjust the maximum number of tokens (words) for the model's response. Max for selected model: {max_tokens_range}"
        )

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    avatar = '🤖' if message["role"] == "assistant" else '👨‍💻'
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

def generate_chat_responses(chat_completion) -> Generator[str, None, None]:
    """Yield chat response content from the Groq API response."""
    for chunk in chat_completion:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

# Chat input with custom styling
with stylable_container(
    key="chat_input",
    css_styles="""
        {
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
    """
):
    if prompt := st.chat_input("Enter your prompt here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user", avatar='👨‍💻'):
            st.markdown(prompt)

        # Fetch response from Groq API
        try:
            chat_completion = client.chat.completions.create(
                model=model_option,
                messages=[
                    {
                        "role": m["role"],
                        "content": m["content"]
                    }
                    for m in st.session_state.messages
                ],
                max_tokens=max_tokens,
                stream=True
            )

            # Use the generator function with st.write_stream
            with st.chat_message("assistant", avatar="🤖"):
                chat_responses_generator = generate_chat_responses(chat_completion)
                full_response = st.write_stream(chat_responses_generator)
        except Exception as e:
            st.error(e, icon="🚨")

        # Append the full response to session_state.messages
        if isinstance(full_response, str):
            st.session_state.messages.append(
                {"role": "assistant", "content": full_response})
        else:
            # Handle the case where full_response is not a string
            combined_response = "\n".join(str(item) for item in full_response)
            st.session_state.messages.append(
                {"role": "assistant", "content": combined_response})

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 30px; padding: 20px; background: linear-gradient(90deg, #4e54c8 0%, #8f94fb 100%); color: white; border-radius: 10px;">
    <p>Powered by Groq API | Created with ❤️ using Streamlit</p>
</div>
""", unsafe_allow_html=True)