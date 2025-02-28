import streamlit as st
import requests

st.set_page_config(
    page_title="Chain-of-Draft Reasoning Optimizer",
    layout="centered",
    initial_sidebar_state="auto",
)

st.title("Chain-of-Draft (CoD) Reasoning Optimizer Demo")

st.sidebar.markdown("## Backend Configuration")
backend_url = st.sidebar.text_input("Backend URL", value="http://127.0.0.1:8000")

question = st.text_input("Enter your question:")
method_choice = st.radio(
    "Select Reasoning Method",
    ["Standard", "Chain of Thought (CoT)", "Chain of Draft (CoD)"]
)

if st.button("Generate Answer"):
    method_map = {
        "Standard": "standard",
        "Chain of Thought (CoT)": "cot",
        "Chain of Draft (CoD)": "cod"
    }
    method = method_map[method_choice]
    payload = {"question": question, "method": method}

    try:
        response = requests.post(f"{backend_url}/ask", json=payload, timeout=60)
        if response.status_code == 200:
            resp_data = response.json()
            st.write("### Answer:")
            st.write(resp_data["response"])
            
            # Show token usage and inference time if available
            st.write("**Token Usage & Time:**")
            st.json({
                "prompt_tokens": resp_data.get("prompt_tokens"),
                "completion_tokens": resp_data.get("completion_tokens"),
                "total_tokens": resp_data.get("total_tokens"),
                "inference_time_s": resp_data.get("inference_time_s")
            })
        else:
            st.error(f"Error {response.status_code}: {response.text}")
    except Exception as e:
        st.error(f"Request failed: {e}")
