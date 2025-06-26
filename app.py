import streamlit as st
import os
from dotenv import load_dotenv
import pandas as pd

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import AzureChatOpenAI
from uqlm import BlackBoxUQ, WhiteBoxUQ, UQEnsemble, LLMPanel

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")
os.environ["AZURE_OPENAI_KEY"] = os.getenv("AZURE_OPENAI_KEY")


st.set_page_config(layout="wide")
# Model options and their constructors
MODEL_OPTIONS = {
    "Gemini (Google)": lambda temp: ChatGoogleGenerativeAI(model='gemini-2.0-flash', temperature=temp),
    "Azure OpenAI GPT-4o": lambda temp: AzureChatOpenAI(
        azure_deployment='gpt-4o',
        api_version="2024-12-01-preview",
        temperature=temp,
        max_tokens=1000,
    ),
}

UQ_METHODS = {
    "BlackBoxUQ (semantic_negentropy)": "blackbox",
    "WhiteBoxUQ (min_probability)": "whitebox",
    "UQEnsemble (exact_match, cosine_sim)": "ensemble",
    "LLMPanel (all supported)": "llmpanel"
}

st.title("LLM UQ Playground: Hallucination & Uncertainty Detection")
st.markdown("Select a model, UQ method, set temperature, and enter your prompt.")

with st.sidebar:
    model_name = st.selectbox("Model", list(MODEL_OPTIONS.keys()), index=0)
    uq_method = st.selectbox("UQ Method", list(UQ_METHODS.keys()), index=0)
    temperature = st.slider("Temperature", 0.0, 1.0, value=0.7, step=0.01)
    num_responses = st.number_input("Num Responses (for BlackBoxUQ)", min_value=1, max_value=10, value=5)

prompt = st.text_area("Prompt", height=100)
run = st.button("Generate")

import asyncio

async def run_uq(prompt, model_name, uq_method, temperature, num_responses):
    llm = MODEL_OPTIONS[model_name](temperature)
    if uq_method == "BlackBoxUQ (semantic_negentropy)":
        uq = BlackBoxUQ(llm=llm, scorers=["semantic_negentropy"], use_best=True)
        results = await uq.generate_and_score(prompts=[prompt], num_responses=num_responses)
        return results.to_df().to_string(index=False)
    elif uq_method == "WhiteBoxUQ (min_probability)":
        uq = WhiteBoxUQ(llm=llm, scorers=["min_probability"])
        results = await uq.generate_and_score(prompts=[prompt])
        return results.to_df().to_string(index=False)
    elif uq_method == "UQEnsemble (exact_match, cosine_sim)":
        uq = UQEnsemble(llm=llm, scorers=["exact_match", "cosine_sim"], use_best=True)
        results = await uq.generate_and_score(prompts=[prompt])
        return results.to_df().to_string(index=False)
    elif uq_method == "LLMPanel (all supported)":
        judge_llm1 = ChatGoogleGenerativeAI(model='gemini-2.0-flash')
        judge_llm2 = ChatGoogleGenerativeAI(model='gemini-2.0-flash')
        components = [
            judge_llm1,
            judge_llm2,
            "semantic_negentropy",
            "noncontradiction",
            "exact_match",
            "cosine_sim",
            "bert_score",
            "bleurt",
            "normalized_probability",
            "min_probability"
        ]
        uq = LLMPanel(llm=llm, judges=components)
        results = await uq.generate_and_score(prompts=[prompt])
        return results.to_df().to_string(index=False)
    else:
        return "Unknown UQ method."

if run and prompt.strip():
    try:
        result = asyncio.run(run_uq(prompt, model_name, uq_method, temperature, num_responses))
        # Try to parse the result as a DataFrame for pretty display
        try:
            # If result is already a DataFrame, display directly
            if isinstance(result, pd.DataFrame):
                st.dataframe(result)
            else:
                # Try to convert string to DataFrame
                from io import StringIO
                df = pd.read_csv(StringIO(result), sep="\s{2,}|\t|,", engine="python")
                st.dataframe(df)
        except Exception:
            st.text_area("UQ Output", value=result, height=300)
    except Exception as e:
        st.error(f"Error: {e}")