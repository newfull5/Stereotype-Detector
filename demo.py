import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
import altair as alt

cols = [
    "stereotype",
    "anti-stereotype",
    "unrelated",
    "profession",
    "race",
    "gender",
    "religion",
]


def inference(inputs: str):
    tokenized = tokenizer(
        inputs,
        return_tensors="pt",
        max_length=512,
        padding="max_length",
        truncation=True,
    ).to(device)
    outputs = model(**tokenized)
    return outputs.logits.tolist()[0]


def write_header():
    st.title("Korean Stereotype Detector")
    st.markdown(
        """
        - Write any sentence containing stereotypes and click the Run button.
        - This application is made with TUNiB-Electra and K-StereoSet.
        - Using CPU, the result might be slow
    """
    )


def get_json_str(lists):
    string = "{\n"
    for i, v in enumerate(lists):
        string += f'\t"{cols[i]}": {v},\n'
    return string + "}"


def write_textbox():
    input_text = st.text_area(label="Write your sentence", key=1, height=40)
    button = st.button(label="Run")
    output = inference(input_text)

    st.markdown(
        """
    #
    ## result
    ***
    """
    )

    if button:
        col1, col2 = st.columns([5, 5])

        with col1:
            st.code(get_json_str(output))

        with col2:
            st.write(
                alt.Chart(
                    pd.DataFrame({"Class": cols, "Logits": output}),
                    width=490,
                    height=360,
                )
                .mark_bar()
                .encode(x="Class", y="Logits")
            )


if __name__ == "__main__":
    st.set_page_config(
        page_title="Korean stereotype detector", page_icon="☮️", layout="wide"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "dhtocks/tunib-electra-stereotype-classifier"
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        "dhtocks/tunib-electra-stereotype-classifier"
    )
    device = "cpu"
    write_header()
    write_textbox()
