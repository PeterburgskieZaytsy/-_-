import gradio as gr
from utils import *


def get_result(text):
    predicted_category = get_bert_prediction(text)
    return predicted_category


iface = gr.Interface(
    get_result,
    "text",
    "text",
    title="Предсказать категорию",
)

if __name__ == "__main__":
    iface.launch()