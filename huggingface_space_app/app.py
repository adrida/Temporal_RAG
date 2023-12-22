
import gradio as gr
from functools import partial
from rag_benchmark import get_benchmark



title = "Prototype Temporal Augmented Retrieval (TAR)"
desc = "Database: 22.4k tweets related to finance dated from July 12,2018 to July 19,2018 - know more about the approach: [link to medium]\ncontact: adrida.github.io"


with gr.Blocks(title=title,theme='nota-ai/theme') as demo:
    gr.Markdown(f"# {title}\n{desc}")
    with gr.Row():
        with gr.Column(scale = 10):
            text_area = gr.Textbox(placeholder="Write here", lines=1, label="Ask anything")
        with gr.Column(scale = 2):
            api_key = gr.Textbox(placeholder="Paste your OpenAI API key here", lines=1)
            search_button = gr.Button(value="Ask")

    with gr.Row():
        with gr.Tab("Dynamic Temporal Augmented Retrieval (ours)"):
    
            gr.Markdown("## Dynamic Temporal Augmented Retrieval (ours)\n---")
            tempo = gr.Markdown()             
        with gr.Tab("Naive Semantic Search"):
            gr.Markdown("## Simple Semantic Search\n---")
            naive = gr.Markdown() 
        with gr.Tab("Traditional RAG (Langchain type)"):
            gr.Markdown("## Augmented Indexed Retrieval\n---")
            classic = gr.Markdown() 
            
    search_function = partial(get_benchmark)

    search_button.click(fn=search_function, inputs=[text_area, api_key], outputs=[tempo, classic, naive],
    )

demo.queue(concurrency_count=100,status_update_rate=500).launch(max_threads=100, show_error=True, debug = True, inline =False)

