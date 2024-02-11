__all__ = ["app"]

import gradio as gr

from demo.gradio_invert import run
from PIL import Image
import torchvision.transforms as transforms

to_pil = transforms.ToPILImage()


def run_detector(input_str, tv):
    for image in run(input_str, tv):
        yield to_pil(image[0])


css = """
.green { color: black!important;line-height:1.9em; padding: 0.2em 0.2em; background: #ccffcc; border-radius:0.5rem;}
.red { color: black!important;line-height:1.9em; padding: 0.2em 0.2em; background: #ffad99; border-radius:0.5rem;}
.hyperlinks {
  display: flex;
  align-items: center;
  align-content: center;
  padding-top: 12px;
  justify-content: flex-end;
  margin: 0 10px; /* Adjust the margin as needed */
  text-decoration: none;
  color: #000; /* Set the desired text color */
}
"""

# Most likely human generated, #most likely AI written

prompt = '''An astronaut exploring an alien planet, discovering a mysterious ancient artifact" for different models.'''
print(prompt)
# default_image = Image.open('figures/astronaut.png')
with gr.Blocks(css=css,
               theme=gr.themes.Default(font=[gr.themes.GoogleFont("Inconsolata"), "Arial", "sans-serif"])) as app:
    with gr.Row():
        with gr.Column(scale=3):
            gr.HTML("<p><h1>What do we learn from inverting CLIP models?</h1></p>")
        with gr.Column(scale=3):
            gr.HTML(
                "<p>This space may generate sexually explicit and NSFW (Not Safe For Work) images.</p>")
        with gr.Column(scale=1):
            gr.HTML("""
            <p>
            <a href="https://openreview.net/forum?id=3SrYqA2NHy&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICML.cc%2F2024%2FConference%2FAuthors%23your-submissions)" target="_blank">paper</a>

            <a href="https://github.com/AHans30/Binoculars" target="_blank">code</a>

            <a href="mailto:hamidkazemi22@gmail.com" target="_blank">contact</a>
            """, elem_classes="hyperlinks")
    with gr.Row():
        input_box = gr.Textbox(value=prompt, placeholder="Enter prompt here", lines=2, label="Prompt", )
    with gr.Row():
        tv_number = gr.Number(0.01, label='tv')
        submit_button = gr.Button("Run Inversion", variant="primary")
        clear_button = gr.ClearButton()
    with gr.Column(scale=3):
        gr.HTML("<p><h1>Generated Image:</h1></p>")
    with gr.Column(scale=1):
        # output_text = gr.Textbox(label="Prediction", value="Most likely AI-Generated")
        # output_text = gr.Image(type='pil', show_label=False, width=224, height=224, value=default_image)
        output_text = gr.Image(type='pil', show_label=False, width=224, height=224)
    with gr.Row():
        gr.HTML("<p><p><p>")
    with gr.Row():
        gr.HTML("<p><p><p>")
    with gr.Row():
        gr.HTML("<p><p><p>")

    with gr.Accordion("Disclaimer", open=False):
        gr.Markdown(
            """
            - `Warning` :
                - Some prompts lead to NSFW images.
            """
        )

    with gr.Accordion("Cite our work", open=False):
        gr.Markdown(
            """
            ```bibtex
                @misc{hans2024spotting,
                      title={Spotting LLMs With Binoculars: Zero-Shot Detection of Machine-Generated Text}, 
                      author={Abhimanyu Hans and Avi Schwarzschild and Valeriia Cherepanova and Hamid Kazemi and Aniruddha Saha and Micah Goldblum and Jonas Geiping and Tom Goldstein},
                      year={2024},
                      eprint={2401.12070},
                      archivePrefix={arXiv},
                      primaryClass={cs.CL}
                }
            """
        )

    submit_button.click(run_detector, inputs=[input_box, tv_number], outputs=output_text, show_progress='hidden')
    clear_button.click(lambda: ("", ""), outputs=[input_box, output_text])
