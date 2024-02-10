from demo.huggingface_gradio import app

if __name__ == "__main__":
    app.launch(show_api=False, debug=True, share=True, enable_queue=True)