import multiprocessing
import gradio as gr
import requests

class ChatUI:
    def __init__(self, api_url: str = "http://localhost:8000/ai/ask"):
        self.api_url = api_url

    def chat_logic(self, user_input, history):
        try:
            # L'endpoint supporta solo "prompt", quindi mandiamo solo l'ultimo messaggio (senza history per ora)
            params = {"prompt": user_input}
            res = requests.get(self.api_url, params=params, verify=False)
            res.raise_for_status()
            return res.json().get("response", "[Nessuna risposta]")
        except Exception as e:
            return f"[Errore]: {e}"


    def chat_fn(self, user_input, history):
        response = self.chat_logic(user_input, history)
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": response})
        return "", history

    def _run_gradio(self):
        with gr.Blocks() as demo:
            gr.Markdown("## Demo Agenti Intelligenti 🤖")
            chatbot = gr.Chatbot(type="messages")
            msg = gr.Textbox(label="Scrivi qui...")
            clear = gr.Button("🧹 Pulisci")
            state = gr.State([])

            msg.submit(self.chat_fn, [msg, state], [msg, chatbot])
            clear.click(lambda: ([], []), None, [chatbot, state])

        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            inbrowser=False,
            show_api=False
        )

    def start(self):
        process = multiprocessing.Process(target=self._run_gradio, daemon=True)
        process.start()
