import multiprocessing
import gradio as gr
import requests

class ChatUI:
    def __init__(self, api_url: str = "http://localhost:8000/bot/ask"):
        self.api_url = api_url

    def chat_logic(self, user_input):
        try:
            payload = {"prompt": user_input}
            res = requests.post(self.api_url, json=payload)
            res.raise_for_status()
            print("response:", res)
            json_response = res.json()
            print("json_response:", json_response)

            # Estrai la risposta principale dal campo 'response'
            inner = json_response.get("response", {})
            answer = inner.get("answer", "[Nessuna risposta]")

            # Aggiungi le fonti, se presenti
            sources = inner.get("sources", [])
            if sources:
                sources_text = "\n\n📚 Fonti:\n" + "\n".join(f"- {s}" for s in sources)
            else:
                sources_text = ""

            return answer + sources_text

        except requests.RequestException as e:
            return f"[Errore HTTP]: {e}"
        except Exception as e:
            return f"[Errore generico]: {e}"



    def chat_fn(self, user_input, history):
        response = self.chat_logic(user_input)
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": response})
        return "", history

    def _run_gradio(self):
        with gr.Blocks() as demo:
            gr.Markdown("## Demo Agente Intelligente 🤖")
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
