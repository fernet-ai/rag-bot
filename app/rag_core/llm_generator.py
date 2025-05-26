import os
from app.config import GOOGLE_API_KEY
from transformers import LlamaTokenizer, pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import pipeline as hf_pipeline
import google.generativeai as genai
import torch


class LLMGenerator:
    def __init__(self, model_type: str = "flan-t5"):
        self.model_type = model_type.lower()
        self.generator = None

        if self.model_type == "flan-t5":
            self._init_flan_t5()

        elif self.model_type == "gemini":
            self._init_gemini()

        elif self.model_type == "hermes":
            self._init_hermes()

        else:
            raise ValueError(f"Modello non supportato: {self.model_type}")

    def _init_flan_t5(self):
        print("🔧 Inizializzo FLAN-T5...")
        self.generator = hf_pipeline(
            "text2text-generation",
            model="google/flan-t5-base"
        )

    def _init_gemini(self):
        print("🔧 Inizializzo Gemini...")
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Imposta la variabile d'ambiente GOOGLE_API_KEY.")
        genai.configure(api_key=api_key)
        self.generator = genai.GenerativeModel("gemini-2.0-flash")

    def _init_hermes(self):
        print("🔧 Inizializzo Nous Hermes 2 Mistral...")
        model_name = "NousResearch/Nous-Hermes-2-Mistral-7B-DPO"
        # bnb_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_quant_type="nf4",
        #     bnb_4bit_use_double_quant=True,
        #     bnb_4bit_compute_dtype=torch.float16
        # )
        tokenizer = LlamaTokenizer.from_pretrained(model_name) 
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float32 #torch_dtype=torch.float16, se sei su CPU, meglio float32
            #quantization_config=bnb_config
        )
        self.generator = hf_pipeline("text-generation", model=model, tokenizer=tokenizer)

    def generate(self, prompt: str, max_new_tokens: int = 200, do_sample: bool = True) -> str:
        if self.model_type == "gemini":
            response = self.generator.generate_content(prompt)
            return response.text
        elif self.model_type == "flan-t5":
            response = self.generator(prompt, max_new_tokens=max_new_tokens, do_sample=do_sample)
            return response[0]["generated_text"]
        elif self.model_type == "hermes":
            response = self.generator(prompt, max_new_tokens=max_new_tokens, do_sample=do_sample)
            return response[0]["generated_text"]
        else:
            raise RuntimeError("Modello non inizializzato correttamente.")
