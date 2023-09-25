import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
import fire
import gradio as gr
import time

def respond(
        message,
        chat_history,
        base_model: str = "EleutherAI/polyglot-ko-12.8b",
        lora_weights: str = "/home/hjcho9510/my_alpaca/output/"
):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(base_model, quantization_config=bnb_config, device_map={"": 0})
    model = PeftModel.from_pretrained(model, lora_weights)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model.eval()

    def gen(x):
        q = f"###명령어: {x}\n\n### 응답:"
        gened = model.generate(
            **tokenizer(
                q,
                return_tensors='pt',
                return_token_type_ids=False
            ).to('cuda'),
            max_new_tokens=100,
            early_stopping=True,
            do_sample=True,
            eos_token_id=2,
        )
        return tokenizer.decode(gened[0])

    bot_message = gen(message)
    chat_history.append((message, bot_message))
    time.sleep(0.5)
    return "", chat_history

with gr.Blocks() as demo:
    # 대충 소개글
    gr.Markdown("데모입니다~")
    # 채팅 화면
    chatbot = gr.Chatbot().style(height=600)
    with gr.Row():
        with gr.Column(scale= 0.9):
            # 입력
            msg = gr.Textbox(
                show_label=False,
                placeholder="Enter text and press enter",
            ).style(container=False)
        with gr.Column(scale=0.1):
            # 버튼
            clear = gr.Button("➤")
    # 버튼 클릭
    clear.click(respond, [msg, chatbot], [msg, chatbot])
    # 엔터키
    msg.submit(respond, [msg, chatbot], [msg,chatbot])

if __name__ == "__main__":
    demo.launch()
