import os
import sys
import gc
import traceback

import fire
import gradio as gr
from queue import Queue
from threading import Thread
import torch
import transformers
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer

if torch.cuda.is_available():
    device = "cpu"
else:
    device = "cpu"


PROMPT_DICT = {
    "prompt_input":
        ("以下はタスクを説明する指示と、さらなる文脈を提供する入力がペアになっています。"
         "適切にリクエストを完了するレスポンスを書いてください。\n\n"
         "### 指示:{instruction}\n\n### 入力:\n{input}\n\n### レスポンス:"),
    "prompt_no_input": ("以下はタスクを説明する指示です。"
                        "適切にリクエストを完了するレスポンスを書いてください。\n\n"
                        "### 指示:\n{instruction}\n\n### レスポンス:"),
}

class Stream(transformers.StoppingCriteria):
    def __init__(self, callback_func=None):
        self.callback_func = callback_func

    def __call__(self, input_ids, scores) -> bool:
        if self.callback_func is not None:
            self.callback_func(input_ids[0])
        return False


class Iteratorize:

    """
    Transforms a function that takes a callback
    into a lazy iterator (generator).
    """

    def __init__(self, func, kwargs={}, callback=None):
        self.mfunc = func
        self.c_callback = callback
        self.q = Queue()
        self.sentinel = object()
        self.kwargs = kwargs
        self.stop_now = False

        def _callback(val):
            if self.stop_now:
                raise ValueError
            self.q.put(val)

        def gentask():
            try:
                ret = self.mfunc(callback=_callback, **self.kwargs)
            except ValueError:
                pass
            except:
                traceback.print_exc()
                pass

            self.q.put(self.sentinel)
            if self.c_callback:
                self.c_callback(ret)

        self.thread = Thread(target=gentask)
        self.thread.start()

    def __iter__(self):
        return self

    def __next__(self):
        obj = self.q.get(True, None)
        if obj is self.sentinel:
            raise StopIteration
        else:
            return obj

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_now = True


def create_prompt(instruction_text, input_text):

    prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
    return (
        prompt_no_input.format(instruction=instruction_text)
        if input_text == ""
        else prompt_input.format(
            instruction=instruction_text, input=input_text
        )
    )


def get_response(output):
    return output.split("### レスポンス:")[1].strip()


def main(
    base_model: str = "abeja/gpt-neox-japanese-6.7b",
    weights_dir: str = "output",
    server_name: str = "0.0.0.0",  # Allows to listen on all interfaces by providing '0.
    share_gradio: bool = False,
):
    """Generate text using a GPT-NeoX model.

    Args:
        base_model (str, optional): The base model to use. Defaults to "abeja/gpt-neox-japanese-6.7b".
        weights_dir (str, optional): The directory to load the weights from. Defaults to "output".
        server_name (str, optional): Allows to listen on all interfaces by providing '
    """
    assert "abeja" in base_model, "Please use a abeja model, e.g. 'abeja/gpt-neox-japanese-6.7b"
    
    if base_model.endswith("2.7b"):
        use_auth_token = False
    else:
        use_auth_token = True

    if not weights_dir.startswith("/"):
        weights_dir = os.path.abspath(weights_dir)

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_auth_token=use_auth_token)
    model = AutoModelForCausalLM.from_pretrained(base_model, use_auth_token=use_auth_token).to(device)#.half()

    # pytorchから始まり、.binで終わるファイル名だけを取得する
    weight_file_names = [file_name for file_name in os.listdir(weights_dir) if file_name.startswith("pytorch") and file_name.endswith(".bin")]
    stata_dict_files = [os.path.join(weights_dir, file_name) for file_name in weight_file_names]

    # Load weights
    for i, state_dict_file in enumerate(stata_dict_files):
        loaded_state_dict = torch.load(state_dict_file, map_location=lambda storage, loc: storage)
        if i == 0:
            state_dict = loaded_state_dict
        else:
            state_dict.update(loaded_state_dict)
    model.load_state_dict(state_dict)

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    def evaluate(
        instruction_text,
        input_text="",
        temperature=0.9,
        top_p=0.75,
        top_k=40,
        max_new_tokens=256,
        min_length=0,
        no_repeat_ngram_size=0,
        stream_output=True,
        **kwargs,
    ):
        prompt = create_prompt(instruction_text, input_text)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)

        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=True,
            min_length=min_length,
            no_repeat_ngram_size=no_repeat_ngram_size,
            pad_token_id=31998,
            **kwargs,
        )

        generate_params = {
            "input_ids": input_ids,
            "generation_config": generation_config,
            "return_dict_in_generate": True,
            "output_scores": True,
            "max_new_tokens": max_new_tokens,
            "min_length": min_length,
        }

        if stream_output:
            # Stream the reply 1 token at a time.
            # This is based on the trick of using 'stopping_criteria' to create an iterator,
            # from https://github.com/oobabooga/text-generation-webui/blob/ad37f396fc8bcbab90e11ecf17c56c97bfbd4a9c/modules/text_generation.py#L216-L243.

            def generate_with_callback(callback=None, **kwargs):
                kwargs.setdefault(
                    "stopping_criteria", transformers.StoppingCriteriaList()
                )
                kwargs["stopping_criteria"].append(
                    Stream(callback_func=callback)
                )
                with torch.no_grad():
                    model.generate(**kwargs)

            def generate_with_streaming(**kwargs):
                return Iteratorize(
                    generate_with_callback, kwargs, callback=None
                )

            with generate_with_streaming(**generate_params) as generator:
                for output in generator:
                    # new_tokens = len(output) - len(input_ids[0])
                    decoded_output = tokenizer.decode(output, skip_special_tokens=True)

                    if output[-1] in [tokenizer.eos_token_id]:
                        break

                    yield get_response(decoded_output)
            return

        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
                min_length=min_length,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s, skip_special_tokens=True)
        print(output)
        print()
        response = get_response(output)
        yield response

    gr.Interface(
        fn=evaluate,
        inputs=[
            gr.components.Textbox(
                lines=2,
                label="指示",
                placeholder="あなたは何でも正確に答えられるAIです。",
            ),
            gr.components.Textbox(
                lines=2,
                label="入力",
                placeholder="日本でいちばん高い山は？"
            ),
            gr.components.Slider(
                minimum=0, maximum=1, value=0.7, label="Temperature"
            ),
            gr.components.Slider(
                minimum=0, maximum=1, value=0.75, label="Top p"
            ),
            gr.components.Slider(
                minimum=0, maximum=100, step=1, value=40, label="Top k"
            ),
            gr.components.Slider(
                minimum=1, maximum=2000, step=1, value=256, label="Max tokens"
            ),
            gr.components.Slider(
                minimum=0, maximum=2000, step=1, value=0, label="Min length"
            ),
            gr.components.Slider(
                minimum=0, maximum=10, step=1, value=0, label="no_repeat_ngram_size"
            ),
            gr.components.Checkbox(value=True, label="Stream output"),
        ],
        outputs=[
            gr.inputs.Textbox(
                lines=5,
                label="Output",
            )
        ],
        title=f"ABEJA GPT-NeoX Japanese {base_model.split('-')[-1].upper()} Fine-tuned on Dolly dataset",
        description=f"databricks-dolly-15k-jaでfinetuneしたABEJA GPTモデル",
    ).queue().launch(server_name="0.0.0.0", share=share_gradio)


if __name__ == "__main__":
    fire.Fire(main)

"""
python generate.py --base_model 'abeja/gpt-neox-japanese-6.7b' --weights_dir 'output-6_7b'
"""