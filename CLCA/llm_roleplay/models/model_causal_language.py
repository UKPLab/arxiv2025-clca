import torch
from llm_roleplay.common.model import Model
from llm_roleplay.common.utils import Device
from transformers import AutoModelForCausalLM, AutoTokenizer


class ModelCausalLanguageSocial(Model):
    SELF_REPLY_TOKENS = {
        "llama": "[INST",
        "vicuna": "### Human:",
    }

    def __init__(self, cfg, role) -> None:
        super().__init__(cfg, role)
        self._tokenizer = None

    def _apply_temp(self):
        chat_template = """{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] | trim + '\n\n' %}{% set messages = messages[1:] %}{% else %}{% set system_message = '' %}{% endif %}{{ bos_token + system_message}}
    {% for message in messages %}
    {% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}
        {{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}
    {% endif %}
    {% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] | trim + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' ' + message['content'] | trim + eos_token }}{% endif %}
    {% endfor %}"""
        self._tokenizer.chat_template = chat_template

    @property
    def model(self):
        if self._model is None:
            _model = AutoModelForCausalLM.from_pretrained(
                self.cfg.name,
                cache_dir=self.cfg.cache_dir,
                local_files_only=True,
                device_map=Device.get_device(),
                load_in_4bit=True,
                token=self.cfg.api_token,
            )
            opt_model = torch.compile(_model, mode='max-autotune')
            self._model = opt_model
            for param in self._model.parameters():
                param.requires_grad = False
            self._model.eval()
        return self._model

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.cfg.name, cache_dir=self.cfg.cache_dir)
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
            if "mistral" in self.cfg.name:
                self._apply_temp()
        return self._tokenizer

    def get_prompt(self, turn, response_msg=None, persona=None, instructions=None):
        return self.conv_template.format()

    @torch.no_grad()
    def generate(self, prompt, generate_cfg):
        self.model.eval()
        if isinstance(prompt, str):
            prompt_tokenized = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)
        else:
            prompt = self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
            if self.tokenizer.bos_token:
                prompt = prompt.replace(self.tokenizer.bos_token, "")
            # prompt = self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True).replace(self.tokenizer.bos_token, "")
            prompt_tokenized = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)

        output_tokenized = self.model.generate(prompt_tokenized, pad_token_id=self.tokenizer.eos_token_id,
                                               **generate_cfg)
        output = self.tokenizer.decode(output_tokenized[0], skip_special_tokens=True)
        output_o = output.replace(str(self.tokenizer.eos_token), "").strip()
        model_prompt_o = prompt.replace(str(self.tokenizer.eos_token), "").strip()
        if self.tokenizer.bos_token:
            output_o = output_o.replace(str(self.tokenizer.bos_token), "").strip()
            model_prompt_o = model_prompt_o.replace(str(self.tokenizer.bos_token), "").strip()
        turn_response = output_o.replace(model_prompt_o, "", 1)

        del output_tokenized

        if not turn_response:
            return "No Output"

        return turn_response


def update_history(self, prompt, output_extract, action):
    self.history.append(f'{prompt}\nAction:{action}\nOutput:"{output_extract}"')
