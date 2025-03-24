import hashlib
import json
import os
from pathlib import Path

import jsonlines
import torch
from llm_roleplay.common.model import Model
from omegaconf import DictConfig

TEMPLATE_INTENT = """
    Here is the basic information about this conversation.
    Scenario: {setting}
    Information about {name}:
    Background: {background}
    Occupation: {occupation}
    Information about {name2}:
    Background: {background2}
    Occupation: {occupation2}
    Both participants are from the {culture} culture, you are an expert in {culture} culture.
    """

COUNTRY2CULTURE = {
    "china": "Chinese",
    "germany": "German",
    "japan": "Japanese",
    "mexico": "Mexican",
    "uk": "British",
}

CULTURE2SUFFIX = {

    "British": """\nPlease predict {name}'s intent in the last turn based on the provided conversation, and reason the prediction with respect to the social or cultural expectations in {culture} that might be influencing the tone and content of this interaction in a short sentence. Don't explain if you are unsure of the reasons, only explain if you are very certain, keep it short.
     Please following the schema:
     INTENT: <intent>
     Please only output the response in English:""",
}


def parse_generated_outputs(prompt, output):
    output_without_prompt = output.split("Please only output the response in English:")[-1].strip()

    output_without_prompt = output_without_prompt.replace("assistant", "").strip()
    output_without_prompt = output_without_prompt.replace("\n\n", "\n").strip()
    output_without_prompt = output_without_prompt.split("INTENT:")
    if len(output_without_prompt) == 2:
        intent = output_without_prompt[-1].strip()
        return intent
    return None


class DialogueGenerator():
    def __init__(self, cfg: DictConfig) -> None:

        self.cfg = cfg
        self.inquirer_cfg = self.cfg.model_inquirer
        self.responder_cfg = self.cfg.model_responder
        self.metrics = {
            "num_no_prompts": 0,
            "num_multiple_prompts": 0,
            "num_non_coherent": 0,
            "num_regenerate_worked": 0,
            "num_self_replies": 0,
            "num_non_coherent_model_responder": 0,
            "personas": {}
        }

        self.records_dir = Path(self.cfg.workdir).joinpath(
            "dialogs", cfg.culture.lower(), cfg.format_template_key,
            f"{self.inquirer_cfg.name.split('/')[-1]}",
            str(hashlib.md5(json.dumps(self.metrics).encode()).hexdigest()),
        )
        os.makedirs(self.records_dir, exist_ok=True)

        self.initialize()

    def track(self, prompt, name, context=None):
        pass

    def load_dataset(self, fn):
        data = []
        with open(fn, 'r') as f:
            for line in f.readlines():
                data.append(json.loads(line))
        return data

    def initialize(self):
        self.cfg.dataset = self.cfg.dataset.format(culture=self.cfg.culture.lower())
        self.dataset = self.load_dataset(self.cfg.dataset)
        if self.inquirer_cfg.name == self.responder_cfg.name:
            print("Agents of the same type!!!\n#######")
            self.model_inquirer = Model.get_model(self.inquirer_cfg, role="model_inquirer")
            self.model_inquirer.spec_tokens = self.cfg.spec_tokens
            self.model_inquirer.metrics = self.metrics
            self.model_responder = self.model_inquirer
        else:
            self.model_inquirer = Model.get_model(self.inquirer_cfg, role="model_inquirer")
            self.model_responder = Model.get_model(self.responder_cfg, role="model_responder")
            self.model_inquirer.spec_tokens = self.cfg.spec_tokens
            self.model_responder.spec_tokens = self.cfg.spec_tokens
            self.model_inquirer.metrics = self.metrics
            self.model_responder.metrics = self.metrics

    @torch.no_grad()
    def _generate(self, prompt, inquirer, generate_cfg=None):
        if inquirer:
            output = self.model_inquirer.generate(
                prompt=prompt,
                generate_cfg=(generate_cfg),
            )
        else:
            output = self.model_responder.generate(
                prompt=prompt,
                generate_cfg=(generate_cfg),
            )

        return output

    def get_prompt_header(self, data):
        personas = data["persona"]
        culture = personas["culture"].strip()
        culture = COUNTRY2CULTURE.get(culture.lower(), culture)
        name1, name2 = personas["persona"].split(" and ")
        name1 = name1.strip()
        name2 = name2.strip()
        person1 = personas["person1"]
        person2 = personas["person2"]

        return TEMPLATE_INTENT.format(setting=personas["setting"], culture=culture,
                                      name=name1, background=person1["background"].strip(),
                                      occupation=person1["occupation"].strip(),
                                      name2=name2, background2=person2["background"].strip(),
                                      occupation2=person2["occupation"].strip(),
                                      ), name1, name2

    def generate(self) -> Path:
        c = 0
        culture = COUNTRY2CULTURE.get(self.cfg.culture.lower(), self.cfg.culture)
        instruction_suffix = """\nPlease predict {name}'s intent in the last turn based on the provided conversation, and reason the prediction with respect to the social or cultural expectations in {culture} that might be influencing the tone and content of this interaction in a short sentence. Don't explain if you are unsure of the reasons, only explain if you are very certain, keep it short.
                                Please follow the schema:
                                INTENT: <intent>
                                Please only output the response in English:
                             """
        for datapoint in self.dataset:
            c += 1
            print(f"Currently processing :{c} / {len(self.dataset)}")
            history = []
            sys_prompt, name1, name2 = self.get_prompt_header(datapoint)
            for turn in datapoint["dialog"]:
                history.append(f"{name1}: " + turn["model_inquirer"].strip())
                inquirer_prompt = [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user",
                     "content": "\n".join(history) + instruction_suffix.format(name=name1, culture=culture)}
                ]
                inquirer_output = self._generate(inquirer_prompt, inquirer=True,
                                                 generate_cfg=self.inquirer_cfg.generate)
                inquirer_intent = parse_generated_outputs("\n".join(history), inquirer_output)

                if not inquirer_intent:
                    print("Invalid intent")
                    print(inquirer_output)
                    break
                turn["model_inquirer_intent"] = inquirer_intent

                history.append(f"{name2}: " + turn["model_responder"].strip())
                responder_prompt = [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user",
                     "content": "\n".join(history) + instruction_suffix.format(name=name2, culture=culture)}
                ]
                responder_output = self._generate(responder_prompt, inquirer=False,
                                                  generate_cfg=self.responder_cfg.generate)
                responder_intent = parse_generated_outputs("\n".join(history), responder_output)
                if not responder_intent:
                    print("Invalid responder intent")
                    print(responder_output)
                    break
                turn["model_responder_intent"] = responder_intent
                torch.cuda.empty_cache()

            with jsonlines.open(self.records_dir.joinpath(f"intent_aug.jsonl"), mode="a") as writer:
                writer.write(datapoint)
