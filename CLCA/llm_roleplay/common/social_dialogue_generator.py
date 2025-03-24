import os
from pathlib import Path

import jsonlines
import torch
from llm_roleplay.common.generate_scenarios import ScenarioGenerator, load_scenarios, get_instruction, IDX2HT
from llm_roleplay.common.model import Model
from omegaconf import DictConfig
from tqdm import tqdm

ID2ACT = {"1": "none", "3": "speak", "2": "action", "4": "leave"}


def parse_generated_outputs(output):
    output_without_prompt = output.split("Please only output the response in English:")[-1].strip()
    output_without_prompt = output_without_prompt.replace("\n\n", "\n").strip()
    if output_without_prompt.startswith("assistant"):
        output_without_prompt = output_without_prompt.replace("assistant", '').strip()
        output_without_prompt = output_without_prompt.replace("：", ':').replace("：", ":").strip()

    if len(output_without_prompt.split("\n")) == 2:
        action, output = output_without_prompt.split("\n")
        if action.strip().startswith("ACTION:"):
            action = action.split("ACTION:")[-1].strip()
        if output.strip().startswith("OUTPUT:"):
            output = output.split("OUTPUT:")[-1].strip()
        return normalize_action(action), output, None
    else:
        if "ACTION:" in output_without_prompt and "OUTPUT:" in output_without_prompt:
            if len(output_without_prompt.split("\n")) == 3:
                intent, action, output = output_without_prompt.split("\n")
            else:
                action = output_without_prompt
                output = output_without_prompt
                intent = output_without_prompt
            if action.strip().startswith("ACTION:"):
                action = action.split("ACTION:")[-1].strip()
            else:
                action = "none"
            if output.strip().startswith("OUTPUT:"):
                output = output.split("OUTPUT:")[-1].strip()
            else:
                output = ""
            if intent.strip().startswith("INTENT:"):
                intent = intent.split("INTENT:")[-1].strip()
            else:
                intent = ""
            return normalize_action(action), output, intent
        if ":" in output_without_prompt:
            fields = output_without_prompt.split(":")
            if len(fields) == 2:
                return normalize_action(fields[0].lower()), ":".join(fields[1:]), None

    return None, None, None


def normalize_action(action):
    if not action.strip():
        return "none"
    if action.isnumeric():
        return ID2ACT.get(action, "1")
    elif action[0].isnumeric():
        return ID2ACT.get(action[0], "1")
    return action.strip()


def check_stop_conversation(action):
    return "leave" in action


def prep_history(name, action, output, format_key="vanilla"):
    history_template = IDX2HT.get(format_key, IDX2HT.get("vanilla"))
    return history_template.format(name=name, action=action, output=output)


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
        )
        os.makedirs(self.records_dir, exist_ok=True)

        self.initialize()

    def track(self, prompt, name, context=None):
        pass

    #     TODO

    def initialize(self):
        self.cfg.dataset = self.cfg.dataset.format(culture=self.cfg.culture.lower())
        self.raw_scenarios = load_scenarios(self.cfg.dataset)
        self.scenarios = ScenarioGenerator.get_scenarios(self.raw_scenarios)

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

    def generate(self) -> Path:
        sample_idx = 0
        for _ in tqdm(range(self.cfg.num_samples), desc="samples"):
            for idx, (persona1, persona2, name1, name2, persona_hash) in enumerate(self.scenarios):
                sample_idx += 1
                self.metrics["personas"][persona_hash] = self.raw_scenarios[idx]
                dialog = []
                history = []
                turn = 0

                with tqdm(total=self.cfg.num_turns, desc="turns", leave=False) as pbar:
                    while turn < self.cfg.num_turns:
                        pbar.set_postfix(turn=turn + 1)
                        # ------------------------------------------ Inquirer Model ------------------------------------------
                        inquirer_prompt = get_instruction(persona=persona1, name=name1,
                                                          history="This is beginning of the interaction, no history available." if turn == 0 else "\n".join(
                                                              history),
                                                          turn_number=turn,
                                                          sys_prompt=self.inquirer_cfg.conv_template.system_msg,
                                                          user_prompt=self.inquirer_cfg.conv_template.user_msg,
                                                          format_template_key=self.cfg.format_template_key,
                                                          model_name=self.inquirer_cfg.name)
                        inquirer_output = self._generate(inquirer_prompt, inquirer=True,
                                                         generate_cfg=self.inquirer_cfg.generate)

                        inquirer_action, parsed_inquirer_output, inquirer_intent = parse_generated_outputs(
                            inquirer_output)

                        if not inquirer_action and not parsed_inquirer_output:
                            print("Invalid inquirer")
                            print(f"Scenario: {idx}")
                            print(inquirer_output)
                            break

                        history.append(prep_history(name1, inquirer_action, parsed_inquirer_output,
                                                    format_key=self.cfg.format_template_key))
                        if check_stop_conversation(inquirer_action):
                            dialog.append(
                                {
                                    "turn": turn,
                                    "model_inquirer_action": inquirer_action,
                                    "model_inquirer": parsed_inquirer_output,
                                    "model_inquirer_intent": inquirer_intent,
                                    "model_responder_action": "leave",
                                    "model_responder": "",
                                    "model_responder_intent": "",
                                })
                            break
                        # ------------------------------------------ Responder Model ------------------------------------------

                        responder_prompt = get_instruction(persona=persona2, name=name2,
                                                           history="\n".join(history),
                                                           turn_number=turn,
                                                           sys_prompt=self.responder_cfg.conv_template.system_msg,
                                                           user_prompt=self.responder_cfg.conv_template.user_msg,
                                                           format_template_key=self.cfg.format_template_key,
                                                           model_name=self.responder_cfg.name)
                        responder_output = self._generate(responder_prompt,
                                                          inquirer=False, generate_cfg=self.responder_cfg.generate)
                        responder_action, parsed_responder_output, responder_intent = parse_generated_outputs(
                            responder_output)

                        if not responder_action and not parsed_responder_output:
                            print("Invalid responder")
                            print(f"Scenario: {idx}")
                            print(responder_output)
                            break

                        history.append(prep_history(name2, responder_action, parsed_responder_output,
                                                    format_key=self.cfg.format_template_key))
                        if check_stop_conversation(responder_action):
                            dialog.append(
                                {
                                    "turn": turn,
                                    "model_inquirer_action": inquirer_action,
                                    "model_inquirer": parsed_inquirer_output,
                                    "model_inquirer_intent": inquirer_intent,
                                    "model_responder_action": responder_action,
                                    "model_responder": parsed_responder_output,
                                    "model_responder_intent": responder_intent,
                                })
                            break

                        # --------------------------------------- Passing histories to the inquirer ---------------------------------------
                        dialog.append(
                            {
                                "turn": turn,
                                "model_inquirer_action": inquirer_action,
                                "model_inquirer": parsed_inquirer_output,
                                "model_inquirer_intent": inquirer_intent,
                                "model_responder_action": responder_action,
                                "model_responder": parsed_responder_output,
                                "model_responder_intent": responder_intent,
                            })
                        torch.cuda.empty_cache()
                        turn += 1
                        pbar.update(1)
                for h in dialog:
                    print(h)
                with jsonlines.open(self.records_dir.joinpath(f"{self.cfg.seed}_data.jsonl"), mode="a") as writer:
                    writer.write(
                        {
                            "persona": self.raw_scenarios[idx],
                            "sample": sample_idx,
                            "num_turns": turn,
                            "dialog": dialog,
                        }
                    )
        if self.responder_cfg.name.startswith("gpt"):
            with open(self.records_dir.joinpath(f"{self.cfg.seed}_cost.jsonl"), 'a') as f:
                total_tokens = sum(self.model_inquirer.total_tokens) + sum(self.model_responder.total_tokens)
                f.write(f"Total tokens: {total_tokens}\n")

        return self.records_dir
