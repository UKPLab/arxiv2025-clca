import logging
import os

import hydra
from omegaconf import DictConfig

from llm_roleplay.common.dialog_filter import filter_dialog
from llm_roleplay.common.dialog_judge import judge_dialog
from llm_roleplay.common.social_dialogue_generator import DialogueGenerator


@hydra.main(version_base=None, config_path="llm_roleplay/configs", config_name="dialogue_generator")
def main(cfg: DictConfig):
    actions = cfg.actions
    dialogues_dir = os.path.join(cfg.workdir,
                                 "dialogs", cfg.culture.lower(), cfg.format_template_key,
                                 f"{cfg.model_inquirer.name.split('/')[-1]}",
                                 )
    print(dialogues_dir)
    if "generate" in actions:
        dialogue_generator = DialogueGenerator(cfg)
        dialogue_generator.initialize()
        dialogues_dir = dialogue_generator.generate()
        logging.info(f"Dialogues succesfully generated and stored in: {dialogues_dir}")
    if "judge" in actions:
        input_file = os.path.join(dialogues_dir, f"{cfg.seed}.jsonl")
        output_file = os.path.join(dialogues_dir, f"{cfg.seed}_judge.jsonl")
        judge_dialog(input_file, output_file, cfg=cfg.model_judge)
    if "filter" in actions:
        input_file = os.path.join(dialogues_dir, f"{cfg.seed}_judge.jsonl")
        output_file = os.path.join(dialogues_dir, f"{cfg.seed}_filtered.jsonl")
        filter_dialog(input_file, output_file)


if __name__ == '__main__':
    main()
