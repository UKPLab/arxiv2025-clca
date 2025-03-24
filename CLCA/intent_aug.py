import logging

import hydra
from omegaconf import DictConfig

from llm_roleplay.common.intent_generator import DialogueGenerator


@hydra.main(version_base=None, config_path="llm_roleplay/configs", config_name="intent_aug")
def main(cfg: DictConfig):
    dialogue_generator = DialogueGenerator(cfg)
    dialogue_generator.initialize()
    dialogues_dir = dialogue_generator.generate()
    logging.info(f"Dialogues succesfully generated and stored in: {dialogues_dir}")


if __name__ == '__main__':
    main()
