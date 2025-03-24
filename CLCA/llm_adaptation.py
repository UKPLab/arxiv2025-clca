import logging

import hydra
from omegaconf import DictConfig

from llm_roleplay.common.model_tuning import train
from llm_roleplay.evaluation.wvs_evaluation import evaluate_model


@hydra.main(version_base=None, config_path="llm_roleplay/configs", config_name="llm_adaptation")
def main(cfg: DictConfig):
    logging.info(f"Start training the model")
    peft_path = train(cfg)
    logging.info(f"Finish training the model")
    cfg.evaluation.pretrained_peft = peft_path
    cfg.evaluation.exp_name = cfg.exp_name
    cfg.evaluation.model_name = cfg.architecture
    cfg.evaluation.model_responder = cfg.model_responder
    logging.info(f"Start evaluating the model")
    evaluate_model(cfg.evaluation)


if __name__ == '__main__':
    main()
