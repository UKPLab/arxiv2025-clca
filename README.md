<p align="center">
  <img src="static/round_clca.jpg" alt="Logo" height="200">
</p>
<div align="center">
    <h1 style="margin: 0">Cultural Learning-Based Culture Adaptation of Language Models</h1>
</div>
<br clear="left"/>
* Icons created with the help of DALL-E.

This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication.

---
### Setup
Install all Python requirements listed in `requirements.txt` (check [here](https://pytorch.org/get-started/locally/) 
to see how to install Pytorch on your system).

You can install the requirements.txt like:
```
pip install --upgrade pip
pip install -r requirements.txt
```
### Usage
To generate data, you need files containing social scenarios. 
An example starting scenario file is in `CLCA/data/scenarios/germany_scenarios.txt` (*coming soon*). 

To evaluate models with WVS data, please prepare the data using [WorldValuesBench](https://github.com/Demon702/WorldValuesBench) (commit: e5d959b0365b45fcae7a5a6668b5789612da336c).
Please put the extracted data of a culture into the corresponding folder `CLCA/data/WorldValuesBench/{culture}`.
To extract data for a specific culture, please refer to the WVS dataset and filter the records by the corresponding country code.
After the data extraction, you should have a `demographic_qa` and a `value_qa` file, in addition, you should have a `question_metadata` file and a `codebook.json`.

The hyper-parameters and paths in this repository are manged using Hydra. 
Please see example config files in `llm_roleplaying/configs`, there are a couple of places requiring updates.

The `model_inquirer` and `model_responder` in the config folder contain configurations for two LLMs used in role-play (Participant 1 and Participant 2).

If two models are the same, only the `model_inquirer` will be loaded to save memory.

The overall workflow of our work:
 1) Generate data (with judge and filtering, `generate_social_dialog.py`) 
 2) Generate intents (`intent_aug.py`) 
 3) Adaptation (`llm_adaptation.py`)

---
## Citation
If you find this repository useful, please cite the following paper:
```
@misc{liu2025clca,
  title={Cultural Learning-Based Culture Adaptation of Language Models},
  author={Chen Cecilia Liu and Anna Korhonen and Iryna Gurevych},
  year={2025},
  eprint={2504.02953},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={http://arxiv.org/abs/2504.02953},
}
```

## Contact
First author: first_name.last_name AT tu-darmstadt.de

https://www.ukp.tu-darmstadt.de/

https://www.tu-darmstadt.de/

Don't hesitate to send us an e-mail or report an issue, if something is broken (and it shouldn't be) or if you have further questions.
This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication. 

## License
CLCA is licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for the full license text.

### Acknowledgement
The role-playing code of this repository is modified upon the [LLM-roleplay](https://github.com/UKPLab/llm-roleplay). 
