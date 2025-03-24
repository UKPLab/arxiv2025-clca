import copy
import logging
import os
import random
import sys
from typing import Dict

import datasets
import numpy as np
import torch
import transformers
from datasets import load_dataset
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


def smart_tokenizer_and_embedding_resize(
        special_tokens_dict: Dict,
        tokenizer: transformers.PreTrainedTokenizer,
        model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.
    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def my_set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    g = torch.Generator()
    g.manual_seed(seed)


TEMPLATE = """
    Scenario: {setting}
    Information about {name}:
    Background: {background}
    Occupation: {occupation}
    Imagine you are {name}, your task is to act/speak like {name}, and a member from the {culture} culture.
    """

CULT2TEPLATE = {
    "vanilla": TEMPLATE,
}

TEMPLATE_INTENT = """
    Scenario: {setting}
    There are two participants in this conversation {name} and {name2}:
    Your task is to predict the intent of participants in the context of social or cultural expectations of {culture} in a short sentence.
    """

CULT2TEPLATE_INTENT = {
    "intent": TEMPLATE_INTENT,
}


def get_sys_prompt_from_personas(data, perspective=1, template_key="vanilla"):
    personas = data["persona"]
    culture = personas["culture"].strip()
    name1, name2 = personas["persona"].split(" and ")
    name1 = name1.strip()
    name2 = name2.strip()
    person1 = personas["person1"]
    person2 = personas["person2"]
    template = CULT2TEPLATE.get(template_key)
    if perspective == 1:
        return template.format(setting=personas["setting"], culture=culture,
                               name=name1, background=person1["background"].strip(),
                               occupation=person1["occupation"].strip())
    else:
        return template.format(setting=personas["setting"], culture=culture,
                               name=name2, background=person2["background"].strip(),
                               occupation=person2["occupation"].strip())


def get_sys_prompt_from_personas_intent(data, perspective=1, template_key="intent"):
    personas = data["persona"]
    culture = personas["culture"].strip()
    name1, name2 = personas["persona"].split(" and ")
    name1 = name1.strip()
    name2 = name2.strip()
    template = CULT2TEPLATE_INTENT.get(template_key)
    return template.format(setting=personas["setting"], culture=culture,
                           name=name1,
                           name2=name2), name1, name2


def get_prompt(sample, perspective=1, model="llama", template_key="vanilla"):
    system_prompt = get_sys_prompt_from_personas(sample, perspective=perspective, template_key=template_key)
    msg = [
        {"role": "system", "content": system_prompt},
    ]
    if perspective == 2 and ("mistral" in model):
        msg.append(
            {"role": "user", "content": " "}
        )
    for i, turns in enumerate(sample["dialog"]):
        if perspective == 1:
            user_content = turns["model_inquirer"].strip()
            agent_content = turns["model_responder"].strip()
            msg.append(
                {"role": "user", "content": user_content}
            )
            msg.append(
                {"role": "assistant", "content": agent_content}
            )
        else:
            user_content = turns["model_responder"].strip()
            agent_content = turns["model_inquirer"].strip()
            msg.append(
                {"role": "assistant", "content": agent_content}
            )
            msg.append(
                {"role": "user", "content": user_content}
            )
    return msg


def get_prompt_intent(sample, perspective=1, model="llama", template_key="intent"):
    system_prompt, name1, name2 = get_sys_prompt_from_personas_intent(sample, perspective=perspective,
                                                                      template_key=template_key)
    msg = [
        {"role": "system", "content": system_prompt},
    ]

    for i, turns in enumerate(sample["dialog"]):
        model_inquirer_intent = turns.get("model_inquirer_intent") if turns.get("model_inquirer_intent") else "unknown"
        model_responder_intent = turns.get("model_responder_intent") if turns.get(
            "model_responder_intent") else "unknown"
        turn_data = []
        turn_data.append({"role": "user", "content": f"{name1}: " + turns["model_inquirer"].strip()})
        turn_data.append({"role": "assistant", "content": "Intent: " + model_inquirer_intent + "\n"})
        turn_data.append({"role": "user", "content": f"{name2}: " + turns["model_responder"].strip()})
        turn_data.append({"role": "assistant", "content": "Intent: " + model_responder_intent + "\n"})
        msg.extend(turn_data)
    return msg


TEMPLATE2FUN = {
    "vanilla": get_prompt,
    "intent": get_prompt_intent,
}

TEMPLATE_CULT2_SUFFIX = {
    "british": "",
    "chinese": "_zh",
    "mexican": "_mx",
    "german": "_de",
    "japanese": "_jp",
}


def train(cfg):
    # HF parser
    cfg.exp_name = cfg.exp_name.format(template_style=cfg.template_style,
                                       suffix=cfg.train_data_type)
    cfg.evaluation.exp_name = cfg.exp_name  # make sure the same name
    template_suffix = TEMPLATE_CULT2_SUFFIX.get(cfg.template_culture.lower(), "")

    training_args = TrainingArguments(
        seed=cfg.seed,
        data_seed=42,
        output_dir=os.path.join(cfg.output_dir, cfg.architecture,
                                cfg.evaluation.culture.lower(),
                                cfg.exp_name),
        learning_rate=cfg.lr,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        eval_steps=cfg.eval_steps,
        num_train_epochs=cfg.num_train_epochs,
        fp16=True
    )
    my_set_seed(cfg.seed)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if cfg.should_log:
        transformers.utils.logging.set_verbosity_info()

    log_level = logging.DEBUG
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Device: {cfg.device}"
        + f"16-bits training: {cfg.fp16}"
    )
    cfg.train_data_path = cfg.train_data_path.format(culture=cfg.evaluation.culture.lower(),
                                                     template_style="intent" if cfg.template_style == "vanilla" else cfg.template_style,
                                                     train_data_type=cfg.train_data_type)

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}

    raw_datasets = load_dataset('json', data_files={"train": cfg.train_data_path})

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name_or_path,
        cache_dir="/storage/ukp/shared/shared_model_weights/",
        local_files_only=True,
        torch_dtype=torch.float16,
        device_map=device_map,
        quantization_config=quantization_config
    )

    if "llama" in cfg.model_name_or_path:
        parsed_model_name = "llama"
    elif "mistral" in cfg.model_name_or_path:
        parsed_model_name = "mistral"
    else:
        parsed_model_name = "qwen"

    tokenizer_class = AutoTokenizer
    tokenizer = tokenizer_class.from_pretrained(
        cfg.model_name_or_path,
        padding_side="left",
    )

    print("MODEL TOKENS")
    print(tokenizer.eos_token, tokenizer.pad_token, tokenizer.bos_token)
    print(len(tokenizer))
    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )
    print("MODEL TOKENS AFTER")
    print(tokenizer.eos_token, tokenizer.pad_token, tokenizer.bos_token)
    print(len(tokenizer))

    model = prepare_model_for_kbit_training(model)

    if not cfg.eval_only:
        lora_config = LoraConfig(
            r=cfg.lora.lora_r,
            lora_alpha=cfg.lora.lora_alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=cfg.lora.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            modules_to_save=["lm_head", "embed_tokens"]
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    def create_conversation(sample):
        msg = TEMPLATE2FUN.get("vanilla")(sample, perspective=1, model=parsed_model_name,
                                          template_key=f"vanilla{template_suffix}")
        return {"messages": msg}

    def create_conversation_rev(sample):
        msg = TEMPLATE2FUN.get("vanilla")(sample, perspective=2, model=parsed_model_name,
                                          template_key=f"vanilla{template_suffix}")
        return {"messages": msg}

    def create_conversation_intent(sample):
        msg = TEMPLATE2FUN.get("intent")(sample, perspective=2, model=parsed_model_name,
                                         template_key=f"intent{template_suffix}")
        return {"messages": msg}

    copy_train_data = copy.deepcopy(raw_datasets["train"])
    copy_train_data2 = copy.deepcopy(raw_datasets["train"])

    copy_train_data = copy_train_data.map(
        create_conversation_rev,
        num_proc=cfg.preprocessing_num_workers,
        remove_columns=next(iter(raw_datasets.values())).column_names,
        desc="preprocess train data set", batched=False)

    train_data = raw_datasets["train"].map(
        create_conversation,
        num_proc=cfg.preprocessing_num_workers,
        remove_columns=next(iter(raw_datasets.values())).column_names,
        desc="preprocess train data set", batched=False)
    if "intent" in cfg.template_style:
        train_data_intent = copy_train_data2.map(
            create_conversation_intent,
            num_proc=cfg.preprocessing_num_workers,
            remove_columns=next(iter(raw_datasets.values())).column_names,
            desc="preprocess train data set", batched=False)

    print("dataset before merge", len(train_data), len(copy_train_data))
    if cfg.template_style == "vanilla":
        train_data = datasets.concatenate_datasets([train_data, copy_train_data])
    else:
        train_data = datasets.concatenate_datasets([train_data, copy_train_data, train_data_intent])
    print("dataset after merge", len(train_data))
    chat_template = None
    if not cfg.eval_only:
        if "llama" in cfg.model_name_or_path:
            instruction_template = "<|start_header_id|>user<|end_header_id|>"
            response_template = "<|start_header_id|>assistant<|end_header_id|>"
        elif "mistral" in cfg.model_name_or_path:
            chat_template = """{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] | trim + '\n\n' %}{% set messages = messages[1:] %}{% else %}{% set system_message = '' %}{% endif %}{{ bos_token + system_message}}
    {% for message in messages %}
    {% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}
        {{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}
    {% endif %}
    {% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] | trim + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' ' + message['content'] | trim + eos_token }}{% endif %}
    {% endfor %}"""
            tokenizer.chat_template = chat_template
            instruction_template = "[INST]"
            response_template = "[/INST]"
        else:  # qwen
            instruction_template = "<|im_start|>user"
            response_template = "<|im_start|>assistant"

        collator = DataCollatorForCompletionOnlyLM(instruction_template=instruction_template,
                                                   response_template=response_template,
                                                   tokenizer=tokenizer, mlm=False)

        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=train_data,
            data_collator=collator,
            max_seq_length=2048,
            peft_config=lora_config,
            tokenizer=tokenizer,
            dataset_kwargs={
                "add_special_tokens": False,  # We template with special tokens
                "append_concat_token": False,  # No need to add additional separator token
            }
        )
        model.config.use_cache = False
        if torch.__version__ >= "2" and sys.platform != "win32":
            model = torch.compile(model)
        trainer.train()
        model.save_pretrained(os.path.join(cfg.output_dir, cfg.architecture,
                                           cfg.evaluation.culture.lower(),
                                           cfg.exp_name, "lora"),
                              save_embedding_layers=True)
    return os.path.join(cfg.output_dir, cfg.architecture, cfg.evaluation.culture.lower(), cfg.exp_name, "lora")
