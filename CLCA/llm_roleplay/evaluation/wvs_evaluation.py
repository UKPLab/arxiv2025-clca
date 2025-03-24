import json
import os
import re

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import tqdm
from llm_roleplay.common.model import Model
from scipy.spatial import distance

PAT = re.compile("([0-9]\s)(.*)")
JP_PAT = re.compile(r"([0-9])\s(在1到|1表示|1 は|から).*")

PERSONA_PROMPT = """
    You are currently living in {country} and here is you basic demographic information:
    Settlement: {settlement}, {region}
    Gender: {gender}
    Age: {age}
    Born in {country}: {born}
    Marital status: {marital}
    Number of people in household: {household}
    Education: {education}
    Profession: {profession}
    Employment: {employeed}
    Class: {classes}
"""

PERSONA_PROMPT_ZH = """
    您目前居住在中国，以下是您的基本人口统计信息：
    居住地：{settlement}, {region}
    性别：{gender}
    年龄：{age}
    出生地是否在中国：{born}
    婚姻状况：{marital}
    家庭成员人数：{household}
    教育程度：{education}
    职业：{profession}
    就业状态：{employeed}
    社会阶层：{classes}
"""

PERSONA_PROMPT_JP = """
    あなたは現在日本に住んでおり、以下があなたの基本的な人口統計情報です：
    居住地：{settlement}、{region}
    性別：{gender}
    年齢：{age}
    出身国：{born}
    配偶者の有無：{marital}
    世帯人数：{household}
    学歴：{education}
    職業：{profession}
    雇用状況：{employeed}
    階級：{classes}
"""

PERSONA_PROMPT_DE = """
    Sie leben derzeit in Deutschland und hier sind Ihre grundlegenden demografischen Informationen:
    Wohnort: {settlement}, {region}
    Geschlecht: {gender}
    Alter: {age}
    Geboren in Deutschland: {born}
    Familienstand: {marital}
    Anzahl der Personen im Haushalt: {household}
    Bildung: {education}
    Beruf: {profession}
    Beschäftigungsstatus: {employeed}
    Klasse: {classes}
"""

PERSONA_PROMPT_MX = """
    Actualmente vives en Mexicanos y aquí está tu información demográfica básica:
    Asentamiento: {settlement}, {region}
    Género: {gender}
    Edad: {age}
    Nacido en Mexicanos: {born}
    Estado civil: {marital}
    Número de personas en el hogar: {household}
    Educación: {education}
    Profesión: {profession}
    Empleo: {employeed}
    Clase: {classes}
"""

DATA_PROMPT = """
    Please answer the following question, output the integer option when instructed, don't explain:
    QUESTION: {question}
    ANSWER: 
"""

DATA_PROMPT_ZH = """
    请回答以下问题，输出整数选项，不要解释：
    QUESTION: {question}
    ANSWER: 
"""

DATA_PROMPT_DE = """
    Bitte beantworten Sie die folgende Frage, geben Sie die Ganzzahl-Option aus, ohne Erklärungen:
    QUESTION: {question}
    ANSWER: 
"""

DATA_PROMPT_JP = """
    次の質問に回答してください。整数の選択肢を出力し、説明はしないでください。
    QUESTION: {question}
    ANSWER: 
"""

DATA_PROMPT_MX = """
    Por favor responda la siguiente pregunta, dé la opción en número entero, no explique:
    QUESTION: {question}
    ANSWER: 
"""

CULT2DATA_PROMPT = {
    "british": DATA_PROMPT,
    "chinese": DATA_PROMPT_ZH,
    "german": DATA_PROMPT_DE,
    "japanese": DATA_PROMPT_JP,
    "mexican": DATA_PROMPT_MX
}

CULT2PERSONA_PROMPT = {
    "british": PERSONA_PROMPT,
    "chinese": PERSONA_PROMPT_ZH,
    "german": PERSONA_PROMPT_DE,
    "japanese": PERSONA_PROMPT_JP,
    "mexican": PERSONA_PROMPT_MX
}

KEY2Q = {
    "gender": "Q260",
    "age": "X003R",
    "born": "Q263",
    "household": "Q270",
    "marital": "Q273",
    "profession": "Q281",
    "education": "Q275",
    "employeed": "Q279",
    "classes": "Q287"
}

# To account for model verbalize the options in multilingual
ANS2NUM = {
    "好": 1,
    "坏": 3,
    "良い": 1,
    "悪い": 3,
    "good": 1,
    "bad": 3,
    "gut": 1,
    "schlecht": 3,
    "bueno": 1,
    "malo": 3,
    "yes": 1,
    "no": 2,
    "no 2": 2,
    "是": 1,
    "1 是": 1,
    "1 ja": 1,
    "1 sí": 1,
    "否": 2,
    "はい": 1,
    "いいえ": 2,
    "ja": 1,
    "nein": 2,
    "si": 1,
    "非常不同意": [4, 5],
    "very不同意": [4, 5],
    "反对": 4,
    "非常同意": 1,
    "強く賛成": 1,
    "強烈に賛成": 1,
    "強く反対": [4, 5],
    "stimme überhaupt nicht zu": [4, 5],
    "totalmente en desacuerdo": [4, 5],
    "en desacuerdo": 4,
    "stimme voll zu": 1,
    "totalmente de acuerdo": 1,
    "de acuerdo": 2,
    "sehr wichtig": 1,
    "überhaupt nicht wichtig": 4,
    "非常に重要": 1,
    "全く重要ではない": 4,
    "非常重要": 1,
    "非常非常重要": 1,
    "非常 important": 1,
    "很重要": 2,
    "一点也不重要": 4,
    "非常不重要": 4,
    "muy importante": 1,
    "nada importante": 4,
}

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"


def smart_tokenizer_and_embedding_resize(
        special_tokens_dict,
        tokenizer,
        model,
):
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def get_prompt_w_system_prompt(persona_prompt, user_prompt):
    return [
        {"role": "system", "content": persona_prompt},
        {"role": "user", "content": user_prompt},
    ]


def get_data_prompt(data, metadata, qid, culture="british"):
    question = metadata.get(qid)["question"]
    label = data[qid]
    data_prompt = CULT2DATA_PROMPT.get(culture, CULT2DATA_PROMPT.get("british"))
    return label, data_prompt.format(question=question)


def get_persona_prompt(data, culture="british"):
    prompt = CULT2PERSONA_PROMPT.get(culture, CULT2PERSONA_PROMPT.get("british"))
    return prompt.format(country=data["B_COUNTRY"], settlement=data["H_SETTLEMENT"], region=data["H_URBRURAL"],
                         gender=data[KEY2Q.get("gender")], born=data[KEY2Q.get("born")],
                         marital=data[KEY2Q.get("marital")], age=data[KEY2Q.get("age")],
                         household=data[KEY2Q.get("household")], education=data[KEY2Q.get("education")],
                         profession=data[KEY2Q.get("profession")], employeed=data[KEY2Q.get("employeed")],
                         classes=data[KEY2Q.get("classes")])


def load_metadata(codebook_fn, qa_metadata_fn):
    with open(codebook_fn, 'r') as f:
        codebook = json.load(f)
    with open(qa_metadata_fn, 'r') as f:
        question_metadata = json.load(f)

    return codebook, question_metadata


def load_data(data_dir):
    persona_data = pd.read_csv(f"{data_dir}/full_demographic_qa.tsv", delimiter="\t")
    value_data = pd.read_csv(f"{data_dir}/full_value_qa.tsv", delimiter="\t")
    return persona_data.merge(value_data, on=['D_INTERVIEW'])


def deep_parsing(answer, min_scale, max_scale):
    ans = re.split(r"[:：]", answer)[-1]
    for phrase in ["。", "**", "la respuesta es", "宗教", "ich bi", "die antwort ist"]:
        ans = ans.replace(phrase, "")
    ans = ans.split("(")[0].strip()
    if ans.isnumeric():
        idx = int(float(ans))
        if idx < 0:
            idx = 0
        if min_scale <= idx <= max_scale:
            return idx
        else:
            return -1
    idx = ANS2NUM.get(ans, ans)
    if not idx:
        return -1
    elif isinstance(idx, list):
        return max_scale
    else:
        return idx


def final_check(answer, min_scale, max_scale):
    match = JP_PAT.match(answer)
    if match:
        first_group = match.group(1)
        if first_group.isdigit():
            if min_scale <= int(first_group) <= max_scale:
                return int(first_group)
            else:
                return -1
    return answer


def parse_answer(answer, min_scale, max_scale, stats):
    raw_answer = answer.split("ANSWER:")[-1].replace("assistant", "").replace(".", "").strip().split("\n")[0]
    raw_answer = raw_answer.replace("I think the answer is", "").strip()
    if raw_answer.isnumeric():
        idx = int(float(raw_answer))
        if idx < 0:
            idx = 0
        if min_scale <= idx <= max_scale:
            stats[idx] += 1
            return idx
    dp_answer = deep_parsing(raw_answer.lower(), min_scale, max_scale)
    if isinstance(dp_answer, str):
        dp_answer = final_check(dp_answer, min_scale, max_scale)
    if isinstance(dp_answer, str):
        print(f"Wrong answer: {raw_answer}", f"Parsed answer: {dp_answer}")
        stats[-1] += 1
        return -1
    else:
        stats[dp_answer] += 1
        return dp_answer
    stats[-1] += 1
    return -1


def get_oracle_answers(full_qa_df, qid, answer_scale_max):
    counts = full_qa_df[qid].value_counts().sort_index()
    vector = np.zeros(answer_scale_max)
    vector[counts.index.astype('int64')] = counts.values
    return vector


def get_person_answer_score(full_qa_df, persona_idx, model_answers, qids):
    row_by_idx = full_qa_df.iloc[int(persona_idx)]
    correct = 0
    t = []
    for c, qid in enumerate(qids):
        t.append(row_by_idx[qid])
        if row_by_idx[qid] == float(model_answers[c]):
            correct += 1
    return correct / len(qids)


@torch.no_grad()
def get_kl_div(qid, oracle, predicted, kl_qid):
    o = oracle / oracle.sum()
    q = predicted / predicted.sum()
    kl_div = F.kl_div(F.log_softmax(q.unsqueeze(0), dim=1), o.unsqueeze(0), reduction="batchmean")
    kl_qid[qid] = "{:.5f}".format(kl_div.item())


@torch.no_grad()
def js_distance(qid, oracle, predicted, tvd_qid):
    p = oracle / oracle.sum()
    q = predicted / predicted.sum()
    jsd = distance.jensenshannon(p.cpu().numpy(), q.cpu().numpy())
    tvd_qid[qid] = "{:.5f}".format(jsd.item())


def evaluate_model(cfg):
    cfg.data_dir = cfg.data_dir.format(culture=cfg.culture)
    df = load_data(data_dir=cfg.data_dir)
    print(f"Dataframe length: {len(df)}")
    cfg.qa_metadata_fn = cfg.qa_metadata_fn.format(culture=cfg.template_culture.lower())
    _, question_metadata = load_metadata(codebook_fn=cfg.codebook_fn, qa_metadata_fn=cfg.qa_metadata_fn)
    model = Model.get_model(cfg.model_responder, role="model_inquirer")
    print(model.tokenizer.eos_token, model.tokenizer.pad_token, model.tokenizer.bos_token)
    print(len(model.tokenizer))  # llama3.1 128256

    directory = f"{cfg.result_dir}/{cfg.exp_name}/{cfg.culture}"
    if not os.path.exists(directory):
        os.makedirs(directory)

    individual_results = open(cfg.result_dir + f"/{cfg.exp_name}/{cfg.culture}/{cfg.model_name}_idv_result.txt", 'w')
    per_person_responses = []
    if cfg.pretrained_peft:
        # to hack the fact that pad_token was set to eos token, but the embedding size not matching
        # for llama model only
        if "llama" in cfg.model_name or "mistral" in cfg.model_name:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
                tokenizer=model.tokenizer,
                model=model.model,
            )
            model.tokenizer.pad_token_id = model.tokenizer.eos_token_id
            print("MODEL TOKENS AFTER THE HACK")
            print(model.tokenizer.eos_token, model.tokenizer.pad_token, model.tokenizer.bos_token)
            print(len(model.tokenizer))  # llama3.1 128257
        model.model.load_adapter(cfg.pretrained_peft)
    cfg.model_responder.generate.do_sample = False
    qids = cfg.qids
    # +2 to shift index, add safegaurd bucket
    results = {q: [0] * (question_metadata[q]["answer_scale_max"] + 2) for q in qids}
    c = 0
    # default without replacements
    for index, row in tqdm.tqdm(df.sample(n=1000).iterrows(), total=1000):
        c += 1
        row_dict = row.to_dict()
        persona_prompt = get_persona_prompt(row_dict, cfg.template_culture.lower())
        per_person_response = [str(index)]
        for qid in qids:
            lable, data_prompt = get_data_prompt(row_dict, question_metadata, qid, cfg.template_culture.lower())
            prompt = get_prompt_w_system_prompt(persona_prompt, data_prompt)
            answer = model.generate(
                prompt=prompt,
                generate_cfg=(cfg.model_responder.generate),
            )
            answer = answer.replace(question_metadata.get(qid)["question"], "")
            idx = parse_answer(answer, question_metadata[qid]["answer_scale_min"],
                               question_metadata[qid]["answer_scale_max"], results[qid])
            per_person_response.append(str(idx))
        per_person_responses.append(per_person_response)
        individual_results.write("\t".join(per_person_response) + "\n")
    individual_results.close()
    json.dump(results, open(cfg.result_dir + f"/{cfg.exp_name}/{cfg.culture}/{cfg.model_name}_result.txt", 'w'))

    kl_qid = {}
    tvd_qid = {}
    for qid in qids:
        oracle = get_oracle_answers(df, qid, question_metadata[qid][
            "answer_scale_max"] + 2)  # +2 to shift index, add safegaurd bucket
        predicted = results[qid]
        p = torch.tensor(predicted[1:], dtype=torch.float32)  # remove the index 0
        q = torch.tensor(oracle[1:], dtype=torch.float32)  # remove the index 0
        get_kl_div(qid, oracle=q, predicted=p, kl_qid=kl_qid)
        js_distance(qid, oracle=q, predicted=p, tvd_qid=tvd_qid)

    json.dump(kl_qid, open(cfg.result_dir + f"/{cfg.exp_name}/{cfg.culture}/{cfg.model_name}_result_kl.txt", 'w'))
    json.dump(tvd_qid, open(cfg.result_dir + f"/{cfg.exp_name}/{cfg.culture}/{cfg.model_name}_result_jsd.txt", 'w'))

    total = []
    for per_person in per_person_responses:
        persona_idx = per_person[0]
        p_score = get_person_answer_score(df, persona_idx, per_person[1:], qids)
        total.append(p_score)
    json.dump(sum(total) / len(total),
              open(cfg.result_dir + f"/{cfg.exp_name}/{cfg.culture}/{cfg.model_name}_person_score.txt", 'w'))
