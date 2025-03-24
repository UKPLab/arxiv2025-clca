import json
import sys

from openai import OpenAI

client = OpenAI(
    api_key=sys.argv["OPENAI_API_KEY"]
)

TEMPLATE_GPT_QUALITY = """
    Please read the provided dialog between two people and their basic information, judge the quality of their conversation, output the quality and the confidence. The conversation is bad in quality if the it has many repeating sentences towards the end or the content doesn't align with the given setting.
    Quality choices: 1. good, 2. bad
    Confidence choices: 1. very confident, 2. confident, 3. not sure
    Here are the basic information of the participants in this conversation:
    {persona}
    Here is the dialog:
    {dialog}
    Please output the choice number only (don't explain) using the following schema:
    Quality: <choice>
    Confidence: <choice>
"""

TEMPLATE_GPT_ALIGNMENT = """
    Please read the provided dialog between two people and their basic information, judge if their conversation aligns with the {culture} culture, output the culture alignment and the confidence. 
    Culture alignment choices: 1. aligned to the culture, 2. not aligned to the culture
    Confidence choices: 1. very confident, 2. confident, 3. not sure
    Here are the basic information of the participants in this conversation:
    {persona}
    Here is the dialog:
    {dialog}
    Please output the choice number only (don't explain) using the following schema:
    Culture alignment: <choice>
    Confidence: <choice>
"""

TEMPLATE_META_QUALITY = """
    Please critic on the previous judgement output a meta label on the conversation's alignment to the {culture} culture and the confidence (don't judge the goal achievement). 
    Meta label choices: 1. good, 2. bad
    Confidence choices: 1. very confident, 2. confident, 3. not sure
    Here are the basic information of the participants in this conversation:
    {persona}
    Here is the dialog:
    {dialog}
    Previous judgement:
    {judgement}
    Please output the choice number only (don't explain) using the following schema:
    Meta label: <choice>
    Confidence: <choice>
    Critic: <critic>
"""


class JudgeModel:
    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def load(self, filename):
        candidate_dialog = []
        with open(filename, 'r') as f:
            for line in f.readlines():
                data = json.loads(line.strip())
                candidate_dialog.append(data)
        return candidate_dialog

    def _request(self, message):
        completion = client.chat.completions.create(model=self.cfg.name,
                                                    temperature=0.0,
                                                    messages=[{"role": "user", "content": message}],
                                                    max_tokens=self.cfg.max_tokens)
        return completion.choices[0].message.content

    def is_align_with_culture(self, dialog, culture, persona):
        calign, confidence = None, None
        msg = TEMPLATE_GPT_ALIGNMENT.format(culture=culture, persona=persona, dialog=dialog)
        response = self._request(msg)
        lines = response.split("\n")
        if lines[-1].startswith("Confidence"):
            confidence = lines[-1].split(":")[-1].strip()
        if lines[-2].startswith("Culture alignment"):
            calign = lines[-2].split(":")[-1].strip()

        return confidence, calign

    def is_good_quality(self, dialog, persona):
        quality, confidence = None, None
        msg = TEMPLATE_GPT_QUALITY.format(persona=persona, dialog=dialog)
        response = self._request(msg)
        lines = response.split("\n")
        if lines[-1].startswith("Confidence"):
            confidence = lines[-1].split(":")[-1].strip()
        if lines[-2].startswith("Quality"):
            quality = lines[-2].split(":")[-1].strip()
        if not confidence or not quality:
            print(response)
        return confidence, quality

    def meta_quality(self, dialog, persona, judge, culture):
        quality, confidence = None, None
        msg = TEMPLATE_META_QUALITY.format(culture=culture, persona=persona, dialog=dialog, judgement=judge)
        response = self._request(msg)
        lines = response.split("\n")
        if lines[0].strip().startswith("Meta label"):
            quality = lines[0].split(":")[-1].strip()
        if lines[1].strip().startswith("Confidence"):
            confidence = lines[1].split(":")[-1].strip()
        if lines[2].strip().startswith("Critic"):
            critic = lines[2].split(":")[-1].strip()
        return quality, confidence, critic

    def _get_persona_from_data(self, dialog):
        return json.dumps(dialog["persona"])

    def _get_dialog_from_data(self, dialog):
        turns = []

        for turn in dialog["dialog"]:
            data = """
                Turn: {turn_no}
                Person 1: {model_inquirer}
                Person 2: {model_responder}
            """.format(
                turn_no=turn["turn"],
                model_inquirer=turn["model_inquirer"],
                model_responder=turn["model_responder"],
            )
            turns.append(data)
        return "".join(turns), len(turns)

    def _normalize_calign(self, confidence, calign):
        if calign == "1" or calign.startswith("1"):
            calign = "aligned"
        elif calign == "2" or calign.startswith("2"):
            calign = "not aligned"
        else:
            pass

        if confidence == "1" or confidence.startswith("1"):
            confidence = "very confident"
        elif confidence == "2" or confidence.startswith("2"):
            confidence = "confident"
        else:
            confidence = "not sure"

        return calign, confidence

    # 1. aligned to the culture, 2. not aligned to the culture

    def _normalize_quality(self, confidence, quality):
        if quality == "1" or quality.startswith("1"):
            quality = "good"
        elif quality == "2" or quality.startswith("2"):
            quality = "bad"
        else:
            pass

        if confidence == "1" or confidence.startswith("1"):
            confidence = "very confident"
        elif confidence == "2" or confidence.startswith("2"):
            confidence = "confident"
        else:
            confidence = "not sure"

        return quality, confidence

    def check_alignment_and_goal(self, dialogs):
        stats = {"aligned": 0, "achieved": 0}
        nturns = []
        for c, dialog in enumerate(dialogs):
            print(f"Sample: {c}")
            culture = dialog["persona"]["culture"]
            persona = self._get_persona_from_data(dialog)
            dialog_flow, nturn = self._get_dialog_from_data(dialog)

            quality_confidence, quality = self.is_good_quality(dialog_flow, persona)
            quality, quality_confidence = self._normalize_quality(quality_confidence, quality)
            confidence, calign = self.is_align_with_culture(dialog_flow, culture, persona)
            calign, confidence = self._normalize_calign(confidence, calign)

            judge = f"culture_alignment: {calign}\nculture_alignment_confidence: {confidence}\n" \
                    f"quality: {quality}\nquality_confidence:{quality_confidence}"
            meta_quality, confidence, critic = self.meta_quality(dialog_flow, persona, judge, culture)
            meta_quality, meta_quality_confidence = self._normalize_quality(confidence, meta_quality)

            nturns.append(nturn)

            if calign == "aligned":
                stats["aligned"] += 1

            dialog["culture_alignment"] = calign
            dialog["culture_alignment_confidence"] = confidence
            dialog["quality"] = quality
            dialog["quality_confidence"] = quality_confidence
            dialog["meta_quality"] = meta_quality
            dialog["meta_critic"] = critic
            dialog["meta_quality_confidence"] = meta_quality_confidence
            print(calign, confidence, quality, quality_confidence, meta_quality, meta_quality_confidence)

        print(stats, sum(nturns) / len(nturns))

    def dump(self, dialogs, out_file):
        with open(out_file, 'w') as f:
            for d in dialogs:
                f.write(json.dumps(d) + "\n")


NAME2MODEL = {
    "openai": JudgeModel,
    "generic": JudgeModel,
}


def judge_dialog(filename, out_file, cfg):
    model_fn = NAME2MODEL.get(cfg.type, NAME2MODEL.get("generic"))
    judge = model_fn(cfg)
    dataset = judge.load(filename)
    judge.check_alignment_and_goal(dataset)
    judge.dump(dataset, out_file)
