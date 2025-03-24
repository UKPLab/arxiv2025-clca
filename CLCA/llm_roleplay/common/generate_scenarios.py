import hashlib
import json
import re
from typing import Dict, List, Tuple

from llm_roleplay.common.culture_persona_var import COUNTRY2INGLEHART, COUNTRY2HOFSTEDE

pat = re.compile("(\[\[([a-zA-Z\s]+)\|([a-zA-Z\s]+)\]\])")

# Template adapted from:
# * Sotopia - Zhou et al., 2023 [https://arxiv.org/abs/2310.11667]
# * Sotopia Pi - Wang et al., 2024 [https://arxiv.org/abs/2403.08715]
FORMAT_TEMPLATE = """
        You can find {name}'s the conversation history here:\n
        {history}
        You are at Turn #{turn_number}.
        Your available action types are: 1. none, 2. non-verbal, 3. speak, 4. leave
        You should output 4. leave as the action as soon as if 1) you have achieved your own social goals, 2) or there is a clear outcome for the goal has achieved regardless of success of failure.
        Please only generate replies including the action type and the argument, following the schema:
        ACTION: <action_type>
        OUTPUT: <your_output>
        <your_output> is the utterance if choose to 3; speak, the physical non-verbal action if choose 2. non-verbal; a closing sentence if choose 4. leave; other wise, it is NONE.\n
        Please only output the response in English:
        """

IDX2FT = {
    "vanilla": FORMAT_TEMPLATE
}

IDX2HT = {
    "vanilla": "{name} - ACTION: {action}\nOUTPUT: {output}",
}

COUNTRY2CULTURE = {
    "china": "Chinese",
    "germany": "German",
    "japan": "Japanese",
    "mexico": "Mexican",
    "uk": "British",
}


def get_instruction(persona, name, history, turn_number, sys_prompt, user_prompt, format_template_key="vanilla",
                    model_name="llama"):
    sp = sys_prompt.format(SYSTEM=persona)
    up = user_prompt.format(USER_PROMPT=IDX2FT.get(format_template_key, 'vanilla').format(name=name, history=history,
                                                                                          turn_number=turn_number))
    if "gemma" in model_name:
        return [
            {"role": "user", "content": sp + "\n" + up},
        ]

    return [
        {"role": "system", "content": sp},
        {"role": "user", "content": up},
    ]


class ScenarioGenerator:
    TEMPLATE = """
    Here is the context of this interaction:
    Scenario: {setting}
    Participants: {persona}
    Both participants are {culture}, their culture dimensions are as follow:
    IngleHart Culture Dimension: {inglehart}
    Hofstade Culture Dimension: {hofstade}
    Information about {name}:
    Background: {background}
    Occupation: {occupation}
    Goal: {goal}
    Secrets: {secrets}
    Information about {name2}:
    Background: {background2}
    Occupation: {occupation2}
    Goal: Unknown\n
    Imagine you are {name}, your task is to act/speak like {name}, and a member from the {culture} culture that aligns with the provided culture dimensions, with {name}'s social goal in mind.
    """

    @staticmethod
    def get_scenarios(raw_scenarios) -> List[Tuple[str, Dict[str, str]]]:
        scenarios = []
        for scenario in raw_scenarios:
            culture = COUNTRY2CULTURE.get(scenario["culture"].lower(), scenario["culture"])
            name1, name2 = scenario["persona"].split(" and ")
            name1 = name1.strip()
            name2 = name2.strip()
            inglehart = COUNTRY2INGLEHART.get(culture)
            hofstade = COUNTRY2HOFSTEDE.get(culture)
            person1 = scenario["person1"]
            person2 = scenario["person2"]

            p1 = ScenarioGenerator.TEMPLATE.format(setting=scenario["setting"], persona=scenario["persona"],
                                                   inglehart=inglehart, hofstade=hofstade,
                                                   name=name1, background=person1["background"],
                                                   occupation=person1["occupation"],
                                                   goal=person1["goal"], secrets=person1["secrets"],
                                                   name2=name2, background2=person2["background"],
                                                   occupation2=person2["occupation"], culture=culture
                                                   )

            p2 = ScenarioGenerator.TEMPLATE.format(setting=scenario["setting"], persona=scenario["persona"],
                                                   inglehart=inglehart, hofstade=hofstade,
                                                   name=name2, background=person2["background"],
                                                   occupation=person2["occupation"],
                                                   goal=person2["goal"], secrets=person2["secrets"],
                                                   name2=name1, background2=person1["background"],
                                                   occupation2=person1["occupation"], culture=culture
                                                   )
            persona_hash = hashlib.md5(json.dumps(scenario).encode()).hexdigest()
            scenarios.append((p1, p2, name1, name2, persona_hash))

        return scenarios


def load_scenarios(filename):
    scenarios = []
    with open(filename, 'r') as f:
        for item in f.readlines():
            scenarios.append(json.loads(item))
    return scenarios
