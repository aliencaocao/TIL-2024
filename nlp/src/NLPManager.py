import logging
import re
from copy import deepcopy
from typing import Optional, Union

import orjson
from exllamav2 import ExLlamaV2, ExLlamaV2Cache, ExLlamaV2Config, ExLlamaV2Tokenizer
from exllamav2.generator import ExLlamaV2BaseGenerator, ExLlamaV2Sampler
from pydantic import BaseModel, ValidationError

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.DEBUG,
    handlers=[
        logging.StreamHandler()
    ])


class control_turret(BaseModel):
    heading: str = ''  # if regex detected then no such field will be exec
    tool: str = ''  # if regex detected then no such field will be exec
    target: str


class control_turret_optional_target(BaseModel):
    heading: str = ''  # if regex detected then no such field will be exec
    tool: str = ''  # if regex detected then no such field will be exec
    target: str = ''  # on retry target may still be empty, in this case we keep the other fields


class NLPManager:
    def __init__(self, model_path: str):
        logging.info(f"Initializing NLPManager using {model_path}")
        self.model_path = model_path
        config = ExLlamaV2Config(self.model_path)
        temperature = 0.0
        batch_size = 4  # from eval runner
        self.max_new_tokens = 128
        config.max_seq_len = 512
        config.max_batch_size = batch_size
        config.max_output_len = 1  # no need logit => just set to 1, reduce VRAM bandwidth usage a lot
        model = ExLlamaV2(config)
        cache = ExLlamaV2Cache(model, lazy=True, batch_size=config.max_batch_size)
        model.load_autosplit(cache)
        self.tokenizer = ExLlamaV2Tokenizer(config)
        self.generator = ExLlamaV2BaseGenerator(model, cache, self.tokenizer)
        self.settings = ExLlamaV2Sampler.Settings()
        self.settings.temperature = temperature

        self.heading_parse_regex = re.compile(r'(?:zero|one|two|three|four|five|six|seven|eight|nine|niner)\s*(?:zero|one|two|three|four|five|six|seven|eight|nine|niner)\s*(?:zero|one|two|three|four|five|six|seven|eight|nine|niner)')
        self.heading_missing_quotes_regex = re.compile(r'heading=(\d{3})')
        self.words_to_numbers = {
            'one': '1',
            'two': '2',
            'three': '3',
            'four': '4',
            'five': '5',
            'six': '6',
            'seven': '7',
            'eight': '8',
            'nine': '9',
            'niner': '9',  # a special one in their training data
            'zero': '0'
        }
        known_tools = {'anti-air artillery', 'drone catcher', 'electromagnetic pulse', 'emp', 'interceptor jets', 'machine gun', 'surface-to-air missiles'}
        self.known_tools_regex = re.compile("|".join(tool.replace(' ', '[ _]?') for tool in known_tools))

        self.give_none_if_not_specified_string = ' Give None if not specified.'
        functions = [
            {
                "name": "control_turret",
                "description": "Control the turret by giving it heading, tool to use and target description",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "heading": {"type": "string", "description": f"Heading of target in three arabic numbers and multiples of five (005 to 360).{self.give_none_if_not_specified_string}"},
                        "tool": {"type": "string", "description": f"Tool to use or deploy.{self.give_none_if_not_specified_string}"},
                        "target": {"type": "string", "description": f"Description of the target or enemy, exclude any quantifiers like 'the' or 'a'. It is a phrase that describe the appearance of the target like its color and type. Include ONLY the appearance and NOTHING else like its heading.{self.give_none_if_not_specified_string}"}
                    },
                    "required": ["heading", "tool", "target"],
                },
            }
        ]
        functions_without_tool = deepcopy(functions)
        del functions_without_tool[0]['parameters']['properties']['tool']
        functions_without_tool[0]['parameters']['required'].remove('tool')

        functions_without_heading = deepcopy(functions)
        del functions_without_heading[0]['parameters']['properties']['heading']
        functions_without_heading[0]['parameters']['required'].remove('heading')

        functions_without_heading_and_tool = deepcopy(functions_without_heading)
        del functions_without_heading_and_tool[0]['parameters']['properties']['tool']
        functions_without_heading_and_tool[0]['parameters']['required'].remove('tool')

        self.functions_string = orjson.dumps(functions).decode('utf-8')
        self.functions_without_heading_string = orjson.dumps(functions_without_heading).decode('utf-8')
        self.functions_without_tool_string = orjson.dumps(functions_without_tool).decode('utf-8')
        self.functions_without_heading_and_tool_string = orjson.dumps(functions_without_heading_and_tool).decode('utf-8')

        self.system_prompt = 'Convert the given instruction to turret into a function call to the control_turret function. Strictly ONLY one call is needed. If there are multiple instructions, use the FIRST one only.'
        if self.system_prompt: self.system_prompt += '\n'

    def format_prompt(self, user_query: str, remove_repeat: bool = True, on_retry: bool = False) -> tuple[str, Optional[str], Optional[str]]:
        user_query = user_query.lower()
        actual_functions_string = self.functions_string

        append_to_query = ''
        maybe_known_heading = None
        # heading_regex_parsed = None
        heading_regex_parsed = self.heading_parse_regex.search(user_query)
        if heading_regex_parsed:
            heading_regex_parsed_loc = heading_regex_parsed.span()
            heading_regex_parsed = heading_regex_parsed.group(0).split()
            heading_regex_parsed = ''.join([self.words_to_numbers[word.lower()] for word in heading_regex_parsed])
            try:
                heading_int = int(heading_regex_parsed)
            except ValueError:
                heading_int = 0
            if (heading_int < 5 or heading_int > 360) or heading_int % 5:  # heading is in multiples of 5 and no 000, so if this line is True then heading is wrong
                logging.info(f"Regex heading failed: {heading_regex_parsed}")
            else:
                maybe_known_heading = heading_regex_parsed
                logging.info(f"Detected known heading: {maybe_known_heading}")
                actual_functions_string = self.functions_without_heading_string
                append_to_query += f' It is known that the heading is "{maybe_known_heading}".'

        # maybe_known_tool = None
        maybe_known_tool = self.known_tools_regex.search(user_query)
        if maybe_known_tool:
            maybe_known_tool_loc = maybe_known_tool.span()
            maybe_known_tool = maybe_known_tool.group(0)
            logging.info(f"Detected known tool: {maybe_known_tool}")
            actual_functions_string = self.functions_without_heading_and_tool_string if maybe_known_heading else self.functions_without_tool_string
            append_to_query += f' It is known that the tool is "{maybe_known_tool}".'

        if 'repeat' in user_query.lower():
            if remove_repeat:
                repeat_loc = user_query.index('repeat')
                if (maybe_known_heading and repeat_loc > heading_regex_parsed_loc[0]) and (maybe_known_tool and repeat_loc > maybe_known_tool_loc[0]):
                    logging.info(f"Removing repeat from query: {user_query}")
                    user_query = user_query.split('repeat')[0].strip()
                    logging.info(f"New query: {user_query}")
                else:
                    logging.info(f"Repeat is after heading or known weapon, not removing repeat")
        elif on_retry:  # means retrying but there was no repeat in the first place so model giving wrong answer is not due to removal of repeat. In this case disable the give None prompt and let it cook
            logging.warning('Retrying but no repeat in query. Disabling give None prompt and let it COOK')
            actual_functions_string = actual_functions_string.replace(self.give_none_if_not_specified_string, '')

        user_query += append_to_query

        return f"{self.system_prompt}### Instruction: <<function>>{actual_functions_string}\n<<question>>{user_query}\n### Response: ", maybe_known_heading, maybe_known_tool

    def qa(self, context: list[str], on_retry: bool = False) -> list[dict[str, str]]:
        result_list = []
        logging.debug('Predicting')
        logging.debug(f'Original prompts: {context}')
        prompt, maybe_known_heading, maybe_known_weapon = zip(*[self.format_prompt(c, remove_repeat=not on_retry, on_retry=on_retry) for c in context])
        # print(len(self.tokenizer.encode(prompt[0], add_bos=True)[0]))
        result = self.generator.generate_simple(list(prompt), self.settings, self.max_new_tokens, add_bos=True, completion_only=True)
        for p, r, h, w in zip(context, result, maybe_known_heading, maybe_known_weapon):
            p = p.lower().strip()
            r = r.strip()[len('<<function>>'):]
            logging.debug(f'Raw response: {r}')
            try:
                if on_retry:
                    r = r.replace('control_turret', 'control_turret_optional_target')
                potential_heading = self.heading_missing_quotes_regex.search(r)
                if potential_heading:
                    logging.info(f'Unquoted heading found, fixing {potential_heading.group(1)} -> "{potential_heading.group(1)}"')
                    potential_heading = potential_heading.group(1)
                    r = r.replace(potential_heading, f'"{potential_heading}"')
                func_call_evaled: Union[control_turret, control_turret_optional_target] = eval(r)
                heading, tool, target = h or func_call_evaled.heading, w or func_call_evaled.tool, func_call_evaled.target
            except (ValidationError, SyntaxError) as e:
                if on_retry:
                    logging.error(f"Error evaluating function call after retry: {e}")
                    return [{"heading": "", "tool": "", "target": ""}]
                else:
                    logging.warning(f'Retrying due to {e}')
                    result_list.extend(self.qa([p], on_retry=True))
            except Exception as e:
                logging.error(f"Error evaluating function call: {e}")
                result_list.append({"heading": "", "tool": "", "target": ""})
            else:
                # remove some common errors
                tool = tool.replace('engage', '').strip()
                tool = tool.replace('deploy', '').strip()
                tool = tool.replace('use', '').strip()
                target = target.replace('target', '').strip()

                # Sometimes it give string "None" so when eval it dont get evaluated to None but a non-empty string, failing the checks
                heading = None if "None" in heading else heading
                tool = None if "None" in tool else tool
                target = None if "None" in target else target

                if tool and tool in p and target and target in p:
                    result_list.append({"heading": heading, "tool": tool, "target": target})
                else:
                    if on_retry:
                        logging.error(f"Tool or target not in query after retry: tool: {tool} target: {target}, just leaving it as it is and huff copium")
                        result_list.append({"heading": heading, "tool": tool, "target": target})
                    else:
                        logging.warning(f"Tool or target not in query: tool: {tool} target: {target}, retrying")
                        result_list.extend(self.qa([p], on_retry=True))
        return result_list


if __name__ == "__main__":
    from tqdm import tqdm
    # model_name = 'gorilla-openfunctions-v2-5.0bpw-h6-exl2'
    model_name = 'gorilla-openfunctions-v2-TIL24-r16-a16-ctx768-v2-5.0bpw-h6-exl2'
    nlp_manager = NLPManager(f"models/{model_name}/")
    # result = nlp_manager.qa(['Control addressing air defense turrets, prepare to deploy surface-to-air missiles. Heading zero five five. I repeat, heading zero five five. Target identified as yellow, orange, and green helicopter. Engage and neutralize the threat. Over.',
    #                          'Control here, deploy an electromagnetic pulse (EMP) in the heading of three four five to neutralize the green fighter jet.',
    #                          'Control to all turrets, I repeat, Control to all turrets. Deploy anti-air artillery towards heading three zero five. Target is green, red, and purple missile. I need immediate response, over.',
    #                          'Control to air defense turrets, I repeat, Control to air defense turrets. Deploy EMP tool against the purple, blue, and silver helicopter at heading zero one five. Engage target immediately. Over.'])
    # print(result)
    # exit()
    all_answers = []

    with open("../../data/nlp.jsonl", "r") as f:
        instances = [orjson.loads(line.strip()) for line in f if line.strip() != ""]
    # with open("gpt_eval.jsonl", "r") as f:
    #     instances = [orjson.loads(line.strip()) for line in f if line.strip() != ""]
    batch_size = 4
    instances = instances  # take the first 400 train samples for now for eval
    for index in tqdm(range(0, len(instances), batch_size)):
        _instances = instances[index: index + batch_size]
        input_data = {
            "instances": [
                {"key": _instance["key"], "transcript": _instance["transcript"]}
                for _instance in _instances
            ]
        }
        transcripts = [instance["transcript"] for instance in input_data["instances"]]
        # each is a dict with one key "transcript" and the transcription as a string
        answers = nlp_manager.qa(transcripts)
        answers = [{"key": _instance["key"], **answer} for _instance, answer in zip(_instances, answers)]
        print(answers)
        all_answers.extend(answers)

    with open(f'eval_outputs/{model_name}-{len(instances)}samples.json', 'wb+') as f:
        f.write(orjson.dumps(all_answers))
    print('Results saved to', f'eval_outputs/{model_name}-{len(instances)}samples.json')
