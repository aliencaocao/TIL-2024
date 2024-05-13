# NLP

## Docker
```shell
docker build -t 12000sgd-nlp .
```
Test:
```shell
docker run -p 5002:5002 --gpus all -d 12000sgd-nlp
```

## Input

Text transcription of a turret voice command.

Example transcription: `"Target is red helicopter, heading is zero one zero, tool to deploy is surface-to-air missiles."`

## Output

JSON object containing three keys:

1. "target", for the target identified
2. "heading", for the heading of the target
3. "tool", for the tool to be deployed to neutralize the target

Example JSON:

```json
{
  "target": "red helicopter",
  "heading": "010",
  "tool": "surface-to-air missiles"
}
```

Function definition:
```python
self.give_none_if_not_specified_string = ' Give None if not specified.'
functions = [
            {
                "name": "control_turret",
                "description": "Control the turret by giving it heading, tool to use and target description",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "heading": {"type": "string", "description": f"Heading of target in three arabic numerals and multiples of five (005 to 360).{self.give_none_if_not_specified_string}"},
                        "tool": {"type": "string", "description": f"Tool to use/deploy.{self.give_none_if_not_specified_string}"},
                        "target": {"type": "string", "description": f"Description of the target/enemy, exclude any quantifiers like 'the' or 'a'.{self.give_none_if_not_specified_string}"}
                    },
                    "required": ["heading", "tool", "target"],
                },
            }
        ]
```

# ExLlamaV2
1. [Download](https://drive.google.com/file/d/1VLaP60DxsysOVPCQGFHGR8AR67hzhML6/view) OR Build wheel on Linux and T4 (same as docker runtime)
```shell
git clone https://github.com/turboderp/exllamav2
cd exllamav2
python setup.py bdist_wheel
```
If need to rebuild, delete the build folder then run build

Make sure wheel is in nlp/
2. Download model:
```shell
# pretrained mode calibrated on default set
huggingface-cli download LoneStriker/gorilla-openfunctions-v2-5.0bpw-h6-exl2 --local-dir models/gorilla-openfunctions-v2-5.0bpw-h6-exl2 --local-dir-use-symlinks False
```


# Evaluations
## Pretrained Gorilla OpenFunctionsV2 EXL 5.0bit hb6 calibrated on default set, eval on full train set
Remove duplicate but retry without removing if model cannot find one or more information. If retrying even though nothing has been removed (no 'repeat' in prompt), then remove the prompt that ask it to give None if not found to allow it to hallucinate (cook).

Using weapon instead of tool
- NLP mean score: 0.9679137333565905
- NLP detailed score: {'heading': 0.9968571428571429, 'target': 0.9515214959643531, 'tool': 0.9553625612482756}

Using tool instead of weapon
- NLP mean score: 0.9688114958493109
- NLP detailed score: {'heading': 0.9965714285714286, 'target': 0.9436904728324896, 'tool': 0.9661725861440147}

Conclusion: Using tool instead of weapon is better, all below result use tool.

With top_p = 0.9
- NLP mean score: 0.9685219499454794
- NLP detailed score: {'heading': 0.9965714285714286, 'target': 0.9433206141197737, 'tool': 0.9656738071452358}

Conclusion: dont use top_p is better, all below result do not specify top_p.

With regex weapon detection (**not** representative as overfitted to train set)
- NLP mean score: 0.9892643084428798
- NLP detailed score: {'heading': 0.9942857142857143, 'target': 0.9740786396143539, 'tool': 0.9994285714285714}

## Pretrained Gorilla OpenFunctionsV2 EXL 5.0bit hb6 calibrated on default set, eval on test set

## Pretrained Gorilla OpenFunctionsV2 EXL 5.0bit hb6 calibrated on default set + train set, eval on full train set

## Pretrained Gorilla OpenFunctionsV2 EXL 5.0bit hb6 calibrated on default set + train set, eval on test set