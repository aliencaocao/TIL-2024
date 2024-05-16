# NLP

## Docker
Build base (only 1-time):
```shell
docker build -f base.dockerfile -t 12000sgd-nlp-base .
```
Build image with model:
```shell
docker build -t 12000sgd-nlp .
```
Test:
```shell
docker run -p 5002:5002 --gpus all -d 12000sgd-nlp
```
Submit:
```shell
docker tag 12000sgd-nlp asia-southeast1-docker.pkg.dev/dsta-angelhack/repository-12000sgdplushie/12000sgd-nlp:latest
docker push asia-southeast1-docker.pkg.dev/dsta-angelhack/repository-12000sgdplushie/12000sgd-nlp:latest
gcloud ai models upload --region asia-southeast1 --display-name '12000sgd-nlp' --container-image-uri asia-southeast1-docker.pkg.dev/dsta-angelhack/repository-12000sgdplushie/12000sgd-nlp:latest --container-health-route /health --container-predict-route /extract --container-ports 5002 --version-aliases default
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
huggingface-cli download LoneStriker/gorilla-openfunctions-v2-5.0bpw-h6-exl2 --local-dir src/models/gorilla-openfunctions-v2-5.0bpw-h6-exl2 --local-dir-use-symlinks False
```
3. Run calibration yourself:
```shell
# make sure you running from root dir instead of root/nlp
python exllamav2/convert.py -i nlp/src/models/gorilla-openfunctions-v2-TIL24-r16-a16-ctx768 -o nlp/src/models/exl2_tmp/ -cf nlp/src/models/gorilla-openfunctions-v2-TIL24-r16-a16-ctx768/ -b 5.0 -hb 6
```

# Evaluations
## Pretrained Gorilla OpenFunctionsV2 EXL 5.0bit hb6 calibrated on default set, eval on full train set
Remove duplicate but retry without removing if model cannot find one or more information. If retrying even though nothing has been removed (no 'repeat' in prompt), then remove the prompt that ask it to give None if not found to allow it to hallucinate (cook).

Using weapon instead of tool
- NLP mean score: 0.9679137333565905
- NLP detailed score: {'heading': 0.9968571428571429, 'target': 0.9515214959643531, 'tool': 0.9553625612482756}

Using tool instead of weapon **(BEST)**
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

With regex weapon detection and noun enum v1:
- NLP mean score: 0.9409983355064514
- NLP detailed score: {'heading': 0.99, 'target': 0.837852149376497, 'tool': 0.9951428571428571}

Conclusion: spacy as enum is bad as it contain too much noise.

With regex weapon detection and noun as check for LLM target v1: if disagree, use the longest noun as target:
- NLP mean score: 0.9423480519480519
- NLP detailed score: {'heading': 0.9942857142857143, 'target': 0.8333298701298701, 'tool': 0.9994285714285714}

With regex weapon detection and noun as check for LLM target v2: v1 & add to function definition that the target is a descriptive phrase stating appearance and form of target, replace LLM target with longest spacy noun after removing tools:
- NLP mean score: 0.9427309283309283
- NLP detailed score: {'heading': 0.9951428571428571, 'target': 0.8333356421356422, 'tool': 0.9997142857142857}

With regex weapon detection and noun as check for LLM target v3: v2 & keep only the longest noun phrase in LLM-detected target, and keep the same for LLM-detected tool (non regex found), do NOT just replace LLM target with spacy target as LLM is often better:
No work as Spacy will give a single letter as noun phrase

With regex weapon detection fresh from non-spacy, just adjusted prompt to be more descriptive on target and tool:
- NLP mean score: 0.9946673358387644
- NLP detailed score: {'heading': 0.9962857142857143, 'target': 0.991716293230579, 'tool': 0.996}

With regex weapon detection, moved regex heading to before LLM generation, improve func def to include optional arg " (some of them may be known already)" **DITCHED**
- NLP mean score: 0.9844925370925371
- NLP detailed score: {'heading': 0.986, 'target': 0.9817633255633256, 'tool': 0.9857142857142858}
**Have bug**  where target field may be missing, causing entire sample to be empty

With regex weapon detection, prompt more descriptive on target and tool + make target not mandatory on retry so allow it to not give target but keep other fields if it fails, instead of whole sample fail **(BEST)**:
NLP mean score: 0.9969136080850367
NLP detailed score: {'heading': 0.9997142857142857, 'target': 0.9915979671122528, 'tool': 0.9994285714285714}

With above + change func description in case of known heading or tool + add " A target can have multiple colors." to target desc:
- NLP mean score: 0.988973401974375
- NLP detailed score: {'heading': 0.9971428571428571, 'target': 0.972920205923125, 'tool': 0.9968571428571429}
Conclusion: worse

With regex weapon detection and color-based regex for target:


## Pretrained Gorilla OpenFunctionsV2 EXL 5.0bit hb6 calibrated on default set, eval on test set
with regex weapon detection
- Accuracy: 0.99190173
- Speed Score: 0.81024160 = 17min4s

With regex weapon detection, moved regex heading to before LLM generation, improve func def to include optional arg " (some of them may be known already)"
**Have bug** where target field may be missing, causing entire sample to be empty
- Accuracy: 0.9763647907647908
- Speed Score: 0.7861219566666666 = 19min15s like due to many retries

## TIL Trained Gorilla OpenFunctionsV2 EXL 5.0bit hb6 calibrated on default set, eval on full train set
**Have bug** where target field may be missing, causing entire sample to be empty
**Eval on train set, not representative!**
- NLP mean score: 0.9984081632653061
- NLP detailed score: {'heading': 0.9982857142857143, 'target': 0.998938775510204, 'tool': 0.998}


## Pretrained Gorilla OpenFunctionsV2 EXL 5.0bit hb6 calibrated on default set + train set, eval on full train set
without regex weapon detection
- NLP mean score: 0.9645239279239279
- NLP detailed score: {'heading': 0.9991428571428571, 'target': 0.9331017744160601, 'tool': 0.9613271522128665}

Conclusion: custom calibration is bad on OpenFunctionsV2

## Pretrained Gorilla OpenFunctionsV2 EXL 5.0bit hb6 calibrated on default set, eval on test set
with regex weapon detection, with `" It is known that the target is NOT {maybe_known_tool}."` instead of `" It is known that the tool is {maybe_known_tool}."`
- NLP mean score: 0.9823668997668997
- Timing score 0.8059028631481482 = 17min28s


## Pretrained Gorilla OpenFunctionsV2 EXL 3.0bit hb6 calibrated on default set, eval on test set
- Accuracy: 0.9813814370814371
- Speed Score: 0.8239338579629629 = 15min50s


## Pretrained Gorilla OpenFunctionsV2 EXL 4.0bit hb6 calibrated on default set, eval on test set
- Accuracy: 0.9862283494283495
- Speed Score: 0.8129256762962963 = 16min50s


## Pretrained Gorilla OpenFunctionsV2 EXL 8.0bit hb8 calibrated on default set, eval on test set
- Accuracy: 0.9867131979131979
- Speed Score: 0.7240222144444444 = 24min50sec


## Pretrained mzbac/Phi-3-mini-4k-instruct-function-calling EXL 5.0bit hb6 calibrated on default set, eval on full train set
Cannot follow instructions.