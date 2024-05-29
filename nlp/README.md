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
docker run -p 5002:5002 --gpus all --network none -d 12000sgd-nlp
```
Submit:
```shell
docker tag 12000sgd-nlp asia-southeast1-docker.pkg.dev/dsta-angelhack/repository-12000sgdplushie/12000sgd-nlp:latest
docker push asia-southeast1-docker.pkg.dev/dsta-angelhack/repository-12000sgdplushie/12000sgd-nlp:latest
gcloud ai models upload --region asia-southeast1 --display-name '12000sgd-nlp' --container-image-uri asia-southeast1-docker.pkg.dev/dsta-angelhack/repository-12000sgdplushie/12000sgd-nlp:latest --container-health-route /health --container-predict-route /extract --container-ports 5002 --version-aliases default
```
Finals submission:
```shell
docker tag 12000sgd-nlp asia-southeast1-docker.pkg.dev/dsta-angelhack/repository-12000sgdplushie/12000sgd-nlp:finals
docker push asia-southeast1-docker.pkg.dev/dsta-angelhack/repository-12000sgdplushie/12000sgd-nlp:finals
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
3. Run calibration (if not using pretrained):
```shell
# make sure you running from root dir instead of root/nlp
python exllamav2/convert.py -i nlp/src/models/gorilla-openfunctions-v2-TIL24-r16-a16-ctx768-v2 -o nlp/src/models/exl2_tmp/ -cf nlp/src/models/gorilla-openfunctions-v2-TIL24-r16-a16-ctx768-v2-5.0bpw-h6-exl2/ -b 5.0 -hb 6
```
Or with existing measurement:
```shell
python exllamav2/convert.py -i nlp/src/models/gorilla-openfunctions-v2-TIL24-r16-a16-ctx768-v2 -o nlp/src/models/exl2_tmp/ -cf nlp/src/models/gorilla-openfunctions-v2-TIL24-r16-a16-ctx768-v2-3.0bpw-h6-exl2/ -b 3.0 -hb 6 -m nlp/src/models/exl2_tmp/measurement.json
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

With regex weapon detection, prompt more descriptive on target and tool + make target not mandatory on retry so allow it to not give target but keep other fields if it fails, instead of whole sample fail:
NLP mean score: 0.9969136080850367
NLP detailed score: {'heading': 0.9997142857142857, 'target': 0.9915979671122528, 'tool': 0.9994285714285714}

Above but without regex weapon detection (to not overfit on train):
- NLP mean score: 0.9865973143258857
- NLP detailed score: {'heading': 0.9994285714285714, 'target': 0.9869926184926185, 'tool': 0.9733707530564674}
  Conclusion: Prompt improved as target tool both improve from 0.94 and 0.96 respectively

With above (with regx weapon) + change func description in case of known heading or tool + add " A target can have multiple colors." to target desc:
- NLP mean score: 0.988973401974375
- NLP detailed score: {'heading': 0.9971428571428571, 'target': 0.972920205923125, 'tool': 0.9968571428571429}
  Conclusion: worse

new prompt format upgraded: prevent missing quote on heading, prevent premature split by repeat when detected heading or tool in the 2nd half, check for existence of target/tool in prompt to prevent hallucination. Retry if not found. Fix rare "None" treated as string but not None **(BEST)**.
- NLP mean score: 0.9972206238206238
- NLP detailed score: {'heading': 0.9997142857142857, 'target': 0.9925190143190143, 'tool': 0.9994285714285714}
  Conclusion: prompt improvement on target is effective

  above but without regex weapon and also heading det (full zero shot):
- NLP mean score: 0.852171472971473
- NLP detailed score: {'heading': 0.6225714285714286, 'target': 0.980303908789623, 'tool': 0.9536390815533673}

above but with regex weapon and replace "target", "deploy", "use" in tool, and "engage" in target as postprocessing:
- NLP mean score: 0.9972978724978725
- NLP detailed score: {'heading': 0.9997142857142857, 'target': 0.9927507603507604, 'tool': 0.9994285714285714}

With regex weapon detection and color-based regex for target:


## Pretrained Gorilla OpenFunctionsV2 EXL 5.0bit hb6 calibrated on default set, eval on test set
with regex weapon detection
- Accuracy: 0.99190173
- Speed Score: 0.81024160 = 17min4s

With regex weapon detection, moved regex heading to before LLM generation, improve func def to include optional arg " (some of them may be known already)"
**Have bug** where target field may be missing, causing entire sample to be empty
- Accuracy: 0.9763647907647908
- Speed Score: 0.7861219566666666 = 19min15s like due to many retries

With regex weapon detection, prompt more descriptive on target and tool + make target not mandatory on retry so allow it to not give target but keep other fields if it fails, instead of whole sample fail **(BEST)**:
- Accuracy: 0.9978597402597402
- Speed Score: 0.8079942874074074 = 17min16s likely due to a few more retries

new prompt format upgraded: prevent missing quote on heading, prevent premature split by repeat when detected heading or tool in the 2nd half, check for existence of target/tool in prompt to prevent hallucination. Retry if not found. Fix rare "None" treated as string but not None.
Not testing. See below.

Above but with regex weapon and replace "target", "deploy", "use" in tool, and "engage" in target as postprocessing:
- Accuracy: 0.996647619047619
- Speed Score: 0.718941949074074


## TIL Trained Gorilla OpenFunctionsV2 EXL 5.0bit hb6 calibrated on default set, eval on full train set
**Have bug** where target field may be missing, causing entire sample to be empty
**Eval on train set, not representative!**
- NLP mean score: 0.9984081632653061
- NLP detailed score: {'heading': 0.9982857142857143, 'target': 0.998938775510204, 'tool': 0.998}


## TIL Trained Gorilla OpenFunctionsV2 EXL 5.0bit hb6 calibrated on default set, eval on full train set
https://huggingface.co/aliencaocao/gorilla-openfunctions-v2-TIL24-r16-a16-ctx768-v1

new prompt format (best on zeroshot)
- NLP mean score: 0.9986441558441559
- NLP detailed score: {'heading': 0.9982857142857143, 'target': 0.9990753246753247, 'tool': 0.9985714285714286}
  Hallucination when removed repeat and no actual heading/tool/target provided in the leftover prompt, thus lower score and cannot detect. Fix: check if target/tool is substring of prompt, if not, retry without repeat. If regex detect heading or known weapon is after repeat, do not remove repeat

new prompt format upgraded: prevent missing quote on heading, prevent premature split by repeat when detected heading or tool in the 2nd half, check for existence of target/tool in prompt to prevent hallucination. Retry if not found. Fix rare "None" treated as string but not None.
- NLP mean score: 0.9999047619047619
- NLP detailed score: {'heading': 1.0, 'target': 1.0, 'tool': 0.9997142857142857}

above but without regex weapon and also heading det (full zero shot):
- NLP mean score: 0.9999047619047619
- NLP detailed score: {'heading': 1.0, 'target': 1.0, 'tool': 0.9997142857142857}
  probably memorized

Above same but 3.0 bit quant:
- NLP mean score: 0.9917142857142857
- NLP detailed score: {'heading': 0.9908571428571429, 'target': 0.9922857142857143, 'tool': 0.992}
Have some cases of partially output string, likely due to quantization, not good

Above same but 4.0 bit quant:
Have some cases of partially output string, likely due to quantization, not good, not testing full

Above same (no weapon regex) but with head bits 8.0:
NLP mean score: 0.9996190476190476
NLP detailed score: {'heading': 0.9997142857142857, 'target': 0.9997142857142857, 'tool': 0.9994285714285714}
somehow worse than hb 6.0 likely because of less bits for the middle layers

## TIL Trained Gorilla OpenFunctionsV2 EXL 6.0bit hb8 calibrated on default set, eval on full train set
above but without regex weapon and also heading det (full zero shot):
- NLP mean score: 0.9999047619047619
- NLP detailed score: {'heading': 1.0, 'target': 1.0, 'tool': 0.9997142857142857}
Conclusion: beyond 5.0 hb6, no more improvement.

## TIL Trained Gorilla OpenFunctionsV2 EXL 4.0bit hb6 calibrated on default set, eval on full train set
func call v2 is essentially training data. V1 was just the json and was not aligned well

Without weapon and heading regex
- NLP mean score: 0.9996190476190476
- NLP detailed score: {'heading': 0.9997142857142857, 'target': 0.9997142857142857, 'tool': 0.9994285714285714}
Calibration in train helps a lot in train set but still have 2 cases of memorization: it outputs the rest of the label instead of the func call

## TIL Trained Gorilla OpenFunctionsV2 EXL 3.0bit hb6 calibrated on default set + func call v2, eval on full train set
NLP mean score: 0.940952380952381
NLP detailed score: {'heading': 0.9411428571428572, 'target': 0.9411428571428572, 'tool': 0.9405714285714286}
A lot worse than calibrating without func call v2
Conclusion: custom calibration on func call v2 is still bad.


## TIL Trained V2 Gorilla OpenFunctionsV2 EXL 5.0bit hb6 calibrated on default set, eval on test set
https://huggingface.co/aliencaocao/gorilla-openfunctions-v2-TIL24-r16-a16-ctx768-v2
`gorilla-openfunctions-v2-TIL24-r16-a16-ctx768-5.0bpw-h6-exl2`

new prompt format (best on zeroshot)
- Accuracy: 0.9955555555555555
- Speed Score: 0.7971747083333334, maybe more retries

without weapon regex:
- Accuracy: 0.9955555555555555
- Speed Score: 0.8122045603703704, maybe not having he prompt "The tool is known to be..." reduces retries

new prompt format upgraded: prevent missing quote on heading, prevent premature split by repeat when detected heading or tool in the 2nd half, check for existence of target/tool in prompt to prevent hallucination. Retry if not found. Fix rare "None" treated as string but not None.
- Accuracy: 0.9993333333333333
- Speed Score: 0.8133207179629629

Above but with regex weapon and replace "target", "deploy", "use" in tool, and "engage" in target as postprocessing:
- Accuracy: 0.9993333333333333
- Speed Score: 0.7122024411111112 (probably a bug or GCP issue as all the changes were 4 str.replace after inference)
  Conclusion: VS 0.996 on pretrained, trained perfs better

Above but without weapon regex or heading (full zero shot on test) **BEST ZERO SHOT**:
- Accuracy: 0.9993333333333333
- Speed Score: 0.808186342037037
  Conclusion: trained model does well even without regex help


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