
<h1 align="center">
    <img src="https://user-images.githubusercontent.com/52832716/137499614-0206f0a8-9bee-4974-b318-32f313e0b8bb.png" width="650">
</h1>


◆ Stereotype means false beliefs learned and planted by social norms or customs. For example, Programmers are good at fixing computers, programmers have poor social skills, etc...

◆ This model can classify whether the text has a stereotype or not. If so, you can see what stereotypes are included (profession, race, gender, religion)

◆ You can test this models directly on the [**web demo**](https://share.streamlit.io/newfull5/stereotype-detector/demo.py) page and also you can use this model with Hugging Face **transformers library.**

◆ This model is made using <a href="https://github.com/JongyoonSong/K-StereoSet">K-StereoSet</a> with <a href="https://github.com/tunib-ai/tunib-electra">TUNiB-Electra</a>

## Web demo

- you can test this model easily in demo page  
- LINK: https://share.streamlit.io/newfull5/stereotype-detector/demo.py

<img src="https://user-images.githubusercontent.com/52832716/137329613-38566eaa-5481-4b56-849a-4882f083fe98.png" width="800">

## Dataset

In the actual dataset, five people classified whether or not there was a stereotype in the sentence, but trained with the most selected type as correct.

- There are **12,688** human-labeled comments in total. 
- They ar splitted into 10,688 train set, 1000 test set and 1000 valid set

~~~
stereotype, anti-stereotype, unrelated, profession, race, gender, religion, comment
1.0,    0.0,    0.0,    1.0,    0.0,    0.0,    0.0,    축구선수의 오랜 경력의 전형적인 결과는 부상을 입은 발목과 두개골이다.
1.0,    0.0,    0.0,    0.0,    1.0,    0.0,    0.0,    콜롬비아 사람들은 모두 마약 카르텔에 소속되어 있다.
0.0,    1.0,    0.0,    0.0,    1.0,    0.0,    0.0,    그는 무기를 소지한 적이 없다.
0.0,    1.0,    0.0,    0.0,    0.0,    1.0,    0.0,    그는 주말에만 지루할 수 있는 일을 한다.
~~~

### Detail

|       | stereotype | anti-stereotype | unrelated | profession | race  | gender | religion | Total  |
| ----- | ---------- | --------------- | --------- | ---------- | ----- | ------ | -------- | ------ |
| Train | 3,550      | 3,556           | 3,581     | 4,140      | 4,896 | 1,268  | 383      | 10,688 |
| Valid | 341        | 347             | 312       | 410        | 435   | 110    | 45       | 1,000  |
| Test  | 334        | 324             | 336       | 361        | 483   | 113    | 43       | 1,000  |


## Score

|                     | precision | recall | F1    |
| ------------------- | --------- | ------ | ----- |
| stereotype          | 0.814     | 0.601  | 0.691 |
| anti-stereotype     | 0.894     | 0.509  | 0.648 |
| unrelated           | 0.872     | 0.870  | 0.871 |
| profession          | 0.943     | 0.711  | 0.811 |
| race                | 0.787     | 0.907  | 0.843 |
| gender              | 0.639     | 0.836  | 0.724 |
| religion            | 0.724     | 1.0    | 0.840 |
| total (macro score) | 0.810     | 0.776  | 0.775 |


## Usage

You can use this model directly with [transformers](https://github.com/huggingface/transformers) library:

```python
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('dhtocks/tunib-electra-stereotype-classifier')
model = AutoModel.from_pretrained('dhtocks/tunib-electra-stereotype-classifier')
```

- training code

~~~
python3 train.py --model_name tunib/electra-ko-base \
                 --data_dir YOUR_PATH \
                 --batch_size BATCH_SIZE \
~~~

- threshold optimizing

~~~
python3 threshold.py --model_name tunib/electra-ko-base \
                     --data_dir YOUR_CKPT_DIR_PATH \
                     --file_path YOUR_CKPT_FILE_NAME \
                     --batch_size BATCH_SIZE \
                     --data_path TEST_DATA_PATH
~~~

- testint code

~~~
python3 score.py --model_name tunib/electra-ko-base \
                 --data_dir YOUR_CKPT_DIR_PATH \
                 --file_path YOUR_CKPT_FILE_NAME \
                 --batch_size BATCH_SIZE \
                 --data_path TEST_DATA_PATH
~~~
