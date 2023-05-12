# KOTE (Korean Online That-gul Emotions) Dataset

## paper
https://arxiv.org/pdf/2205.05300.pdf

## dataset
- 다양한 플랫폼에서 수집한 50,000개의 댓글에 44개 정서로 레이블링한 데이터셋.
    * 한 댓글 당 5명이 레이블링 --> 25만 케이스
    * 이것 저것 해 보기 좋은 25만 케이스에 관한 raw data: [raw.json](https://huggingface.co/datasets/searle-j/kote/blob/main/raw.json)

    * 정서 레이블 종류 ['불평/불만',
 '환영/호의',
 '감동/감탄',
 '지긋지긋',
 '고마움',
 '슬픔',
 '화남/분노',
 '존경',
 '기대감',
 '우쭐댐/무시함',
 '안타까움/실망',
 '비장함',
 '의심/불신',
 '뿌듯함',
 '편안/쾌적',
 '신기함/관심',
 '아껴주는',
 '부끄러움',
 '공포/무서움',
 '절망',
 '한심함',
 '역겨움/징그러움',
 '짜증',
 '어이없음',
 '없음',
 '패배/자기혐오',
 '귀찮음',
 '힘듦/지침',
 '즐거움/신남',
 '깨달음',
 '죄책감',
 '증오/혐오',
 '흐뭇함(귀여움/예쁨)',
 '당황/난처',
 '경악',
 '부담/안_내킴',
 '서러움',
 '재미없음',
 '불쌍함/연민',
 '놀람',
 '행복',
 '불안/걱정',
 '기쁨',
 '안심/신뢰']
 
- huggingface datasets으로 데이터셋 내려받기
```python
from datasets import load_dataset

dataset = load_dataset("searle-j/kote")
print(dataset)

# output
DatasetDict({
    train: Dataset({
        features: ['ID', 'text', 'labels'],
        num_rows: 40000
    })
    test: Dataset({
        features: ['ID', 'text', 'labels'],
        num_rows: 5000
    })
    validation: Dataset({
        features: ['ID', 'text', 'labels'],
        num_rows: 5000
    })
})
```

## Models
- 바쁜 사람들을 위한 huggingface Trainer 버전
    * huggingface 디폴트를 사용해서 논문에서 사용한 pytorch_lightning 버전과 파라미터와 아키텍쳐가 약간 다릅니다. 성능도 약간 낮습니다. (macro F1: 0.56 vs 0.55)
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline

model_name = "searle-j/kote_for_easygoing_people"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

pipe = TextClassificationPipeline(
        model=model,
        tokenizer=tokenizer,
        device=0, # gpu number, -1 if cpu used
        return_all_scores=True,
        function_to_apply='sigmoid'
    )

for output in pipe("""재미있어요! 재미는 확실히 있는데 뭐랄까... 너무 정신 없달까...ㅋㅋ""")[0]:
    if output["score"]>0.4:
        print(output)

#outputs
{'label': '안타까움/실망', 'score': 0.7091670632362366}
{'label': '즐거움/신남', 'score': 0.8421422243118286}
{'label': '당황/난처', 'score': 0.44475653767585754}
{'label': '행복', 'score': 0.46991464495658875}
{'label': '기쁨', 'score': 0.7035757303237915}
```

- 꼼꼼한 사람들을 위한 pytorch lightning 논문 버전
    * ---------------------------------------------------------------------------------------------------------------
    * [해당 이슈](https://github.com/searle-j/KOTE/issues/3)로 작동하지 않을 수 있습니다. 곧 업데이트하도록 하겠습니다.
    * ---------------------------------------------------------------------------------------------------------------
    * 논문에 사용한 weights가 담겨 있는 바이너리 파일: [kote_pytorch_lightning.bin](https://huggingface.co/searle-j/kote_for_meticulous_people/blob/main/kote_pytorch_lightning.bin)
    * 내려받은 이후에 파일 이름을 {파일_이름}, 확장자는 bin으로 설정해 주세요. --> {파일_이름}.bin
    * 돌려보기
```python
import pytorch_lightning as pl
import torch.nn as nn
from transformers import ElectraModel, AutoTokenizer
import torch

LABELS = ['불평/불만',
 '환영/호의',
 '감동/감탄',
 '지긋지긋',
 '고마움',
 '슬픔',
 '화남/분노',
 '존경',
 '기대감',
 '우쭐댐/무시함',
 '안타까움/실망',
 '비장함',
 '의심/불신',
 '뿌듯함',
 '편안/쾌적',
 '신기함/관심',
 '아껴주는',
 '부끄러움',
 '공포/무서움',
 '절망',
 '한심함',
 '역겨움/징그러움',
 '짜증',
 '어이없음',
 '없음',
 '패배/자기혐오',
 '귀찮음',
 '힘듦/지침',
 '즐거움/신남',
 '깨달음',
 '죄책감',
 '증오/혐오',
 '흐뭇함(귀여움/예쁨)',
 '당황/난처',
 '경악',
 '부담/안_내킴',
 '서러움',
 '재미없음',
 '불쌍함/연민',
 '놀람',
 '행복',
 '불안/걱정',
 '기쁨',
 '안심/신뢰']

device = "cuda" if torch.cuda.is_available() else "cpu"

class KOTEtagger(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.electra = ElectraModel.from_pretrained("beomi/KcELECTRA-base").to(device)
        self.tokenizer = AutoTokenizer.from_pretrained("beomi/KcELECTRA-base")
        self.classifier = nn.Linear(self.electra.config.hidden_size, 44).to(device)
        
    def forward(self, text:str):
        encoding = self.tokenizer.encode_plus(
          text,
          add_special_tokens=True,
          max_length=512,
          return_token_type_ids=False,
          padding="max_length",
          return_attention_mask=True,
          return_tensors='pt',
        ).to(device)
        output = self.electra(encoding["input_ids"], attention_mask=encoding["attention_mask"])
        output = output.last_hidden_state[:,0,:]
        output = self.classifier(output)
        output = torch.sigmoid(output)
        torch.cuda.empty_cache()
        
        return output

trained_model = KOTEtagger()
trained_model.load_state_dict(torch.load("{PATH}/{파일_이름}.bin")) # <All keys matched successfully>라는 결과가 나오는지 확인!

preds = trained_model(
"""재미있어요! 재미는 확실히 있는데 뭐랄까... 너무 정신 없달까...ㅋㅋ"""
)[0]

for l, p in zip(LABELS, preds):
    if p>0.4:
        print(f"{l}: {p}")

# outputs
불평/불만: 0.6502879858016968
안타까움/실망: 0.7633695006370544
어이없음: 0.4875696301460266
즐거움/신남: 0.8589422106742859
당황/난처: 0.6053884029388428
재미없음: 0.48843538761138916
기쁨: 0.4896940588951111
```

## 댓글 말고 다른 도메인에서도 통할까?
- 위의 huggingface Trainer 모델을 이용한 결과.

```python
def top_k_emotions(texts:list, threshold:float, k:int):
    if not 0 <= threshold <=1:
        raise ValueError("theshold must be a float b/w 0 ~ 1.")
    results = {}
    for text in texts:
        cur_result = {}
        for out in pipe(text)[0]:
            if out["score"] > threshold:
                cur_result[out["label"]] = round(out["score"], 2)
        cur_result = sorted(cur_result.items(), key=lambda x: x[1], reverse=True)
        results[text] = cur_result[:k]
        
    return results
```

#### poem
```python
poem_texts = [
    """내가 그다지 사랑하든 그대여 내한평생(平生)에 차마 그대를 잊을수없소이다.
    내차례에 못올사랑인줄은 알면서도 나혼자는 꾸준히생각하리다. 자그러면 내내어여쁘소서""",
    """13인의아해가도로로질주하오.(길은막다른골목이적당하오.) 제1의아해가무섭다고그리오.""",
    """자세히 보아야 예쁘다. 오래 보아야 사랑스럽다. 너도 그렇다.""",
    """너에게로 가지 않으려고 미친 듯 걸었던 그 무수한 길도 실은 네게로 향한 것이었다.""",
    """가야 할 때가 언제인가를 분명히 알고 가는 이의 뒷모습은 얼마나 아름다운가.
    봄 한 철 격정을 인내한 나의 사랑은 지고 있다.""",
]
top_k_emotions(texts=poem_texts, threshold=0.3, k=5)

# outputs
{'내가 그다지 사랑하든 그대여 내한평생(平生)에 차마 그대를 잊을수없소이다.
내차례에 못올사랑인줄은 알면서도 나혼자는 꾸준히생각하리다. 자그러면 내내어여쁘소서':
[('아껴주는', 0.92),
  ('슬픔', 0.8),
  ('고마움', 0.75),
  ('감동/감탄', 0.73),
  ('불쌍함/연민', 0.71)],
 '13인의아해가도로로질주하오.(길은막다른골목이적당하오.) 제1의아해가무섭다고그리오.':
 [('불안/걱정', 0.81),
  ('없음', 0.65),
  ('공포/무서움', 0.52),
  ('슬픔', 0.47),
  ('불쌍함/연민', 0.41)],
 '자세히 보아야 예쁘다. 오래 보아야 사랑스럽다. 너도 그렇다.':
 [('행복', 0.84),
  ('감동/감탄', 0.76),
  ('기쁨', 0.7),
  ('아껴주는', 0.69),
  ('흐뭇함(귀여움/예쁨)', 0.68)],
 '너에게로 가지 않으려고 미친 듯 걸었던 그 무수한 길도 실은 네게로 향한 것이었다.':
 [('슬픔', 0.89),
  ('안타까움/실망', 0.77),
  ('불쌍함/연민', 0.76),
  ('깨달음', 0.72),
  ('절망', 0.62)],
 '가야 할 때가 언제인가를 분명히 알고 가는 이의 뒷모습은 얼마나 아름다운가.
 봄 한 철 격정을 인내한 나의 사랑은 지고 있다.':
 [('깨달음', 0.86),
  ('슬픔', 0.74),
  ('감동/감탄', 0.68),
  ('행복', 0.55),
  ('불쌍함/연민', 0.51)]}
```

#### novel
```python
novel_texts = [
    """부끄럼 많은 생애를 보냈습니다. 저는 인간의 삶이라는 것을 도저히 이해할 수 없습니다.""",
    """대저 천하의 대세란 오랫동안 나뉘면 반드시 합하게 되고, 오랫동안 합쳐져 있다면 반드시 나뉘게 된다.""",
    """국경의 긴 터널을 빠져나오자, 설국이었다.""",
    """그다지 오래되지 않은 옛날,
    라 만차 지방의 이름도 기억나지 않는 어느 마을에서 선반에 창과 낡은 방패를 두고,
    야윈 말과 경주를 위한 사냥개를 가진 신사가 살고 있었다.""",
    """속았구나! 속았구나! 한번 야간 비상종이 잘못 울린 것을 따랐더니 결코 다시는 돌이킬 수가 없구나!""",
]
top_k_emotions(texts=novel_texts, threshold=0.3, k=5)

# outputs
{'부끄럼 많은 생애를 보냈습니다. 저는 인간의 삶이라는 것을 도저히 이해할 수 없습니다.':
[('안타까움/실망', 0.83),
  ('슬픔', 0.8),
  ('절망', 0.8),
  ('힘듦/지침', 0.67),
  ('패배/자기혐오', 0.58)],
 '대저 천하의 대세란 오랫동안 나뉘면 반드시 합하게 되고, 오랫동안 합쳐져 있다면 반드시 나뉘게 된다.':
 [('없음', 0.93),
  ('깨달음', 0.63),
  ('비장함', 0.49)],
 '국경의 긴 터널을 빠져나오자, 설국이었다.':
 [('깨달음', 0.72),
  ('없음', 0.57),
  ('슬픔', 0.46),
  ('감동/감탄', 0.45),
  ('힘듦/지침', 0.36)],
 '그다지 오래되지 않은 옛날,
 라 만차 지방의 이름도 기억나지 않는 어느 마을에서 선반에 창과 낡은 방패를 두고,
 야윈 말과 경주를 위한 사냥개를 가진 신사가 살고 있었다.':
 [('없음', 0.83),
  ('깨달음', 0.56),
  ('신기함/관심', 0.55),
  ('기대감', 0.34),
  ('감동/감탄', 0.3)],
 '속았구나! 속았구나! 한번 야간 비상종이 잘못 울린 것을 따랐더니 결코 다시는 돌이킬 수가 없구나!':
 [('안타까움/실망', 0.89),
  ('슬픔', 0.85),
  ('절망', 0.78),
  ('힘듦/지침', 0.65),
  ('서러움', 0.6)]}
```

#### movie
```python
movie_texts = [
    """천하의 아귀가 혓바닥이 왜 이렇게 길어? 후달리냐?""",
    """매너가 사람을 만든다. 이게 무슨 뜻인지 알고 있나?""",
    """내일은 또 내일의 해가 뜨는 법이니까.""",
    """내 사랑의 유통기한은 만 년으로 하고 싶다.""",
    """내가 잠들기 전에 마지막으로 이야기 하고 싶은 사람이 바로 너이기에 널 사랑해!""",
]
top_k_emotions(texts=movie_texts, threshold=0.3, k=5)

{'천하의 아귀가 혓바닥이 왜 이렇게 길어? 후달리냐?':
 [('어이없음', 0.95),
  ('한심함', 0.93),
  ('불평/불만', 0.85),
  ('의심/불신', 0.84),
  ('짜증', 0.83)],
 '매너가 사람을 만든다. 이게 무슨 뜻인지 알고 있나?':
 [('우쭐댐/무시함', 0.74),
  ('한심함', 0.6),
  ('깨달음', 0.59),
  ('비장함', 0.57),
  ('없음', 0.56)],
 '내일은 또 내일의 해가 뜨는 법이니까.':
 [('기대감', 0.76),
  ('없음', 0.68),
  ('깨달음', 0.47),
  ('즐거움/신남', 0.45),
  ('비장함', 0.38)],
 '내 사랑의 유통기한은 만 년으로 하고 싶다.':
 [('슬픔', 0.78),
  ('절망', 0.75),
  ('힘듦/지침', 0.7),
  ('안타까움/실망', 0.63),
  ('불안/걱정', 0.49)],
 '내가 잠들기 전에 마지막으로 이야기 하고 싶은 사람이 바로 너이기에 널 사랑해!':
 [('아껴주는', 0.95),
  ('행복', 0.83),
  ('고마움', 0.78),
  ('환영/호의', 0.76),
  ('기쁨', 0.76)]}
```

## References
[KcELECTRA](https://github.com/Beomi/KcELECTRA)
