# KOTE (Korean Online That-gul Emotions) Dataset

## paper

## dataset
- 다양한 플랫폼에서 수집한 50,000개의 댓글에 44개 정서로 레이블링한 데이터셋.
    * 한 댓글 당 5명이 레이블링 --> 25만 케이스. 이것 저것 해 보기 좋은 25만 케이스에 관한 raw data: [raw.json](https://huggingface.co/datasets/searle-j/kote/blob/main/raw.json)
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
        model = model,
        tokenizer = tokenizer,
        device = 0, # gpu number, -1 if cpu used
        return_all_scores = True,
        function_to_apply = 'sigmoid'
    )

pipe("""자 떠나자 동해바다로 삼등삼등 완행열차 기차를 타고""")

#outputs
[[{'label': '불평/불만', 'score': 0.02542797103524208},
  {'label': '환영/호의', 'score': 0.3231465816497803},
  {'label': '감동/감탄', 'score': 0.19002869725227356},
  {'label': '지긋지긋', 'score': 0.07263068854808807},
  {'label': '고마움', 'score': 0.10604046285152435},
  {'label': '슬픔', 'score': 0.0321921668946743},
  {'label': '화남/분노', 'score': 0.025590604171156883},
  {'label': '존경', 'score': 0.035038430243730545},
  {'label': '기대감', 'score': 0.8417094945907593},
  {'label': '우쭐댐/무시함', 'score': 0.1559317708015442},
  {'label': '안타까움/실망', 'score': 0.027600685134530067},
  {'label': '비장함', 'score': 0.549821674823761},
  {'label': '의심/불신', 'score': 0.017292646691203117},
  {'label': '뿌듯함', 'score': 0.2693062424659729},
  {'label': '편안/쾌적', 'score': 0.2215314656496048},
  {'label': '신기함/관심', 'score': 0.19473326206207275},
  {'label': '아껴주는', 'score': 0.23799468576908112},
  {'label': '부끄러움', 'score': 0.01746259070932865},
  {'label': '공포/무서움', 'score': 0.01043156161904335},
  {'label': '절망', 'score': 0.019326098263263702},
  {'label': '한심함', 'score': 0.03933953121304512},
  {'label': '역겨움/징그러움', 'score': 0.007506131660193205},
  {'label': '짜증', 'score': 0.03316599875688553},
  {'label': '어이없음', 'score': 0.019064713269472122},
  {'label': '없음', 'score': 0.5564616918563843},
  {'label': '패배/자기혐오', 'score': 0.026847394183278084},
  {'label': '귀찮음', 'score': 0.06647266447544098},
  {'label': '힘듦/지침', 'score': 0.18237516283988953},
  {'label': '즐거움/신남', 'score': 0.7502350211143494},
  {'label': '깨달음', 'score': 0.26856672763824463},
  {'label': '죄책감', 'score': 0.006604543421417475},
  {'label': '증오/혐오', 'score': 0.015338188968598843},
  {'label': '흐뭇함(귀여움/예쁨)', 'score': 0.04649550840258598},
  {'label': '당황/난처', 'score': 0.010948682203888893},
  {'label': '경악', 'score': 0.0059306081384420395},
  {'label': '부담/안_내킴', 'score': 0.033722419291734695},
  {'label': '서러움', 'score': 0.023046258836984634},
  {'label': '재미없음', 'score': 0.041899073868989944},
  {'label': '불쌍함/연민', 'score': 0.0388372428715229},
  {'label': '놀람', 'score': 0.02612595073878765},
  {'label': '행복', 'score': 0.5933531522750854},
  {'label': '불안/걱정', 'score': 0.04089110344648361},
  {'label': '기쁨', 'score': 0.5395115613937378},
  {'label': '안심/신뢰', 'score': 0.16369180381298065}]]
```

- 꼼꼼한 사람들을 위한 pytorch lightning 논문 버전
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
  """자 떠나자 동해바다로 삼등삼등 완행열차 기차를 타고"""
  )[0]

for label, pred in zip(LABELS, preds):
    if pred>0.4:
        print(f"{label}: {pred}")

# outputs
기대감: 0.7901794910430908
비장함: 0.5611065626144409
없음: 0.566945493221283
즐거움/신남: 0.6499309539794922
```

