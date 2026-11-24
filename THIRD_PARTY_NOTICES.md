# 출처와 제삼자 자료 고지

## COCO 사진과 주석

COCO Consortium의 캡션 주석은 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)이다. [COCO 이용 조건](https://cocodataset.org/#termsofuse)에 따라 사진 저작권은 원저작자에게 있다. 이 실험은 COCO val2014 메타데이터에 license ID 4, [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/)으로 기재된 사진만 선택했다. 전체 100장의 원출처와 개별 라이선스는 [data/manifest.json](data/manifest.json)에 있다.

아래는 공개 그림에 재현한 사진이다. 저작자와 제목은 Flickr의 공개 oEmbed 응답에서 확인했다. 사진은 크기를 조정하고 도표·문장을 배치했으며, Attention 패널에는 반투명 색상 가중치 지도를 덧씌웠다. 저작자의 보증이나 승인을 의미하지 않는다. 원사진의 워터마크는 제거하지 않았다.

| COCO ID | 제목 | 저작자 | 원사진 | 사용 |
|---|---|---|---|---|
| 4972 | A Knight and his Horse | [Sean Tbear](https://www.flickr.com/photos/sptphotographe/) | [Flickr](https://flic.kr/p/fdjVTi) | Attention, 번역 |
| 19888 | Maaike's bed in Yantai | [selmerv](https://www.flickr.com/photos/selmer/) | [Flickr](https://flic.kr/p/e53AiW) | Attention, 번역 |
| 406976 | Image010 | [Uriel 1998](https://www.flickr.com/photos/uriel1998/) | [Flickr](https://flic.kr/p/4osPuL) | 추론 사례 |
| 515540 | IMG_5649.jpg | [James Buck](https://www.flickr.com/photos/jameskarlbuck/) | [Flickr](https://flic.kr/p/a1Cxcr) | 추론 사례 |

위 네 사진의 라이선스는 모두 CC BY 2.0이며, 링크와 저작자 표기를 유지해야 한다. manifest의 원본 파일 해시로 사용한 COCO 배포 버전을 식별할 수 있다. 주석 발췌는 참조 캡션으로 표시했다.

## 모델과 화면

- [Salesforce/blip-image-captioning-base](https://huggingface.co/Salesforce/blip-image-captioning-base): 모델 카드의 BSD-3-Clause. [공식 BLIP 라이선스 전문](third_party/BLIP-LICENSE.txt)을 보존했다. 가중치는 재배포하지 않고 고정 revision으로 다운로드한다. 설정·토크나이저는 해당 모델에서 저장한 재현용 파일이다. 모델의 라이선스는 저장소 원본 코드 MIT와 별개다.
- [facebook/m2m100_418M](https://huggingface.co/facebook/m2m100_418M): 모델 카드의 MIT. 가중치를 재배포하지 않는다.
- `figures/model_card_excerpt.png`: 위 BLIP Hugging Face 모델 카드의 실제 화면을 교육적 출처 확인을 위해 캡처하고 헤더 영역으로 자른 발췌다. Hugging Face 및 Salesforce의 이름·표시 요소는 각 권리자에게 속한다.
- `figures/code_excerpt.png`: 이 저장소에서 실제 실행한 API와 저장 결과의 편집 발췌다. 콘솔 원본 화면으로 표시하지 않았다.

## 평가 패키지와 폰트

- [pycocoevalcap](https://github.com/salaniz/pycocoevalcap), [원 COCO-caption](https://github.com/tylin/coco-caption): 의존 패키지로 설치해 사용한다. Stanford PTBTokenizer와 BLEU/CIDEr 구현은 각 원저작자의 조건을 따른다. 해당 소스·JAR를 이 저장소에 복제하지 않는다. 독립 검증 스크립트는 표준 수식의 별도 구현이다.
- Transformers, PyTorch, torchvision, Pillow, NumPy, Matplotlib, SentencePiece, Sacremoses 등은 각 패키지의 라이선스를 따른다. requirements와 environment에 버전을 기록했다.
- 실제 한국어 그림은 설치된 Microsoft Word의 맑은 고딕으로 렌더링했다. 폰트 파일은 복사하거나 공개하지 않았다. 재실행 코드의 대체 폰트는 [Noto Sans CJK KR](https://github.com/notofonts/noto-cjk), [SIL Open Font License 1.1](https://github.com/notofonts/noto-cjk/blob/main/LICENSE)이다. 대체 폰트를 다운로드하면 해당 라이선스를 따른다.

이 저장소의 MIT 라이선스는 원본 코드에 적용하며 제삼자 자료의 이용 조건을 대체하지 않는다.
