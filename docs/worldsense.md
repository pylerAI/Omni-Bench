# WorldSense

## 공식 링크

- Official GitHub: [JaaackHongggg/WorldSense](https://github.com/JaaackHongggg/WorldSense)
- arXiv: [WorldSense: Evaluating Real-world Omnimodal Understanding for Multimodal LLMs](https://arxiv.org/abs/2502.04326)
- Project page: [WorldSense project page](https://jaaackhongggg.github.io/WorldSense/)
- Dataset: [honglyhly/WorldSense](https://huggingface.co/datasets/honglyhly/WorldSense)

![WorldSense task category results](assets/worldsense_fine_task.png)

## 목적

WorldSense는 synchronized video와 audio 기반의 real-world omnimodal understanding을 평가합니다. 모델이 visual, audio, text cue를 결합해 grounded reasoning을 수행할 수 있는지를 측정합니다.

## 데이터셋

- Local path: `/gpfs/public/datasets/WorldSense/`
- Official repo: `submodules/WorldSense`
- Annotation: `/gpfs/public/datasets/WorldSense/worldsense_qa.json`
- 1,662개 audio-visual synchronized video
- 3,172개 multiple-choice QA pair
- 8개 domain, 67개 subcategory, 26개 task type

![WorldSense dataset distribution](assets/worldsense_distribution.png)

video는 zip archive 형태로 배포되므로 평가 전에 한 번 압축을 해제합니다.

```bash
bash scripts/extract_worldsense_videos.sh
```

## 평가 방식

WorldSense official repository는 VLMEvalKit 기반 재현을 안내합니다. local adapter는 VLMEvalKit의 `WorldSense` 구현을 참고해 prompt, exact matching parser, dimension aggregation을 맞춥니다.

기본 설정은 VLMEvalKit의 `WorldSense_8frame_audio` variant에 맞춥니다. video를 그대로 넣지 않고 8개 frame을 image input으로 추출하며, audio는 WAV로 분리해 전달합니다. 응답은 VLMEvalKit의 `extract_characters_regex` 방식으로 A-D letter를 추출하고, duration/domain/sub-category/task-domain/task-type/audio-class별 rating을 `vlmeval_rating.json`에 저장합니다.

참조한 official 구현은 VLMEvalKit의 [`WorldSense_8frame_audio` dataset config](https://github.com/open-compass/VLMEvalKit/blob/0bfa830fa42fd1d5ac485bbb0bbc78e16c079e18/vlmeval/dataset/video_dataset_config.py#L204-L216), [`WorldSense.build_prompt`](https://github.com/open-compass/VLMEvalKit/blob/0bfa830fa42fd1d5ac485bbb0bbc78e16c079e18/vlmeval/dataset/worldsense.py#L243-L292), [`WorldSense.evaluate`](https://github.com/open-compass/VLMEvalKit/blob/0bfa830fa42fd1d5ac485bbb0bbc78e16c079e18/vlmeval/dataset/worldsense.py#L296-L349), [`extract_characters_regex`](https://github.com/open-compass/VLMEvalKit/blob/0bfa830fa42fd1d5ac485bbb0bbc78e16c079e18/vlmeval/dataset/utils/worldsense.py#L218-L240), [`get_dimension_rating`](https://github.com/open-compass/VLMEvalKit/blob/0bfa830fa42fd1d5ac485bbb0bbc78e16c079e18/vlmeval/dataset/utils/worldsense.py#L142-L206)입니다.

## Metric

- Overall accuracy: 전체 MCQ 중 정답 option이 일치한 비율입니다. real-world omnimodal understanding의 전체 성능을 나타냅니다.
- Domain accuracy: video domain별 정확도입니다. 예를 들어 music, sports, culture/politics, tech/science 등 real-world domain에 따라 모델이 어떤 상황에서 강하거나 약한지 확인합니다.
- Sub-category accuracy: domain보다 더 세밀한 scene/topic 단위 정확도입니다. 특정 콘텐츠 유형에서의 모델 편향과 실패 양상을 분석합니다.
- Task-type accuracy: benchmark가 정의한 26개 task 유형별 정확도입니다. audio change, temporal localization, audio counting, spatial relation 등 어떤 reasoning operation에서 성능이 떨어지는지 확인합니다.
- Audio-class accuracy: `Speech`, `Music`, `Event` 등 audio cue 유형별 성능을 분석할 때 사용합니다.

![WorldSense audio type results](assets/worldsense_fine_audio.png)

## 최종 Output Metric Table

`results/<model>/worldsense/summary.json` 기준 최종 결과는 official overall performance table과 유사하게 domain별 accuracy를 나열합니다.

| Model | LLM Size | Tech & Science | Culture & Politics | Daily Life | Film & TV | Performance | Games | Sports | Music | Avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-Omni | 30B | - | - | - | - | - | - | - | - | - |
| Nemotron-3-Nano-Omni | 30B | - | - | - | - | - | - | - | - | - |

per-sample 예측과 raw response는 `records.jsonl`에 저장하고, VLMEvalKit 형식의 dimension rating은 `vlmeval_rating.json`에 저장합니다.
