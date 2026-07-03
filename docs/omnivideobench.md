# OmniVideoBench

## 공식 링크

- Official GitHub: [NJU-LINK/OmniVideoBench](https://github.com/NJU-LINK/OmniVideoBench)
- arXiv: [OmniVideoBench: Towards Audio-Visual Understanding Evaluation for Omni MLLMs](https://arxiv.org/abs/2510.10689)
- Project page: [OmniVideoBench](https://omnivideobench.github.io/omnivideobench_home/)
- Dataset: [NJU-LINK/OmniVideoBench](https://huggingface.co/datasets/NJU-LINK/OmniVideoBench)

![OmniVideoBench examples](assets/omnivideobench_examples.png)

## 목적

OmniVideoBench는 audio-visual reasoning, modality complementarity, logical consistency, long-term temporal reasoning을 평가합니다.

## 데이터셋

- Local path: `/gpfs/public/datasets/OmniVideoBench/`
- Official repo: `submodules/OmniVideoBench`
- Annotation: `/gpfs/public/datasets/OmniVideoBench/data.parquet`
- 628개 video
- 1,000개 QA pair
- 8개 video category
- 13개 question type

![OmniVideoBench dataset statistics](assets/omnivideobench_main.png)

## 평가 방식

official Qwen3-Omni evaluation code는 visual frame을 샘플링하고 audio track을 별도로 추출합니다. adapter는 이 구조에 맞춰 다음을 수행합니다.

- visual 입력은 최대 `120`개 frame으로 균일 샘플링합니다.
- audio 입력은 원본 full audio track을 WAV로 추출해 별도 audio input으로 전달합니다.
- 응답은 official-style `extract_model_answer` / `clean_text` 흐름으로 정리해 gold answer와 비교합니다.

## Metric

- Overall accuracy: 전체 QA pair 기준 정답률입니다. audio-visual joint reasoning의 전반적 성능을 확인합니다.
- Video type accuracy: `Vlog`, `Movie`, `Cartoon`, `Education`, `Sports` 등 video type별 성능입니다. 장르나 촬영 양식에 따른 편차를 봅니다.
- Question type accuracy: 13개 reasoning type별 성능입니다. 예를 들어 causal reasoning, spatial understanding, temporal sequencing, attribute comparison, counting, fine-grained perception 등이 포함됩니다. 모델이 어떤 reasoning operation에서 약한지 확인합니다.
- Audio type accuracy: `Speech`, `Sound`, `Music` 등 audio cue 유형별 성능입니다. speech grounding과 non-speech sound 이해를 분리해서 분석합니다.
- Duration accuracy: 짧은 clip과 긴 clip에서 context handling 차이를 확인할 수 있습니다.

![OmniVideoBench performance overview](assets/omnivideobench_results.png)

## 최종 Output Metric Table

`results/<model>/omnivideobench/summary.json` 기준 최종 결과는 audio type과 주요 reasoning type을 함께 보여주는 table로 정리합니다.

| Model | Size | Overall | Speech | Sound | Music | Causal Reasoning | Spatial Understanding | Temporal Sequencing | Attribute Comparison | Counting | Fine-grained Perception | Relationship Reasoning | Summarization |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-Omni | 30B | - | - | - | - | - | - | - | - | - | - | - | - |
| Nemotron-3-Nano-Omni | 30B | - | - | - | - | - | - | - | - | - | - | - | - |

per-question raw record는 `records.jsonl`에 저장합니다.
