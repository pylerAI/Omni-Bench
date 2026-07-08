# Video-MME

## 공식 링크

- Official GitHub: [MME-Benchmarks/Video-MME](https://github.com/MME-Benchmarks/Video-MME)
- arXiv: [Video-MME: The First-Ever Comprehensive Evaluation Benchmark of Multi-modal LLMs in Video Analysis](https://arxiv.org/abs/2405.21075)
- Project page: [Video-MME](https://video-mme.github.io/)
- Dataset: [lmms-lab/Video-MME](https://huggingface.co/datasets/lmms-lab/Video-MME)

![Video-MME model results](assets/videomme_results_of_various_models.png)

## 목적

Video-MME는 long-video multimodal understanding benchmark입니다. short, medium, long video 구간별 long-video 이해 성능을 확인하는 데 적합합니다.

## 데이터셋

- Local path: `/gpfs/public/datasets/Video-MME/`
- Official repo: `submodules/Video-MME`
- 900개 video
- 2,700개 human-annotated multiple-choice QA pair
- short, medium, long duration group 포함

![Video-MME dataset statistics](assets/videomme_statistics.jpg)

## 평가 방식

adapter는 official evaluation script가 기대하는 nested response format에 맞춰 `official_results.json`을 저장합니다. accuracy는 official Video-MME evaluator로 계산하는 것을 기준으로 합니다.

`use_subtitles: true`일 때는 prompt 앞에 subtitle text를 포함합니다. 기본값은 `false`입니다. adapter는 모델 응답을 official `output_test_template.json`과 같은 nested JSON 구조의 `official_results.json`으로 저장하고, accuracy는 official evaluator로 계산합니다.

## Throughput

Video-MME adapter는 accuracy만 측정합니다. Video-MME의 입력은 프레임을 client-side에서 샘플링하므로 처리량이 model이 아닌 frame 디코딩(CPU)에 좌우되어, 모델 throughput 지표로는 부적합합니다. 모델 생성 throughput은 `vllm bench throughput`으로 별도 측정하며 (`results/throughput.json`, HTML 리포트의 Model throughput 섹션 참고), input/output 길이를 고정한 text-generation 기준으로 양자화 variant 간 속도를 공정하게 비교합니다.

## Accuracy Metric

정확도는 official evaluator를 기준으로 계산합니다.

| Metric | 평가 목적 |
| --- | --- |
| Overall accuracy | 전체 MCQ 성능 |
| Duration accuracy | short/medium/long video별 long-context 성능 |
| Domain accuracy | visual domain별 성능 차이 |
| Sub-category accuracy | 세부 주제별 취약점 분석 |
| Task-type accuracy | counting, action reasoning, information synopsis 등 task 유형별 성능 |

## 최종 Output Metric Table

Video-MME accuracy는 official evaluator output을 기준으로 정리합니다.

| Model | Size | Overall | Short | Medium | Long | Knowledge | Film & TV | Sports | Artistic Performance | Life Record | Multilingual |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-Omni | 30B | - | - | - | - | - | - | - | - | - | - |
| Nemotron-3-Nano-Omni | 30B | - | - | - | - | - | - | - | - | - | - |

official evaluator 입력 파일은 `official_results.json`, per-question raw record는 `records.jsonl`에 저장합니다.
