# AV-SpeakerBench

## Official Links

- Official GitHub: [plnguyen2908/AV-SpeakerBench](https://github.com/plnguyen2908/AV-SpeakerBench)
- arXiv: [See, Hear, and Understand: Benchmarking Audiovisual Human Speech Understanding in Multimodal Large Language Models](https://arxiv.org/abs/2512.02231)
- Project page: [AV-SpeakerBench project page](https://plnguyen2908.github.io/AV-SpeakerBench-project-page/)
- Dataset: [plnguyen2908/AV-SpeakerBench](https://huggingface.co/datasets/plnguyen2908/AV-SpeakerBench)

![AV-SpeakerBench question design](assets/av_speakerbench_question_design.png)

## Purpose

AV-SpeakerBench evaluates speaker-centric audiovisual reasoning on real-world video. The core task is deciding who spoke, what was said, and when a speech or visual event occurred, using audio and visual cues together.

## Dataset

- Local path: `/gpfs/public/datasets/AV-SpeakerBench/`
- Official repo: `submodules/AV-SpeakerBench`
- 3,212 multiple-choice questions
- Main tasks: speaker detection, speaker recognition, speech recognition, speech duration/rate/intensity/pitch, visual counting, attribute recognition, activity recognition

![AV-SpeakerBench dataset statistics](assets/av_speakerbench_dataset_stat.png)

## Evaluation Method

The adapter follows the benchmark prompt.

```text
Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option.
...
The best answer is:
```

Responses are parsed for an A-D letter following the official repo's `extract_characters_regex` approach; if no valid option is found, the answer is counted as incorrect.

## Metrics

- Overall accuracy: the fraction of the 3,212 MCQs whose predicted letter matches the gold answer. It reflects overall speaker-centric audiovisual reasoning ability.
- Category accuracy: accuracy grouped by which modality cue is central — `Audio-centric`, `Visual-centric`, `Speaker-centric`. It exposes the performance gap between audio-dependent and visual-dependent questions.
- Sub-category accuracy: accuracy per fine-grained task. Examples below.

| Sub-category | What it measures |
| --- | --- |
| Speaker Detection | Whether someone is speaking in the video and who the speech source is |
| Speaker Recognition | Distinguishing speaker identity or description from audio-visual cues |
| Speaker Counting | Inferring how many people are speaking |
| Speech Recognition | Recognizing the spoken content |
| Speech Duration | Understanding how long a speech lasts and over which span |
| Speech Pitch | Comparing the pitch of speech |
| Speech Rate | Comparing speaking rate |
| Speech Intensity | Comparing loudness/intensity of the voice |
| Visual Counting | Counting visual objects/people |
| Attribute Recognition | Recognizing attributes of people or objects |
| Activity Recognition | Recognizing human activity |

## Final Output Metric Table

Final results, based on `results/<model>/av_speakerbench/summary.json`, are reported in the following form.

| Model | Size | Speaker Detection | Speaker Recognition | Speaker Counting | Attribute Recognition | Activity Recognition | Visual Counting | Speech Recognition | Speech Duration | Speech Pitch | Speech Rate | Speech Intensity | Speech Counting | Overall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-Omni | 30B | - | - | - | - | - | - | - | - | - | - | - | - | - |
| Nemotron-3-Nano-Omni | 30B | - | - | - | - | - | - | - | - | - | - | - | - | - |

Per-sample predictions and raw responses are stored in `records.jsonl`.
