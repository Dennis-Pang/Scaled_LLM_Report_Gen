# GenAI VLM Summary Workflow

Local SGLang client + helpers for generating VLM summaries from sample data.

## Directory Layout

```
sample/
  sample1/
    extracted_data/        # JSON + images (input to VLM)
      patient_info.json
      anamnesis.json
      therapy_goals.json
      assessments.json
      therapy_program.json
      scientific_background.json
      lokomat_assessment/  # images referenced by JSON
      scientific_background/
    Verlauf26.pdf          # original PDF (outside extracted_data)
    summary.txt            # generated output (optional)
LLM/
  client.py                # SGLang client + server helpers
  chat.py                  # interactive CLI chat
scripts/
  vlm_data_loader.py       # builds OpenAI/SGLang messages
  pdf_extractor.py         # extracts images from a PDF
generate_sum.py            # generate summary from a sample
```

## Quick Start

1) Start the SGLang server (interactive GPU selection):
```
python -m LLM.client start
```

2) Chat (text or multimodal):
```
python -m LLM.chat
python -m LLM.chat --text "Describe this image" --image sample/sample1/extracted_data/lokomat_assessment/cmill_butterfly_june2022.jpg
```

3) Generate a summary:
```
python generate_sum.py --sample-id sample1 --output sample/sample1/summary.txt
```

## Common Commands

### Client (server management)
```
python -m LLM.client gpus
python -m LLM.client check --model Qwen/Qwen3-VL-32B-Instruct
python -m LLM.client test --base-url http://127.0.0.1:30000/v1
python -m LLM.client start --model Qwen/Qwen3-VL-32B-Instruct --gpus 0,1
```

### Chat (interactive or one-shot)
```
python -m LLM.chat
python -m LLM.chat --text "Hello"
python -m LLM.chat --text "Describe this image" --image /path/to/image.png
```

### Build Messages in Python
```
from LLM.client import SGLangLLM
from scripts.vlm_data_loader import PatientDataLoader

loader = PatientDataLoader("sample1")
messages = loader.build_vlm_messages()

llm = SGLangLLM(auto_start=False, gpu_ids=[0])
print(llm.chat_messages(messages))
```

## Data Notes

- All JSON and images must live under `sample/<id>/extracted_data`.
- Image paths inside JSON are relative to `extracted_data/`.
- The original PDF (if present) should sit in `sample/<id>/`.

## Proxy Note

If your environment uses a proxy, local calls should bypass it:
```
NO_PROXY=127.0.0.1,localhost python -m LLM.client test
```
