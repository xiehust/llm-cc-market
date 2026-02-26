# Dataset Formats

LLaMA-Factory supports two primary formats: **Alpaca** and **ShareGPT**. Datasets are registered via `dataset_info.json` in the dataset directory.

Supported file types: JSON, JSONL, CSV, Parquet, Arrow, TXT (pre-training only).

## Alpaca Format (default)

Set `formatting: alpaca` in `dataset_info.json` (or omit -- it's the default).

### SFT Dataset
```json
[
  {
    "instruction": "What is the capital of France?",
    "input": "",
    "output": "The capital of France is Paris.",
    "system": "You are a helpful assistant.",
    "history": [
      ["Previous question", "Previous answer"]
    ]
  }
]
```

- `instruction` (required): User instruction
- `input` (optional): Additional user input (concatenated with instruction)
- `output` (required): Model response
- `system` (optional): System prompt
- `history` (optional): Previous conversation turns (also trained on)

### Pre-training Dataset
```json
[
  {"text": "Raw text document for pre-training..."},
  {"text": "Another document..."}
]
```

### Preference Dataset (for DPO/ORPO/SimPO/RM)
```json
[
  {
    "instruction": "What is AI?",
    "input": "",
    "chosen": "AI is artificial intelligence...",
    "rejected": "I don't know."
  }
]
```

Register with `ranking: true` in `dataset_info.json`.

## ShareGPT Format

Set `formatting: sharegpt` in `dataset_info.json`.

### SFT Dataset (Single-turn)
```json
[
  {
    "conversations": [
      {"from": "human", "value": "What is the capital of France?"},
      {"from": "gpt", "value": "The capital of France is Paris."}
    ],
    "system": "You are a helpful assistant."
  }
]
```

### SFT Dataset (Multi-turn)
```json
[
  {
    "conversations": [
      {"from": "human", "value": "Hi!"},
      {"from": "gpt", "value": "Hello! How can I help?"},
      {"from": "human", "value": "What is AI?"},
      {"from": "gpt", "value": "AI is artificial intelligence."}
    ]
  }
]
```

Roles: `human` (user), `gpt` (assistant). Human/observation must appear in odd positions, gpt/function_call in even positions.

### Preference Dataset (for DPO/ORPO/SimPO/RM)
```json
[
  {
    "conversations": [
      {"from": "human", "value": "What is AI?"},
      {"from": "gpt", "value": "Context response..."},
      {"from": "human", "value": "Tell me more."}
    ],
    "chosen": {"from": "gpt", "value": "Good detailed answer..."},
    "rejected": {"from": "gpt", "value": "Bad short answer..."}
  }
]
```

Register with `ranking: true` in `dataset_info.json`.

### KTO Dataset
```json
[
  {
    "conversations": [
      {"from": "human", "value": "What is AI?"},
      {"from": "gpt", "value": "AI is artificial intelligence."}
    ],
    "kto_tag": true
  }
]
```

`kto_tag: true` = desirable, `kto_tag: false` = undesirable.

### Tool-Calling / Agent Dataset
```json
[
  {
    "conversations": [
      {"from": "human", "value": "What's the weather in NYC?"},
      {"from": "function_call", "value": "{\"name\": \"get_weather\", \"arguments\": {\"city\": \"NYC\"}}"},
      {"from": "observation", "value": "{\"temperature\": 72, \"condition\": \"sunny\"}"},
      {"from": "gpt", "value": "It's 72 degrees and sunny in NYC."}
    ],
    "tools": "[{\"name\": \"get_weather\", \"description\": \"Get current weather\", \"parameters\": {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\"}}}}]"
  }
]
```

- `function_call`: Model's tool invocation (trained on)
- `observation`: Tool result (NOT trained on, treated like user input)
- `tools`: Tool definitions (injected into system prompt)

### Multimodal Image Dataset
```json
[
  {
    "conversations": [
      {"from": "human", "value": "<image>Describe this image."},
      {"from": "gpt", "value": "This image shows a cat sitting on a windowsill."}
    ],
    "images": ["path/to/image.jpg"]
  }
]
```

Number of `<image>` tokens must match length of `images` array.

### Multimodal Video Dataset
```json
[
  {
    "conversations": [
      {"from": "human", "value": "<video>What happens in this video?"},
      {"from": "gpt", "value": "A person walks through a park."}
    ],
    "videos": ["path/to/video.mp4"]
  }
]
```

### Multimodal Audio Dataset
```json
[
  {
    "conversations": [
      {"from": "human", "value": "<audio>Transcribe this audio."},
      {"from": "gpt", "value": "Hello, this is a test recording."}
    ],
    "audios": ["path/to/audio.wav"]
  }
]
```

### Reasoning / CoT Dataset
For thinking models (e.g., Qwen3, DeepSeek-R1), place chain-of-thought in the output:

```json
[
  {
    "instruction": "Solve: 15 * 23",
    "input": "",
    "output": "<think>15 * 23 = 15 * 20 + 15 * 3 = 300 + 45 = 345</think>345"
  }
]
```

## OpenAI Messages Format

A special case of ShareGPT with different role/content tags:

```json
[
  {
    "messages": [
      {"role": "system", "content": "You are helpful."},
      {"role": "user", "content": "What is AI?"},
      {"role": "assistant", "content": "AI is artificial intelligence."}
    ]
  }
]
```

Register in `dataset_info.json` with custom tags:
```json
{
  "my_openai_data": {
    "file_name": "data.jsonl",
    "formatting": "sharegpt",
    "columns": {
      "messages": "messages"
    },
    "tags": {
      "role_tag": "role",
      "content_tag": "content",
      "user_tag": "user",
      "assistant_tag": "assistant",
      "system_tag": "system"
    }
  }
}
```

## Registering Custom Datasets (dataset_info.json)

Place `dataset_info.json` in the dataset directory (default: `./data`). Each entry describes one dataset.

### Local Alpaca SFT
```json
{
  "my_sft_data": {
    "file_name": "my_data.json",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output"
    }
  }
}
```

### Local ShareGPT SFT
```json
{
  "my_sharegpt_data": {
    "file_name": "my_data.jsonl",
    "formatting": "sharegpt",
    "columns": {
      "messages": "conversations"
    }
  }
}
```

### HuggingFace Hub Dataset
```json
{
  "my_hf_data": {
    "hf_hub_url": "username/dataset-name",
    "formatting": "sharegpt",
    "columns": {
      "messages": "conversations"
    }
  }
}
```

### ModelScope Hub Dataset
```json
{
  "my_ms_data": {
    "ms_hub_url": "username/dataset-name",
    "formatting": "alpaca",
    "columns": {
      "prompt": "instruction",
      "response": "output"
    }
  }
}
```

### Preference Dataset (DPO)
```json
{
  "my_dpo_data": {
    "file_name": "dpo_data.jsonl",
    "formatting": "sharegpt",
    "ranking": true,
    "columns": {
      "messages": "conversations",
      "chosen": "chosen",
      "rejected": "rejected"
    }
  }
}
```

### KTO Dataset
```json
{
  "my_kto_data": {
    "file_name": "kto_data.jsonl",
    "formatting": "sharegpt",
    "columns": {
      "messages": "conversations",
      "kto_tag": "kto_tag"
    }
  }
}
```

### Multimodal Dataset
```json
{
  "my_vl_data": {
    "file_name": "vl_data.jsonl",
    "formatting": "sharegpt",
    "columns": {
      "messages": "conversations",
      "images": "images"
    }
  }
}
```

### Tool-Calling Dataset
```json
{
  "my_tool_data": {
    "file_name": "tool_data.json",
    "formatting": "sharegpt",
    "columns": {
      "messages": "conversations",
      "tools": "tools"
    }
  }
}
```

### All dataset_info.json Fields

| Field | Description | Default |
|-------|-------------|---------|
| `file_name` | Local file path | none |
| `hf_hub_url` | HuggingFace Hub repo | none |
| `ms_hub_url` | ModelScope Hub repo | none |
| `cloud_file_name` | S3/GCS URL | none |
| `formatting` | `alpaca` or `sharegpt` | `alpaca` |
| `ranking` | `true` for preference datasets | `false` |
| `subset` | Dataset subset name | none |
| `split` | Dataset split | `train` |
| `num_samples` | Limit sample count | none |
| `columns` | Column name mapping | varies |
| `tags` | ShareGPT role/content tag mapping | varies |

Data source priority: `hf_hub_url` > `ms_hub_url` > `script_url` > `cloud_file_name` > `file_name`

## Using Datasets in Training

Reference datasets by their `dataset_info.json` key names:

```yaml
dataset: my_sft_data,alpaca_en_demo    # Comma-separated dataset names
dataset_dir: /path/to/data/folder       # Folder containing dataset_info.json
```

### Dataset Mixing Strategies

| `mix_strategy` | Description |
|----------------|-------------|
| `concat` | Concatenate all datasets (default) |
| `interleave_under` | Interleave, undersample to shortest |
| `interleave_over` | Interleave, oversample to longest |
| `interleave_once` | Interleave, stop at shortest |

```yaml
dataset: data_a,data_b,data_c
mix_strategy: interleave_over
interleave_probs: 0.5,0.3,0.2    # Optional sampling weights
```
