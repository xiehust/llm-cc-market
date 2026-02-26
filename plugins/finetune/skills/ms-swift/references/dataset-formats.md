# Dataset Formats

ms-swift auto-detects four input formats via `AutoPreprocessor`. All can be used directly with `--dataset <path>`.

## Four Auto-Detected Input Formats

### Format 1: Messages (Standard / OpenAI-style)
```jsonl
{"messages": [{"role": "system", "content": "You are helpful"}, {"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]}
```

### Format 2: ShareGPT
```jsonl
{"system": "You are helpful", "conversation": [{"human": "Q", "assistant": "A"}]}
```

### Format 3: Query-Response
```jsonl
{"system": "You are helpful", "query": "Q", "response": "A", "history": [["Q1", "A1"]]}
```
Auto-detected field names — query: `query`, `prompt`, `input`, `instruction`, `question`, `problem`; response: `response`, `answer`, `output`, `targets`, `solution`, `text`, `completion`, `content`.

### Format 4: Alpaca
```jsonl
{"system": "You are helpful", "instruction": "Summarize this", "input": "long text...", "output": "summary"}
```
Query becomes `f'{instruction}\n{input}'` when both are non-empty.

## Task-Specific Standard Formats

### Pre-training
```jsonl
{"messages": [{"role": "assistant", "content": "Raw text for pre-training..."}]}
```

### SFT (Single-turn)
```jsonl
{"messages": [{"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": "4"}]}
```

### SFT (Multi-turn)
```jsonl
{"messages": [{"role": "user", "content": "Q1"}, {"role": "assistant", "content": "A1"}, {"role": "user", "content": "Q2"}, {"role": "assistant", "content": "A2"}]}
```

### SFT with Per-Message Loss Control (v3.8+)
Set `"loss": false` on assistant messages to exclude from loss computation:
```jsonl
{"messages": [{"role": "user", "content": "Q1"}, {"role": "assistant", "content": "A1", "loss": false}, {"role": "user", "content": "Q2"}, {"role": "assistant", "content": "A2", "loss": true}]}
```

### DPO / ORPO / CPO / SimPO / RM (Preference Data)
```jsonl
{"messages": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "chosen answer"}], "rejected_response": "rejected answer"}
```
Or with full rejected messages:
```jsonl
{"messages": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "chosen"}], "rejected_messages": [{"role": "assistant", "content": "rejected"}]}
```

### KTO (Binary Preference)
```jsonl
{"messages": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}], "label": true}
```
`label: true` = desirable, `label: false` = undesirable.

### PPO / GRPO (Query Only — Model Generates Completions)
```jsonl
{"messages": [{"role": "user", "content": "Solve: what is 15 * 23?"}]}
```
GRPO passes all extra fields to the reward function. Add a `"solution"` field for the `accuracy` reward:
```jsonl
{"messages": [{"role": "user", "content": "Solve: 15 * 23 = ?"}], "solution": "345"}
```

### GKD (Knowledge Distillation)
Same as SFT format. With `--seq_kd true`, can also use prompts-only format (teacher generates).

### Sequence Classification
```jsonl
{"messages": [{"role": "user", "content": "This movie is great!"}], "label": 1}
```
Supports single-label (int), multi-label (list of ints), regression (float).

## Multimodal Data Formats

Add media fields to any format above. Tags (`<image>`, `<video>`, `<audio>`) mark insertion points in content.

### Image
```jsonl
{"messages": [{"role": "user", "content": "<image>What is in this image?"}, {"role": "assistant", "content": "A cat."}], "images": ["/path/to/cat.jpg"]}
```

### Video
```jsonl
{"messages": [{"role": "user", "content": "<video>Describe this video."}, {"role": "assistant", "content": "A person walking."}], "videos": ["/path/to/clip.mp4"]}
```

### Audio
```jsonl
{"messages": [{"role": "user", "content": "<audio>Transcribe this."}, {"role": "assistant", "content": "Hello world."}], "audios": ["/path/to/audio.wav"]}
```

### Multiple Images
```jsonl
{"messages": [{"role": "user", "content": "<image><image>Compare these two images."}, {"role": "assistant", "content": "The first shows..."}], "images": ["/path/to/img1.jpg", "/path/to/img2.jpg"]}
```

Media fields accept: file paths, URLs, base64 strings, or video frames as image lists.

### Multimodal RLHF
Include `rejected_images` / `rejected_videos` / `rejected_audios` for preference data with different media:
```jsonl
{"messages": [...], "rejected_response": "...", "images": [...], "rejected_images": [...]}
```

## Agent / Tool-Calling Data Format

Use with `--agent_template hermes` (or `react_en`, `glm4`, `qwen_en`, `toolbench`).

```jsonl
{"tools": "[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"description\":\"Get weather\",\"parameters\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}}}}}]", "messages": [{"role": "user", "content": "What's the weather in Beijing?"}, {"role": "tool_call", "content": "{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Beijing\"}}"}, {"role": "tool_response", "content": "{\"temp\": 25, \"condition\": \"sunny\"}"}, {"role": "assistant", "content": "It's 25 degrees and sunny in Beijing."}]}
```

Key behaviors:
- `tools` field is combined with the system prompt per agent_template
- `tool_call` messages are auto-converted to assistant format
- `tool_response` messages are treated as user input (excluded from loss)
- Supports parallel tool calls (multiple tool_call/tool_response sequences)
- Supports multimodal agents (add `images`/`videos`/`audios` fields)

## Grounding Data Format

```jsonl
{"messages": [{"role": "user", "content": "<image>Find the dog"}, {"role": "assistant", "content": "The <ref>dog</ref> is here."}], "images": ["/path/to/img.jpg"], "objects": {"ref": ["dog"], "bbox": [[100, 200, 300, 400]], "bbox_type": "real"}}
```

## Custom Column Mapping

Remap column names via CLI:
```bash
~/swift-env/bin/swift sft --dataset data.jsonl --columns '{"problem": "query", "solution": "response"}'
```

Or register datasets via `dataset_info.json`:
```json
[{"dataset_name": "my_data", "dataset_path": "/path/to/data", "columns": {"input": "query", "output": "response"}}]
```
Usage: `--custom_dataset_info my_info.json --dataset my_data`

## Dataset Sampling Syntax

```bash
--dataset 'dataset_id#500'                    # Sample 500 rows
--dataset 'dataset_id:subset_name'            # Specific subset
--dataset 'dataset_id:subset1/subset2'        # Multiple subsets
--dataset 'dataset_id:all'                    # All subsets
--dataset data1.jsonl data2.jsonl             # Multiple datasets
```
