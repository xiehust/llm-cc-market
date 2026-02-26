import time
import statistics
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

prompts = [
    "Explain the water cycle in one paragraph.",
    "What are the benefits of exercise?",
    "Write a short poem about the ocean.",
    "What is machine learning? Explain simply.",
    "List 5 tips for better sleep.",
]

print("=" * 60)
print("Performance Benchmark — Qwen3.5-35B-A3B (LoRA merged)")
print("=" * 60)

# --- Latency test (sequential, streaming TTFT) ---
print("\n[1] Latency Test (5 requests, sequential, max_tokens=200)\n")
latencies = []
ttfts = []
token_counts = []

for i, prompt in enumerate(prompts):
    t0 = time.perf_counter()
    first_token_time = None
    content = ""
    stream = client.chat.completions.create(
        model="merged",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        if delta and first_token_time is None:
            first_token_time = time.perf_counter()
        content += delta
    t1 = time.perf_counter()
    latency = t1 - t0
    ttft = (first_token_time - t0) if first_token_time else 0
    tokens = len(content.split())
    latencies.append(latency)
    ttfts.append(ttft)
    token_counts.append(tokens)
    print(f"  [{i+1}] TTFT: {ttft*1000:.0f}ms | Total: {latency:.2f}s | ~{tokens} words | prompt: \"{prompt[:40]}...\"")

print(f"\n  Avg TTFT:    {statistics.mean(ttfts)*1000:.0f} ms")
print(f"  Avg latency: {statistics.mean(latencies):.2f} s")
print(f"  Avg output:  ~{statistics.mean(token_counts):.0f} words")

# --- Throughput test (concurrent) ---
print("\n[2] Throughput Test (10 concurrent requests, max_tokens=100)\n")
import threading

results = []
errors = []

def send_request(idx):
    prompt = prompts[idx % len(prompts)]
    t0 = time.perf_counter()
    try:
        resp = client.chat.completions.create(
            model="merged",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
        )
        t1 = time.perf_counter()
        out_tokens = resp.usage.completion_tokens
        results.append((t1 - t0, out_tokens))
    except Exception as e:
        errors.append(str(e))

threads = [threading.Thread(target=send_request, args=(i,)) for i in range(10)]
t_start = time.perf_counter()
for t in threads: t.start()
for t in threads: t.join()
t_total = time.perf_counter() - t_start

if results:
    total_tokens = sum(r[1] for r in results)
    avg_lat = statistics.mean(r[0] for r in results)
    throughput = total_tokens / t_total
    print(f"  Requests:       10")
    print(f"  Wall time:      {t_total:.2f} s")
    print(f"  Avg latency:    {avg_lat:.2f} s")
    print(f"  Total tokens:   {total_tokens}")
    print(f"  Throughput:     {throughput:.1f} tokens/s")
if errors:
    print(f"  Errors: {errors}")

print("\n" + "=" * 60)
print("Benchmark complete.")
print("=" * 60)