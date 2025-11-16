from huggingface_hub import snapshot_download

print("Downloading Qwen3-Embedding-0.6B...")
snapshot_download(
    repo_id="Qwen/Qwen3-Embedding-0.6B",
    local_dir="models/Qwen3-Embedding-0.6B"
)
print("✓ Qwen3-Embedding-0.6B downloaded")

print("\nDownloading Qwen3-Reranker-0.6B...")
snapshot_download(
    repo_id="Qwen/Qwen3-Reranker-0.6B",
    local_dir="models/Qwen3-Reranker-0.6B"
)
print("✓ Qwen3-Reranker-0.6B downloaded")

print("\nAll models downloaded successfully!")
