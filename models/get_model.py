from fastembed import TextEmbedding

model = TextEmbedding(
    model_name="BAAI/bge-small-en-v1.5",
    cache_dir="vendor/fastembed_cache",
    local_files_only=False,
)

# Force actual download + model initialization
_ = list(model.embed(["test"]))
print("model cached") 