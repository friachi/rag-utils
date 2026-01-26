# Build
docker buildx build --platform linux/amd64 -t falriachi/rag-utils:amd64 --load --progress=plain .

# Start a Test container
# docker run --rm -p 8181:8181 falriachi/rag-utils:amd64