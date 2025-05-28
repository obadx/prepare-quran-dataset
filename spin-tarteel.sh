docker build -t vllm-audio .
docker run --gpus all \
    	-v ~/.cache/huggingface:/root/.cache/huggingface \
    	-p 8000:8000 \
    	--ipc=host \
    	vllm-audio \
	--no-enable-prefix-caching \
    	--model tarteel-ai/whisper-base-ar-quran \
	--gpu-memory-utilization 0.9 \
	--dtype float32 \
	# --cpu-offload-gb 4 \
	# --max-model-len 10 \


# --env "HUGGING_FACE_HUB_TOKEN=<secret>" \
