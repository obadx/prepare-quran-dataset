FROM vllm/vllm-openai:latest
RUN pip install vllm[audio]
