import gradio as gr
import librosa
import numpy as np

from prepare_quran_dataset.modeling_streaming_rnn.inference import (
    Wav2Vec2BertForRNNStreamingMultilevelCTCInference,
)

MODEL_PATH = "./results-streaming-rnn-v2/checkpoint-35480"
model = Wav2Vec2BertForRNNStreamingMultilevelCTCInference(MODEL_PATH, device="cpu")

processed_up_to = 0
chunk_idx = 0
results_lines = []


def on_stream(state_acc, audio_chunk):
    global processed_up_to, chunk_idx, results_lines

    if audio_chunk is None:
        return state_acc, "", ""

    sr, y = audio_chunk

    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)

    if state_acc is not None:
        state_acc = np.concatenate([state_acc, y])
    else:
        state_acc = y

    while True:
        is_first = chunk_idx == 0
        n = model.get_first_input_samples() if is_first else model.get_chunk_samples()

        if processed_up_to + n > len(state_acc):
            break

        wav = state_acc[processed_up_to : processed_up_to + n]
        for out in model(wav, is_first=is_first):
            results_lines.append(
                f"chunk {chunk_idx:02d}: "
                + " ".join(str(id_) for id_ in out.phonemes_ids)
            )

        processed_up_to += n
        chunk_idx += 1

    text = "\n".join(results_lines[-100:])
    summary = (
        f"chunks: {len(results_lines)} | processed: {processed_up_to / 16000:.1f}s"
    )
    return state_acc, text, summary


def on_start_recording():
    global processed_up_to, chunk_idx, results_lines
    processed_up_to = 0
    chunk_idx = 0
    results_lines = []
    model.reset()
    return None, "", "recording..."


def on_reset():
    global processed_up_to, chunk_idx, results_lines
    processed_up_to = 0
    chunk_idx = 0
    results_lines = []
    model.reset()
    return None, "", "reset OK"


with gr.Blocks(title="Streaming RNN Phoneme Inference") as demo:
    gr.Markdown("## Streaming RNN Phoneme Inference\nHold mic to record")

    audio_input = gr.Audio(
        sources=["microphone"],
        streaming=True,
        type="numpy",
        label="Hold to record",
    )

    state = gr.State()

    with gr.Row():
        reset_btn = gr.Button("Reset")
        summary_output = gr.Textbox(label="Summary", lines=1)

    output = gr.Textbox(label="Phoneme IDs (last 100 chunks)", lines=25)

    audio_input.start_recording(
        fn=on_start_recording,
        outputs=[state, output, summary_output],
    )

    audio_input.stream(
        fn=on_stream,
        inputs=[state, audio_input],
        outputs=[state, output, summary_output],
        stream_every=0.25,
    )

    reset_btn.click(
        fn=on_reset,
        outputs=[state, output, summary_output],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=True)
