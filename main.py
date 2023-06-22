import streamlit as st
import os
import argparse
from pydub import AudioSegment
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import whisper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models = T5ForConditionalGeneration.from_pretrained("Michau/t5-base-en-generate-headline")
tokenizer = T5Tokenizer.from_pretrained("Michau/t5-base-en-generate-headline")
models = models.to(device)
model = whisper.load_model("base")

st.title("GPTTUBE")

# Arguments input
st.subheader("Enter YouTube link and file name:")
url = st.text_input("YouTube link")
name = st.text_input("File name")

# Process audio and generate headings
if st.button("Process"):
    if os.path.exists("audio.mp3"):
        os.remove("audio.mp3")

    os.system("youtube-dl "+"--write-thumbnail "+"--skip-download "+url + " -o logo.png")
    os.system("yt-dlp -f 140 -o audio.mp3 " + url)

    while not os.path.exists("audio.mp3"):
        continue

    if os.path.exists("segments"):
        os.system("rm -rf segments")

    audio = AudioSegment.from_file("audio.mp3")
    segment_length = 30 * 1000

    if not os.path.exists("segments"):
        os.makedirs("segments")

    for i, segment in enumerate(audio[::segment_length]):
        segment.export(f"segments/{i}.mp3", format="mp3")

    original_text = ""
    audio_list = os.listdir("segments")
    headings = []
    original_texts = []
    dataForWeb = {}

    for i in range(len(audio_list)):
        st.write(f"Processing segment {i+1}/{len(audio_list)}")
        audio = whisper.load_audio(f"segments/{i}.mp3")
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(model.device)
        _, probs = model.detect_language(mel)
        options = whisper.DecodingOptions(fp16=False)
        result = whisper.decode(model, mel, options)

        text = "headline: " + result.text
        max_len = 256
        encoding = tokenizer.encode_plus(text, return_tensors="pt")
        input_ids = encoding["input_ids"].to(device)
        attention_masks = encoding["attention_mask"].to(device)
        beam_outputs = models.generate(
            input_ids=input_ids,
            attention_mask=attention_masks,
            max_length=64,
            num_beams=3,
            early_stopping=True,
        )
        generated_heading = tokenizer.decode(beam_outputs[0])
        headings.append(generated_heading)
        original_texts.append(result.text)
        dataForWeb[i] = {
            "heading": generated_heading,
            "text": result.text
        }

        original_text += "\n"
        original_text += "<h3>" + generated_heading + "</h3>"
        original_text += "\n"
        original_text += "<p>" + result.text + "</p>"

    with open(name, "w") as f:
        f.write(original_text)

    st.success("Audio processing completed!")

    # Display results
    st.subheader("Generated Headings and Text:")
    for i, heading in enumerate(headings):
        st.write(f"Segment {i+1}:")
        st.write("Heading:", heading)
        st.write("Text:", original_texts[i])
        st.write("-----------")
