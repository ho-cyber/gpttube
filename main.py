from flask import Flask, render_template, request
import os
import argparse
from pydub import AudioSegment
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import whisper

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models = T5ForConditionalGeneration.from_pretrained("Michau/t5-base-en-generate-headline")
tokenizer = T5Tokenizer.from_pretrained("Michau/t5-base-en-generate-headline")
models = models.to(device)
model = whisper.load_model("base")

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        url = request.form['url']
        name = request.form['name']

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

        for i in range(len(audio_list)):
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

            original_text += "\n"
            original_text += "<h3>" + generated_heading + "</h3>"
            original_text += "\n"
            original_text += "<p>" + result.text + "</p>"

        with open(name, "w") as f:
            f.write(original_text)

        return render_template('result.html', headings=headings, texts=original_texts)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
