from pathlib import Path
import kagglehub

download_dir = Path("/projects/EEG-foundation-model/RECH202/speech_raw_datasets").expanduser()
download_dir.mkdir(parents=True, exist_ok=True)

path = kagglehub.dataset_download(
    "abdulkareembageri/imagined-speech-eeg-signal-bci2020",
    output_dir=str(download_dir),
)

print("Path to dataset files:", path)