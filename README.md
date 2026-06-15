# kr-sc-srt

Two-pass workflow for public SOOP VOD Korean subtitles and Simplified-Chinese hard-sub videos.

The project is designed for Colab first, with the same CLI available on Linux and Windows.

## Workflow

1. First pass, `prepare`
   - Downloads the lowest available video quality.
   - Extracts 16 kHz mono WAV audio.
   - Uses FunASR `iic/SenseVoiceSmall` to create `ko.srt`.
   - Stops there so you can download `ko.srt` and translate it locally with the helper command.
   - Does not create a subtitled video.
2. Second pass, `render`
   - Downloads the highest available video quality.
   - Reads your locally translated `zh.srt` from the job output directory.
   - Reads a same-job CSV file describing segments.
   - Cuts each segment, retimes subtitles, and burns Chinese subtitles into MP4 files.

The pipeline is resumable. Every stage writes state into `run.json`; reruns skip completed stages when outputs still exist and parameters have not changed.

## Colab Usage

Use `notebooks/colab_soop_kr_to_zh.ipynb`.

Default persistent paths:

- Root: `/content/drive/MyDrive/kr-sc-srt`
- Model cache: `/content/drive/MyDrive/kr-sc-srt/models`
- Outputs: `/content/drive/MyDrive/kr-sc-srt/outputs/<job-name>`
- Last job: `/content/drive/MyDrive/kr-sc-srt/last_job.json`

On the first run, set `VOD_URL`. If a later run fails or the Colab runtime restarts, leave `VOD_URL` empty and run with resume enabled; the notebook uses the URL from `last_job.json`.

## Local Install

Install Python dependencies:

```bash
python -m pip install -e .
```

Install system tools:

- `ffmpeg` and `ffprobe`
- `yt-dlp`

Example:

```bash
kr-sc-srt prepare "https://vod.sooplive.com/player/198391511" --root ./work
```

Resume the same first pass without typing the URL again:

```bash
kr-sc-srt prepare --root ./work --resume-last
```

Translate the Korean SRT on your local machine:

```bash
kr-sc-srt translate-srt ./work/outputs/<job-name>/ko.srt ./work/outputs/<job-name>/zh.srt \
  --api-key-env OPENAI_API_KEY
```

Or translate using DeepSeek API:

```bash
kr-sc-srt translate-srt ./work/outputs/<job-name>/ko.srt ./work/outputs/<job-name>/zh.srt \
  --api-base "https://api.deepseek.com/v1" \
  --translation-model "deepseek-chat" \
  --api-key "YOUR_DEEPSEEK_API_KEY"
```


Create a segment CSV at `./work/outputs/<job-name>/<job-name>.csv`:

```csv
name,start,end
part-01,00:01:30,00:05:00
part-02,00:05:00,00:08:30.500
```

Run the second pass:

Before rendering, upload or save the translated Chinese file as `zh.srt` in the same job output directory.

```bash
kr-sc-srt render --root ./work --resume-last
```

## Important Options

- `--model-cache-dir`: persistent FunASR/model cache. In Colab, use Google Drive.
- `--resume-last`: load the URL and output directory from `<root>/last_job.json`.
- `--force-stage STAGE`: rerun one stage, for example `--force-stage asr`.
- `--force-all`: rerun every stage.
- `--cookies cookies.txt`: optional yt-dlp cookies file. First version only guarantees public SOOP VODs.
- `--segments file.csv`: explicit segment CSV for `render`.
- `--asr-chunk-s SECONDS`: ASR audio chunk size in seconds (default: 180).
- `--test`: test mode, only extract and process the first 10 minutes of audio.

For local translation:

- `translate-srt INPUT OUTPUT`: translate a Korean SRT into Chinese SRT without running video processing.
- `--api-key-env`: environment variable containing the OpenAI-compatible API key.
- `--api-base`: OpenAI-compatible API base URL.
- `--translation-model`: chat model used for translation.

## Outputs

First pass:

- `source.low.mp4` or equivalent yt-dlp output
- `audio.aac`
- `ko.srt`
- `run.json`

You create `zh.srt` locally with `translate-srt`, then upload or save it beside `ko.srt`.

Second pass:

- `source.high.mp4` or equivalent yt-dlp output
- `segments/<name>.zh.srt`
- `segments/<name>.mp4`

## Notes

The download layer uses yt-dlp. Public SOOP VOD support depends on yt-dlp and the current SOOP site behavior. If a video requires login, pass a cookies file or download the video manually and use the local file path as the source.
