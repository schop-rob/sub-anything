# sub-anything

A command-line tool to transcribe any audio or video file to SRT subtitles using state-of-the-art speech recognition models.

## Features

- **Multiple transcription backends** - Choose between Google Chirp 3, Google Long, or WhisperX (via Replicate)
- **Multi-language support** - Auto-detect or specify source language (70+ languages with Chirp 3)
- **Translation** - Translate subtitles to any language using OpenAI GPT
- **Speaker diarization** - Label who is speaking in multi-speaker audio
- **Smart chunking** - Automatically handles files longer than 1 hour with overlap merging
- **Subtitle embedding** - Optionally mux subtitles directly into video files
- **Word-level timestamps** - Accurate timing for subtitle synchronization
- **Cost estimation** - Shows estimated API cost after each run

## Installation

### Prerequisites

1. **Python 3.10+**

2. **ffmpeg** - Required for audio extraction and processing
   ```bash
   # macOS
   brew install ffmpeg

   # Ubuntu/Debian
   sudo apt install ffmpeg

   # Windows
   choco install ffmpeg
   ```

### Setup

```bash
# Clone or download the repository
cd sub-anything

# Install Python dependencies
pip install -r requirements.txt

# Make executable (optional)
chmod +x sub-anything

# Add to PATH (optional)
ln -s "$(pwd)/sub-anything" /usr/local/bin/sub-anything
```

## Configuration

### Environment Variables

| Variable | Required For | Description |
|----------|--------------|-------------|
| `GOOGLE_APPLICATION_CREDENTIALS` | chirp3, long | Path to Google Cloud service account JSON key |
| `REPLICATE_API_TOKEN` | whisperx | Replicate API token |
| `OPENAI_API_KEY` | --translate | OpenAI API key for translation |
| `HF_TOKEN` | --diarize (whisperx) | HuggingFace token for speaker diarization |

### First Run Setup (Google models)

On first run with a Google model, you'll be prompted for:
- **GCS Bucket Name** - A Google Cloud Storage bucket for temporary audio uploads (audio is uploaded, transcribed, then deleted)
- **Project ID** - Your Google Cloud project ID (auto-detected from credentials if possible)

These are saved to `config.json` in the script directory.

## Usage

### Basic Examples

```bash
# Transcribe a video (uses Chirp 3 by default)
sub-anything video.mp4

# Transcribe an audio file
sub-anything podcast.mp3

# Use WhisperX model (via Replicate)
sub-anything audio.mp3 --model whisperx

# Specify source language
sub-anything lecture.mp4 --language en-US

# Auto-detect language (default)
sub-anything foreign_film.mkv --language auto
```

### Translation

```bash
# Transcribe and translate to Spanish
sub-anything video.mp4 --translate es

# Transcribe Chinese audio and translate to English
sub-anything chinese_interview.mp3 --translate en

# Verbose mode shows detected languages
sub-anything mixed_language.mp4 --translate en -v
```

### Speaker Diarization

```bash
# Label speakers with WhisperX (recommended)
sub-anything interview.wav --model whisperx --diarize

# Diarization with Google models
sub-anything meeting.mp4 --model chirp3 --diarize
```

### Embedding Subtitles

```bash
# Create video with embedded subtitle track (soft subs)
sub-anything movie.mkv --mux
# Output: movie_subtitled.mkv + movie.srt
```

### All Options

```bash
sub-anything [OPTIONS] INPUT_FILE

Options:
  -v, --verbose          Show detailed progress and debug info
  --model MODEL          Transcription model: chirp3, long, whisperx (default: chirp3)
  --google-location LOC  Google Speech-to-Text location for chirp3/long (e.g., us, eu, asia-northeast1)
  --language LANG        Source language hint (default: auto)
  --translate LANG       Translate subtitles to target language
  --translate-model      OpenAI model for --translate (default: gpt-4o-mini)
  --translate-batch-size Subtitle segments per translation request (default: 20)
  --save-original        When using --translate, also save the original transcript as *.orig.srt/.orig.txt
  --diarize              Enable speaker diarization
  --mux                  Embed subtitles into video file (soft subs)
  --no-timestamps        Output plain text (.txt) instead of SRT subtitles (.srt)
  -h, --help             Show help message
```

## Models

| Model | Provider | Strengths | Limitations |
|-------|----------|-----------|-------------|
| `chirp3` | Google Cloud | Best accuracy, 70+ languages, good for accents | May have occasional timestamp gaps |
| `long` | Google Cloud | Guaranteed accurate timestamps | Slightly less accurate transcription |
| `whisperx` | Replicate | Excellent timestamps, fast, great diarization | English-focused (supports others) |

### Model Selection Guide

- **General use**: `chirp3` (default) - best overall quality
- **Guaranteed timestamps**: `long` - when subtitle timing is critical
- **Interviews/meetings**: `whisperx --diarize` - best speaker separation
- **Quick processing**: `whisperx` - fastest turnaround

## Output

By default, the tool generates an SRT file in the same directory as the input:

```
input: /path/to/video.mp4
output: /path/to/video.srt
```

With `--no-timestamps`, it writes plain text instead:

```
output: /path/to/video.txt
```

With `--translate --save-original`, it also writes the untranslated version as `*.orig.srt` (or `*.orig.txt`).

### SRT Format Features

- **42-character line wrapping** - Standard broadcast subtitle width
- **Low confidence marking** - Uncertain segments prefixed with `[?]`
- **Speaker labels** - When diarization is enabled: `[Speaker 1] Hello...`

## Supported Formats

### Video
`.mp4`, `.mkv`, `.avi`, `.mov`, `.webm`, `.flv`, `.wmv`, `.m4v`

### Audio
`.mp3`, `.wav`, `.flac`, `.ogg`, `.m4a`, `.aac`, `.wma`, `.opus`

## How It Works

1. **Audio Extraction** - Extracts audio from video files (skipped for audio-only inputs)
2. **Chunking** - Splits files >1 hour into chunks with 10-second overlap
3. **Upload** - Uploads audio to cloud storage (GCS for Google, direct for Replicate)
4. **Transcription** - Sends to speech recognition API
5. **Merging** - Combines chunks using confidence-based overlap resolution
6. **Translation** - Optionally translates via OpenAI GPT with timing reflow
7. **SRT Generation** - Formats output with proper timestamps and line wrapping
8. **Cleanup** - Removes temporary files from cloud storage

## Cost Estimation

Approximate costs per minute of audio:

| Model | Cost/min |
|-------|----------|
| chirp3 | ~$0.016 |
| long | ~$0.016 |
| whisperx | ~$0.006 |

Translation adds ~$0.001-0.005/min depending on text length.

## Troubleshooting

### "No speech detected"
- Check that the audio contains clear speech
- Try a different model (`--model whisperx` or `--model long`)
- Specify the language explicitly (`--language en-US`)

### Timestamp gaps with Chirp 3
- Chirp 3 may occasionally skip timestamps for paraphrased content
- Use `--model long` for guaranteed timestamps
- Use `--model whisperx` for most reliable timing

### Diarization not working
- WhisperX requires `HF_TOKEN` environment variable
- Google diarization only supports 15 languages

### Rate limiting
- The tool automatically retries with exponential backoff
- For large batches, add delays between files

### Language code errors (Google models)
- Google models expect BCP-47 language codes (e.g. `en-US`, `cmn-Hans-CN`)
- If you see an error mentioning a specific location (e.g. `location named "us"`), try `--google-location eu`
- If unsure, omit `--language` (auto-detect) or use `--model whisperx`

## License

MIT License - See LICENSE file for details.

## Credits

- [Google Cloud Speech-to-Text](https://cloud.google.com/speech-to-text) - Chirp 3 and Long models
- [WhisperX](https://github.com/m-bain/whisperX) - Word-level timestamp alignment
- [Replicate](https://replicate.com) - WhisperX hosting
- [OpenAI](https://openai.com) - Translation via GPT
