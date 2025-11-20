```markdown
# Indonesian End-to-End Text-to-Speech System

A high-quality, monolingual Text-to-Speech (TTS) system for Bahasa Indonesia built with PyTorch. The model combines a shared CNN backbone operating on rich acoustic features (Mel-spectrogram + MFCC + deltas) with an autoregressive acoustic model and a HiFi-GAN vocoder.

## Key Features

- Full support for Indonesian phonology (including digraphs: ng, ny, kh, sy, etc.)
- Automatic number-to-word conversion during text normalization
- 119-channel input features (80 Mel + 13 MFCC + 13 Δ + 13 ΔΔ)
- Mixed-precision training with GradScaler
- HiFi-GAN-based neural vocoder for natural waveform synthesis
- Clean, modular code structure

## Project Structure

```
.
├── config.py                  # Global configuration and hyperparameters
├── audio_utils.py             # WAV loading, normalization, and saving
├── text/
│   ├── symbols.py             # Phoneme/symbol vocabulary
│   ├── cleaners.py            # Indonesian text normalization & num-to-word
│   └── g2p_id.py              # Wrapper for g2p-id grapheme-to-phoneme
├── feature_extractor.py       # Extraction of 119-channel acoustic features
├── model/
│   ├── shared_cnn_backbone.py # CNN feature extractor (shared)
│   ├── tts_head.py            # Autoregressive Mel-spectrogram predictor
│   ├── hifigan_generator.py   # HiFi-GAN vocoder
│   └── discriminators.py      # Multi-Period & Multi-Scale discriminators (for future GAN training)
├── dataset.py                 # Custom Dataset and collate function
├── train.py                   # Training script (currently supervised L1 on Mel)
├── inference.py               # Simple inference script
└── checkpoints_tts/           # Saved model weights
```

## Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.0 (CUDA recommended)
- librosa
- soundfile
- torchaudio
- python_speech_features
- tqdm
- g2p-id

```bash
pip install torch torchvision torchaudio
pip install librosa soundfile python_speech_features tqdm g2p-id
```

## Dataset Format

Create a `metadata.txt` file with one line per sample:

```
wavs/001.wav|Selamat pagi, apa kabar hari ini?
wavs/002.wav|Saya sedang belajar text to speech.
...
```

Audio files must be mono WAVs (any sample rate; will be resampled to 22.05 kHz).

## Training

```bash
python train.py
```

- Checkpoints are saved every 50 epochs in `checkpoints_tts/`
- Current training objective: L1 reconstruction loss on Mel-spectrograms
- Discriminator modules are included but not yet used (ready for future adversarial training)

## Inference

```bash
python inference.py
```

This will synthesize the example sentence and save the result as `output_tts_indonesia.wav`.

The `synthesize(text)` function in `inference.py` can also be imported and used programmatically.
