# config.py
class Config:
    sr = 22050
    n_mels = 80
    n_mfcc = 13
    total_channels = 119
    hop_length = 256
    win_length = 1024
    n_fft = 1024
    fmin = 0
    fmax = 8000
    max_len_sec = 15.0
    batch_size = 32
    lr_backbone = 3e-4
    lr_tts_head = 3e-4
    lr_vocoder = 2e-4
    epochs = 1000
    checkpoint_dir = "checkpoints_tts"
    log_dir = "logs_tts"
    seed = 42

config = Config()