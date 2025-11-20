# text/g2p_id.py
from g2p_id import G2P
g2p = G2P()

def text_to_phonemes(text):
    cleaned = indonesian_cleaners(text)
    return g2p(cleaned)