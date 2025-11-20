# text/cleaners.py
import re
def indonesian_cleaners(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[0-9]+', lambda m: num_to_words(int(m.group())), text)
    return text.strip()

def num_to_words(n):
    ones = ["","satu","dua","tiga","empat","lima","enam","tujuh","delapan","sembilan"]
    teens = ["sepuluh","sebelas","dua belas","tiga belas","empat belas","lima belas","enam belas","tujuh belas","delapan belas","sembilan belas"]
    tens = ["","","dua puluh","tiga puluh","empat puluh","lima puluh","enam puluh","tujuh puluh","delapan puluh","sembilan puluh"]
    if n < 10:
        return ones[n]
    if n < 20:
        return teens[n-10]
    if n < 100:
        return tens[n//10] + ("" if n%10==0 else " " + ones[n%10])
    return str(n)