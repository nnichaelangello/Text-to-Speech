# text/symbols.py
symbols = ['PAD','EOS','UNK',' ','a','i','u','e','ə','o','ɛ','ɔ','b','c','d','f','g','h','j','k','l','m','n','ŋ','ɲ','p','q','r','s','t','v','w','x','y','z','dʒ','tʃ','ʃ','ʒ','kh','sy','ng','ny',',','.','!','?','-']
symbol_to_id = {s:i for i,s in enumerate(symbols)}
id_to_symbol = {i:s for i,s in enumerate(symbols)}