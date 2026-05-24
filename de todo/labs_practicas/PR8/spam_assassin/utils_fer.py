import re

def preProcess(email):

    hdrstart = email.find("\n\n")
    if hdrstart != -1:
        email = email[hdrstart:]

    email = email.lower()
    # Strip html tags. replace with a space
    email = re.sub(r'<[^<>]+>', ' ', email)
    # Any numbers get replaced with the string 'number'
    email = re.sub(r'[0-9]+', 'number', email)
    # Anything starting with http or https:// replaced with 'httpaddr'
    email = re.sub(r'(http|https)://[^\s]*', 'httpaddr', email)
    # Strings with "@" in the middle are considered emails --> 'emailaddr'
    email = re.sub(r'[^\s]+@[^\s]+', 'emailaddr', email)
    # The '$' sign gets replaced with 'dollar'
    email = re.sub(r'[$]+', 'dollar', email)
    return email


def simple_stem(token: str) -> str:
    """
    Stemmer muy simple para salir del paso.
    No es Porter, pero reduce formas típicas.
    """
    suffixes = ["ing", "ed", "ly", "es", "s"]
    for suf in suffixes:
        if token.endswith(suf) and len(token) > len(suf) + 2:
            return token[:-len(suf)]
    return token


def email2TokenList(raw_email):
    """
    Function that takes in a raw email, preprocesses it, tokenizes it,
    'stems' each word with simple_stem, and returns a list of tokens.
    """

    email = preProcess(raw_email)

    # Split the e-mail into individual words (tokens)
    tokens = re.split(r'[ \@\$\/\#\.\-\:\&\*\+\=\[\]\?\!\(\)\{\}\,\'\"\>\_\<\;\%]',
                      email)

    tokenlist = []
    for token in tokens:
        token = re.sub(r'[^a-zA-Z0-9]', '', token)
        if not token:
            continue
        stemmed = simple_stem(token)
        tokenlist.append(stemmed)

    return tokenlist


def getVocabDict(reverse=False):
    """
    Read vocab list text file into a dictionary.
    key = stemmed word, value = index (or al revés si reverse=True)
    """
    vocab_dict = {}
    import os
    BASE_DIR = os.path.dirname(__file__)
    VOCAB_PATH = os.path.join(BASE_DIR, "vocab.txt")
    with open(VOCAB_PATH) as f:
        for line in f:
            (val, key) = line.split()
            if not reverse:
                vocab_dict[key] = int(val)
            else:
                vocab_dict[int(val)] = key

    return vocab_dict
