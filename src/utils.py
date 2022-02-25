
def pad_tokenize(text):
    """Simple whitespace tokenization that includes punctuation padding."""
    new_text = ''
    chars = list(text)
    for i, char in enumerate(chars):
        cp = ord(char)
        if (cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or \
           (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126):
            char = ' ' + char
            try:
                if chars[i + 1] != ' ':
                    char += ' '
            except IndexError:
                pass
        new_text += char
    return new_text.split()


def replace_word(text, replace_at, replace_with=None):
    """Tokenize string in text and replace_with string at replace_at index."""
    text = pad_tokenize(text)
    text[replace_at] = replace_with
    return ' '.join(text)
