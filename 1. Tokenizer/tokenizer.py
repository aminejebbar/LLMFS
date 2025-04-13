"""
What's a Tokenizer and why do we need it?

A tokenizer is a really important part of any NLP project when we want to process text.

The reason for that is because computers cannot understand our language, letters or whatever.

They speak in numbers, a good example is Unicode. Unicode is the way that a computer shows letters. You see letters because we're Human and it's more easier for us. When
they see numbers, numbers because they're computer and it's easier for them.

A tokenizer is just a kind of split function that will cut our words and transform them as a bunch of numbers.
"""

# Test Spliting for Tokenizer Understanding
import re

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
    
print("Total number of characters:", len(raw_text))
print(raw_text[:99])

text = "Hello, world. This, is a test."
result = re.split(r'([,.]|\s)', text)
result = [item for item in result if item.strip()]
print(result)
# Result: ['Hello', ',', 'world', '.', 'This', ',', 'is', 'a', 'test', '.']

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(preprocessed[:30])
# Result: ['I', 'HAD', 'always', 'thought', 'Jack', 'Gisburn', 'rather', 'a', 'cheap', 'genius', '--', 'though', 'a', 'good', 'fellow', 'enough', '--', 'so', 'it', 'was', 'no', 'great', 'surprise', 'to', 'me', 'to', 'hear', 'that', ',', 'in']

print(len(preprocessed))
# Result: 4690

# We remove duplicates using the "set" function and then sort in alphabetical order to get a list of all the words, 
# and we calculate the total number of words.

all_words = sorted(set(preprocessed))
vocab_size = len(all_words)

print(vocab_size)
# Result: 1130

vocab = {token:integer for integer,token in enumerate(all_words)}

class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}
    
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
                                
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
        
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text
    

tokenizer = SimpleTokenizerV1(vocab)

text = """"It's the last he painted, you know," 
           Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
print(ids)
# Result: [1, 56, 2, 850, 988, 602, 533, 746, 5, 1126, 596, 5, 1, 67, 7, 38, 851, 1108, 754, 793, 7]

decoded_text = tokenizer.decode(ids)
print(decoded_text)
# Result: '" It\'s the last he painted, you know," Mrs. Gisburn said with pardonable pride.'

# We add two more tokens for handling the endoftext in a sentence and the unknown token for avoiding errors while encoding.
all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])

vocab = {token:integer for integer,token in enumerate(all_tokens)}

class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}
    
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [
            item if item in self.str_to_int 
            else "<|unk|>" for item in preprocessed
        ]

        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
        
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text
    
tokenizer = SimpleTokenizerV2(vocab)

text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."

text = " <|endoftext|> ".join((text1, text2))

print(text)
# Result: Hello, do you like tea? <|endoftext|> In the sunlit terraces of the palace.

encoded_text = tokenizer.encode(text)
print(encoded_text)
# Result of encoding: [encoded IDs representing '<|unk|>, do you like tea? <|endoftext|> In the sunlit terraces of the <|unk|>.']

decoded_text = tokenizer.decode(tokenizer.encode(text))
print(decoded_text)
# Result of decoding: '<|unk|>, do you like tea? <|endoftext|> In the sunlit terraces of the <|unk|>.'

"""
The problem with those kind of tokenizers is that we need to make the computer learn all the words or possibilities. It's possible but not optimal.

So that's why we use a BPE encoder.

Instead of splitting by word, we splitting the word itself as many pieces.
"""

# Example of Tiktoken BPE Encoder/Decoder:

import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces "
    "of someunknownPlace."
)

integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})

print(integers)
# Result: [15496, 11, 466, 345, 588, 8887, 30, 220, 50256, 554, 262, 4252, 18250, 8812, 2114, 1659, 617, 34680, 27271, 13]

strings = tokenizer.decode(integers)

print(strings)
# Result: Hello, do you like tea? <|endoftext|> In the sunlit terraces of someunknownPlace.

# Trying to encode "Akwirw ier"
# Exercise
exercice_Akwirw_ier = tokenizer.encode("Akwirw ier", allowed_special={"<|endoftext|>"})

print(exercice_Akwirw_ier)
# Result: [33901, 86, 343, 86, 220, 959]

# Decoding Exercise
decode_exercice = tokenizer.decode(exercice_Akwirw_ier)

print(decode_exercice)
# Result: Akwirw ier