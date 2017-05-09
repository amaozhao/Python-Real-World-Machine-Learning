from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
# from nltk.tokenize import PunktWordTokenizer
from nltk.tokenize.punkt import PunktSentenceTokenizer
from nltk.tokenize import WordPunctTokenizer

text = ("Are you curious about tokenization? " +
        "Let's see how it works! " +
        "We need to analyze a couple of " +
        "sentences with punctuations to see it in action.")


sent_tokenize_list = sent_tokenize(text)
print ("Sentence tokenizer:")
print (sent_tokenize_list)


print ("Word tokenizer:")
print (word_tokenize(text))

# Create a new punkt word tokenizer


punkt_sent_tokenizer = PunktSentenceTokenizer()
print ("Punkt word tokenizer:")
print (punkt_sent_tokenizer.tokenize(text))


word_punct_tokenizer = WordPunctTokenizer()
print ("Word punct tokenizer:")
print (word_punct_tokenizer.tokenize(text))
