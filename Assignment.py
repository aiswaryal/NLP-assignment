
import nltk 


paragraph = """I have three visions for India. In 3000 years of our history, people from all over the world have come and invaded us, captured our lands, conquered our minds. From Alexander onwards, the Greeks, the Turks, the Moguls, the Portuguese, the British, the French, the Dutch, all of them came and looted us, took over what was ours. Yet, we have not done this to any other nation. We have not conquered anyone. We have not grabbed their land, their culture, their history and tried to enforce our way of life on them. Why? Because we respect the freedom of others.

That is why my first vision is that of FREEDOM.

I believe that India got its first vision of this in 1857, when we started the war of Independence. It is this freedom that we must protect and nurture and build on. If we are not free, no one will respect us.

My second vision for India’s DEVELOPMENT.

For fifty years we have been a developing nation. It is time we see ourselves as a developed nation. We are among top five nations of the world in terms of GDP. We have 10 per cent growth rate in most areas. Our poverty levels are falling. Our achievements are being globally recognised today. Yet we lack the self-confidence to see ourselves as a developed nation, self-reliant and self-assured. Isn’t this incorrect?

I have a THIRD vision.

India must stand up to the world. Because I believe that, unless India stands up to the world, no one will respect us. Only strength respects strength. We must be strong not only as a military power but also as an economic power. Both must go hand-in-hand.

My good fortune was to have worked with three great minds. Dr. Vikram Sarabhai of the Department of Space, Professor Satish Dhawan, who succeeded him and Dr.Brahm Prakash, the father of nuclear material. I was lucky to have worked with all three of them closely and consider this the great opportunity of my life.

I see four milestones in my career: """

#cleaning the texts
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

ps = PorterStemmer()
wordnet=WordNetLemmatizer()
sentences = nltk.sent_tokenize(paragraph)
corpus = []
for i in range(len(sentences)):
    review = re.sub('[^a-zA-Z]', ' ', sentences[i])
    review = review.lower()
    review = review.split()
    review = [wordnet.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
# Creating TFIDF
from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer()
X = cv.fit_transform(corpus).toarray()

#To find frequency of words in corpus
 
wordfreq = {}
for sentence in corpus:
    tokens = nltk.word_tokenize(sentence)
    for token in tokens:
        if token not in wordfreq.keys():
            wordfreq[token] = 1
        else:
            wordfreq[token] += 1
            
#In the most frequency list we get the frequency of words in descending order
import heapq
most_freq = heapq.nlargest(200, wordfreq, key=wordfreq.get)

sentence_vectors = []
for sentence in corpus:
    sentence_tokens = nltk.word_tokenize(sentence)
    sent_vec = []
    for token in most_freq:
        if token in sentence_tokens:
            sent_vec.append(1)
        else:
            sent_vec.append(0)
    sentence_vectors.append(sent_vec)
    
#Sentiment analysis of the paragraph
#Polarity means emotions expressed in a text
#Subjectivty means expressing personal feelings, views, or beliefs
from textblob import TextBlob
blob1 = TextBlob(paragraph)
print(blob1.sentiment)

#Parts of Speech

nltk.pos_tag(most_freq)
# print(text_tok)
pos_tagged = nltk.pos_tag(most_freq)
 # print the list of tuples: (word,word_class)
print(pos_tagged)

nouns= []
for word,pos in pos_tagged:
    if pos in ['NN',"NNP"]:
        nouns.append(word)
freq_nouns=nltk.FreqDist(nouns)
print (freq_nouns)





    




