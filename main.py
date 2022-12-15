import os
import warnings
from ontology_dc8f06af066e4a7880a5938933236037.simple_text import SimpleText
from openfabric_pysdk.context import OpenfabricExecutionRay
from openfabric_pysdk.loader import ConfigClass
from time import time
import numpy as np
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
nltk.download('punkt')
nltk.download('wordnet')


############################################################
# Callback function called on update config
############################################################
def config(configuration: ConfigClass):
    # TODO Add code here
    pass


############################################################
# Callback function called on each execution pass
############################################################
def execute(request: SimpleText, ray: OpenfabricExecutionRay) -> SimpleText:
    output = []
    for text in request.text:
        # TODO Add code here
        text = text.lower()
        file =open('data/text_corpus.txt','r',errors = 'ignore')
        raw=file.read()
        raw = raw.lower()
        sent_tokens = nltk.sent_tokenize(raw) 
        word_tokens = nltk.word_tokenize(raw)
        lemmer = nltk.stem.WordNetLemmatizer()
        def LemTokens(tokens):
            return [lemmer.lemmatize(token) for token in tokens]
        remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
        def LemNormalize(text_input):
            return LemTokens(nltk.word_tokenize(text_input.lower().translate(remove_punct_dict)))
        def chat_response(user_response):
            robot_response=''
            sent_tokens.append(user_response)
            TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
            tfidf = TfidfVec.fit_transform(sent_tokens)
            vals = cosine_similarity(tfidf[-1], tfidf)
            idx=vals.argsort()[0][-2]
            flat = vals.flatten()
            flat.sort()
            req_tfidf = flat[-2]
            if(req_tfidf==0):
                robot_response=robot_response+"Sorry I am not sure."
                return robot_response
            else:
                robot_response = robot_response+sent_tokens[idx]
                return robot_response
        response = ''+chat_response(text)
        output.append(response)
    return SimpleText(dict(text=output))