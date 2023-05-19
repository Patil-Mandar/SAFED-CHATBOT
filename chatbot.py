import pickle
import numpy as np
from keras.models import load_model
# from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
label_encoder.fit_transform(['Documents',
 'Documents',
 'Documents',
 'Documents',
 'Documents',
 'Documents',
 'Documents',
 'Standards of exporting',
 'Standards of exporting',
 'Standards of exporting',
 'Standards of exporting',
 'government schemes',
 'government schemes',
 'government schemes',
 'government schemes',
 'government schemes',
 'government schemes',
 'government schemes'])
vocab_size = 1000 
embedding_dim = 32 
max_len = 20 
oov_token = "<OOV>"

tokenizer = Tokenizer(num_words = vocab_size , oov_token = oov_token , lower = True , filters='!"#$%&()*+,-./:;<=>@[\\]^_`{|}~\t\n' , split = ' ' )
tokenizer.fit_on_texts(list(['What are the documents required for the export of milk ?',
 'What paperwork is necessary for milk exports?',
 'What documentation is needed for milk export?',
 'Documentation required for milk export.',
 'Which paperwork is needed for milk exports?',
 'Documents',
 'Documents for export',
 'What are the standards of exporting milk?',
 'What criteria are there for milk exports?',
 'What standards apply to milk exports?',
 'Criteria for milk export',
 'What are the government schemes to support export of milk ?',
 "What are the government's programmes to encourage milk exports?",
 'What initiatives does the government have in place to promote milk exports?',
 'How does the government help in the export of milk ?',
 'What is the contribution of government in the export of milk?',
 'Does the government help in the export of milk?',
 ''])) 


def chatbot(inp):   
    with open("./intents2.json") as file:
        data = json.load(file)
    chat_model = load_model('./bestv2.h5')
    max_len = 10
    print(tokenizer.texts_to_sequences([inp]))
    result = chat_model.predict(pad_sequences(tokenizer.texts_to_sequences([inp]),
                                             truncating='post', maxlen=max_len))
    # print(result)
    # print(np.argmax(result))
    tag = label_encoder.inverse_transform([np.argmax(result)])

    for i in data['intents']:
        if i['tag'] == tag:
            a=np.random.choice(i['responses'])
            return a
        











# from tensorflow import keras
# import random
# import numpy as np
# import pickle
# import json

# path = './intents2.json'

# with open(path) as file : 
#   data = json.load(file)

# def chat():
#     # load trained model
#     chat_model = keras.models.load_model('./bestv2.h5')

#     # load tokenizer object
#     with open('./tokenizer.pickle', 'rb') as handle:
#         tokenizer = pickle.load(handle)

#     # load label encoder object
#     with open('./label_encoder.pickle', 'rb') as enc:
#         onehot_encoded = pickle.load(enc)

#     # parameters
#     max_len = 10
    
#     while True:
        
#         print('Enter')
#         inp = input()
#         if inp.lower() == "quit":
#             break

#         result = chat_model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([inp]),
#                                              truncating='post', maxlen=max_len))
#         tag = onehot_encoded.inverse_transform([np.argmax(result)])
        

#         for i in data['intents']:
#             if i['tag'] == tag:
#                 print("ChatBot:", np.random.choice(i['responses']))
# chat()