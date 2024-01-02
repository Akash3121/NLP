from transformers import TFAutoModel
from transformers import AutoTokenizer
from datasets import load_dataset
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Dense
# import numpy as np    q
import numpy as np 
import pandas as pd
from math import sqrt

bert_model = TFAutoModel.from_pretrained("distilbert-base-uncased")
bert_model.trainable = False
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

#load dataset
# dataset = load_dataset("amazon_polarity")
dataset = load_dataset("yelp_polarity")
# shuffling and then splitting into train and test 
tr_dataset = dataset['train'][:1000]
te_dataset = dataset['test'][2000:3000]

tokenized_train = tokenizer(tr_dataset["text"][:1000] ,max_length=512, truncation=True, padding='max_length',return_tensors="tf")
tokenized_test = tokenizer(te_dataset["text"][:1000] ,max_length=512, truncation=True, padding='max_length',return_tensors="tf")

print(tokenized_train)
print(tokenized_test)

train_y = to_categorical(tr_dataset["label"])
test_y = to_categorical(te_dataset["label"])

maxlen = 512
token_ids =  Input(shape=(maxlen,), dtype=tf.int32,name="token_ids")
attention_masks = Input(shape=(maxlen,), dtype=tf.int32,name="attention_masks")
#new respresentations
bert_output = bert_model(token_ids,attention_mask=attention_masks)

#adding new layers
dense_layer = Dense(64,activation="relu")(bert_output[0][:,0])
#adding output softmax layer
output = Dense(2,activation="softmax")(dense_layer)

model = Model(inputs=[token_ids,attention_masks],outputs=output)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
print(tokenized_train["input_ids"])
print(tokenized_train["attention_mask"])

model.fit([tokenized_train["input_ids"],tokenized_train["attention_mask"]],train_y, batch_size=25, epochs=3)
print("mode summary:")
print(model.summary())
score =model.evaluate([tokenized_test["input_ids"],tokenized_test["attention_mask"]], test_y,verbose=0)
print("Accuracy on test data:", score[1])
a = model.predict([tokenized_test["input_ids"], tokenized_test["attention_mask"]])

_text = np.transpose(np.array(te_dataset["text"], ndmin=2))
_label = np.transpose(np.array(te_dataset["label"],ndmin=2))

predictions = np.concatenate((_text, _label, a), axis=1)
df = pd.DataFrame(predictions, columns=["text", "actual", "POS", "NEG"])
def determine_class_from_prob(row):
    if(row[2] > row[3]):
        # print(row[2], " : ", row[3])
        return 1
    else:
        return 0
df['predicted'] = df.apply(determine_class_from_prob, axis=1)
df['actual'] = df['actual'].astype('int64')
print(df.head(10))
df.query('actual != predicted')[:10]
print(df.query('actual == predicted')[:10])

sentences = [["The scientist conducted the experiment meticulously", "revealing groundbreaking results that reshaped our understanding of the subject matter"], 
         ["The chef prepared the dish with precision", "presenting a culinary masterpiece that delighted the taste buds of every diner"], 
         ["After weeks of preparation, the team executed the project flawlessly", "earning praise and recognition from their peers"],
         ["The photographer captured the moment perfectly", "freezing a fleeting instant in time that would become a cherished memory"],
         ["The author penned the final chapter with a sense of closure", "concluding the epic tale in a way that left readers satisfied and contemplative"]
        ]

tokenizers_output = [tokenizer(i, max_length=9, truncation=True, padding='max_length',return_tensors="tf") for i in sents]
print(tokenizers_output)

output1 = bert_model(tokenizers_output[0]["input_ids"],attention_mask=tokenizers_output[0]["attention_mask"])
output2 = bert_model(tokenizers_output[1]["input_ids"],attention_mask=tokenizers_output[1]["attention_mask"])
output3 = bert_model(tokenizers_output[2]["input_ids"],attention_mask=tokenizers_output[2]["attention_mask"])
output4 = bert_model(tokenizers_output[3]["input_ids"],attention_mask=tokenizers_output[3]["attention_mask"])
output5 = bert_model(tokenizers_output[4]["input_ids"],attention_mask=tokenizers_output[4]["attention_mask"])

print(output1)
print(output2)
print(output3)
print(output4)
print(output5)

def cosine_similarity(a, b):
    return np.dot(a,b)/(sqrt(np.dot(a,a))*sqrt(np.dot(b,b)) )

print("scientist vs groundbreaking: ", cosine_similarity(output1[0][0][3],output1[0][1][3]))

print("chef vs culinary: ", cosine_similarity(output2[0][0][3],output2[0][1][4]))

print("preparation vs recognition: ", cosine_similarity(output3[0][0][5],output3[0][1][5]))

print("photographer vs instant: ", cosine_similarity(output4[0][0][3],output3[0][1][5]))

print("author vs concluding: ", cosine_similarity(output5[0][0][3],output3[0][1][2]))
