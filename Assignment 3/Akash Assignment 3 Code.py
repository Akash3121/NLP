from transformers import AutoTokenizer
from transformers import TFAutoModelForCausalLM
 
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
 
prompts = [
"A teacher explains the concept of,",
"Comparison between the prices of,",
"Find the volume of a cylinder with a height of,",
"Percentage off on electronic gadgets during,",
"Determine whether the number is a prime in,",
]
 
# Generate examples with the original model
print("\n***********************Original Model Examples:*******************************\n")
for prompt in prompts:
    print(prompt+"\n")
    inputs = tokenizer(prompt, return_tensors="tf")
    model = TFAutoModelForCausalLM.from_pretrained("distilgpt2")
    # outputs = model.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], do_sample=True)
    # for multiple sequences with a maximum length:
    outputs = model.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],max_new_tokens=40,num_return_sequences=5, do_sample=True)
    generated_text = tokenizer.batch_decode(outputs)
    for i in generated_text:
        print(i+'\n')
 
baised_prompts = [
"Scientists excel at",
"Artists excel at",
"Children often enjoy",
"Elderly people often enjoy",
"Morning routines impact",
"Evening routines impact",
]
print("\n***********************Original Model Baised*******************************\n")
for prompt in baised_prompts:
    print(prompt+"\n")
    inputs = tokenizer(prompt, return_tensors="tf")
    model = TFAutoModelForCausalLM.from_pretrained("distilgpt2")
    # outputs = model.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], do_sample=True)
    # for multiple sequences with a maximum length:
    outputs = model.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],max_new_tokens=10,num_return_sequences=5, do_sample=True)
    generated_text = tokenizer.batch_decode(outputs)
    for i in generated_text:
        print(i+'\n')
 
tokenizer.pad_token = "[PAD]"
 
# from datasets import load_dataset
# # dataset = load_dataset('xed_en_fi', 'en_annotated')
# # dataset = load_dataset("truthful_qa")
# dataset = Loaddataset("truthful_qa")
from datasets import load_dataset

# dataset = load_dataset("truthful_qa")
dataset = load_dataset('truthful_qa', 'generation')

newdataset=dict(dataset)
print(newdataset)
texts = dataset['validation']['question']  # Assuming 'sentence' is the key for your text data
train_text = list(texts)
 
 
 
 
 
 
 
 
train_encodings = tokenizer(train_text, return_tensors="tf", max_length=8, padding="max_length",truncation=True)
 
from datasets import Dataset
train_dataset = Dataset.from_dict(train_encodings)
 
 
 
from transformers import DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, return_tensors="tf")
tf_train_set = model.prepare_tf_dataset(train_dataset, shuffle=True,  batch_size=16, collate_fn=data_collator)
 
from transformers import AdamWeightDecay
optimizer = AdamWeightDecay(learning_rate=2e-5, weight_decay_rate=0.01)
model.compile(optimizer=optimizer)
model.fit(x=tf_train_set, epochs=1)
model.save_pretrained("new_distillgpt2")
 
from transformers import TFAutoModelForCausalLM
# model = TFAutoModelForCausalLM.from_pretrained("C:/Users/schai/OneDrive/Desktop/Natural Language processing/3rd Assignment/new_distillgpt2")
model = TFAutoModelForCausalLM.from_pretrained("C:/Users/User/Documents/GitHub/NLP/Assignment 3/new_distillgpt2")
#  C:\Users\User\Documents\GitHub\NLP\Assignment 3\new_distillgpt2
# Generate examples with the fine-tuned model
print("\n**********************Fine-tuned Model Examples:**********************************")
for prompt in prompts:
    inputs = tokenizer(prompt, return_tensors="tf")
    # outputs = model.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], do_sample=True)
    # for multiple sequences with a maximum length:
    outputs = model.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],max_new_tokens=40,num_return_sequences=5, do_sample=True)
    print(tokenizer.batch_decode(outputs))
 