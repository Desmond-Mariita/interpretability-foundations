import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import transformers
import datasets
import evaluate
from collections import Counter

# Load Dataset

train_data = datasets.load_dataset("ag_news", split = "train")
test_data = datasets.load_dataset("ag_news", split = "test")

# Labeling

TEXT_LABELS = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tech"}

# Convert to dataframe

train = pd.DataFrame({'text':train_data['text'],'label':train_data['label']})
test = pd.DataFrame({'text':test_data['text'],'label':test_data['label']})

# Full Data

data = datasets.DatasetDict({"train":train_data,"test":test_data})

# Check token distribution

seq_len = [len(i.split()) for i in train['text']]

d = Counter(seq_len)
            
lists = sorted(d.items()) # sorted by key, return a list of tuples

x, y = zip(*lists) # unpack a list of pairs into two tuples

#plt.plot(x, y)
plt.bar(x, y, width=4,facecolor='g')
plt.xlabel("Token Count")
plt.ylabel("Number of Examples")
plt.title("Token Distribution")

plt.savefig('token_distribution.png')

# Config
id2label = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
label2id = {"World": 0, "Sports": 1, "Business": 2, "Sci/Tech": 3}

# Load Model

download = False

if download:
    pass

tokenizer = transformers.AutoTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True)
model = transformers.AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased',
                                                                        num_labels=4,
                                                                        id2label=id2label,
                                                                        label2id=label2id)

def tokenize_function(example):
    return tokenizer(example["text"], truncation=True)


tokenized_data = data.map(tokenize_function, batched=True)

data_collator = transformers.DataCollatorWithPadding(tokenizer=tokenizer)

# Training

def compute_metrics(eval_pred):
    metric1 = evaluate.load("precision")
    metric2 = evaluate.load("recall")
    metric3 = evaluate.load("accuracy")
    metric4 = evaluate.load("f1")
    
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    precision = metric1.compute(predictions=predictions,
                                references=labels,
                                average='macro')["precision"]
    recall = metric2.compute(predictions=predictions,
                             references=labels,
                             average='macro')["recall"]
    accuracy = metric3.compute(predictions=predictions,
                               references=labels)["accuracy"]
    f1 = metric4.compute(predictions=predictions,
                         references=labels,
                         average='macro')["f1"]

    return {"Accuracy": accuracy,"Precision": precision, "Recall": recall, "F1": f1,}

# Training Arguments

training_args = transformers.TrainingArguments(
    output_dir="finetuned_distilbert_model",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=10,
    fp16=True,
    push_to_hub=True,
    hub_token="<REDACTED>",  # 2023 file contained a real HF token here; revoked. Use the HF_TOKEN env var if re-running.
)

# Trainer

trainer = transformers.Trainer(
    model,
    training_args,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    
)

if __name__ == '__main__':
    
    print('Training Model !!! \n')
    
    trainer.train()
    
    trainer.push_to_hub()
    
    print("Done!!!")







    