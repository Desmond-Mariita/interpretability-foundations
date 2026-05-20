import os
import re
from glob import glob
import pandas as pd
from PIL import Image
import numpy as np
from datasets import DatasetDict, Dataset
from tqdm import tqdm
from lime import lime_text
import torch
import clip
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer



import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")





### Read Data

root_folder = os.getcwd()

def create_dataset(mode='train'):
    # Get list of all filepaths
    files = glob(f'{root_folder}/MVSA/splits/{mode}*', recursive = True)
    files = [s.replace('\\','/') for s in files]
    # Create dataframe
    df = pd.concat([pd.read_json(f_name, lines=True) for f_name in files],
                   ignore_index=True)
    
    id_list = df['id'].tolist()
    label_list = df['label'].tolist()
    text_list = df['text'].tolist()
    image_paths = [f"{root_folder}/MVSA/images/{img}" for img in id_list]
    
    return id_list, label_list, text_list, image_paths

def label2id(labels):
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    return label_to_id

def id2label(labels):
    id_to_label = {idx: label for idx, label in enumerate(labels)}
    return id_to_label

def map_label2id(labels):
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    return label_to_id

class_labels = ['Positive', 'Neutral', 'Negative']

train_id_list, train_label_list, train_text_list,train_image_paths = create_dataset(mode='train')
val_id_list, val_label_list, val_text_list, val_image_paths = create_dataset(mode='val')
test_id_list, test_label_list, test_text_list, test_image_paths = create_dataset(mode='test')

id2label = id2label(class_labels)
label2id = label2id(class_labels)

label_to_id_mapping = map_label2id(class_labels)

# Map labels to numerical IDs

train_labels = [label_to_id_mapping[label] for label in train_label_list]
test_labels = [label_to_id_mapping[label] for label in test_label_list]
val_labels = [label_to_id_mapping[label] for label in val_label_list]

# Full Data

train_dict = {"text": train_text_list, "label": train_labels, "image_id": train_id_list}
test_dict = {"text": test_text_list, "label": test_labels, "image_id": test_id_list}
validation_dict = {"text": val_text_list, "label": val_labels, "image_id": val_id_list}

data = DatasetDict({"train": Dataset.from_dict(train_dict),
                    "test": Dataset.from_dict(test_dict),
                    "validation": Dataset.from_dict(validation_dict)})

# For Lime
train_dictionary = dict(zip(train_text_list,train_labels))
val_dictionary = dict(zip(val_text_list,val_labels))
test_dictionary = dict(zip(test_text_list,test_labels))



### Image Features

# 1. Get Caption for each image

max_length = 16
num_beams = 3
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

caption_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
caption_feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
caption_tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
caption_model.to(device)

def predict_step(image_path):
    i_image = Image.open(image_path)
    pixel_values = caption_feature_extractor(images=i_image,
                                             return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)
    output_ids = caption_model.generate(pixel_values, **gen_kwargs)

    preds = caption_tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    
    return preds[0]

# 2. Remove the stop words from the captions

def remove_stop_words(sentence):
    stop_words = set(stopwords.words('english'))
    word_tokens = nltk.word_tokenize(sentence)
    filtered_sentence = [word for word in word_tokens if word.lower() not in stop_words]
    
    return ' '.join(filtered_sentence)

# 3. Create ngrams for comparison

def create_ngrams(string, n):
    words = string.split()
    ngrams = []
    for i in range(1, n + 1):
        ngrams.extend([' '.join(words[j:j+i]) for j in range(len(words) - i + 1)])
    return ngrams

# 4. Get score from Clip

def get_max_value_and_index(arr):
    max_value = np.max(arr)
    max_index = np.argmax(arr)
    return max_value, max_index

model, preprocess = clip.load("ViT-B/32", device=device)

def get_score(img_path, ngrams):
    image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
    text = clip.tokenize(ngrams).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        score, class_index = get_max_value_and_index(probs)
        
    return [ngrams[class_index], class_index]

train_image_features = []

for i in tqdm(range(len(train_image_paths))):
    images = predict_step(train_image_paths[i])
    images = remove_stop_words(images)
    images = create_ngrams(images, len(re.split(" ", images)))
    images = get_score(train_image_paths[i], images)
    train_image_features.append(images)

def create_dataframe(data):    
    df = pd.DataFrame()
    for row in data:
        string = row[0]
        number = row[1]
        if string not in df.columns:
            df[string] = 0
        df.loc[len(df), string] = number
    df = df.fillna(0)
    return df

df = create_dataframe(train_image_features)
df.to_csv("train_image_features.csv", index=False)

### Text Features

# Helper Function

def get_top_scores(scores):
    "Returns top 5 features from our text"
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    top_scores = sorted_scores[:1]
    return top_scores

## vectorize to tf-idf vectors

tfidf_vc = TfidfVectorizer(min_df = 10,
                           max_features = 100000,
                           analyzer = "word",
                           ngram_range = (1, 2),
                           stop_words = 'english',
                           lowercase = True)

# Transform data

train_vc = tfidf_vc.fit_transform(train_text_list)
val_vc = tfidf_vc.transform(val_text_list)

# Model

model = LogisticRegression(C = 0.5, solver = "sag")
model = model.fit(train_vc, train_labels)
val_pred = model.predict(val_vc)

c = make_pipeline(tfidf_vc, model)

# Explain text feature importance with lime

def explain(text):
    explainer = lime_text.LimeTextExplainer(class_names = class_labels)
    exp = explainer.explain_instance(text,
                                     c.predict_proba,
                                     num_features = 10)
    scores = exp.as_list()   
    return get_top_scores(scores)

def detect_anomaly(text):
    # Define the patterns for detecting anomalies
    patterns = [r'^\?.*',
                r'^\- ?.*',
                r', ,.*']

    # Check if any of the patterns match the text
    for pattern in patterns:
        if re.search(pattern, text):
            return True
    return False

    
def replace_string_at_index(lst, index, new_string):
    if 0 <= index < len(lst):
        lst[index] = new_string

# Create list of text features

val_scores = []

bad_idx = []

new_string = "Text was not found. This is random filler text. Text was not found"

for i in tqdm(range(len(val_text_list))):
    try:
        scores = explain(val_text_list[i])
        val_scores.append(scores)
    except :
        bad_idx.append(i)
        replace_string_at_index(val_text_list,i,new_string)
        val_scores.append(explain(val_text_list[i]))
        pass


# Create Dataframe

def create_dataframe(data):
    df = pd.DataFrame()
    for row in data:
        for i in row:
            string = i[0]
            number = i[1]
            if string not in df.columns:
                df[string] = 0
            df.loc[len(df), string] = number
    df = df.fillna(0)
    return df

df = create_dataframe(val_scores)
df.to_csv("val_text_features.csv", index=False)


