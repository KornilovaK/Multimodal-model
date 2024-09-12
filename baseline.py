import pandas as pd
import numpy as np
import ast
import re

import torch
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import PCA


import joblib

pd.options.mode.chained_assignment = None

def load_data():
    resnet_path = './data/test/resnet_test.parquet'
    text_and_bert_path = './data/test/text_and_bert_test.parquet'
    test_path = './data/test/test.parquet'
    
    test = pd.read_parquet(test_path, engine='pyarrow')
    resnet = process_resnet(pd.read_parquet(resnet_path, engine='pyarrow'), test)
    text_and_bert = process_text(pd.read_parquet(text_and_bert_path, engine='pyarrow'), test)

    return resnet, text_and_bert, test

def clean_description(text, model, tokenizer):
    text = re.sub(r'<[^>]+>', '', text)
    emoji_pattern = re.compile("[" 
                               u"\U0001F600-\U0001F64F"  
                               u"\U0001F300-\U0001F5FF"  
                               u"\U0001F680-\U0001F6FF"  
                               u"\U0001F1E0-\U0001F1FF"  
                               u"\U00002702-\U000027B0"  
                               u"\U000024C2-\U0001F251" 
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    text = re.sub(r'[$#@]', '', text)
    text = re.sub(r'[^\w\s\-.]', '', text)
    
    return embed_bert_cls(text, model, tokenizer)

def embed_bert_cls(text, model, tokenizer):
    t = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**{k: v.to(model.device) for k, v in t.items()})
    embeddings = model_output.last_hidden_state[:, 0, :]
    embeddings = torch.nn.functional.normalize(embeddings)
    return embeddings[0].cpu().numpy()

def process_text(df, test):
    tokenizer = AutoTokenizer.from_pretrained("./rubert_tiny2/tokenizer")
    model = AutoModel.from_pretrained("./rubert_tiny2/model")

    df = df[(df.variantid.isin(test.variantid1) | df.variantid.isin(test.variantid2))].fillna('')
    df.description = df.description.apply(lambda x: clean_description(x, model, tokenizer))
    
    return df[['variantid', 'name_bert_64', 'description']]

def process_resnet(resnet, test):
    resnet = resnet[(resnet.variantid.isin(test.variantid1) | resnet.variantid.isin(test.variantid2))].fillna('')
    
    resnet['len_emb_not_main'] = resnet.pic_embeddings_resnet_v1.apply(lambda x: len(x))   
    resnet['pca_not_main'] = resnet.apply(lambda x: pca_transform(x.pic_embeddings_resnet_v1) if x.len_emb_not_main > 0 else np.zeros(128), axis=1)
    resnet.main_pic_embeddings_resnet_v1 = resnet.main_pic_embeddings_resnet_v1.apply(lambda x: x[0])
    resnet = resnet[['variantid', 'main_pic_embeddings_resnet_v1', 'pca_not_main']]

    return resnet

def pca_transform(row):
    n = row.shape[0]
    x = np.concatenate(row, axis=0).reshape(n, 128).T
    pca = PCA(n_components=1)
    x_transformed = pca.fit_transform(x)
    
    return np.concatenate(x_transformed)

def dot_product(emb1, emb2, length, n):
    dot = []
    for i in range(n):
        dot.append(np.dot(emb1[i], emb2[i]) / np.sqrt(length))
    return np.array(dot)

def merge_and_calculate(resnet, text_and_bert, test):
    temp1 = resnet.rename(columns={'variantid': 'variantid1', 'main_pic_embeddings_resnet_v1': 'main_image_embedding1', 'pca_not_main': 'not_main_image_embedding1'})
    temp1 = test.merge(temp1, how='inner', on='variantid1')
    temp2 = resnet.rename(columns={'variantid': 'variantid2', 'main_pic_embeddings_resnet_v1': 'main_image_embedding2', 'pca_not_main': 'not_main_image_embedding2'})
    df = temp1.merge(temp2, how='inner', on='variantid2')

    temp1 = text_and_bert.rename(columns={'variantid': 'variantid1', 'name_bert_64': 'name1', 'description': 'description1'})
    temp1 = df.merge(temp1, how='inner', on='variantid1')
    temp2 = text_and_bert.rename(columns={'variantid': 'variantid2', 'name_bert_64': 'name2', 'description': 'description2'})
    df = temp1.merge(temp2, how='inner', on='variantid2')

    return df

def distances(df):
    n = len(df)
    len_image = 128
    len_description = 312
    len_name = 64

    emb1 = np.concatenate(df.main_image_embedding1.to_numpy(), axis=0).reshape(n, len_image)
    emb2 = np.concatenate(df.main_image_embedding2.to_numpy(), axis=0).reshape(n, len_image)
    embedding1, embedding2 = torch.Tensor(emb1), torch.Tensor(emb2)
    df['main_image_cos_distance'] = F.cosine_similarity(embedding1, embedding2).numpy()
    df['main_image_eucl_distance'] = F.pairwise_distance(embedding1, embedding2).numpy()
    df['main_image_dot_distance'] = dot_product(emb1, emb2, len_image, n)

    emb1 = np.concatenate(df.not_main_image_embedding1.to_numpy(), axis=0).reshape(n, len_image)
    emb2 = np.concatenate(df.not_main_image_embedding2.to_numpy(), axis=0).reshape(n, len_image)
    embedding1, embedding2 = torch.Tensor(emb1), torch.Tensor(emb2)
    df['not_main_image_cos_distance'] = F.cosine_similarity(embedding1, embedding2).numpy()
    df['not_main_image_eucl_distance'] = F.pairwise_distance(embedding1, embedding2).numpy()
    df['not_main_image_dot_distance'] = dot_product(emb1, emb2, len_image, n)
    
    emb1 = np.concatenate(df.name1.to_numpy(), axis=0).reshape(n, len_name)
    emb2 = np.concatenate(df.name2.to_numpy(), axis=0).reshape(n, len_name)
    name1, name2 = torch.Tensor(emb1), torch.Tensor(emb2)
    df['name_cos_distance'] =  F.cosine_similarity(name1, name2).numpy()
    df['name_eucl_distance'] = F.pairwise_distance(name1, name2).numpy()
    df['name_dot_distance'] = dot_product(emb1, emb2, len_name, n)

    emb1 = np.concatenate(df.description1.to_numpy(), axis=0).reshape(n, len_description)
    emb2 = np.concatenate(df.description2.to_numpy(), axis=0).reshape(n, len_description)
    description1, description2 = torch.Tensor(emb1), torch.Tensor(emb2)
    df['description_cos_distance'] = F.cosine_similarity(description1, description2).numpy()
    df['description_eucl_distance'] = F.pairwise_distance(description1, description2).numpy()
    df['description_dot_distance'] = dot_product(emb1, emb2, len_description, n)
    
    columns = ['variantid1', 'variantid2', 'main_image_cos_distance', 'main_image_eucl_distance', 'main_image_dot_distance', \
                                                       'not_main_image_cos_distance', 'not_main_image_eucl_distance', 'not_main_image_dot_distance', \
                                                       'name_cos_distance', 'name_eucl_distance', 'name_dot_distance', \
                                                       'description_cos_distance', 'description_eucl_distance', 'description_dot_distance']
    return df[columns]

def make_dataset():
    resnet, text_and_bert, test = load_data()
    df = merge_and_calculate(resnet, text_and_bert, test)
    df = distances(df)
    return df