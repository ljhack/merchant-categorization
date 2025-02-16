'''
Functions to preprocess sms text and data before categorization
'''

from typing import Union
import regex as re
from datetime import datetime
import pandas as pd
import fasttext
from sklearn.preprocessing import StandardScaler

from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True)

import configs as CFG
import numpy as np

CURRENT_TIME = datetime.now().strftime("%Y-%m-%d_%H_%M")
FASTTEXT_EMBEDDING_MODEL = fasttext.load_model(CFG.FASTTEXT_MODEL_PATH)


def wordify(text) -> Union[None, str]:
    '''
    Returns only alphabetical part from text
    '''
    response = None
    pattern = r'[a-z]+'
    parts = re.findall(pattern, str(text), re.IGNORECASE)
    ## Keep only words with at least 2 letters
    if parts := [x.lower() for x in parts if len(x) > 1]:
        response = " ".join(parts)
    return response


# Function to get embeddings for a text
def get_embedding(text):
    '''
    Returns embeddings for a sentence
    '''
    return FASTTEXT_EMBEDDING_MODEL.get_sentence_vector(text)



def preprocess_data(**kwargs):
    '''
    Preprocesses data
    '''
    data = kwargs.get('data')
    text_col = kwargs.get('text_col', 'pos')

    print('Preprocessing data...')
    print(f'Number of rows before preprocessing: {len(data)}')

    ## Remove unnecessary rows
    data = data[
        (data[text_col].notna()) &      ## remove null values
        (data[text_col] != '-1') &      ## remove -1 values
        (data[text_col] != '')          ## remove empty strings
    ]
    print(f'Number of rows after removing unnecessary rows: {len(data)}')

    ## Clean text
    print('Cleaning text...')
    data[f'cleaned_{text_col}'] = data[text_col].parallel_apply(wordify)

    ## Deduplicate on cleaned text and remove unnecessary rows
    data = data[data[f'cleaned_{text_col}'].notna()]
    data = data.drop_duplicates(subset=f'cleaned_{text_col}')
    print(f'Number of rows after deduplication: {len(data)}')

    ## Create word count
    data['word_count'] = data[f'cleaned_{text_col}'].parallel_apply(lambda x: len(x.split()))
    print(f"Word count dstribution:-\n{data['word_count'].describe(percentiles=[i/10 for i in range(11)]) * 100}")

    ## Remove rows with word count less than 2
    data = data[data['word_count'] >= 2]
    print(f'Number of rows after removing rows with word count less than 2: {len(data)}')
    
    ## Create embeddings
    print('Creating embeddings...')
    data['embedding'] = data[f'cleaned_{text_col}'].parallel_apply(get_embedding)

    ## Save only cleaned text and embeddings in appropriate file
    print('Saving data...')
    cleaned_text = data[f'cleaned_{text_col}'].tolist()
    embeddings = np.array(data['embedding'].tolist())
    np.save(f'{CFG.REPORTS_PATH}cleaned_{text_col}_embeddings_{CURRENT_TIME}.npy', embeddings)
    np.save(f'{CFG.REPORTS_PATH}cleaned_{text_col}_{CURRENT_TIME}.npy', cleaned_text)
    print('Data saved successfully!')

    ## Scale embeddings
    print('Scaling embeddings...')
    scaler = StandardScaler()
    embeddings = scaler.fit_transform(embeddings)
    np.save(f'{CFG.REPORTS_PATH}cleaned_{text_col}_embeddings_scaled_{CURRENT_TIME}.npy', embeddings)
    print('Embeddings scaled successfully!')


    return embeddings