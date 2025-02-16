'''
Functions to separate merchant names from human names
'''

from collections import Counter


def segregate_merchant_names(**kwargs):
    '''
    Function to segregate merchant names from human names

    Algorithms:
        1. Check for cleaned text with only two words
        2. Divide into first word and second word
        3. Create frequencies of both first word and second words
        4. Rare human first names and second names will have very low frequence, hence remove them
        5. To check for common human names in high frequency first and second word, we can look at outliers in embeddings (merchant names would generally cluster together), and remove them

    Caution!!!:
        Outliers are highly dependent on input data, and many merchant names that are cutoff or wrongly tokenized can easily be removed
    '''
    data = kwargs.get('data')
    text_col = kwargs.get('text_col', 'pos')
    print(f'Number of rows before removing rows with more than 2 words: {len(data)}')

    ## Check for cleaned text with only two words
    data['cleaned_text_split'] = data[text_col].parallel_apply(lambda x: x.split())
    data['cleaned_text_split_len'] = data['cleaned_text_split'].parallel_apply(len)

    ## Only keep rows with 2 words in clean text
    data = data[data['cleaned_text_split_len'] == 2]
    print(f'Number of rows after removing rows with more than 2 words: {len(data)}')

    ## First and second word frequencies
    first_word_freq = Counter(data['cleaned_text_split'].parallel_apply(lambda x: x[0]).tolist())
    second_word_freq = Counter(data['cleaned_text_split'].parallel_apply(lambda x: x[1]).tolist())

    ## Visualize low first word and high first word frequencies
    print('First word frequency distribution:')

    ## Choose display threshold based on number of rows
    ## Visualize low second word and high second word frequencies
    print('Second word frequency distribution:')
    ## Remove rows with low first word frequency
    data = data[data['cleaned_text_split'].parallel_apply(lambda x: x[0]).parallel_apply(lambda x: first_word_freq[x]) > 10]
    



    ## Create wordcloud for first and second words
    first_word_freq = dict(first_word_freq)
    second_word_freq = dict(second_word_freq)




    return data