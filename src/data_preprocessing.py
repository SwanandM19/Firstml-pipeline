# import os
# import logging
# import pandas as pd
# from sklearn.preprocessing import LabelEncoder
# from nltk.stem.porter import PorterStemmer
# from nltk.corpus import stopwords
# import string
# import nltk
# nltk.download('stopwords')
# nltk.download('punkt')


# #""the entire logger code is similar to the data_ingestion file check it out from their""
# # Ensure the "logs" directory exists
# log_dir = 'logs'
# os.makedirs(log_dir, exist_ok=True)

# # Setting up logger
# logger = logging.getLogger('data_preprocessing')
# logger.setLevel('DEBUG')

# console_handler = logging.StreamHandler()
# console_handler.setLevel('DEBUG')

# log_file_path = os.path.join(log_dir, 'data_preprocessing.log')
# file_handler = logging.FileHandler(log_file_path)
# file_handler.setLevel('DEBUG')

# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# console_handler.setFormatter(formatter)
# file_handler.setFormatter(formatter)

# logger.addHandler(console_handler)
# logger.addHandler(file_handler)

# #"this function is use to more pre process the data making it lower case ,removal of punctuation,tokenizing sentences,removing alphnumeric tokens ,stemming and then finally joining these tokens into single line"
# def transform_text(text):
#     """
#     Transforms the input text by converting it to lowercase, tokenizing, removing stopwords and punctuation, and stemming.
#     """
#     ps = PorterStemmer()
#     # Convert to lowercase
#     text = text.lower()
#     # Tokenize the text
#     text = nltk.word_tokenize(text)
#     # Remove non-alphanumeric tokens
#     text = [word for word in text if word.isalnum()]
#     # Remove stopwords and punctuation
#     text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]
#     # Stem the words
#     text = [ps.stem(word) for word in text]
#     # Join the tokens back into a single string
#     return " ".join(text)


# #"read the docstring its nothing more than that"
# def preprocess_df(df, text_column='text', target_column='target'):
#     """
#     Preprocesses the DataFrame by encoding the target column, removing duplicates, and transforming the text column.
#     """
#     try:
#         logger.debug('Starting preprocessing for DataFrame')
#         # Encode the target column
#         encoder = LabelEncoder()
#         df[target_column] = encoder.fit_transform(df[target_column])
#         logger.debug('Target column encoded')

#         # Remove duplicate rows
#         df = df.drop_duplicates(keep='first')
#         logger.debug('Duplicates removed')
        
#         # Apply text transformation to the specified text column
#         df.loc[:, text_column] = df[text_column].apply(transform_text)
#         logger.debug('Text column transformed')
#         return df
    
#     except KeyError as e:
#         logger.error('Column not found: %s', e)
#         raise
#     except Exception as e:
#         logger.error('Error during text normalization: %s', e)
#         raise

# def main(text_column='text', target_column='target'):
#     """
#     Main function to load raw data, preprocess it, and save the processed data.
#     """
#     try:
#         # Fetch the data from data/raw
#         train_data = pd.read_csv('./data/raw/train.csv')
#         test_data = pd.read_csv('./data/raw/test.csv')
#         logger.debug('Data loaded properly')

#         # Transform the data
#         train_processed_data = preprocess_df(train_data, text_column, target_column)
#         test_processed_data = preprocess_df(test_data, text_column, target_column)

#         # Store the data inside data/processed
#         data_path = os.path.join("./data", "interim")#"INSIDE THIS NEW DATA/INTERIAM FOLDER THEIR WILL BE MOW THE PRE PROCESSED DATA PRESENT CLEAN"
#         os.makedirs(data_path, exist_ok=True)
        
#         train_processed_data.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
#         test_processed_data.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)
        
#         logger.debug('Processed data saved to %s', data_path)
#     except FileNotFoundError as e:
#         logger.error('File not found: %s', e)
#     except pd.errors.EmptyDataError as e:
#         logger.error('No data: %s', e)
#     except Exception as e:
#         logger.error('Failed to complete the data transformation process: %s', e)
#         print(f"Error: {e}")

# if __name__ == '__main__':
#     main()


import os
import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import spacy
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import re

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Ensure the "logs" directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# Setting up logger
logger = logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'data_preprocessing.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def transform_text(text):
    """
    Transforms the input text by converting to lowercase, removing punctuation,
    removing stopwords, lemmatizing, and removing non-alphanumerics.
    """
    # Remove punctuation and make lowercase
    text = re.sub(r'[^\w\s]', '', text.lower())

    # Process using spaCy
    doc = nlp(text)

    # Lemmatize, remove stopwords and non-alpha words
    tokens = [token.lemma_ for token in doc if token.is_alpha and token.lemma_ not in ENGLISH_STOP_WORDS]

    return " ".join(tokens)

def preprocess_df(df, text_column='text', target_column='target'):
    """
    Preprocesses the DataFrame by encoding the target column, removing duplicates,
    and transforming the text column.
    """
    try:
        logger.debug('Starting preprocessing for DataFrame')

        # Encode the target column
        encoder = LabelEncoder()
        df[target_column] = encoder.fit_transform(df[target_column])
        logger.debug('Target column encoded')

        # Remove duplicate rows
        df = df.drop_duplicates(keep='first')
        logger.debug('Duplicates removed')

        # Apply text transformation to the specified text column
        df.loc[:, text_column] = df[text_column].apply(transform_text)
        logger.debug('Text column transformed')

        return df

    except KeyError as e:
        logger.error('Column not found: %s', e)
        raise
    except Exception as e:
        logger.error('Error during text normalization: %s', e)
        raise

def main(text_column='text', target_column='target'):
    """
    Main function to load raw data, preprocess it, and save the processed data.
    """
    try:
        # Load raw data
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        logger.debug('Data loaded properly')

        # Preprocess data
        train_processed_data = preprocess_df(train_data, text_column, target_column)
        test_processed_data = preprocess_df(test_data, text_column, target_column)

        # Save processed data
        data_path = os.path.join("./data", "interim")
        os.makedirs(data_path, exist_ok=True)

        train_processed_data.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
        test_processed_data.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)

        logger.debug('Processed data saved to %s', data_path)

    except FileNotFoundError as e:
        logger.error('File not found: %s', e)
    except pd.errors.EmptyDataError as e:
        logger.error('No data: %s', e)
    except Exception as e:
        logger.error('Failed to complete the data transformation process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
