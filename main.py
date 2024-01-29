import string
from datasets import load_dataset
import nltk
from nltk.corpus import stopwords

def remove_punctuation(dataset):
    remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
    clean_dataset = dataset.map(lambda example: {'text': example['text'].translate(remove_punctuation_map)})
    return clean_dataset

def whitespace_tokenize(dataset):
    tokenized_dataset = dataset.map(lambda example: {'text': example['text'].lower().split()})
    return tokenized_dataset
    # TODO maybe use bert tokenizer

def remove_stopwords(dataset):
    """
    Using nltk to remove stopwords
    """
    stop_words = stopwords.words('english')
    clean_dataset = dataset.map(lambda example: {'text': [word for word in example['text'] if not word in stop_words]})
    return clean_dataset

def lemmanize(dataset):
    """
    Using nltk to lemmatize
    """
    lemmatizer = nltk.stem.WordNetLemmatizer()
    clean_dataset = dataset.map(lambda example: {'text': [lemmatizer.lemmatize(word) for word in example['text']]})
    return clean_dataset

def stem(dataset):
    """
    Using nltk to stem
    """
    stemmer = nltk.stem.PorterStemmer()
    clean_dataset = dataset.map(lambda example: {'text': [stemmer.stem(word) for word in example['text']]})
    return clean_dataset

def remove_url(dataset):
    """
    Remove url
    """
    clean_dataset = dataset.map(lambda example: {'text': example['text'].replace(r"http\S+", "")})
    return clean_dataset

def remove_hashtag(dataset):
    """
    Remove hashtag
    """
    clean_dataset = dataset.map(lambda example: {'text': example['text'].replace(r"#\S+", "")})
    return clean_dataset

def remove_emoji(dataset):
    """
    Remove emoji
    """
    clean_dataset = dataset.map(lambda example: {'text': example['text'].encode('ascii', 'ignore').decode('ascii')})
    return clean_dataset

def remove_numbers(dataset):
    """
    Remove numbers
    """
    clean_dataset = dataset.map(lambda example: {'text': example['text'].replace(r"\d+", "")})
    return clean_dataset

def main():
    dataset = load_dataset("carblacac/twitter-sentiment-analysis")

    # TODO remove mentions
    clean_dataset = remove_url(dataset)
    clean_dataset = remove_hashtag(clean_dataset)
    clean_dataset = remove_emoji(clean_dataset)
    clean_dataset = remove_numbers(clean_dataset)
    clean_dataset = remove_punctuation(dataset)
    clean_dataset = whitespace_tokenize(clean_dataset)
    clean_dataset = remove_stopwords(clean_dataset)
    # clean_dataset = lemmanize(clean_dataset)
    clean_dataset = stem(clean_dataset)
    
    # print 10 examples side by side to compare
    for i in range(10):
        print(clean_dataset['train'][i]['text'])
        print(dataset['train'][i]['text'])
        

if __name__ == "__main__":
    nltk.download('stopwords')
    nltk.download('wordnet')
    main()
