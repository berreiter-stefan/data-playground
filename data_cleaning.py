from typing import List, Set
import nltk
nltk.download('stopwords')


class TextProcessor:
    def __init__(self):
        self._w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
        self._lemmatizer = nltk.stem.WordNetLemmatizer()
        self._stop_words = set(nltk.corpus.stopwords.words('english'))
        self._unwanted_characters = ['!', '.', ';', '-', ':', ',']

    def clean_text(self, text: str) -> str:
        """ Makes text be adhere to a desired format. """
        text = text.lower()
        for val in self._unwanted_characters:
            text = text.replace(val, '')
        return text

    @staticmethod
    def unique_words(words: List[str]) -> List[str]:
        """ Returns unique words of a list in a list. """
        return list(set(words))

    @staticmethod
    def count_len_of_words(word_list: List[str]) -> int:
        """ Counts the number of words in a list of words. """
        return len([word for word in word_list])

    def relevant_tokens(self, tokens: List[str]):
        return [token for token in tokens if token not in self._stop_words]

    def add_stop_words(self, new_stop_words):
        self._stop_words.update(new_stop_words)

    def lemmatize_text(self, text: str) -> List[str]:
        """ Lemmatize words. """
        tokens = self._w_tokenizer.tokenize(text)
        return [self._lemmatizer.lemmatize(token, pos="v") for token in tokens]

    @property
    def stop_words(self) -> Set[str]:
        return self._stop_words

