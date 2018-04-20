from keras.datasets import imdb
from IPython import embed

#IMDB source: http://ai.stanford.edu/~amaas/data/sentiment/

def create_vocabularies():
    word_to_id = imdb.get_word_index()
    id_to_word = {_id: word for (word, _id) in word_to_id.items()}

    return word_to_id, id_to_word

def prepare_imdb(word_to_id, id_to_word):

    (x_train, y_train), (x_test, y_test) = imdb.load_data()
    postive = x_train[y_train]
    first = postive[0]
    first_words = [id_to_word[_id] for _id in first]
    embed()


    return x_train, y_train, x_test, y_test


if __name__ == "__main__":
    word_to_id, id_to_word = create_vocabularies()
    x_train, y_train, x_test, y_test = prepare_imdb(word_to_id, id_to_word)