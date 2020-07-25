# this library is used for selecting random numbers from a list
import random
#  this library is used for reading an entire article from the given URL.
from newspaper import Article
# this library is used for applying TFIDF vectorization on the document.
from sklearn.feature_extraction.text import TfidfVectorizer
#  this library is used for finding the cosine similarity between sentences
from sklearn.metrics.pairwise import cosine_similarity
#  this library is used to deal with human language data in python
import nltk


def main():
    """
    This is the main method. It downloads the article and apply the NLP processing
    on the same. It also performs tokenization on the article and starts the chatbot.
    :return: None
    """
    # Get the article URL
    data = Article(
        'https://github.com/d3/d3/blob/master/API.md')
    data.download()  # Download the article
    data.parse()  # Parse the article
    data.nlp()  # Apply Natural Language Processing (NLP)
    corpus = data.text  # Store the article text into corpus
    # print(corpus)
    # Tokenization
    text = corpus
    sent_tokens = nltk.sent_tokenize(text)  # txt to a list of sentences

    # Start the chat
    print(
        "D3 Helper:  I'm here to help you with d3.js APIs! If you want to exit, type Bye!")
    exit_list = ['exit', 'see you later', 'bye', 'quit', 'break']
    while True:
        user_input = input()
        if user_input.lower() in exit_list:
            print("D3 Helper: Chat with you later !")
            break
        else:
            if greeting_response(user_input) is not None:
                print("D3 Helper: " + greeting_response(user_input))
            else:
                print(" D3 Helper: " + generate_response(user_input, sent_tokens))


def greeting_response(text):
    """
    This method returns a random greeting response to users input.
    :param text: user's input
    :return: greeting response of a bot
    """
    text = text.lower()
    bot_greetings = ["howdy", "hi", "hey", "hello", "hey there"]
    user_greetings = ["hi", "hello", "hola", "greetings", "wassup", "hey"]

    for word in text.split():
        if word in user_greetings:
            return random.choice(bot_greetings)



def generate_response(user_input, sentence_list):
    """
    This method is to calculate the word counts from the article specified and
    calculate the similarity between the user's input and the document provided
    and generates a response to display.
    :param user_input: User's question
    :param sentence_list: tokenized sentences from the document
    :return: D3 Helper response
    """
    user_input = user_input.lower()
    sentence_list.append(user_input)
    bot_response = ''
    cm = TfidfVectorizer().fit_transform(sentence_list)
    similarity_scores = cosine_similarity(cm[-1], cm)
    flatten = similarity_scores.flatten()
    index = sorting_index(flatten)
    index = index[1:]
    response_flag = 0
    j = 0
    for i in range(0, len(index)):
        if flatten[index[i]] > 0.0:
            bot_response = bot_response + '\n' + sentence_list[index[i]]
            response_flag = 1
            j = j + 1
        if j == 2:
            sentence_list.remove(user_input)
            break
    # if no sentence contains a similarity score greater than 0.4 then
    # print "I'm afraid I'm not able to find what you need. Please try again!"
    if response_flag == 0:
        bot_response = bot_response + ' ' + "I'm afraid I'm not able to find what you need. Please try again!"
        sentence_list.remove(user_input)

    return bot_response


def sorting_index(list_var):
    """
    This method is used to sort the indices of the values to order them by values
    and returns the same.
    :param list_var: list of indices unsorted
    :return: sorted list of indices
    """
    length = len(list_var)
    list_index = list(range(0, length))
    x = list_var
    for i in range(length):
        for j in range(length):
            if x[list_index[i]] > x[list_index[j]]:
                temp = list_index[i]
                list_index[i] = list_index[j]
                list_index[j] = temp
    return list_index


if __name__ == '__main__':
    main()
