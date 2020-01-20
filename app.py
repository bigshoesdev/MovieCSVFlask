from flask import Flask
from flask import render_template
from flask import request
from collections import Counter
from math import log10, log
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from numpy import dot
from numpy.linalg import norm
import csv

import nltk
nltk.download('stopwords')

class Engine:
    def __init__(self):
        self.movie_data = []

        with open('static/movie.csv', 'r', encoding='utf-8') as csvfile:
            moviereader = csv.reader(csvfile)
            index = 0
            for movie_line in moviereader:
                if index == 500:
                    break
                try:
                    if (movie_line[20] == 'title'):
                        pass
                    else:
                        self.movie_data.append({'overview': movie_line[9], 'id': movie_line[5], 'title': movie_line[20],
                                          'original_title': movie_line[8], 'poster_path': movie_line[11]})
                except:
                    pass
                index += 1

        self.stop_words = stopwords.words('english')
        self.unique_words = set()
        self.word_document_frequency = {}
        self.word_frequency_documents = {}
        self.document_lengths = {}
        length = len(self.movie_data)

        for index in range(length):
            row = self.movie_data[index]

            document = row['overview']
            title = row['title']
            movie_id = row['id']
            tokens = self.stem_tokenize(str(document))
            self.unique_words = self.unique_words.union(set(tokens))
            for term in set(tokens):
                if term not in self.word_frequency_documents:
                    self.word_frequency_documents[term] = {}
                self.word_frequency_documents[term][movie_id] = tokens.count(term)

            for term in self.word_frequency_documents:
                self.word_document_frequency[term] = len(self.word_frequency_documents[term])
            self.document_lengths[movie_id] = len(tokens)

    def stem_tokenize(self, document):
        stemmer = PorterStemmer()
        tokenizer = RegexpTokenizer(r'[a-zA-Z]+')

        terms = tokenizer.tokenize(document.lower())
        filtered = [stemmer.stem(word) for word in terms if not word in self.stop_words]
        return filtered

    def make_tf_idf_table(self, tf_scores, idf_scores, movie_id, query_words):
        table = '<table class="table table-striped"><tr>'
        table += '<th class="col_token">Token</th><th class="col_tf">TF</th><th class="col_idf">IDF</th><th class="col_tf_idf">TF * IDF</th>'
        table += '</tr>'
        final_score = 0.0
        for token in query_words:
            tf_score = tf_scores[movie_id][token] if token in tf_scores[movie_id] else 0.0
            idf_score = idf_scores[token] if token in idf_scores else 0.0
            final_score = final_score + (tf_score * idf_score)
            table += '<tr><td>' + token + '</td><td>' + str(tf_score) + '</td><td>' + str(
                idf_score) + '</td><td>' + str(tf_score * idf_score) + '</td></tr>'
        table += '</table>'
        return table

    def search_movie(self, query_string):
        query_words = self.stem_tokenize(query_string)

        tf_scores = {}
        for word in query_words:
            for movie_id in self.word_frequency_documents[word]:
                if movie_id not in tf_scores:
                    tf_scores[movie_id] = {}
                tf_scores[movie_id][word] = self.word_frequency_documents[word][movie_id] / self.document_lengths[
                    movie_id]

        idf_scores = {}
        for word in set(query_words):
            if word in self.word_document_frequency:
                idf_scores[word] = log10(len(self.document_lengths) / self.word_document_frequency[word])
            else:
                idf_scores[word] = 0

        query_length = len(query_words)
        query_vector = []
        for term in query_words:
            term_count = query_words.count(term)
            term_F = term_count / query_length
            query_idf = 0
            if term in query_words:
                query_idf = log10(query_length / term_count)

            tf_idf = term_F * query_idf
            query_vector.append(tf_idf)

        document_similarity = Counter()

        document_vectors = {}
        for movie_id in tf_scores:
            document_vectors[movie_id] = []
            for query_term in query_words:
                if query_term in self.unique_words:
                    if movie_id in self.word_frequency_documents[query_term]:
                        tf = self.word_frequency_documents[query_term][movie_id] / self.document_lengths[movie_id]
                        idf = 0
                        if query_term in self.word_document_frequency:
                            idf = log10(len(self.document_lengths) / self.word_document_frequency[query_term])
                        else:
                            idf = 0
                        tf_idf = tf * idf
                        document_vectors[movie_id].append(tf_idf)
                    else:
                        document_vectors[movie_id].append(0)
                else:
                    document_vectors[movie_id].append(0)

            document_similarity += {movie_id: dot(query_vector, document_vectors[movie_id]) / (
                    norm(query_vector) * norm(document_vectors[movie_id]))}

        views = []
        for item in document_similarity.most_common(5):
            row = [d for d in self.movie_data if (d['id'] == item[0])][0]

            view = {
                "title_eng": row['title'],
                "title_orig": row['original_title'],
                "overview": row['overview'],
                "poster": "https://image.tmdb.org/t/p/w300_and_h450_bestv2" + row['poster_path'],
                "score_table": self.make_tf_idf_table(tf_scores, idf_scores, item[0], query_words),
            }

            views.append(view)

        return views, query_words


engine = Engine()
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/search', methods=['POST'])
def movie_search():
    query_string = request.form['query_string']
    results, query_words = engine.search_movie(query_string)
    return render_template('index.html', query_string=query_string, query_words=query_words, res=results)


if __name__ == '__main__':
    app.run()
