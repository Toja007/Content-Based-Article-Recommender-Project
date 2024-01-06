
# Importing Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Load data
data = pd.read_csv('medium_data.csv')
data.head()

# clean up columns
def cleanup (column):
    new_col = column.str.replace('[^a-zA-Z0-9]', ' ', regex=True)
    return new_col
data['title'] = cleanup(data['title'])
data['subtitle'] = cleanup(data['subtitle'])

data.info()

# descriptive statistics
data.describe(include='object').T.style.background_gradient(cmap='viridis')
 
# checking for null values
data.isnull().sum()

# filling nan values
data['subtitle'] = data['subtitle'].fillna(data['title'])

# drop unnecessary columns
data.drop(columns=['image', 'id'], inplace = True)

data[data['responses'] == 'Read']

# drop wrong enteries
data.drop(index = [3977, 6392], inplace=True)

def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""
    # Number of data points: n
    n = len(data)

    # x-data for the ECDF: x
    x = np.sort(data)

    # y-data for the ECDF: y
    y = np.arange(1, n+1) / n

    return x, y
# labels
sample = np.random.normal(data["reading_time"].mean(), data["reading_time"].std(), size=10000)

x, y = ecdf(data["reading_time"])
x_theor, y_theor = ecdf(sample)

plt.plot(x, y, marker=".", linestyle="none", label="Data Distribution")
plt.plot(x_theor, y_theor, color="red", label="Theoritical Normal Distribution")

# labels
plt.xlabel("Reading time")
plt.ylabel("ECDF")
plt.title("ECDF of reading_time")
plt.legend()

plt.show()


plt.hist(data["reading_time"], bins=100, stacked=True)
# labels
plt.title("Reading time Histogram")
plt.xlabel("Reading time")
plt.ylabel("Frequency")
plt.show()

sns.countplot(y="publication", data=data, order=data['publication'].value_counts().index)
plt.show()

sns.kdeplot(x="reading_time", hue="publication", data=data)
plt.xlim(0,25)
plt.show()

data["responses"] = data["responses"].astype(int)
data["responses"].dtype

sns.regplot(x="claps", y="responses", data=data)
plt.title("Linear Regression fit")
plt.show()

 
# joining the title and subtitle columns
data['soup'] = data['title'] + ' ' + data['subtitle']

# instatiating vectorizer
vectorizer = TfidfVectorizer(stop_words='english')

# fitting data['soup']
vectorizer_metrix = vectorizer.fit_transform(data['soup'])

# Anguler distances
cosin_sim = cosine_similarity(vectorizer_metrix, vectorizer_metrix)

# resetting index
data = data.reset_index()
indices = pd.Series(data.index, index=data['title'])

def recommend(title, cosin_sim=cosin_sim):
    
    idx = indices[title]
    
    sim_score = list(enumerate(cosin_sim[idx]))
    
    sim_score = sorted(sim_score, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar articles
    sim_score = sim_score[1:6]

    # Get the article indices
    #data_indices = [i[0] for i in sim_scores]
    for i in sim_score:
        print(data.iloc[i[0]].title)

recommend('A Beginner s Guide to Word Embedding with Gensim Word2Vec Model')

recommend('How to Automate Hyperparameter Optimization')

pickle.dump(data, open('articles.pkl', 'wb'))
pickle.dump(cosin_sim, open('simi.pkl', 'wb'))
pickle.dump(indices, open('indx.pkl', 'wb'))