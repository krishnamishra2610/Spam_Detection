from flask import Flask,render_template,request
import numpy as np
import pickle
import nltk
nltk.download('stopwords')
import re 
import pandas as pd
from nltk.corpus import stopwords  
from nltk.stem import PorterStemmer  
ps = PorterStemmer() 

app = Flask(__name__)
model_cv =pickle.load(open('Bag_of_words_Spam_det','rb'))
model_spam = pickle.load(open('model_Spam_det','rb'))


df1 = pd.read_csv('corpus.csv')  

corpus1 =[]
for i in range(len(df1)):
    review = re.sub('a-zA-Z'," ",df1['0'][i])

    corpus1.append(review) 
    
@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/predict',methods=['POST',"GET"])
def predict():
    
    nt = request.form.get('new_text')

    new_text=nt

    review = re.sub('a-zA-Z'," ",new_text) 
    review = review.lower() 
    review = review.split() 

    review = [ps.stem(word) for word in review if word not in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus1.append(review)
    
    X_new = model_cv.fit_transform(corpus1).toarray() 
    spam = model_spam.predict([X_new[-1]])[0] 
    if(spam):
        return render_template('index.html', pred="Spam Message")
    else:
         return render_template('index.html', pred1="Not A Spam Message ") 

   
    
if __name__ == '__main__':
    app.run(debug=True)