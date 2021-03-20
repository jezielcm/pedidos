import nltk
import random
import statistics
import pandas as pd
from nltk import tokenize
from FeatsExtractor import FeatsExtractor
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB,MultinomialNB
from sklearn import tree,svm
from sklearn.metrics import balanced_accuracy_score
from imblearn.over_sampling import ADASYN


def get_corpus():
    ironico = pd.read_csv('Corpus/ironia.csv', encoding='utf-8', sep=';',error_bad_lines=False)
    ironico['irony'] = 1 #inserindo rótulo 1 para tweets irônicos
    
    nao_ironico = pd.read_csv('Corpus/nao-ironico.csv', encoding='utf-8', sep=';',error_bad_lines=False)
    nao_ironico['irony'] = 0 #inserindo rótulo 0 para tweets não irônicos
    
    corpus = pd.concat([ironico,nao_ironico])
    corpus = corpus.drop(columns=['username', 'date', 'retweets','favorites','geo','mentions','hashtags','id','permalink'])

    return corpus


if __name__ == '__main__':
    corpus = get_corpus()
    
    #recebe o array de features convertido para um array numpy e o array de targets
    fe = FeatsExtractor()
    featset, targetset = fe.extractCorpus(corpus, 'irony') 
    
    #faz o balanceamento
    X_resampled, y_resampled = ADASYN().fit_resample(featset, targetset) 

    f = open('resultados.txt', 'w')

    f.write(' ==> Corpus não balanceado <== \n')
    t = corpus.groupby('irony').count()
    f.write(t.to_string())
    
    f.write('\n \n ==> Corpus balanceado <== \n')
    f.write('Irônicos: {} \n'.format(y_resampled.count(1)))
    f.write('Não irônicos: {} \n'.format(y_resampled.count(0)))

    #monsta os splits de treino e teste
    df_train, df_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=(1.0 - 0.9), random_state=None) 
    
    #treina usando o Gaussian Naive bayes
    gnb = GaussianNB()
    y_pred = gnb.fit(df_train, y_train).predict(df_test)
    acc = balanced_accuracy_score(y_test, y_pred)
    print("Gaussian Naive bayes - Acurácia balanceada: {}".format(acc)) 
    f.write('\n \n ==> Modelo treinado com Gaussian Naive Bayes <== \n')
    f.write("Acurácia balanceada: {}".format(acc) )

    #treina usando o Multinomial Naive bayes
    mnb = MultinomialNB()
    y_pred = mnb.fit(df_train, y_train).predict(df_test)
    acc = balanced_accuracy_score(y_test, y_pred)
    print("Multinomial Naive bayes - Acurácia balanceada: {}".format(acc) )
    f.write('\n \n ==> Modelo treinado com Multinomial Naive Bayes <== \n')
    f.write("Acurácia balanceada: {}".format(acc) )

    #treina usando Decision tree
    dtc = tree.DecisionTreeClassifier()
    y_pred = dtc.fit(df_train, y_train).predict(df_test)
    acc = balanced_accuracy_score(y_test, y_pred)
    print("Decision tree - Acurácia balanceada: {}".format(acc) )
    f.write('\n \n ==> Modelo treinado com Decision tree <== \n')
    f.write("Acurácia balanceada: {}".format(acc) )

    #treina usando SVM
    svmc = svm.SVC()
    y_pred = svmc.fit(df_train, y_train).predict(df_test)
    acc = balanced_accuracy_score(y_test, y_pred)
    print("SVM- Acurácia balanceada: {}".format(acc) )
    f.write('\n \n ==> Modelo treinado com SVM <== \n')
    f.write("Acurácia balanceada: {}".format(acc) )

    f.close()


    

    
    
    

    