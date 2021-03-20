from nltk import word_tokenize
import hunspell
import spacy
import csv
from sklearn.feature_extraction import DictVectorizer

class FeatsExtractor:
    
    def __init__(self):
        self.spellchecker = hunspell.HunSpell('dicionarios/hunspell/pt_BR.dic','dicionarios/hunspell/pt_BR.aff') 
        self.spellchecker.add_dic('dicionarios/DELAF_PB_v2/Delaf2015v04.dic')
        self.entset = spacy.load("pt_core_news_lg")  
                
    def extractText(self, texto):
        dictfeats =  {}

        #tamanho do tweet
        tkens = [p for p in word_tokenize(texto, language='portuguese') if p.isalpha()]
        dictfeats['tamanho'] = len(tkens) 
        
        #Quantidade de entidades nomeadas  
        ents = []
        en = self.entset(texto)
        for ent in en.ents:
            ents.append(ent.text)
        dictfeats['quantNE'] = len(ents)  
        
        #quantidade de palavras erradas fora da lista de entidades nomeadas  
        totalerr = 0
        for word in tkens:
            if word.isalpha():
                if not self.spellchecker.spell(word.encode('utf-8')) and word not in ents:
                    totalerr += 1
        errpercent = 0
        if len(tkens) > 0:
            errpercent = (totalerr * 100)/len(tkens)
        dictfeats['misspellingpercent'] = errpercent
        dictfeats['misspelling'] = totalerr  
     
        return dictfeats
           

    def extractCorpus(self, corpus, y):
        featsets = []
        targetset = []
        i = 1
        for t in corpus.itertuples():
            #converte pandas para dicionário
            dct = t._asdict()
            tw = dict(dct)

            print('---- Tweet ',i,' ----\n')
            i += 1
            if tw['text'] != "":
                f = self.extractText(tw['text']) #obtem dicionário de features do tweet
                featsets.append(f) #adiciona no vetor de features
                targetset.append(tw[y]) #adiciona o target no vetor de targets
                
        vec = DictVectorizer()
        vecfeats = vec.fit_transform(featsets).toarray() #converte o vetor de features para um array numpy

        return vecfeats, targetset
    

    # def toCsv(self, nfile, dictfeats):
    #     with open(nfile,'w') as f:
    #         w = csv.DictWriter(f,dictfeats.keys())
    #         w.writerow(dictfeats)
    #     f.close()
