from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

import os


# This is how the Naive Bayes classifier expects the input
# return a dictionary 
def create_word_features(words):
    useful_words = [word for word in words if word not in stopwords.words("english")]
    my_dict = dict([(word, True) for word in useful_words])
    return my_dict


ham_list = []
spam_list = []

flag1 = False
flag2 = False

count1 = 0
count2 = 0

for root, dirs, files in os.walk("."):
    if flag1 and flag2:
        break
    # print(root)
    path_lst = root.split(os.sep)
    # print(path_lst)
    if path_lst[-1] == "ham":   
        for filename in files:
            if count1 > 50:
                flag1 = True
                continue
            with open(os.path.join(root, filename), encoding="latin-1") as f:
                data = f.read()
                words = word_tokenize(data)
                words = create_word_features(words)
                print(words)
                ham_list.append((words,"ham"))
                count1+=1
    
    if path_lst[-1] == "spam":
        for filename in files:
            if count2 > 50:
                flag2 = True
                continue
            with open(os.path.join(root, filename), encoding="latin-1") as f:
                data = f.read()
                words = word_tokenize(data)
                words = create_word_features(words)
                print(words)
                spam_list.append((words,"spam"))
                count2 +=1

print("*"*100)
print(ham_list[0])    
print(spam_list[0])    
















