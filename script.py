import os



ham_list = []
spam_list = []

for root, dirs, files in os.walk("."):
    # print(root)
    path_lst = root.split(os.sep)
    # print(path_lst)
    if path_lst[-1] == "ham":   
        for filename in files:
            with open(os.path.join(root, filename), encoding="latin-1") as f:
                data = f.read()
                ham_list.append(data)
    
    if path_lst[-1] == "spam":
        for filename in files:
            with open(os.path.join(root, filename), encoding="latin-1") as f:
                data = f.read()
                spam_list.append(data)


print(ham_list[0])    
print(spam_list[0])    

# This is how the Naive Bayes classifier expects the input
# return a dictionary 
def create_word_features(words):
    useful_words = [word for word in words if word not in stopwords.words("english")]
    my_dict = dict([(word, True) for word in useful_words])
    return my_dict














