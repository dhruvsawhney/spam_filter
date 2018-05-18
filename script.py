import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

import os
import random

# code to remove import issues with nltk libraries
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


# This is how the Naive Bayes classifier expects the input
# return a dictionary 
def create_word_features(words):
    useful_words = [word for word in words if word not in stopwords.words("english")]
    my_dict = dict([(word, True) for word in useful_words])
    return my_dict



# UNCOMMENT CODE in for-loop to perform the task below
# Method to short-circuit heavy data-training
# hard-coded numbers for "break" can be modified

# collect smaller-subset of data
flag1 = False
flag2 = False
count1 = 0
count2 = 0

# hold the data for classifier in required format of tuple where first element is a dictionary and second element is "ham"/"spam"
ham_list = []
spam_list = []

for root, dirs, files in os.walk("."):
    # if flag1 and flag2:
    #     break
    path_lst = root.split(os.sep)
    if path_lst[-1] == "ham":   
        for filename in files:
            # if count1 > 50:
            #     flag1 = True
            #     continue
            with open(os.path.join(root, filename), encoding="latin-1") as f:
                data = f.read()
                words = word_tokenize(data)
                words = create_word_features(words)
                ham_list.append((words,"ham"))
                # count1+=1
    
    if path_lst[-1] == "spam":
        for filename in files:
            # if count2 > 100:
            #     flag2 = True
            #     continue
            with open(os.path.join(root, filename), encoding="latin-1") as f:
                data = f.read()
                words = word_tokenize(data)
                words = create_word_features(words)
                spam_list.append((words,"spam"))
                # count2 +=1


# split the data-set with 70/30 split
combined_list = ham_list + spam_list 
random.shuffle(combined_list)

split = int(len(combined_list)*(0.7))

train_set = combined_list[:split]
test_set = combined_list[split:]

# create the classifier
classifier = NaiveBayesClassifier.train(train_set)
accuracy = nltk.classify.util.accuracy(classifier, test_set)
print("The accuracy of model is: " + str(accuracy * 100))


# Test the classifier

msg1 = '''Hello th̓ere seُx master :-)
i need c0ck ri͏ght noِw ..͏. don't tell my hǔbbٚy.ٚ. ))
My sc͕rٞeٚe̻nname is Dorry.
My accֺo֔unt is h֯ere: http:nxusxbnd.GirlsBadoo.ru
C u late٘r!'''
 
 
msg2 = '''As one of our top customers we are providing 10% OFF the total of your next used book purchase from www.letthestoriesliveon.com. Please use the promotional code, TOPTENOFF at checkout. Limited to 1 use per customer. All books have free shipping within the contiguous 48 United States and there is no minimum purchase.
 
We have millions of used books in stock that are up to 90% off MRSP and add tens of thousands of new items every day. Don’t forget to check back frequently for new arrivals.'''
 
 
 
msg3 = '''To start off, I have a 6 new videos + transcripts in the members section. In it, we analyse the Enron email dataset, half a million files, spread over 2.5GB. It's about 1.5 hours of  video.
 
I have also created a Conda environment for running the code (both free and member lessons). This is to ensure everyone is running the same version of libraries, preventing the Works on my machine problems. If you get a second, do you mind trying it here?'''



def process_text(text):

    words = word_tokenize(text)
    words = create_word_features(words)
    alg_result = classifier.classify(words)
    return alg_result

# Uncomment to test data

# print(process_text(msg1))
# print(process_text(msg2))
# print(process_text(msg3))


