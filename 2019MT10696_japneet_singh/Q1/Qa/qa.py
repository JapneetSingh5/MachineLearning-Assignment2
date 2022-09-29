import sys
import os
import math
from collections import defaultdict
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

def def_value():
    return 0

def process_command():
    args_length = len(sys.argv);
    if(args_length<3):
        print("Insufficient arguments provided, exiting")
        sys.exit(1)
    path_train_data = sys.argv[1]
    path_test_data = sys.argv[2]
    return path_train_data, path_test_data

def build_phi(neg_path, pos_path):
    neg_count = 0
    pos_count = 0
    total_count = 0
    for _ in os.listdir(neg_path):
        neg_count+=1
        total_count+=1
    for _ in os.listdir(pos_path):
        pos_count+=1
        total_count+=1
    return (pos_count/total_count), (neg_count/total_count), pos_count, neg_count

def main():
    path_train_data ,path_test_data = process_command()
    # print(path_train_data, path_test_data)
    neg_suffix= "/neg"
    pos_suffix= "/pos"
    train_neg_folder = path_train_data + neg_suffix
    train_pos_folder = path_test_data + pos_suffix
    test_neg_folder = path_test_data + neg_suffix
    test_pos_folder = path_test_data + pos_suffix
    pos_vocabulary_dict = defaultdict(def_value)
    neg_vocabulary_dict = defaultdict(def_value)
    ps = PorterStemmer()
    stopwords_set = set(stopwords.words('english'))
    # preprocessing
    # 1. lowercase
    # 2. tokenize
    # 3. stemming
    # -> can remove all special characters 
    for txt_file_name in os.listdir(train_neg_folder):
        txt_file_path = os.path.join(train_neg_folder, txt_file_name)
        review = ""
        with open(txt_file_path, 'r') as f:
            review = f.read()
        tokenized_review = word_tokenize(review.lower())
        for word in set(tokenized_review):
            if word in stopwords_set:
                continue
            stemmed_word = ps.stem(word)
            pos_vocabulary_dict[stemmed_word]+=1;
    for txt_file_name in os.listdir(train_pos_folder):
        txt_file_path = os.path.join(train_pos_folder, txt_file_name)
        review = ""
        with open(txt_file_path, 'r') as f:
            review = f.read()
        tokenized_review = word_tokenize(review.lower())
        for word in set(tokenized_review):
            if word in stopwords_set:
                continue
            stemmed_word = ps.stem(word)
            neg_vocabulary_dict[stemmed_word]+=1;            
    # print(vocabulary_dict)
    # print('years', vocabulary_dict[vocabulary[0]])
    phi_pos, phi_neg, pos_count, neg_count = build_phi(train_neg_folder, train_pos_folder)
    # print(phi_pos, phi_neg, pos_count, neg_count)
    plt.figure(1)
    positive_wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate_from_frequencies(pos_vocabulary_dict)
    plt.figure(figsize=(15,8))
    plt.imshow(positive_wordcloud)
    plt.savefig("positive_wordcloud.png")
    plt.figure(2)
    negative_worldcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate_from_frequencies(neg_vocabulary_dict)
    plt.figure(figsize=(15,8))
    plt.imshow(negative_worldcloud)
    plt.savefig("negative_wordcloud.png")
    # we have phi_pos, phi_neg, count of positive and negative reviews in the training dataset
    # at test time, calculate p(y=1/x) = p(y=1)*product(i=1, n){p(xi/y=1)}/p(x)
    # p(xi/y=1) will be the number of positive documents in which the word xi appears / by the number of total positive documents 
    correct_class = 0
    incorrect_class = 0
    total_class = 0
    for txt_file_name in os.listdir(test_neg_folder):
        prob_pos = math.log(phi_pos)
        prob_neg = math.log(phi_neg)
        txt_file_path = os.path.join(test_neg_folder, txt_file_name)
        review = ""
        with open(txt_file_path, 'r') as f:
            review = f.read()
        tokenized_review = word_tokenize(review.lower())
        for word in set(tokenized_review):
            if word in stopwords_set:
                continue
            stemmed_word = ps.stem(word)
            prob_pos+=math.log((pos_vocabulary_dict[stemmed_word] + 1)/(pos_count +2));
            prob_neg+=math.log((neg_vocabulary_dict[stemmed_word] + 1)/(neg_count +2));
        print(prob_pos, prob_neg)
        if prob_pos > prob_neg :
            # print("Was Neg, Classified Pos")
            incorrect_class+=1
        else:
            # print("Was Neg, Classified Neg")
            correct_class+=1
        total_class+=1
        # pos_vocabulary_dict[stemmed_word]+=1;
    for txt_file_name in os.listdir(test_pos_folder):
        prob_pos = math.log(phi_pos)
        prob_neg = math.log(phi_neg)
        txt_file_path = os.path.join(test_pos_folder, txt_file_name)
        review = ""
        with open(txt_file_path, 'r') as f:
            review = f.read()
        tokenized_review = word_tokenize(review.lower())
        for word in set(tokenized_review):
            if word in stopwords_set:
                continue
            stemmed_word = ps.stem(word)
            prob_pos+=math.log((pos_vocabulary_dict[stemmed_word] + 1)/(pos_count +2));
            prob_neg+=math.log((neg_vocabulary_dict[stemmed_word] + 1)/(neg_count +2)); 
        print(prob_pos, prob_neg)         
        if prob_pos > prob_neg :
            # print("Was Pos, Classified Pos")
            correct_class+=1
        else:
            incorrect_class+=1
            # print("Was Pos, Classified Neg")
        total_class+=1
    print("Accuracy: ", (correct_class/total_class))  








if __name__ == "__main__":
    main()