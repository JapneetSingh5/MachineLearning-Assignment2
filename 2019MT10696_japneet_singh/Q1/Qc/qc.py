import sys
import os
import math
from collections import defaultdict
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import seaborn as sb

# for last obtained train and test set accuracies, see a.txt 

def build_confusion_matrix(actual_p_predicted_p, actual_n_predicted_p, actual_p_predicted_n, actual_n_predicted_n):
    fig = plt.figure(figsize=(8, 6))
    print(actual_p_predicted_p, actual_n_predicted_p, actual_p_predicted_n, actual_n_predicted_n)
    matrix = [[actual_p_predicted_p, actual_n_predicted_p], [actual_p_predicted_n, actual_n_predicted_n]]
    plotit = sb.heatmap(matrix, annot=True, cmap="Greens", fmt='g')
    ax = fig.gca()
    ax.xaxis.tick_top()
    ax.set_xlabel("Actual Class")
    ax.set_ylabel("Predicted Class")
    plt.title("Confusion Matrix (for Test Data Only)")
    ax.xaxis.set_label_position('top')
    plt.savefig("confusion_matrix.png")
    plt.show()

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
    train_pos_folder = path_train_data + pos_suffix
    test_neg_folder = path_test_data + neg_suffix
    test_pos_folder = path_test_data + pos_suffix
    pos_vocabulary_dict = defaultdict(def_value)
    neg_vocabulary_dict = defaultdict(def_value)
    # preprocessing
    # 1. lowercase
    # 2. tokenize
    # 3. remove non-alphanumeric/special characters
    review = ""
    # TRAIN - NEGATIVE REVIEWS
    for txt_file_name in os.listdir(train_neg_folder):
        txt_file_path = os.path.join(train_neg_folder, txt_file_name)
        # Read review from txt file
        with open(txt_file_path, 'r') as f:
            review = f.read()
        # tokenize thereview using nltk's tokenizer
        pre_tokenized_review = word_tokenize(review.lower())
        tokenized_review = []
        # remove non-alphanumeric/special characters from the tokenised array 
        for word in pre_tokenized_review:
            if word.isalnum():
                tokenized_review.append(word)
        # add all the words thus obtained to the vocabulary
        # as we are processing negative reviews right now, we use the negative vocabulary dict
        for word in set(tokenized_review):
            neg_vocabulary_dict[word]+=1;
    # TRAIN - POSITIVE REVIEWS
    for txt_file_name in os.listdir(train_pos_folder):
        txt_file_path = os.path.join(train_pos_folder, txt_file_name)
        # read reviews from txt file
        with open(txt_file_path, 'r') as f:
            review = f.read()
        # tokenize the words using nltk's tokenizer
        pre_tokenized_review = word_tokenize(review.lower())
        tokenized_review = []
        # remove non-alphanumeric/special characters from the tokenised array
        for word in pre_tokenized_review:
            if word.isalnum():
                tokenized_review.append(word)
        # add all the words thus obtained to the vocabulary
        # as we are processing negative reviews right now, we use the positive vocabulary dict
        for word in set(tokenized_review):
            pos_vocabulary_dict[word]+=1;            
    phi_pos, phi_neg, pos_count, neg_count = build_phi(train_neg_folder, train_pos_folder)
    # build word cloud for positive review vocabulary
    plt.figure(1)
    positive_wordcloud = WordCloud(max_font_size=50, max_words=100).generate_from_frequencies(pos_vocabulary_dict)
    plt.figure(figsize=(15,8))
    plt.axis('off')
    plt.imshow(positive_wordcloud, interpolation='bilinear')
    plt.savefig("positive_wordcloud.png")
    # build word cloud for negative review vocabulary
    plt.figure(2)
    negative_worldcloud = WordCloud(max_font_size=50, max_words=100).generate_from_frequencies(neg_vocabulary_dict)
    plt.figure(figsize=(15,8))
    plt.axis('off')
    plt.imshow(negative_worldcloud, interpolation='bilinear')
    plt.savefig("negative_wordcloud.png")

    # we have phi_pos, phi_neg, count of positive and negative reviews in the training dataset
    # at test time, calculate p(y=1/x) = p(y=1)*product(i=1, n){p(xi/y=1)}/p(x)
    # p(xi/y=1) will be the number of positive documents in which the word xi appears / by the number of total positive documents 

    train_correct_class = 0
    train_total_class = 0

    # TRAIN - NEGATIVE REVIEWS
    for txt_file_name in os.listdir(train_neg_folder):
        prob_pos = math.log(phi_pos)
        prob_neg = math.log(phi_neg)
        txt_file_path = os.path.join(train_neg_folder, txt_file_name)
        with open(txt_file_path, 'r') as f:
            review = f.read()
        pre_tokenized_review = word_tokenize(review.lower())
        tokenized_review = []
        for word in pre_tokenized_review:
            if word.isalnum():
                tokenized_review.append(word)
        for word in set(tokenized_review):
            prob_pos+=math.log((pos_vocabulary_dict[word] + 1)/(pos_count + 2));
            prob_neg+=math.log((neg_vocabulary_dict[word] + 1)/(neg_count + 2));
        if prob_neg > prob_pos :
            train_correct_class+=1
            print("Was Neg, Classified Neg")
        else :
            print("Was Neg, Classified Pos")
        train_total_class+=1
    # TRAIN - POSITIVE REVIEWS
    for txt_file_name in os.listdir(train_pos_folder):
        prob_pos = math.log(phi_pos)
        prob_neg = math.log(phi_neg)
        txt_file_path = os.path.join(train_pos_folder, txt_file_name)
        with open(txt_file_path, 'r') as f:
            review = f.read()
        pre_tokenized_review = word_tokenize(review.lower())
        tokenized_review = []
        for word in pre_tokenized_review:
            if word.isalnum():
                tokenized_review.append(word)
        for word in set(tokenized_review):
            prob_pos+=math.log((pos_vocabulary_dict[word] + 1)/(pos_count + 2));
            prob_neg+=math.log((neg_vocabulary_dict[word] + 1)/(neg_count + 2)); 
        # print(prob_pos, prob_neg)         
        if prob_pos > prob_neg :
            train_correct_class+=1
            print("Was Pos, Classified Pos")
        else :
            print("Was Pos, Classified Neg")
        train_total_class+=1
    print("Train Set Accuracy: ", (train_correct_class/train_total_class)*100) 

    correct_class = 0
    total_class = 0
    actual_p_predicted_p = 0
    actual_n_predicted_p = 0
    actual_p_predicted_n = 0
    actual_n_predicted_n = 0
    # TEST SET - NEGATIVE REVIEWS
    for txt_file_name in os.listdir(test_neg_folder):
        prob_pos = math.log(phi_pos)
        prob_neg = math.log(phi_neg)
        txt_file_path = os.path.join(test_neg_folder, txt_file_name)
        review = ""
        with open(txt_file_path, 'r') as f:
            review = f.read()
        pre_tokenized_review = word_tokenize(review.lower())
        tokenized_review = []
        for word in pre_tokenized_review:
            if word.isalnum():
                tokenized_review.append(word)
        for word in set(tokenized_review):
            prob_pos+=math.log((pos_vocabulary_dict[word] + 1)/(pos_count +2));
            prob_neg+=math.log((neg_vocabulary_dict[word] + 1)/(neg_count +2));
        if prob_neg > prob_pos :
            correct_class+=1
            print("Was Neg, Classified Neg")
            actual_n_predicted_n += 1
        else :
            actual_n_predicted_p += 1
            print("Was Neg, Classified Pos")
        total_class+=1
    # TEST SET - POSITIVE REVIEWS
    for txt_file_name in os.listdir(test_pos_folder):
        prob_pos = math.log(phi_pos)
        prob_neg = math.log(phi_neg)
        txt_file_path = os.path.join(test_pos_folder, txt_file_name)
        review = ""
        with open(txt_file_path, 'r') as f:
            review = f.read()
        pre_tokenized_review = word_tokenize(review.lower())
        tokenized_review = []
        for word in pre_tokenized_review:
            if word.isalnum():
                tokenized_review.append(word)
        for word in set(tokenized_review):
            prob_pos+=math.log((pos_vocabulary_dict[word] + 1)/(pos_count +2));
            prob_neg+=math.log((neg_vocabulary_dict[word] + 1)/(neg_count +2));         
        if prob_pos > prob_neg :
            correct_class+=1
            print("Was Pos, Classified Pos")
            actual_p_predicted_p += 1
        else :
            print("Was Pos, Classified Neg")
            actual_p_predicted_n += 1
        total_class+=1
    accuracy = correct_class/total_class
    print("Test Set Accuracy: ", accuracy) 
    precision = (actual_p_predicted_p/(actual_p_predicted_p + actual_n_predicted_p));
    print("Test Set Precision: ", precision)
    recall =  (actual_p_predicted_p/(actual_p_predicted_p + actual_p_predicted_n));
    print("Test Set Recall: ", recall)
    f1 = 2*precision*recall/(precision+recall)
    print("Test Set F1: ", f1)
    build_confusion_matrix(actual_p_predicted_p, actual_n_predicted_p, actual_p_predicted_n, actual_n_predicted_n)  

if __name__ == "__main__":
    main()