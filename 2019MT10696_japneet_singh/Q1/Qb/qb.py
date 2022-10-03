import sys
import os
import math
import random


def process_command():
    args_length = len(sys.argv);
    if(args_length<3):
        print("Insufficient arguments provided, exiting")
        sys.exit(1)
    path_train_data = sys.argv[1]
    path_test_data = sys.argv[2]
    return path_train_data, path_test_data


def main():
    path_train_data ,path_test_data = process_command()
    # print(path_train_data, path_test_data)
    neg_suffix= "/neg"
    pos_suffix= "/pos"
    train_neg_folder = path_train_data + neg_suffix
    train_pos_folder = path_train_data + pos_suffix
    test_neg_folder = path_test_data + neg_suffix
    test_pos_folder = path_test_data + pos_suffix
    # we have phi_pos, phi_neg, count of positive and negative reviews in the training dataset
    # at test time, calculate p(y=1/x) = p(y=1)*product(i=1, n){p(xi/y=1)}/p(x)
    # p(xi/y=1) will be the number of positive documents in which the word xi appears / by the number of total positive documents 
    correct_class = 0
    incorrect_class = 0
    total_class = 0
    pos_count = 0
    neg_count = 0
    for _ in os.listdir(test_neg_folder):
        neg_count+=1
        prob_pos = random.randint(1, 100)
        prob_neg = 100 - prob_pos       
        if prob_pos > prob_neg :
            # print("Was Neg, Classified Pos")
            incorrect_class+=1
        else:
            # print("Was Neg, Classified Neg")
            correct_class+=1
        total_class+=1
        # pos_vocabulary_dict[stemmed_word]+=1;
    for _ in os.listdir(test_pos_folder):
        pos_count+=1
        prob_pos = random.randint(1, 100)
        prob_neg = 100 - prob_pos
        if prob_pos > prob_neg :
            # print("Was Pos, Classified Pos")
            correct_class+=1
        else:
            incorrect_class+=1
            # print("Was Pos, Classified Neg")
        total_class+=1
    print("Test Set Accuracy by Random Prediction: ", (correct_class/total_class)*100)  
    print("Test Set Accuracy by Always Predicting Positive: ", (pos_count/total_class)*100)  

if __name__ == "__main__":
    main()