import sys
import os
import math
import random
import matplotlib.pyplot as plt
import seaborn as sb


def process_command():
    args_length = len(sys.argv);
    if(args_length<3):
        print("Insufficient arguments provided, exiting")
        sys.exit(1)
    path_train_data = sys.argv[1]
    path_test_data = sys.argv[2]
    return path_train_data, path_test_data

def build_confusion_matrix(actual_p_predicted_p, actual_n_predicted_p, actual_p_predicted_n, actual_n_predicted_n,name):
    fig = plt.figure(figsize=(8, 6))
    # print(actual_p_predicted_p, actual_n_predicted_p, actual_p_predicted_n, actual_n_predicted_n)
    matrix = [[actual_p_predicted_p, actual_n_predicted_p], [actual_p_predicted_n, actual_n_predicted_n]]
    plotit = sb.heatmap(matrix, annot=True, cmap="Greens", fmt='g')
    ax = fig.gca()
    ax.xaxis.tick_top()
    ax.set_xlabel("Actual Class")
    ax.set_ylabel("Predicted Class")
    plt.title("Confusion Matrix (for Test Data Only)")
    ax.xaxis.set_label_position('top')
    plt.savefig(name+"_confusion_matrix.png")
    plt.show()    


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
    actual_p_predicted_p = 0
    actual_n_predicted_p = 0
    actual_p_predicted_n = 0
    actual_n_predicted_n = 0
    for _ in os.listdir(test_neg_folder):
        neg_count+=1
        prob_pos = random.randint(1, 100)
        prob_neg = 100 - prob_pos       
        if prob_pos > prob_neg :
            # print("Was Neg, Classified Pos")
            actual_n_predicted_p+=1
            incorrect_class+=1
        else:
            # print("Was Neg, Classified Neg")
            actual_n_predicted_n+=1
            correct_class+=1
        total_class+=1
        # pos_vocabulary_dict[stemmed_word]+=1;
    for _ in os.listdir(test_pos_folder):
        pos_count+=1
        prob_pos = random.randint(1, 100)
        prob_neg = 100 - prob_pos
        if prob_pos > prob_neg :
            # print("Was Pos, Classified Pos")
            actual_p_predicted_p+=1
            correct_class+=1
        else:
            incorrect_class+=1
            actual_p_predicted_n+=1
            # print("Was Pos, Classified Neg")
        total_class+=1
    print("Test Set Accuracy by Random Prediction: ", (correct_class/total_class)*100)  
    print("Test Set Accuracy by Always Predicting Positive: ", (pos_count/total_class)*100)  
    build_confusion_matrix(actual_p_predicted_p, actual_n_predicted_p, actual_p_predicted_n, actual_n_predicted_n, "random") 
    build_confusion_matrix(pos_count, neg_count, 0, 0, "always_positive") 

    
if __name__ == "__main__":
    main()