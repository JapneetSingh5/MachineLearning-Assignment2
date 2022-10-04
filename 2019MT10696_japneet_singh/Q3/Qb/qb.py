import sys
import pickle
import numpy as np
import cvxopt
import time
from scipy import spatial
from sklearn import svm

def process_command():
    args_length = len(sys.argv);
    if(args_length<3):
        print("Insufficient arguments provided, exiting")
        sys.exit(1)
    path_train_data = sys.argv[1]
    path_test_data = sys.argv[2]
    return path_train_data, path_test_data

def transform_label(label, class1, class2):
    if(label==class1):
        return -1.0
    else :
        return 1.0

def build_train_test_data(train_data, test_data, class1, class2):
    train_data_X = np.array([ele[0] for ele in train_data if (ele[1][0]==class1 or ele[1][0]==class2)], dtype=float)
    train_data_Y = np.array([[transform_label(ele[1][0], class1, class2)] for ele in train_data if (ele[1][0]==class1 or ele[1][0]==class2)])
    test_data_X = np.array([ele[0] for ele in test_data if (ele[1][0]==class1 or ele[1][0]==class2)], dtype=float)
    test_data_Y = np.array([[transform_label(ele[1][0], class1, class2)] for ele in test_data if (ele[1][0]==class1 or ele[1][0]==class2)])
    return train_data_X, train_data_Y, test_data_X, test_data_Y

def build_all_test_data(train_data, test_data):
    test_data_all_X = np.array([ele[0] for ele in test_data], dtype=float)
    test_data_all_Y = np.array([[ele[1][0]] for ele in test_data])
    return test_data_all_X, test_data_all_Y

def main():
    path_train_data ,path_test_data= process_command()
    train_data_file = path_train_data+ "/train_data.pickle"
    test_data_file = path_test_data+ "/test_data.pickle"
    train_data_file_loaded = pickle.load(open(train_data_file,"rb"))
    test_data_file_loaded = pickle.load(open(test_data_file,"rb"))
    train_data = [[(train_data_file_loaded['data'][i].flatten()/255).tolist(), train_data_file_loaded['labels'][i].tolist()] for i in range(len(train_data_file_loaded['data']))]
    test_data = [[(test_data_file_loaded['data'][i].flatten()/255).tolist(), test_data_file_loaded['labels'][i].tolist()] for i in range(len(test_data_file_loaded['data']))]
    gamma = 0.001
    test_data_all_X, test_data_all_Y = build_all_test_data(train_data, test_data)
    m,n = test_data_all_X.shape
    confusion_matrix_gaussian = np.zeros((5,5))
    predictions = np.zeros((m, 5))
    predictors = np.empty((5,5), dtype=svm.SVC)
    for i in range(5):
        for j in range(i+1, 5):
            print("Classifier for " + str(i) + " and " + str(j))
            train_data_X, train_data_Y, test_data_X, test_data_Y = build_train_test_data(train_data, test_data, i, j)
            svm_gaussian = svm.SVC(C=1.0, kernel="rbf", gamma=gamma)
            svm_gaussian.fit(train_data_X, train_data_Y.ravel())
            predictors[i,j] = svm_gaussian
            # train_accuracy_gaussian_skl_svm = 100.0*sum(x == y for x,y in zip(svm_gaussian.predict(train_data_X).reshape(-1,1), train_data_Y))/len(train_data_Y)
            # print(train_accuracy_gaussian_skl_svm)
            # test_predictions = svm_gaussian.predict(test_data_X)
            # test_accuracy_gaussian_skl_svm = 100.0*sum(x == y for x,y in zip(test_predictions.reshape(-1,1), test_data_Y))/len(test_data_Y)
            # print(test_accuracy_gaussian_skl_svm)
            # print(test_predictions, test_data_Y)
    for i in range(5):
        for j in range(i+1, 5):
            test_predictions = predictors[i,j].predict(test_data_all_X)
            for k in range(len(test_predictions)):
                if test_predictions[k]==-1:
                    predictions[k,i]+=1
                else:
                    predictions[k, j]+=1
    max_score = [0]*m
    # for i in range(len(predictions)):
        
    #     max_score[i] = final
    print(predictions)
    for i in range(len(predictions)):
        temp_max_index = 0
        temp_max_score = predictions[i,0]
        for j in range(len(predictions[i])):
            if(predictions[i,j]>temp_max_score):
                temp_max_score = predictions[i,j]
                temp_max_index = j
        max_score[i] = temp_max_index
    test_accuracy_gaussian_skl_svm = 100.0*sum(x == y for x,y in zip(np.array(max_score).reshape(-1,1), test_data_all_Y))/len(test_data_all_Y)
    print(test_accuracy_gaussian_skl_svm)
    print(max_score, test_data_all_Y)



if __name__ == "__main__":
    main()