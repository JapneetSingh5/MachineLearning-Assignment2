import sys
import pickle
from sklearn.model_selection import KFold
import numpy as np
import random
import sys
import pickle
import numpy as np
import cvxopt
import time
from scipy import spatial
from sklearn import svm
# np.set_printoptions(threshold=np.inf, linewidth=200)

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

def predict_class(train_data, val_data, gamma, c):
    predictors = np.empty((5,5), dtype=svm.SVC)
    for i in range(5):
        for j in range(i+1, 5):
            print("Classifier for " + str(i) + " and " + str(j))
            train_data_X, train_data_Y = build_train_data(train_data, i, j)
            svm_gaussian = svm.SVC(C=c, kernel="rbf", gamma=gamma)
            svm_gaussian.fit(train_data_X, train_data_Y.ravel())
            predictors[i,j] = svm_gaussian
            train_accuracy_gaussian_skl_svm = 100.0*sum(x == y for x,y in zip(svm_gaussian.predict(train_data_X).reshape(-1,1), train_data_Y))/len(train_data_Y)
            print(train_accuracy_gaussian_skl_svm)
    val_data_X, val_data_Y = build_all_train_data(val_data, None)
    m = len(val_data_X)
    predictions = np.zeros((m, 5))
    for i in range(5):
        for j in range(i+1, 5):
            test_predictions = predictors[i,j].predict(val_data_X)
            for k in range(len(test_predictions)):
                if test_predictions[k]==-1:
                    predictions[k,i]+=1
                else:
                    predictions[k, j]+=1
    max_score = [0]*m
    # print(max_score, val_data_Y)
    # print(predictions)
    for i in range(len(predictions)):
        temp_max_index = 0
        temp_max_score = predictions[i,0]
        for j in range(len(predictions[i])):
            if(predictions[i,j]>temp_max_score):
                temp_max_score = predictions[i,j]
                temp_max_index = j
        max_score[i] = temp_max_index
    test_accuracy_gaussian_skl_svm = 100.0*sum(x == y for x,y in zip(np.array(max_score).reshape(-1,1), val_data_Y))/len(val_data_Y)
    print("Accuracy", c, test_accuracy_gaussian_skl_svm)
    return test_accuracy_gaussian_skl_svm

def build_train_data(train_data, class1, class2):
    train_data_X = np.array([ele[0] for ele in train_data if (ele[1][0]==class1 or ele[1][0]==class2)], dtype=float)
    train_data_Y = np.array([[transform_label(ele[1][0], class1, class2)] for ele in train_data if (ele[1][0]==class1 or ele[1][0]==class2)])
    return train_data_X, train_data_Y

def build_all_test_data(train_data, test_data):
    test_data_all_X = np.array([ele[0] for ele in test_data], dtype=float)
    test_data_all_Y = np.array([[ele[1][0]] for ele in test_data])
    return test_data_all_X, test_data_all_Y

def build_all_train_data(train_data, test_data):
    test_data_all_X = np.array([ele[0] for ele in train_data], dtype=float)
    test_data_all_Y = np.array([[ele[1][0]] for ele in train_data])
    return test_data_all_X, test_data_all_Y

def main():
    path_train_data ,path_test_data= process_command()
    train_data_file = path_train_data+ "/train_data.pickle"
    test_data_file = path_test_data+ "/test_data.pickle"
    train_data_file_loaded = pickle.load(open(train_data_file,"rb"))
    test_data_file_loaded = pickle.load(open(test_data_file,"rb"))
    train_data = [[(train_data_file_loaded['data'][i].flatten()/255).tolist(), train_data_file_loaded['labels'][i].tolist()] for i in range(len(train_data_file_loaded['data']))]
    random.shuffle(train_data)
    test_data = [[(test_data_file_loaded['data'][i].flatten()/255).tolist(), test_data_file_loaded['labels'][i].tolist()] for i in range(len(test_data_file_loaded['data']))]
    # train_data_X, train_data_Y = build_all_train_data(train_data,test_data)
    gamma =0.001
    c_values = [1e-5, 1e-3, 1,5,10]
    kf = KFold(n_splits=5,shuffle=False)
    split = kf.split(train_data)
    train_list = []
    val_list = []
    for train_index, test_index in split:
        temp_train = [train_data[ele] for ele in train_index]
        temp_val = [train_data[ele] for ele in test_index]
        train_list.append(temp_train)
        val_list.append(temp_val)
        # print(train_index, test_index, len(train_index), len(test_index))
    # print(len(train_list[0]), len(val_list[0]))
    print("Here")
    val_accuracy = [0,0,0,0,0]
    for c_index in range(5):
        for i in range(5):
            val_acc = predict_class(train_list[i], val_list[i], gamma, c_values[c_index])
            print(c_index,i,val_acc)
            val_accuracy[c_index]+= val_acc
    print(val_accuracy)


if __name__ == "__main__":
    main()