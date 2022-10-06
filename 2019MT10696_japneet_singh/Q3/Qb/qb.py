import sys
import pickle
import numpy as np
import cvxopt
import time
from scipy import spatial
from sklearn import svm
import matplotlib.pyplot as plt
import seaborn as sb
np.set_printoptions(threshold=sys.maxsize)

# for latest output and accuracies,  see b.txt 

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
    m,_ = test_data_all_X.shape
    confusion_matrix_gaussian = np.zeros((5,5))
    predictions = np.zeros((m, 5))
    predictors = np.empty((5,5), dtype=svm.SVC)
    c = 1
    for i in range(5):
        for j in range(i+1, 5):
            print("Classifier for " + str(i) + " and " + str(j))
            train_data_X, train_data_Y, test_data_X, test_data_Y = build_train_test_data(train_data, test_data, i, j)
            start_time = time.time()
            svm_gaussian = svm.SVC(C=c, kernel="rbf", gamma=gamma)
            svm_gaussian.fit(train_data_X, train_data_Y.ravel())
            end_time = time.time()
            print("Time taken to train classifier", (end_time-start_time))
            predictors[i,j] = svm_gaussian
            # train_accuracy_gaussian_skl_svm = 100.0*sum(x == y for x,y in zip(svm_gaussian.predict(train_data_X).reshape(-1,1), train_data_Y))/len(train_data_Y)
            # print(train_accuracy_gaussian_skl_svm)
            # test_predictions = svm_gaussian.predict(test_data_X)
            # test_accuracy_gaussian_skl_svm = 100.0*sum(x == y for x,y in zip(test_predictions.reshape(-1,1), test_data_Y))/len(test_data_Y)
            # print(test_accuracy_gaussian_skl_svm)
            # print(test_predictions, test_data_Y)
            test_predictions = predictors[i,j].predict(test_data_all_X)
            for k in range(len(test_predictions)):
                if test_predictions[k]==-1:
                    predictions[k,i]+=1
                else:
                    predictions[k, j]+=1

    max_score = [0]*m
    # print(predictions)
    for i in range(len(predictions)):
        temp_max_index = 0
        temp_max_score = predictions[i,0]
        for j in range(len(predictions[i])):
            if(predictions[i,j]>temp_max_score):
                temp_max_score = predictions[i,j]
                temp_max_index = j
        max_score[i] = temp_max_index
    test_accuracy_gaussian_skl_svm = 100.0*sum(predicted == actual for predicted,actual in zip(np.array(max_score).reshape(-1,1), test_data_all_Y))/len(test_data_all_Y)
    print("Test accuracy sklearn multiclass", test_accuracy_gaussian_skl_svm)
    # print(max_score, test_data_all_Y)
    mis_4_to_2 = []
    mis_2_to_4 = []
    for i in range(0, len(test_data_all_Y)):
        # (predicted, actual)
        if(max_score[i]==2 and test_data_all_Y[i]==4 and len(mis_4_to_2)<5):
            mis_4_to_2.append(np.array(test_data_all_X[i]).reshape(32,32,3))
        if(max_score[i]==4 and test_data_all_Y[i]==2 and len(mis_2_to_4)<5):
            mis_2_to_4.append(np.array(test_data_all_X[i]).reshape(32,32,3))
        confusion_matrix_gaussian[max_score[i],test_data_all_Y[i]] += 1
    fig = plt.figure(figsize=(16, 12))
    # print(actual_p_predicted_p, actual_n_predicted_p, actual_p_predicted_n, actual_n_predicted_n)
    _ = sb.heatmap(confusion_matrix_gaussian, annot=True, cmap="Greens", fmt='g')
    ax = fig.gca()
    ax.xaxis.tick_top()
    ax.set_xlabel("Actual Class")
    ax.set_ylabel("Predicted Class")
    plt.title("Confusion Matrix (for Test Data Only)")
    ax.xaxis.set_label_position('top')
    plt.savefig("confusion_matrix.png")
    plt.show()
    count = 0
    for imagearr in mis_2_to_4:
        plt.imshow(imagearr)
        plt.savefig("2to4"+str(count)+".png")
        count+=1
    for imagearr in mis_4_to_2:
        plt.imshow(imagearr)
        plt.savefig("4to2"+str(count)+".png")
        count+=1
    print(mis_2_to_4)
    print(mis_4_to_2)




if __name__ == "__main__":
    main()