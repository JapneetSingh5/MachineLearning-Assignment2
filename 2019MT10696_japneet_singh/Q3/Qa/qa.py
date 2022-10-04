import sys
import pickle
import numpy as np
import cvxopt
import time
from scipy import spatial
from sklearn import svm
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

def transform_label(label, class1, class2):
    if(label==class1):
        return -1.0
    else :
        return 1.0

def build_train_data(train_data, class1, class2):
    train_data_X = np.array([ele[0] for ele in train_data if (ele[1][0]==class1 or ele[1][0]==class2)], dtype=float)
    train_data_Y = np.array([[transform_label(ele[1][0], class1, class2)] for ele in train_data if (ele[1][0]==class1 or ele[1][0]==class2)])
    return train_data_X, train_data_Y

def build_all_test_data(train_data, test_data):
    test_data_all_X = np.array([ele[0] for ele in test_data], dtype=float)
    test_data_all_Y = np.array([[ele[1][0]] for ele in test_data])
    return test_data_all_X, test_data_all_Y

def gaussian_cvxopt(train_data_X, train_data_Y):
    start_time = time.time()
    gamma = 0.001
    c = 1
    m,_ = train_data_X.shape
    pdist = spatial.distance.pdist(train_data_X, 'sqeuclidean')
    K = np.exp(-1*gamma*spatial.distance.squareform(pdist))
    P = cvxopt.matrix(np.outer(train_data_Y, train_data_Y)*K)
    # Qm×1[i] = 1
    q = cvxopt.matrix(-1.0*np.ones((m, 1)))
    # A1×m[i] = y(i)
    A = cvxopt.matrix(train_data_Y.reshape(1,-1))
    # b=0
    b = cvxopt.matrix(np.zeros(1))
    # G2m×m = [−Identity(m); Identity(m)]
    G = cvxopt.matrix(np.vstack((-1.0*np.eye(m),c*1.0*np.eye(m))))
    # h2m×1 = [Zerosm×1; C × Onesm×1]
    h = cvxopt.matrix(np.hstack((np.zeros(m),c*1.0*np.ones(m))))
    cvxopt.solvers.options['show_progress'] = True
    sol = cvxopt.solvers.qp(P, q, G, h, A, b)
    end_time = time.time()
    print("Time taken to train classifier", (end_time-start_time))
    return sol

def predict_class(train_data_X, train_data_Y, test_data_all_X, sol_gaussian):
    gamma = 0.001
    alphas_gaussian = np.array(sol_gaussian['x'])
    support_vectors_gaussian = (alphas_gaussian > 1e-4)
    supp_vec_ind = np.where(support_vectors_gaussian == True)[0]
    support_vectors_gaussian = support_vectors_gaussian.flatten()
    pdist_train = spatial.distance.pdist(train_data_X[supp_vec_ind], 'sqeuclidean')
    K_train = np.exp(-1*gamma*spatial.distance.squareform(pdist_train))
    w_train = np.dot(K_train.T, (alphas_gaussian[support_vectors_gaussian]*train_data_Y[support_vectors_gaussian]))
    bias = train_data_Y[support_vectors_gaussian] - w_train
    b_gaussian = np.mean(bias)
    cdist_test = spatial.distance.cdist(train_data_X[supp_vec_ind], test_data_all_X, 'sqeuclidean')
    K_test = np.exp(-1*gamma*(cdist_test))
    w_gaussian_test = np.dot(K_test.T, (alphas_gaussian[supp_vec_ind]*train_data_Y[supp_vec_ind]))
    test_prediction_gaussian = w_gaussian_test + b_gaussian
    test_predictions = np.array([1.0 if x >= 0 else -1.0 for x in test_prediction_gaussian])
    return test_predictions


def main():
    path_train_data ,path_test_data= process_command()
    train_data_file = path_train_data+ "/train_data.pickle"
    test_data_file = path_test_data+ "/test_data.pickle"
    train_data_file_loaded = pickle.load(open(train_data_file,"rb"))
    test_data_file_loaded = pickle.load(open(test_data_file,"rb"))
    train_data = [[(train_data_file_loaded['data'][i].flatten()/255).tolist(), train_data_file_loaded['labels'][i].tolist()] for i in range(len(train_data_file_loaded['data']))]
    test_data = [[(test_data_file_loaded['data'][i].flatten()/255).tolist(), test_data_file_loaded['labels'][i].tolist()] for i in range(len(test_data_file_loaded['data']))]
    test_data_all_X, test_data_all_Y = build_all_test_data(train_data, test_data)
    m,n = test_data_all_X.shape
    
    confusion_matrix_gaussian = np.zeros((5,5))
    predictions = np.zeros((m, 5))
    predictors = np.empty((5,5), dtype=dict)
    for i in range(5):
        for j in range(i+1, 5):
            print("Classifier for " + str(i) + " and " + str(j))
            train_data_X, train_data_Y = build_train_data(train_data, i, j)
            predictors[i,j] = gaussian_cvxopt(train_data_X, train_data_Y)
            sol_gaussian = predictors[i,j]
            test_predictions = predict_class(train_data_X, train_data_Y, test_data_all_X, sol_gaussian)
            for k in range(len(test_predictions)):
                if test_predictions[k]==-1:
                    predictions[k,i]+=1
                else:
                    predictions[k, j]+=1
    max_score = [0]*m
    print(len(max_score), len(predictions))
    print(predictions)
    for i in range(m):
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
    test_data_all_Y = test_data_all_Y.ravel()
    for i in range(0, len(test_data_all_Y)):
        # (predicted, actual)
        confusion_matrix_gaussian[max_score[i],test_data_all_Y[i]] += 1
    fig = plt.figure(figsize=(16, 12))
    # print(actual_p_predicted_p, actual_n_predicted_p, actual_p_predicted_n, actual_n_predicted_n)
    plotit = sb.heatmap(confusion_matrix_gaussian, annot=True, cmap="Greens", fmt='g')
    ax = fig.gca()
    ax.xaxis.tick_top()
    ax.set_xlabel("Actual Class")
    ax.set_ylabel("Predicted Class")
    plt.title("Confusion Matrix (for Test Data Only)")
    ax.xaxis.set_label_position('top')
    plt.savefig("confusion_matrix.png")
    plt.show()



if __name__ == "__main__":
    main()