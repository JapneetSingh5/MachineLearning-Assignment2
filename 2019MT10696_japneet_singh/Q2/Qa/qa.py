import sys
import pickle
import numpy as np
import cvxopt
import time
from scipy import spatial
from sklearn import svm

class1 = 1 # (6 mod5)=1
class2 = 2 # (6+1 mod5)=2

def process_command():
    args_length = len(sys.argv);
    if(args_length<3):
        print("Insufficient arguments provided, exiting")
        sys.exit(1)
    path_train_data = sys.argv[1]
    path_test_data = sys.argv[2]
    return path_train_data, path_test_data

def transform_label(label):
    if(label==class1):
        return -1.0
    else :
        return 1.0

def main():
    path_train_data ,path_test_data= process_command()
    train_data_file = path_train_data+ "/train_data.pickle"
    test_data_file = path_test_data+ "/test_data.pickle"
    train_data_file_loaded = pickle.load(open(train_data_file,"rb"))
    test_data_file_loaded = pickle.load(open(test_data_file,"rb"))
    # print(len(train_data_loaded['data'][0]), train_data_loaded['labels'][1])
    # print(len(train_data_loaded['data'][0].flatten()), train_data_loaded['labels'][1])
    train_data = [[(train_data_file_loaded['data'][i].flatten()/255).tolist(), train_data_file_loaded['labels'][i].tolist()] for i in range(len(train_data_file_loaded['data']))]
    train_data_X = np.array([ele[0] for ele in train_data if (ele[1][0]==class1 or ele[1][0]==class2)], dtype=float)
    train_data_Y = np.array([[transform_label(ele[1][0])] for ele in train_data if (ele[1][0]==class1 or ele[1][0]==class2)])
    test_data = [[(test_data_file_loaded['data'][i].flatten()/255).tolist(), test_data_file_loaded['labels'][i].tolist()] for i in range(len(test_data_file_loaded['data']))]
    test_data_X = np.array([ele[0] for ele in test_data if (ele[1][0]==class1 or ele[1][0]==class2)], dtype=float)
    test_data_Y = np.array([[transform_label(ele[1][0])] for ele in test_data if (ele[1][0]==class1 or ele[1][0]==class2)])
    # print(train_data_X, train_data_X.shape, train_data_Y, train_data_Y.shape)
    
    # Linear CVXOPT
    start_linear = time.time()
    c_linear = 1
    m,n = train_data_X.shape
    P_temp = np.multiply(train_data_Y,train_data_X)
    P_linear = cvxopt.matrix(np.dot(P_temp, P_temp.T))
    # Qm×1[i] = 1
    q_linear = cvxopt.matrix(-1.0*np.ones((m, 1)))
    # A1×m[i] = y(i)
    A_linear = cvxopt.matrix(train_data_Y.reshape(1,-1))
    # b=0
    b_linear = cvxopt.matrix(np.zeros(1))
    # G2m×m = [−Identity(m); Identity(m)]
    G_linear = cvxopt.matrix(np.vstack((-1.0*np.eye(m),c_linear*1.0*np.eye(m))))
    # h2m×1 = [Zerosm×1; C × Onesm×1]
    h_linear = cvxopt.matrix(np.hstack((np.zeros(m),c_linear*1.0*np.ones(m))))
    cvxopt.solvers.options['show_progress'] = True
    # cvxopt.solvers.options['abstol'] = 1e-10
    # cvxopt.solvers.options['reltol'] = 1e-10
    # cvxopt.solvers.options['feastol'] = 1e-10
    sol_linear = cvxopt.solvers.qp(P_linear, q_linear, G_linear, h_linear, A_linear, b_linear)
    print(f"Time taken to solve using cvxopt with linear kernel: {time.time() - start_linear}")
    # print(sol_linear)
    # print(sol_linear['x'])
    alphas = np.array(sol_linear['x'])
    support_vectors_linear = (alphas > 1e-4).flatten()
    # print(alphas)
    # print(support_vectors_linear)
    w_linear = ((train_data_Y[support_vectors_linear] * alphas[support_vectors_linear]).T @ train_data_X[support_vectors_linear]).reshape(-1, 1)
    b_linear = np.mean(train_data_Y[support_vectors_linear] - np.dot(train_data_X[support_vectors_linear], w_linear))
    pos_svs = len(train_data_Y[support_vectors_linear] == 1)
    neg_svs = len(train_data_Y[support_vectors_linear] == -1)
    print("w", w_linear)
    print("b", b_linear)
    print("alphas", alphas.size, alphas)
    print("support vectors", support_vectors_linear.size, support_vectors_linear)
    print("pos_svs", pos_svs)
    print("neg_svs", neg_svs)
    train_prediction_linear = np.dot(train_data_X, w_linear) + b_linear
    train_prediction_linear_final = np.array([1.0 if x >= 0 else -1.0 for x in train_prediction_linear])
    train_accuracy_linear = 100.0*sum(x == y for x,y in zip(train_prediction_linear_final, train_data_Y))/len(train_data_Y)
    print(train_prediction_linear_final, train_accuracy_linear)
    test_prediction_linear = np.dot(test_data_X, w_linear) + b_linear
    test_prediction_linear_final = np.array([1.0 if x >= 0 else -1.0 for x in test_prediction_linear])
    test_accuracy_linear = 100.0*sum(x == y for x,y in zip(test_prediction_linear_final, test_data_Y))/len(test_data_Y)
    print(test_prediction_linear_final, test_accuracy_linear)
    # print(w.size, train_data_Y.size)
    # print(w)

    # Gaussian CVXOPT
    gamma = 0.001
    start_gaussian = time.time()
    pdist = spatial.distance.pdist(train_data_X, 'sqeuclidean')
    K = np.exp(-1*gamma*spatial.distance.squareform(pdist))
    # K = np.zeros((m, m))
    # for i, x1 in enumerate(train_data_X):
    #     for j, x2 in enumerate(train_data_X):
    #         K[i][j] = np.exp(-gamma * np.linalg.norm(x1 - x2)**2)
    P_gaussian = cvxopt.matrix(np.outer(train_data_Y, train_data_Y)*K)
    b_cvxopt_gaussian = cvxopt.matrix(np.zeros(1))
    # cvxopt.solvers.options['abstol'] = 1e-10
    # cvxopt.solvers.options['reltol'] = 1e-10
    # cvxopt.solvers.options['feastol'] = 1e-10
    sol_gaussian = cvxopt.solvers.qp(P_gaussian, q_linear, G_linear, h_linear, A_linear, b_cvxopt_gaussian)
    print(f"Time taken to solve using cvxopt with gaussian kernel: {time.time() - start_gaussian}")
    alphas_gaussian = np.array(sol_gaussian['x'])
    support_vectors_gaussian = (alphas_gaussian > 1e-4)
    supp_vec_ind = np.where(support_vectors_gaussian == True)[0]
    support_vectors_gaussian = support_vectors_gaussian.flatten()
    pos_svs = len(train_data_Y[support_vectors_gaussian] == 1)
    # print(alphas)
    # print(support_vectors_linear)
    pdist = spatial.distance.cdist(train_data_X[supp_vec_ind], train_data_X, 'sqeuclidean')
    K_train = np.exp(-1*0.05*(pdist))
    w_gaussian = np.dot(K_train.T, (alphas_gaussian[support_vectors_gaussian]*train_data_Y[support_vectors_gaussian]))
    # w_gaussian = ((train_data_Y[support_vectors_gaussian] * alphas_gaussian[support_vectors_gaussian]).T @ train_data_X[support_vectors_gaussian]).reshape(-1, 1)
    bias = train_data_Y[support_vectors_gaussian] - w_gaussian
    b_gaussian = np.mean(bias)
    # a = np.multiply(train_data_Y.reshape(-1, 1), alphas_gaussian)
    # b_gaussian = (train_data_Y - w_gaussian)
    # b_gaussian = np.mean(b_gaussian)
    print("w", w_gaussian)
    print("b", b_gaussian)
    print("alphas", alphas_gaussian.size, alphas_gaussian)
    print("support vectors", support_vectors_gaussian.size, support_vectors_gaussian)
    print("pos_svs", pos_svs)
    cdist = spatial.distance.cdist(train_data_X[supp_vec_ind], train_data_X, 'sqeuclidean')
    K_train2 = np.exp(-1*0.05*(cdist))
    w2 = np.dot(K_train2.T, (alphas_gaussian[supp_vec_ind]*train_data_Y[supp_vec_ind]))
    train_prediction_gaussian = w2 + b_gaussian
    # train_prediction_gaussian = w_gaussian + b_gaussian
    # train_prediction_gaussian = np.dot(train_data_X, w_gaussian) + b_gaussian
    print("train pred gaussian", train_prediction_gaussian)
    train_prediction_gaussian_final = np.array([1.0 if x >= 0 else -1.0 for x in train_prediction_gaussian])
    train_accuracy_gaussian = 100.0*sum(x == y for x,y in zip(train_prediction_gaussian_final, train_data_Y))/len(train_data_Y)
    print(train_prediction_gaussian_final, train_accuracy_gaussian)
    test_prediction_gaussian = np.dot(test_data_X, w_gaussian) + b_gaussian
    test_prediction_gaussian_final = np.array([1.0 if x >= 0 else -1.0 for x in test_prediction_gaussian])
    test_accuracy_gaussian = 100.0*sum(x == y for x,y in zip(test_prediction_gaussian_final, test_data_Y))/len(test_data_Y)
    print(test_prediction_gaussian_final, test_accuracy_gaussian)

    start_svm_linear = time.time()
    svm_linear = svm.SVC(C=1.0, kernel="linear")
    svm_linear.fit(train_data_X, train_data_Y.ravel())
    print(f"Time taken to solve using skl svm with linear kernel: {time.time() - start_svm_linear}")
    print(svm_linear.support_vectors_, len(svm_linear.support_vectors_))
    print(svm_linear.intercept_)
    train_accuracy_linear_skl_svm = 100.0*sum(x == y for x,y in zip(svm_linear.predict(train_data_X).reshape(-1,1), train_data_Y))/len(train_data_Y)
    print(train_accuracy_linear_skl_svm)
    test_accuracy_linear_skl_svm = 100.0*sum(x == y for x,y in zip(svm_linear.predict(test_data_X).reshape(-1,1), test_data_Y))/len(test_data_Y)
    print(test_accuracy_linear_skl_svm)
    start_svm_gaussian =time.time()
    svm_gaussian= svm.SVC(C=1.0, kernel="rbf", gamma=gamma)
    svm_gaussian.fit(train_data_X, train_data_Y.ravel())
    print(f"Time taken to solve using skl svm with gaussian kernel: {time.time() - start_svm_gaussian}")
    print(svm_gaussian.support_vectors_, len(svm_gaussian.support_vectors_))
    print(svm_gaussian.intercept_)
    train_accuracy_gaussian_skl_svm = 100.0*sum(x == y for x,y in zip(svm_gaussian.predict(train_data_X).reshape(-1,1), train_data_Y))/len(train_data_Y)
    print(train_accuracy_gaussian_skl_svm)
    test_accuracy_gaussian_skl_svm = 100.0*sum(x == y for x,y in zip(svm_gaussian.predict(test_data_X).reshape(-1,1), test_data_Y))/len(test_data_Y)
    print(test_accuracy_gaussian_skl_svm)




if __name__ == "__main__":
    main()