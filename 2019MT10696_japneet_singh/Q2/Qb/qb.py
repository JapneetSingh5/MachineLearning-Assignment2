import sys
import pickle
import numpy as np
import cvxopt
import time
from scipy import spatial
from sklearn import svm
import matplotlib.pyplot as plt

# for latest output and accuracies,  see b.txt 

# Entry number 2019MT10696, last digit is 6
class1 = 1 # (6 mod5) = 1
class2 = 2 # (6+1 mod5) = 2

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
    train_data = [[(train_data_file_loaded['data'][i].flatten()/255).tolist(), train_data_file_loaded['labels'][i].tolist()] for i in range(len(train_data_file_loaded['data']))]
    train_data_X = np.array([ele[0] for ele in train_data if (ele[1][0]==class1 or ele[1][0]==class2)], dtype=float)
    train_data_Y = np.array([[transform_label(ele[1][0])] for ele in train_data if (ele[1][0]==class1 or ele[1][0]==class2)])
    test_data = [[(test_data_file_loaded['data'][i].flatten()/255).tolist(), test_data_file_loaded['labels'][i].tolist()] for i in range(len(test_data_file_loaded['data']))]
    test_data_X = np.array([ele[0] for ele in test_data if (ele[1][0]==class1 or ele[1][0]==class2)], dtype=float)
    test_data_Y = np.array([[transform_label(ele[1][0])] for ele in test_data if (ele[1][0]==class1 or ele[1][0]==class2)])
    
    # Linear CVXOPT
    start_linear = time.time()
    c_linear = 1
    m,_ = train_data_X.shape
    P_temp = np.multiply(train_data_Y,train_data_X)
    P_linear = cvxopt.matrix(np.dot(P_temp, P_temp.T))
    # Q(mx1) = [1;1;1....1]
    q_linear = cvxopt.matrix(-1.0*np.ones((m, 1)))
    # G(2mxm) = [−I(mxm); I(mxm)]
    G_linear = cvxopt.matrix(np.vstack((-1.0*np.eye(m),c_linear*1.0*np.eye(m))))
    # h(2mx1) = [Zeros(mx1); C x (Ones(mx1))]
    h_linear = cvxopt.matrix(np.hstack((np.zeros(m),c_linear*1.0*np.ones(m))))
    # A(1xm)[i] = y(i)
    A_linear = cvxopt.matrix(train_data_Y.reshape(1,-1))
    # b=0
    b_linear = cvxopt.matrix(np.zeros(1))
    # cvxopt.solvers.options['show_progress'] = False
    # cvxopt.solvers.options['abstol'] = 1e-15
    sol_linear = cvxopt.solvers.qp(P_linear, q_linear, G_linear, h_linear, A_linear, b_linear)
    print("Time taken for CVXOPT - Linear Kernel", time.time() - start_linear)
    alphas = np.array(sol_linear['x'])
    support_vectors_linear = (alphas > 1e-4).flatten()
    # count the number of support vectors found
    count_svs = len(train_data_Y[support_vectors_linear] == 1)
    w_linear = ((train_data_Y[support_vectors_linear] * alphas[support_vectors_linear]).T @ train_data_X[support_vectors_linear]).reshape(-1, 1)
    b_linear = np.mean(train_data_Y[support_vectors_linear] - np.dot(train_data_X[support_vectors_linear], w_linear))
    print("w for  CVXOPT Linear Kernel", w_linear)
    print("b for CVXOPT Linear Kernel", b_linear)
    print("alphas for CVXOPT Linear Kernel (count) ", alphas.size)
    print("support vectors for CVXOPT Linear Kernel (count)", count_svs)
    # predictions for training set
    train_prediction_linear = np.dot(train_data_X, w_linear) + b_linear
    # classify predictions into classes for train set
    train_prediction_linear_final = np.array([1.0 if wtb >= 0 else -1.0 for wtb in train_prediction_linear])
    # calculate training accuracy
    train_accuracy_linear = 100.0*sum(predcited == actual for predcited,actual in zip(train_prediction_linear_final, train_data_Y))/len(train_data_Y)
    print("Train Accuracy CVXOPT Linear Kernel", train_accuracy_linear)
    # predictions for testing set
    test_prediction_linear = np.dot(test_data_X, w_linear) + b_linear
    # classify predictions into classes for test set
    test_prediction_linear_final = np.array([1.0 if wtb >= 0 else -1.0 for wtb in test_prediction_linear])
    # calculate test set accuracy
    test_accuracy_linear = 100.0*sum(x == y for x,y in zip(test_prediction_linear_final, test_data_Y))/len(test_data_Y)
    print("Test Accuracy CVXOPT Linear Kernel", test_accuracy_linear)
    # Gaussian CVXOPT
    gamma = 0.001
    start_gaussian = time.time()
    # norm of distances to be used in gaussian kernel found using pdist function
    training_data_pdist = spatial.distance.pdist(train_data_X, 'sqeuclidean')
    # build gaussian kernel 
    gaussian_kernel = np.exp(-1*gamma*spatial.distance.squareform(training_data_pdist))
    # looping to calculate Kernel was too slow, much much faster solution found on stackoverflow
    # ref: https://stackoverflow.com/questions/47271662/what-is-the-fastest-way-to-compute-an-rbf-kernel-in-python
    # Only P changes for gaussian coming from linear , b reinit'd as it was changed before
    P_gaussian = cvxopt.matrix(np.outer(train_data_Y, train_data_Y)*gaussian_kernel)
    b_cvxopt_gaussian = cvxopt.matrix(np.zeros(1))
    # cvxopt.solvers.options['show_progress'] = False
    # cvxopt.solvers.options['abstol'] = 1e-15
    sol_gaussian = cvxopt.solvers.qp(P_gaussian, q_linear, G_linear, h_linear, A_linear, b_cvxopt_gaussian)
    print("Time taken for CVXOPT - Gaussiann Kernel", time.time() - start_gaussian)
    alphas_gaussian = np.array(sol_gaussian['x'])
    # 1e-4 gives almost 'exactly' the same number of support vectors as sklearn
    # 1e-5 is too small
    support_vectors_gaussian = (alphas_gaussian > 1e-4)
    indices = np.where(support_vectors_gaussian == True)[0]
    support_vectors_gaussian = support_vectors_gaussian.flatten()
    # count the number of support vectors for the gaussian case
    count_svs = len(train_data_Y[support_vectors_gaussian] == 1)
    training_data_svs_pdist = spatial.distance.pdist(train_data_X[indices], 'sqeuclidean')
    kernel_train_svs = np.exp(-1*gamma*spatial.distance.squareform(training_data_svs_pdist))
    w_train = np.dot(kernel_train_svs.T, (alphas_gaussian[support_vectors_gaussian]*train_data_Y[support_vectors_gaussian]))
    b_gaussian = np.mean(train_data_Y[support_vectors_gaussian] - w_train)
    training_data_svs_cdist = spatial.distance.cdist(train_data_X[indices], train_data_X, 'sqeuclidean')
    kernel_train_svs2 = np.exp(-1*gamma*(training_data_svs_cdist))
    w_gaussian = np.dot(kernel_train_svs2.T, (alphas_gaussian[indices]*train_data_Y[indices]))
    # raw predictions for gaussian cvxopt case
    train_prediction_gaussian = w_gaussian + b_gaussian
    print("b for CVXOPT Gaussian Kernel", b_gaussian)
    print("alphas for CVXOPT Gaussian Kernel", alphas_gaussian.size, alphas_gaussian)
    print("support vectors for CVXOPT Gaussian Kernel", count_svs, support_vectors_gaussian)
    # classify predictions into classes
    train_prediction_gaussian_final = np.array([1.0 if x >= 0 else -1.0 for x in train_prediction_gaussian])
    # calc accuracy for gaussian predictions
    train_accuracy_gaussian = 100.0*sum(predicted == actual for predicted,actual in zip(train_prediction_gaussian_final, train_data_Y))/len(train_data_Y)
    print("Train Accuracy CVXOPT Gaussian Kernel", train_accuracy_gaussian)
    test_data_cdist = spatial.distance.cdist(train_data_X[indices], test_data_X, 'sqeuclidean')
    kernel_test = np.exp(-1*gamma*(test_data_cdist))
    w_gaussian_test = np.dot(kernel_test.T, (alphas_gaussian[indices]*train_data_Y[indices]))
    # raw predictions for test set
    test_prediction_gaussian = w_gaussian_test + b_gaussian
    # classify test set predictions into classes
    test_prediction_gaussian_final = np.array([1.0 if x >= 0 else -1.0 for x in test_prediction_gaussian])
    # calculate test set accuracy
    test_accuracy_gaussian = 100.0*sum(x == y for x,y in zip(test_prediction_gaussian_final, test_data_Y))/len(test_data_Y)
    print("Test Accuracy CVXOPT Gaussian Kernel", test_accuracy_gaussian)

    # sklearn SVM Linear Kernel
    start_svm_linear = time.time()
    svm_linear = svm.SVC(C=1.0, kernel="linear")
    svm_linear.fit(train_data_X, train_data_Y.ravel())
    print("Time taken for scikit-learn - Linear Kernel", time.time() - start_svm_linear)
    # print(svm_linear.support_vectors_, len(svm_linear.support_vectors_))
    print("b for sklearn Linear Kernel", svm_linear.intercept_)
    train_accuracy_linear_skl_svm = 100.0*sum(x == y for x,y in zip(svm_linear.predict(train_data_X).reshape(-1,1), train_data_Y))/len(train_data_Y)
    print("Train Accuracy sklearn linear kernel" , train_accuracy_linear_skl_svm)
    test_accuracy_linear_skl_svm = 100.0*sum(x == y for x,y in zip(svm_linear.predict(test_data_X).reshape(-1,1), test_data_Y))/len(test_data_Y)
    print("Test Accuracy sklearn linear kernel", test_accuracy_linear_skl_svm)
    start_svm_gaussian =time.time()
    svm_gaussian= svm.SVC(C=1.0, kernel="rbf", gamma=gamma)
    svm_gaussian.fit(train_data_X, train_data_Y.ravel())
    print("Time taken for scikit-learn - Gaussian Kernel", time.time() - start_svm_gaussian)
    # print(svm_gaussian.support_vectors_, len(svm_gaussian.support_vectors_))
    print("b for sklearn gaussian kernel",svm_gaussian.intercept_)
    train_accuracy_gaussian_skl_svm = 100.0*sum(x == y for x,y in zip(svm_gaussian.predict(train_data_X).reshape(-1,1), train_data_Y))/len(train_data_Y)
    print("Train Accuracy sklearn gaussian kernel" ,train_accuracy_gaussian_skl_svm)
    test_accuracy_gaussian_skl_svm = 100.0*sum(x == y for x,y in zip(svm_gaussian.predict(test_data_X).reshape(-1,1), test_data_Y))/len(test_data_Y)
    print("Test Accuracy sklearn gaussian kernel" ,test_accuracy_gaussian_skl_svm)
    common_support_vectors_cvxopt = 0
    for i in range(len(support_vectors_linear)):
        if(support_vectors_linear[i] and support_vectors_gaussian[i]):
            common_support_vectors_cvxopt+=1
    print(common_support_vectors_cvxopt, " is the count of support vectors common in linear and gaussian case of cvxopt implementation ")
    # takes too much time hence commented out, run once before submission, result is documented in report
    # common_sv_linear = 0
    # common_sv_gaussian = 0
    # for i in range(len(alphas_gaussian)):
    #     if(alphas_gaussian[i]>1e-4):
    #         if(alphas_gaussian[i] in svm_gaussian.support_vectors_):
    #             common_sv_gaussian+=1
    #         if(alphas[i] in svm_linear.support_vectors_):
    #             common_sv_linear+=1
    # print("common svs in linear cvxopt and sklearn", common_sv_linear)
    # print("common svs in gaussian cvxopt and sklearn", common_sv_gaussian)
    # alphas_indices = np.flip(np.argsort(alphas,axis = 0))
    count = 0
    # plotting top 5 coefficients
    # for i in range(5):
    #     imagearr = np.array(train_data_X[alphas_indices[i]]).reshape(32,32,3)
    #     plt.imshow(imagearr)
    #     plt.savefig("alphas_linear"+str(count)+".png")
    #     count+=1
    alphas_g_indices = np.flip(np.argsort(alphas_gaussian,axis = 0))
    for i in range(5):
        imagearr = np.array(train_data_X[alphas_g_indices[i]]).reshape(32,32,3)
        plt.imshow(imagearr)
        plt.savefig("alphas_gaussian"+str(count)+".png")
        count+=1

            
if __name__ == "__main__":
    main()