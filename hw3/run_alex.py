#!/usr/bin/python
from sklearn.externals import joblib
# from import_data import import_data
import matplotlib.pyplot as plt
# from svm_alex import SVC
import math, time
import numpy as np
from nn import *
from helperFunctions import *
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report

def create_dataset(path,filename):
    # read in ds1 data
    # [odometry, groundtruth, measurements, landmark_groundtruth, barcodes] = import_data(path)

    # Find the ground position of the robot at each sensor measurment
    ground_truth_idxs, measurement_idxs = [], []
    for idx in range(measurements.shape[0]):
        time_diffs = np.tile(measurements[idx,0],groundtruth.shape[0]) - groundtruth[:,0]
        abs_time_diffs = abs(time_diffs)
        time_diff_val = min(abs_time_diffs)
        if (time_diff_val < math.pow(10,-2)):
            ground_truth_idx = abs_time_diffs.argmin()
            ground_truth_idxs.append(ground_truth_idx)
            measurement_idxs.append(idx)
    groundtruth_sub = groundtruth[ground_truth_idxs,1:4] # x, y, theta
    measurement_sub = measurements[measurement_idxs,1] # subject_id
    landmark_id_sub, row_idxs = [], []
    for idx in range(len(measurement_sub)):
        barcode_num = measurement_sub[idx]
        row_idx = np.where(barcodes[:,1] == barcode_num)[0][0]
        if row_idx < 5:
            # Other robots are assigned subject numbers from 1-5, this removes the cases where they are affecting the results
            continue
        row_idxs.append(idx)
        subject_num = barcodes[row_idx, 0]-5
        landmark_id_sub.append(subject_num)
    groundtruth_sub_landmarks = groundtruth_sub[row_idxs,:]
    np_positions = np.array(groundtruth_sub_landmarks)

    np_landmarks = np.zeros((len(landmark_id_sub), 15))
    for idx in range(len(landmark_id_sub)):
        np_landmarks[idx,(landmark_id_sub[idx]-1)] = 1 # -1 because of zero indexing

    training_matrix = np.concatenate((np_positions, np_landmarks), axis=1)
    joblib.dump(training_matrix, filename)
    print('done')

def create_binned_dataset(dataset, filename):
    scale_factor = 10
    binned_dataset = []
    for idx in range(1,dataset.shape[0]):
        y = math.floor(dataset[idx,1]*scale_factor)/float(scale_factor)
        x = math.floor(dataset[idx,0]*scale_factor)/float(scale_factor)
        theta = math.floor(dataset[idx,2]*scale_factor)/float(scale_factor)
        landmarks = dataset[idx,3:]
        binned_dataset.append([x, y, theta] + [x for x in landmarks])
    binned_dataset_np = np.array(binned_dataset)
    joblib.dump(binned_dataset_np, filename)
    print('done')

def create_multiclass_dataset(dataset, filename):
    multiclass_dataset = []
    for idx in range(1,dataset.shape[0]):
        y = dataset[idx,1]
        x = dataset[idx,0]
        theta = dataset[idx,2]
        landmarks = dataset[idx,3:]
        num = np.where(landmarks == 1)
        multiclass_dataset.append([x, y, theta] + [num[0][0]])
    multiclass_dataset_np = np.array(multiclass_dataset)
    joblib.dump(multiclass_dataset_np, filename)
    print('done')

def get_dataset(filename):
    dataset = joblib.load(filename)
    return dataset

def create_svm(X, y):
    clf = SVC()
    clf.train(X,y)
    return clf

def create_random_test_train_split(X,y,split_per):
    list_idxs = list(range(X.shape[0]))
    np.random.shuffle(list_idxs)
    split_idx = int(math.floor(len(list_idxs)*split_per))
    train_idxs = list_idxs[:split_idx]
    test_idxs = list_idxs[split_idx:]
    X_train = X[train_idxs]
    X_test = X[test_idxs]
    y_train = y[train_idxs]
    y_test = y[test_idxs]
    return [X_train, X_test, y_train, y_test]

def visualize(X, y, clf, title):
    # create a mesh to plot in
    h = .02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    print(np.c_[xx.ravel(), yy.ravel()])
    title = [title]

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
    plt.xlabel('Map Width')
    plt.ylabel('Map Height')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title(title)

    plt.show()

def calc_accuracy(label,pred):
    total_num = label.shape[0]
    total_num_ones = sum(abs(label[label > 0]))
    # total_num_zeros = sum(abs(label[label == 0]))
    total_num_zeros = total_num- total_num_ones

    num_correct_ones, num_correct_zeros = 0, 0
    for idx in range(total_num):
        if (label[idx] == 0 and pred[idx] == 0):
            num_correct_zeros = num_correct_zeros + 1
        if (label[idx] == 1 and pred[idx] == 1):
            num_correct_ones = num_correct_ones + 1
    num_correct_total = num_correct_zeros + num_correct_ones
    percent_correct_total = num_correct_total/float(total_num)
    percent_correct_zeros = num_correct_zeros/float(total_num_zeros)
    percent_correct_ones = num_correct_ones/float(total_num_ones)

    return [percent_correct_total, percent_correct_zeros, percent_correct_ones]

def main(path):

    t1 = time.time()
    dataset_filename = 'alexs_dataset.pkl_01.npy'
    binned_dataset_filename = 'alexs_dataset_binned.pkl'
    multiclass_dataset_filename = 'alexs_dataset_multiclass.pkl'

    # You only need to run these functions once to create the training dataset, after that you can just read it in
    # create_dataset(path, dataset_filename)
    # create_binned_dataset(full_dataset, binned_dataset_filename)
    # create_multiclass_dataset(full_dataset, multiclass_dataset_filename)

    full_dataset = np.load(dataset_filename)

    split_per = 0.4
    percent_correct = []

    ROCytest = []
    ROCyscore = []
    for landmark in range(3,full_dataset.shape[1]):
    # for landmark in range(3,4):
        np.random.shuffle(full_dataset)
        X = full_dataset[:,0:3]
        y = full_dataset[:,landmark]
        y[y<0] = 0

        minMaxDataset = getListMinMax(X.tolist())

        myInputNorm = normalize(X.tolist(), minMaxDataset)
        minMaxDatasetNorm = getListMinMax(myInputNorm)


        split_idx = int(math.floor(len(full_dataset)*split_per))
        X_train = X[:split_idx]
        X_test = X[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]

        target = y_train.tolist()

        target = [[i] for i in target]

        myInputNorm = X_train.tolist()
       
        net = NN([3,15, 1])
        error = net.trainBP(myInputNorm, target, targetSSE=1.0, lr = 1.0, maxIter = 150, show = 10)

        # clf = create_svm(X_train, y_train)


        X_test = X_test.tolist()

        preds = []
        for idx in range(len(X_test)):
            pred = net.computeOutput(X_test[idx])
            if(pred[0] >= 0.1):
                pred = 1
            else:
                pred = 0
            preds.append(pred)

        ROCyscore.extend(preds)
        ROCytest.extend(y_test.tolist())

        np_preds = np.array(preds)

        np_ys = np.array(y_test)
        percent_correct.append(calc_accuracy(np_ys, np_preds))


        # only run in x,y input space version
        # s = 'SVM Output, Landmark # ' + str(landmark-2)
        # visualize(X, y, clf, s)

    print('Average accuracies : ', np.average(percent_correct, axis=0))
    print('Maximum accuracies : ', np.amax(percent_correct, axis=0))
    print('Minimum accuracies : ', np.amin(percent_correct, axis=0))
    t2 = time.time()
    print('Finished in ' + str(t2-t1) + ' seconds')


    print classification_report(ROCytest, ROCyscore)
    ##Make ROC Curve
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    fpr[0], tpr[0], _ = roc_curve(ROCytest,ROCyscore)
    roc_auc[0] = auc(fpr[0], tpr[0])

    plt.figure()
    plt.plot(fpr[0], tpr[0], label='ROC curve (area = %0.2f)' % roc_auc[0])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

if __name__=='__main__':
    path = 'ds1/'
    main(path)
