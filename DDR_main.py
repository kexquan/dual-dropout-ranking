import numpy as np
import os
from src.FeatureSelector import FeatureSelector
from DataGenerator import generate_data, get_one_hot
from sklearn.preprocessing import StandardScaler
from itertools import repeat
from torch.multiprocessing import Pool
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import argparse
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_selection import f_classif, mutual_info_classif
from tensorflow.keras import datasets
import torch
import cv2
import warnings
warnings.filterwarnings('ignore')

def mutual_info(trn_feats, trn_labels, fs):

    importances = mutual_info_classif(trn_feats, trn_labels, random_state=0)
    index_selected = np.argsort(-importances)[0:fs]

    return trn_feats[:, index_selected], index_selected, importances

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def classifier(kernel, C):
    return SVC(kernel=kernel, C=C, probability=True, gamma='auto', random_state=0)
def classification(X_train, y_train, X_test, y_test, args):
    clf = classifier(args.kernel_type, args.SVM_C)
    clf.fit(X_train, y_train)
    pre_dict_label = clf.predict(X_test)
    predict_pro_tst = clf.predict_proba(X_test)
    return pre_dict_label, predict_pro_tst

def Standardize(X_train, X_test):
    std = StandardScaler().fit(X_train)
    X_train = std.transform(X_train)
    X_test = std.transform(X_test)
    return X_train, X_test

def cross_val_index_split(trn_feats, n_folds):
    cv_outer = KFold(n_splits=n_folds, shuffle=True, random_state=0)
    train_index_all = []
    test_index_all = []
    for train_index, test_index in cv_outer.split(trn_feats):
        train_index_all.append(train_index)
        test_index_all.append(test_index)
    return train_index_all, test_index_all

def get_nfs_top_rank_index(rank, nfs):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    preserver_rate = sigmoid(rank)
    rank_index = np.argsort(-preserver_rate)
    return rank_index[0: nfs]

def cross_val_run(X, y, train_ix, test_ix, args):

    X_train, X_test = X[train_ix, :], X[test_ix, :]
    y_train, y_test = y[train_ix], y[test_ix]
    X_train, X_test = Standardize(X_train, X_test)
    importances = DDR_run(X_train, y_train, args)

    feature_selected = get_nfs_top_rank_index(importances, args.num_fs)          # The selected features

    predicted_label, _ = classification(X_train[:, feature_selected], y_train, X_test[:, feature_selected], y_test, args) # Classification
    ACC = accuracy_score(y_test, predicted_label)

    return ACC, importances, feature_selected

def cross_validation(X, y, args):

    importances_all = []
    ACC_all = []
    feature_selected_all = []

    train_index_all, test_index_all = cross_val_index_split(X, args.n_folds)

    if args.multi_thread:
        # Run folds in multi-threading
        thread_agrs = list(zip(repeat(X), repeat(y), train_index_all, test_index_all, repeat(args)))
        pool = Pool(args.num_workers)
        results = pool.starmap(cross_val_run, thread_agrs)
        for result in results:
            ACC_all.append(result[0])
            importances_all.append(result[1])
            feature_selected_all.append(result[2])
    else:
        # Run each fold
        for i in range(len(train_index_all)):
            ACC, importances, feature_selected = cross_val_run(X, y, train_index_all[i], test_index_all[i], args)
            ACC_all.append(ACC)
            importances_all.append(importances)
            feature_selected_all.append(feature_selected)

    return ACC_all, importances_all, feature_selected_all

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def DDR_run(train_features, train_labels, args):

    data_batch_size = args.data_batch_size
    mask_batch_size = args.mask_batch_size
    init_dropout_rate = args.init_dropout_rate
    phase_2_start = args.phase_2_start
    max_batches = args.max_batches

    operator_arch = args.operator_arch
    selector_arch = args.selector_arch

    FEATURE_SHAPE = (train_features.shape[1],)

    X_tr = train_features
    y_tr = train_labels

    y_tr = get_one_hot(y_tr.astype(np.int8), len(np.unique(train_labels)))  # Get one-hot labels

    fs = FeatureSelector(FEATURE_SHAPE, data_batch_size, mask_batch_size,
                         epoch_on_which_selector_trained=args.epoch_on_which_selector_trained)

    fs.create_dense_operator(operator_arch)                      # Create operator

    fs.create_dense_selector(selector_arch, init_dropout_rate)   # Create operator

    fs.create_mask_optimizer(epoch_condition=phase_2_start)

    fs.train_networks_on_data(X_tr, y_tr, max_batches)           # Train

    preserve_logit_p = fs.get_dropout_logit_p()

    return preserve_logit_p

def normalize_0_1(importances_):
    return (importances_ - importances_.min()) / (importances_.max() - importances_.min())

if __name__ == '__main__':

    # python DDR_main.py --run_example1 --operator_arch 128 32 4 --num_fs 3  --multi_thread
    # python DDR_main.py --run_example2 --operator_arch 128 32 2 --num_fs 50 --multi_thread
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_example1', action="store_true")              # XOR synthetic dataset classification
    parser.add_argument('--num_training_samples', type=int, default=2500)   # Number of training samples in XOR datset
    parser.add_argument('--run_example2', action="store_true")              # Mnist hand-written digit feature importance visulization
    parser.add_argument('--num_fs', type=int, default=3)                    # Number of selected features in each fold
    parser.add_argument('--n_folds', type=int, default=5)                   # Number of folds in cross-validations
    parser.add_argument('--kernel_type', type=str, default="rbf")           # Kernel type in SVM
    parser.add_argument('--SVM_C', type=float, default=1.0)                 # C in SVM
    parser.add_argument('--multi_thread', action="store_true")              # Run in multi-threading
    parser.add_argument('--num_workers', type=int, default=5)               # Number of workers in multi-threading
    # parameter for DDR
    parser.add_argument('--data_batch_size', type=int, default=32)                        # Batch size
    parser.add_argument('--mask_batch_size', type=int, default=32)                        # The size of the dropout mask subset, e.g., |Z|
    parser.add_argument('--init_dropout_rate', type=float, default=0.35)                  # Initializing dropout rates
    parser.add_argument('--phase_2_start', type=int, default=6000)
    parser.add_argument('--max_batches', type=int, default=25000)                         # Number of iterations
    parser.add_argument('--epoch_on_which_selector_trained', type=int, default=2)
    parser.add_argument('--operator_arch', nargs='+', type=int, default=[128, 32, 4])     # Operator's architecture
    parser.add_argument('--selector_arch', nargs='+', type=int, default=[128, 32, 1])     # Selector's architecture

    parser.add_argument('--CUDA_VISIBLE_DEVICES', type=str, default="0")
    parser.add_argument('--seed', type=int, default=8888)                                 # seed
    args = parser.parse_args()

    torch.set_default_dtype(torch.float64)

    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))
    for CUDA_VISIBLE_DEVICES in args.CUDA_VISIBLE_DEVICES:
        os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
    seed_everything(args.seed)

    if args.run_example1:
        # XOR synthetic dataset classification
        X_tr, y_tr = generate_data(n=args.num_training_samples, seed=args.seed)
        ACC_all, importances_all, feature_selected_all = cross_validation(X_tr, y_tr, args)
        print("Cross-validation ACC mean and std: %.3f %.3f" % (np.mean(ACC_all), np.std(ACC_all)))

        fontsize = 15
        importance_mean = np.array(importances_all).mean(axis=0).flatten()
        importance_mean = sigmoid(importance_mean)

        count = pd.value_counts(np.concatenate((feature_selected_all)))
        count_index = np.array(count.index)
        count_values = np.array(count.values)
        count_values_sorted = count_values[np.argsort(count_index)]
        count_index_sorted = count_index[np.argsort(count_index)]
        complete_index = []
        complete_values = []

        j = 0
        for i in range(10):
            complete_index.append(i)
            if i in count_index_sorted:
                complete_values.append(count_values_sorted[j])
                j = j + 1
            else:
                complete_values.append(0)

        colours = []
        for i in range(10):
            if complete_values[i] == args.n_folds:
                colours.append('r')
            else:
                colours.append('g')

        plt.bar(x=range(10), height=importance_mean, align="center", color=colours)
        plt.ylabel('Feature importance', fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.xticks(range(10), fontsize=fontsize)
        plt.tight_layout()
        plt.savefig('XOR_classification.pdf', dpi=600)
        plt.show()

    elif args.run_example2:
        # Mnist hand-written digit feature importance visulization
        (x_train, y_train), (_, _) = datasets.mnist.load_data()
        digit3_index = np.where(y_train == 3)[0]
        digit8_index = np.where(y_train == 8)[0]

        train_data = np.concatenate((x_train[digit3_index, :], x_train[digit8_index, :]), axis=0).reshape(-1, 784).astype(np.float32) / 255
        train_labels = np.concatenate((np.zeros(shape=digit3_index.shape), np.ones(shape=digit8_index.shape)))

        ACC_all, importances_all, feature_selected_all = cross_validation(train_data, train_labels, args)
        print("Cross-validation ACC mean and std: %.3f %.3f" % (np.mean(ACC_all), np.std(ACC_all)))
        np.save('importances_all.npy', np.array(importances_all))

        # save feature importance map obtained from DDR
        importances_ = np.array(importances_all).mean(axis=0)
        importances_ = normalize_0_1(importances_)
        importances_reshaped = np.reshape(importances_, (28, 28))
        sc = plt.imshow(importances_reshaped)
        plt.axis('off')
        plt.savefig('feature_importance_map1.jpg', dpi=600, bbox_inches='tight', pad_inches=0)
        sc.set_cmap('hot')
        plt.savefig('feature_importance_map2.jpg', dpi=600, bbox_inches='tight', pad_inches=0)

        # save the mean images of digit ‘3’
        digit3_image = normalize_0_1(x_train[digit3_index, :].mean(axis=0))
        sc = plt.imshow(digit3_image)
        sc.set_cmap('Greens')
        plt.axis('off')
        plt.savefig('digit3_mean.jpg', dpi=600, bbox_inches='tight', pad_inches=0)

        # save the mean images of digit ‘8’
        digit3_image = normalize_0_1(x_train[digit8_index, :].mean(axis=0))
        sc = plt.imshow(digit3_image)
        sc.set_cmap('Greens')  # Greens, Blues, Purples
        plt.axis('off')
        plt.savefig('digit8_mean.jpg', dpi=600, bbox_inches='tight', pad_inches=0)

        # Feature importance map is superimposed on the mean images of digit ‘3’
        img1 = cv2.addWeighted(cv2.imread('feature_importance_map2.jpg'), 0.5, cv2.imread('digit3_mean.jpg'), 0.5, 0)
        plt.imshow(img1)
        plt.axis('off')
        plt.savefig('digit3_mean_plus_feature_importance_map.jpg', dpi=600, bbox_inches='tight', pad_inches=0)

        # Feature importance map is superimposed on the mean images of digit ‘8’
        img2 = cv2.addWeighted(cv2.imread('feature_importance_map2.jpg'), 0.5, cv2.imread('digit8_mean.jpg'), 0.5, 0)
        plt.imshow(img2)
        plt.axis('off')
        plt.savefig('digit8_mean_plus_feature_importance_map.jpg', dpi=600, bbox_inches='tight', pad_inches=0)

        # show images:
        print("")
        print("left:   The feature importance map, \n"
              "middle: The feature importance map uperimposed on the mean images of digit ‘3’, \n"
              "right:  The feature importance map uperimposed on the mean images of digit ‘8’")
        plt.subplot(1, 3, 1)
        img1 = cv2.imread('feature_importance_map1.jpg')
        plt.imshow(img1[:, :, [2, 1, 0]])
        plt.axis('off')

        plt.subplot(1, 3, 2)
        img2 = cv2.imread('digit3_mean_plus_feature_importance_map.jpg')
        plt.imshow(img2[:, :, [2, 1, 0]])
        plt.axis('off')

        plt.subplot(1, 3, 3)
        img3 = cv2.imread('digit8_mean_plus_feature_importance_map.jpg')
        plt.imshow(img3[:, :, [2, 1, 0]])
        plt.axis('off')

        plt.tight_layout()
        plt.savefig('mnist_feature_importance.pdf', dpi=600)
        plt.show()
