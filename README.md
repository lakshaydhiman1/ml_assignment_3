# ml_assignment_3
Ques 1 
    ======================Using Equation==========================
    accuracy for fold: 1 91.57894736842105
    accuracy for fold: 2 90.0
    accuracy for fold: 3 91.005291005291
    Overall accuracy for logistic 90.86141279123734
    ======================Using Autograd==========================
    accuracy for fold: 1 92.10526315789474
    accuracy for fold: 2 88.42105263157895
    accuracy for fold: 3 88.88888888888889
    Overall accuracy for logistic 89.80506822612085

Ques 2
    ======================Printing for L1 normalized==========================
    lambdas [0.0001, 0.001, 0.1, 1, 5, 10, 50, 100, 500, 1000]
    Accuracies for L1 Norm [62.7438967789845, 62.7438967789845, 62.7438967789845, 54.322844147405554, 56.57105727281166, 44.624524273647076, 46.60354590179151, 53.39645409820849, 45.20003712986169, 62.7438967789845]

    Theta for large value of lambda = 20000
    Theta =  array([5.81718008, 1.13498375, 1.23207181, 1.85113163, 4.43192514,
       1.00128616, 1.00076896, 1.00010024, 1.00004423, 1.00237209,
       1.00090042, 1.00295815, 1.01956408, 1.01936388, 1.09347548,
       1.00011676, 1.00025651, 1.00037625, 1.0001183 , 1.00031338,
       1.00005553, 1.14165204, 1.29959326, 1.89582391, 4.27386347,
       1.00172172, 1.00134261, 1.0008613 , 1.00044625, 1.00349786,
       1.00106084])

    Increasing lambda further led to overflow. 
    

    We can see that theta are close even with large values of lambda

    ====================== Printing for L2 normalized ==========================
    Accuracies for L2 Norm [62.7438967789845, 62.7438967789845, 62.7438967789845, 62.7438967789845, 62.7438967789845, 62.7438967789845, 62.7438967789845, 62.7438967789845, 62.7438967789845, 62.7438967789845]

Ques 3
    The accuracy results while testing
        accuracy for fold: 1 96.44444444444444
        accuracy for fold: 2 96.65924276169265
        accuracy for fold: 3 94.87750556792874
        accuracy for fold: 4 96.88195991091314
        Overall accuracy for logistic 96.21578817124475
    The data when visualized using only 2 most component we can see that data is very well clustered and seperated

    Confusion Matrix for digits

        TP [53, 31, 46, 42, 37, 54, 41, 45, 46, 38]
        TN [395, 414, 401, 402, 408, 393, 407, 403, 395, 407]
        FP [0, 0, 2, 1, 2, 1, 1, 0, 7, 2]
        FN [1, 4, 0, 4, 2, 1, 0, 1, 1, 2]

    ith value is the number for ith digit is the value . We see 8 is the most confused digit and 0 is the least confused digit.

Ques 4
    Train time complexity for logistic regression is O(ND)
    Space complexity for logitic regression is O(D)
    Where D is the dimension of the input and N is the number of samples.

Ques 6.

    I trained a neural network Model with 2 hiddden with 2,5 hidden nodes for boston price dataset the following is the result
    Root Mean Square Error for a fold 5.967727763043771
    Root Mean Square Error for a fold 5.569464056443299
    Root Mean Square Error for a fold 6.178288112271409
    Overall Mean square error for Network 5.9051599772528265 

    For digit dataset accuracy of 16% with 2 hidden layer with 3 and 5 nodes respectively, activation function relu and sigmoid.