import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn import metrics


def binary_classification_metrics(y_true, y_pred, class1, class2):
    conf_mat = confusion_matrix(y_true=y_true, y_pred=y_pred)

    #--------accuracy, precision, recall, fscore---------
    TP = conf_mat[0,0]
    TN = conf_mat[1,1]
    FN = conf_mat[1,0]
    FP = conf_mat[0,1]

    accuracy = (TP+TN)/(TP + TN + FN + FP)
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    fscore = (2*precision*recall)/(precision+recall)

    print("accuracy: %.2f%%" % (accuracy*100))
    print("precision: %.2f%%" % (precision*100))
    print("recall: %.2f%%" % (recall*100))
    print("fscore: %.2f%%" % (fscore*100))  

    #-------------------Confusion matrix----------------
    print('Confusion matrix:\n', conf_mat)

    labels = [class1, class2]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(conf_mat, cmap=plt.cm.Blues)
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicted')
    plt.ylabel('Expected')
    plt.show()
    
#---------------------roc_auc---------------------
def rocauc(y_true, prob_y_pred):
    rocauc= roc_auc_score(y_true, prob_y_pred)    
    print("rocauc score: %.2f%%" % (rocauc*100))     

    fpr, tpr, _ = metrics.roc_curve(y_true,  prob_y_pred)
    auc = metrics.roc_auc_score(y_true, prob_y_pred)
    plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
    plt.legend(loc=4)
    plt.show()    