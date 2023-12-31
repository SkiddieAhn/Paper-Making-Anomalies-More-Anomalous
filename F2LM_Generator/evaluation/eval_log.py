import matplotlib.pyplot as plt
import numpy as np

def save_text(text, file_path):
    with open(file_path, 'a+') as file:  
        file.write(text + "\n") 


def save_graph(answers_idx, scores, file_path, threshold=-1, x='Frame', y='PSNR'):
    length = len(scores)
    plt.clf()
    plt.plot([num for num in range(length)],[score for score in scores]) # plotting
    plt.bar(answers_idx, max(scores), width=1, color='r',alpha=0.5) # check answer
    plt.xlabel(x)
    plt.ylabel(y)
    if threshold != -1:
        plt.axhline(threshold, color='green', linestyle='--', label='Optimal Threshold')
    plt.savefig(file_path)


def save_auc_graph_test(fpr, tpr, auc, file_path, eer=None):
    plt.clf()
    plt.plot(fpr, tpr, label=f'ROC curve (area = {auc:.3f})')
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel('False Positive Rate (FRR)')
    plt.ylabel('True Positive Rate (TRR)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    if eer != None:        
        plt.scatter([eer], [1 - eer], color='r', marker='o', label=f'EER ({eer:.3f})')
    plt.legend()
    plt.savefig(file_path)


def save_auc_graph_train(iters, scores, file_path):
    plt.clf()
    plt.plot(iters, scores, c='royalblue') # auc

    # check best score
    scores_np = np.array(scores)
    best_idx = np.argmax(scores_np)
    best_itr = iters[best_idx]
    best_score = scores[best_idx]
    plt.scatter([best_itr],[best_score],c='darkorange',s=25, edgecolors='royalblue')
    plt.text(best_itr, best_score, f'{best_itr}: {best_score:.3f}', ha='left', va='bottom')

    plt.xlabel('Iteration')
    plt.ylabel('AUC')
    plt.savefig(file_path)


def save_psnr_graph_train(iters, all_scores, n_scores, an_scores, file_path):
    plt.clf()
    plt.plot(iters, all_scores, c='darkviolet', label="All") # all score
    plt.plot(iters, n_scores, c='royalblue', label="Normal") # normal score
    plt.plot(iters, an_scores, c='red', label="Anomaly")     # anomaly score

    # check best score
    all_scores_np = np.array(all_scores)
    best_idx = np.argmax(all_scores_np)
    best_itr = iters[best_idx]

    best_all_score = all_scores[best_idx]
    best_n_score = n_scores[best_idx]
    best_an_score = an_scores[best_idx]

    plt.scatter([best_itr],[best_all_score],c='darkorange',s=25, edgecolors='darkviolet')
    plt.text(best_itr, best_all_score, f'{best_itr}: {best_all_score:.3f}', ha='left', va='bottom')
    plt.scatter([best_itr],[best_n_score],c='darkorange',s=25, edgecolors='darkviolet')
    plt.text(best_itr, best_n_score, f'{best_itr}: {best_n_score:.3f}', ha='left', va='bottom')
    plt.scatter([best_itr],[best_an_score],c='darkorange',s=25, edgecolors='darkviolet')
    plt.text(best_itr, best_an_score, f'{best_itr}: {best_an_score:.3f}', ha='left', va='bottom')

    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('PSNR Avg')
    plt.savefig(file_path)


def save_dist_graph_train(iters, n_scores, an_scores, file_path):
    plt.clf()

    # draw graph 
    plt.plot(iters, n_scores, c='royalblue', label="Normal") # normal score
    plt.plot(iters, an_scores, c='red', label="Anomaly")     # anomaly score

    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('Distance (Avg PSNR)')
    plt.savefig(file_path)