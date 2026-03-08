import numpy as np
import time


#Ορισμος unpickle απο site cifar 10
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dct = pickle.load(fo, encoding='bytes')
    return dct


#ορισμος load batch οπου κανει unpickle σε καθε αρχειο batch ξεχωριστα   X(50000,3072) Y(50000,1)
def load_batch():
    X_list, Y_list = [], []
    for i in range(1,6):
       #Παιρνω τα data καθενος
       info = unpickle(f"cifar-10-batches-py/data_batch_{i}")
       X_batch = info[b'data'].astype(np.float32)
       Y_batch = np.array(info[b'labels'],dtype=np.int32)
       #τα αποθηκευω να μη χανονται σε καθε loop στις λιστες Χ_list Y_list
       X_list.append(X_batch)
       Y_list.append(Y_batch)
    #ενωνω τις πανω λιστες στις Χ Υ που εχουν αντίστοιχα συνολικα και τα 50.000 των data και label του train   
    X = np.concatenate(X_list, axis=0)   
    Y = np.concatenate(Y_list, axis=0)   
    return X, Y

#ορισμος του loader των test δεδομενων για data και labels   X_test(10000,3082), Y_test (10000,1)
def load_test():
    info = unpickle(f"cifar-10-batches-py/test_batch")
    X_test = info[b'data'].astype(np.float32)          
    Y_test = np.array(info[b'labels'],dtype=np.int32) 
    return X_test, Y_test

#καλω τις συναρτησεις για τα train και test  
X, Y = load_batch()          
X_test, Y_test = load_test() 

#ελεγχος για να βεβαιωθω οτι εχουν σωστες διαστασεις και περασαν ολα τα δεδομένα σωστά στο πρόγραμμα
# print(X.shape)         
# print(Y.shape)          
# print(X_test.shape) 
# print(Y_test.shape) 

# επιλογη για λιγοτερα δειγματα, τα πρωτα απο καθε λιστα ώστε να μη χρειαζεται καθε φορα να τρεχω ολα τα δεδομένα που θα αργούσε
X_small = X[:50000]
Y_small = Y[:50000]
X_test_small  = X_test[:10000]
Y_test_small  = Y_test[:10000]

#οριζω την συνάρτηση που υπολογίζει τη ευκλείδια αποσταση, παίρνω το αποτέλεσμα απο τη διαφορά τετραγώνων ταυτοτητα,
#  καθως συμφωνα με βιβλιογραφία ειναι πιο αποδοτικο + δεν εβγαινε σωστο με απλή αφαίρεση
def distances(A, B):
    A2 = (A**2).sum(1)[:, None]     
    B2 = (B**2).sum(1)[None, :]     
    AB = A @ B.T                                 
    D2 = A2 + B2 - 2*AB
    return np.sqrt(D2)   

#αρχιζω μετρησιες για 1ΝΝ
print(f'calculating 1NN')
start_D = time.time()
D = distances(X_test_small, X_small)            #παιρνω αποσταση των test και train για τα Χ
end_D=time.time()
D_time= end_D-start_D
start_1nn = time.time()
nn_idx = D.argmin(axis=1)                       # δείκτης του κοντινότερου γείτονα για κάθε test
Y_pred = Y_small[nn_idx]                        # κανω predict ότι το label t του τεστ είναι το label του κοντινότερου

acc = (Y_pred == Y_test_small).mean()           #μεσος όρος οπου το predicted label είναι ίσο με το label του test
print("1-NN accuracy:", round(float(acc), 3))   
end_1nn = time.time()
print(f'1NN time: {(end_1nn-start_1nn)+D_time:.2f}seconds')

#3NN
k = 3
start_3NN=time.time()
# δείκτης των k μικρότερων αποστάσεων ,σε αυξουσα σειρά
idx_k = np.argsort(D, axis=1)[:, :k]   

#δημιουργώ άδειο πινακα για να μπουν μετα τα δεδομένα
Y_pred_3nn = np.empty(Y_test_small.shape[0])

#ο κώδικας που αποφασίζει ποιο είναι το label της πλειοψηφίας
for i, neigh in enumerate(idx_k):
     labs = Y_small[neigh]  # labels των k κοντινότερων αυξουσα

     if np.unique(labs).size == k:
         # αν και οι 3 κλασεις διαφορετικες παιρνω το πιο κοντινο γείτονα
         Y_pred_3nn[i] = labs[0]
     else:
         # με πλειοψηφια
         votes = np.bincount(labs, minlength=10)  
         Y_pred_3nn[i] = votes.argmax()

# accuracy 3Nearest Neighbor
acc_3NN = (Y_pred_3nn == Y_test_small).mean()
print("3-NN accuracy:", round(float(acc_3NN), 3))
end_3NN=time.time()
print(f'3NN time: {(end_3NN-start_3NN)+D_time:.2f}seconds')
#NCC , Nearest Class Centroid
print(f'calculating NCC')
start_NCC = time.time()
K = 10  
centroids = np.vstack([X_small[Y_small == c].mean(axis=0) for c in range(K)])  

# αποστασεις και προβλεψη
D_cent = distances(X_test_small, centroids)   
Y_pred_ncc = D_cent.argmin(axis=1)            

# accuracy
acc_ncc = (Y_pred_ncc == Y_test_small).mean()
print("NCC accuracy:", round(float(acc_ncc), 3))
end_NCC = time.time()
print(f'NCC time: {end_NCC-start_NCC:.2f}seconds')
