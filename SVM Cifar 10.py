import numpy as np
import time                             #για να μετραω το χρονο καθε προσωμοίωσης
import matplotlib.pyplot as plt         #βιβλιοθηκη για γραφηματα
from sklearn.decomposition import PCA   #pca για να μειωθεί η διασταση των δεδομένων και τρεξει πιο γρηγορα
from sklearn.svm import SVC    #βιβλιοθηκη για αλγοριθμο suppoer vector classifier

#φορτωση δεδομένων
#Ορισμος unpickle απο site cifar 10
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dct = pickle.load(fo, encoding='bytes')
    return dct

#ορισμος load batch οπου κανει unpickle σε καθε αρχειο batch ξεχωριστα   X(50000,3072) Y(50000,)
def load_batch():
    X_list, Y_list = [], []
    for i in range(1, 6):
        #Παιρνω τα data καθενος
        info = unpickle(f"cifar-10-batches-py/data_batch_{i}")
        X_batch = info[b'data'].astype(np.float32)
        Y_batch = np.array(info[b'labels'], dtype=np.int32)
        #τα αποθηκευω να μη χανονται σε καθε loop στις λιστες Χ_list Y_list
        X_list.append(X_batch)
        Y_list.append(Y_batch)
    #ενωνω τις πανω λιστες στις Χ Υ που εχουν αντίστοιχα συνολικα και τα 50.000 των data και label του train 
    X = np.concatenate(X_list, axis=0)
    Y = np.concatenate(Y_list, axis=0)
    return X, Y

#ορισμος του loader των test δεδομενων για data και labels   X_test(10000,3082), Y_test (10000,)
def load_test():
    info = unpickle(f"cifar-10-batches-py/test_batch")
    X_test = info[b'data'].astype(np.float32)
    Y_test = np.array(info[b'labels'], dtype=np.int32)
    return X_test, Y_test

#καλω τις συναρτησεις για τα train και test  
X_train_full, Y_train = load_batch()
X_test_full, Y_test = load_test()

#κραταω μια αντιγραφη των τεστ δεδομενων πριν κανω κανονικοποιηση για να τις βαλω μετα στην οπτικοιποιηση αποτελεσματων
X_test_visualize = X_test_full.copy()           #με το copy κανω αντιγραφη των δεδομενων σε μια καινουργια θεση μνήμης

#εφαρμοζω το pca

# Κανονικοποίηση στο διάστημα [0, 1] διαιρώντας με το 255 καθώς αυτές ειναι οι τιμές που παίρνουν τα pixel για καθε rgb χρώμα αυτο βοηθάει το SVM να γινει πιο γρήγορα
#καθως αφου μετραει αποστάσεις ειναι πιο ευκολο καθε φορα να υπολογίζει μεταξυ 0-1 παρα 0-255. Το ιδιο ειχα κανει και στο mlp αλλα μέσα στην εντολή transform.ToTensor
X_train_full /= 255.0
X_test_full /= 255.0

# Οριζω το PCA ωστε να κρατήσει οσες διαστάσεις χρειαζονται ωστε να διατηρηθεί το 90% της πληροφορίας, αυτο θα μειώσει αισθητα τον χρόνο εκτέλεσης
pca = PCA(n_components=0.90)
pca.fit(X_train_full)

# μετασχηματιζω τα δεδομενα με το pca ωστε να μειωθούν οι διαστάσεις
X_train_pca = pca.transform(X_train_full)
X_test_pca = pca.transform(X_test_full)

print(f"Original dimensions: {X_train_full.shape[1]}") #τυπώνει τις διαστασεις πριν το pca 3072 πρεπει να δείξει
print(f"Dimensions after PCA: {X_train_pca.shape[1]}") #τυπώνει τις διαστασεις μετα το pca 

#εκπαιδευση του SVM μιας μεθοδου μαθηματικης βελτιστοποιησης

# θα τρέξουμε 3 δομές οπου τις βαζουμε σε μια λιστα δομης (Τύπος Πυρήνα, Παράμετρος C)
# πρωτη Linear δηλαδη γραμμικός διαχωρισμός 
# δευτερη RBF (C=1) μη γραμμικός διαχωρισμος για τα περιπλοκα δεδομενα της cifar 10 περιμένω η μη γραμμικοτητα να βοηθήσει στο χωρισμό των κλάσεων
# τριτη RBF (C=10) μη γραμμικός με πιο αυστηρο C και θα δουμε αν το test πέσει σημαίνει οτι παπαγαλίζει αν βελτιωθεί το C=1 ηταν πολυ χαλαρό
parameters = [
    ('linear', 1.0),
    ('rbf', 1.0),                 
    ('rbf', 10.0)
]
# μεταβλητες για να κρατησω τις προβλεψεις του καλυτερου για την οπτικοποιηση
best_acc = 0.0          #accuracy καλυτερου
best_pred = None        # προβλεψεις του καλυτερου
#τρεξιμο της καθε αρχιτεκτονικής

for kernel_type, C_value in parameters:
    print(f"\nSVM Kernel='{kernel_type}', C={C_value} ")
    
    # ορισμος του SVM με βαση τις αναλογες παραμετρους
    svm = SVC(kernel=kernel_type, C=C_value, cache_size=1000)
    
    #Μετρηση χρόνου 
    start_time = time.time()
    # εκπαιδευση του μοντελου, βλεπει τα δεδομενα που μπαινουν και ψαχνει να βρει τα καταλληλα support vector και το αντιστοιχο ευρος
    svm.fit(X_train_pca, Y_train)
    train_time = time.time() - start_time
    
    print(f"Training time: {train_time:.2f} seconds")
    
    # προβλεψη για το test
    y_pred = svm.predict(X_test_pca)
    acc = (y_pred == Y_test).mean()
    print(f"Test Accuracy: {acc:.4f}")
    #ελεγχος αν ειναι το καλυτερο μοντελο
    if acc > best_acc:
        best_acc = acc           # ανανεωνω το καλυτερο accuracy
        best_pred = y_pred       # κραταω τις προβλεψεις για την οπτικοποίηση
    
#δειγματοληψια για οπτικοποιηση
#Το μοντελο ως κλαση βγαζει ενα αριθμο απο 0-9 αρα εδω το μεταφραζουμε σε λεξεις
classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

#η συναρτηση που δεχεται τις αρχικες εικονες , τις σωστες κλασεις και τις προβλέψεις του svm
def visualize_svm(X_raw, y_true, y_pred):
    correct_samples = [] #λιστα για να αποθηκευονται οι σωστες εικόνες
    wrong_samples = [] #λιστα για να αποθηκευονται οι λαθος εικόνες

    # θα παρουμε 5 σωστες και 5 λαθος 
    for i in range(len(y_true)):
        # Το X_raw[i] είναι ένα flat vector 3072 στοιχείων άρα  tο επαναφέρουμε σε εικόνα (3, 32, 32) και μετά transpose σε (32, 32, 3) για το γραφημα στο matplot
        #και τελος τους μετατρεπω σε οκταμπιτους αριθμους δηλαδη απο 0 - 255 οπως τα χρωματα των pixel
        img_final = X_raw[i].reshape(3, 32, 32).transpose(1, 2, 0).astype(np.uint8)


        # αν σωστη προβλεψη
        if y_pred[i] == y_true[i]:
            if len(correct_samples) < 5:
                correct_samples.append((img_final, y_pred[i], y_true[i]))
        # αν λαθος
        else:
            if len(wrong_samples) < 5:
                wrong_samples.append((img_final, y_pred[i], y_true[i]))
        
        # αν βρω 5 απο καθενα σταματαω
        if len(correct_samples) >= 5 and len(wrong_samples) >= 5:
            break

    #εμφάνιση εικονών
    def plot_samples(samples, title):
        plt.figure(figsize=(10, 4))#καμβας 10 χ 4 ιντσες
        plt.suptitle(title, fontsize=16)#τιτλος
        for i, (img, pred_idx, true_idx) in enumerate(samples):
            plt.subplot(1, 5, i+1)#1 γραμμη 5 στηλες, σχεδιαζω τωρα την i+1
            #Το pytorch εχει σχημα (Channels, Height, Width) (3, 32, 32)
            #Το Matplotlib θελει (Height, Width, Channels) (32, 32, 3)
            plt.imshow(img)
            plt.axis('off')
            #χρώμα τιτλου
            color_title = 'green' if pred_idx == true_idx else 'red'
            plt.title(f'P: {classes[pred_idx]}\nT: {classes[true_idx]}', color=color_title)
        plt.show()

    #εμφανιση των παραδειγματων
    plot_samples(correct_samples, 'Παραδείγματα Σωστής Κατηγοριοποίησης SVM')
    plot_samples(wrong_samples, 'Παραδείγματα Εσφαλμένης Κατηγοριοποίησης SVM')

# Καλούμε τη συνάρτηση οπτικοποίησης χρησιμοποιώντας τις αρχικές εικόνες (X_test_images)
# και τις προβλέψεις του καλύτερου μοντέλου (best_pred)
visualize_svm(X_test_visualize, Y_test, best_pred)