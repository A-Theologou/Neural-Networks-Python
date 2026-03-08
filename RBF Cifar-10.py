import numpy as np
import matplotlib.pyplot as plt         #βιβλιοθηκη για γραφηματα
from sklearn.decomposition import PCA   #pca για να μειωθεί η διασταση των δεδομένων
from sklearn.cluster import KMeans      #για την επιλογη κεντρων του RBF 
import torch                            #βιβλιοθηκες torch για να τρεξώ το νευρωνικό 
import torch.nn as nn                   #εχουν περιγραφει εκτενώς με σχολια την 1 εργασια με το mlp ο κωδικας για την αρχιτεκτονικη ειναι ιδιος
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset 
from torch.utils.tensorboard import SummaryWriter  #για να μπορω να δημιουργήσω τα διαγραμματα loss accuracy στο περιβάλλον του TensorBoard
from torch.utils.data import random_split

#για αρχη φορτώνουμε τα δεδομενα και θα εφαρμοσουμε το pca ο τροπος ειναι ο ίδιος με την 1η ενδίαμεση εργασία ή το svm της δεύτερης παραπανω σχόλια βρίσκονται εκει
#Ορισμος unpickle απο site cifar 10
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dct = pickle.load(fo, encoding='bytes')
    return dct

#ορισμος load batch οπου κανει unpickle σε καθε αρχειο batch ξεχωριστα X(50000,3072) Y(50000,1)
def load_batch():
    X_list, Y_list = [], []
    for i in range(1, 6):
        #Παιρνω τα data καθενος
        info = unpickle(f"cifar-10-batches-py/data_batch_{i}")
        X_batch = info[b'data'].astype(np.float32)
        Y_batch = np.array(info[b'labels'], dtype=np.int64) #εδω ειναι η μονο αλλαγη απο int32 σε Int64 γιατι αλλιως δε θα δουλεψει το crossenetropy loss θελει long tensor της pytorch
        #τα αποθηκευω να μη χανονται σε καθε loop στις λιστες Χ_list Y_list
        X_list.append(X_batch)
        Y_list.append(Y_batch)
    #ενωνω τις πανω λιστες στις Χ Υ που εχουν αντίστοιχα συνολικα και τα 50.000 των data και label του train
    X = np.concatenate(X_list, axis=0)
    Y = np.concatenate(Y_list, axis=0)
    return X, Y

#ορισμος του loader των test δεδομενων X_test(10000,3082), Y_test (10000,1)
def load_test():
    info = unpickle(f"cifar-10-batches-py/test_batch")
    X_test = info[b'data'].astype(np.float32)
    Y_test = np.array(info[b'labels'], dtype=np.int64)
    return X_test, Y_test


#καλω τις συναρτησεις για τα train και test  
X_train_full, Y_train = load_batch()
X_test_full, Y_test = load_test()

#κραταω μια αντιγραφη των τεστ δεδομενων πριν κανω κανονικοποιηση για την οπτικοιποιηση
X_test_visualize = X_test_full.copy()            #με το copy κανω αντιγραφη των δεδομενων σε μια καινουργια θεση μνήμης

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
#print(f"{X_train_pca.dtype}") για να δουμε τι τυπο βγαζει το pca 

print(f"Original dimensions: {X_train_full.shape[1]}") #τυπώνει τις διαστασεις πριν το pca 3072 πρεπει να δείξει
print(f"Dimensions after PCA: {X_train_pca.shape[1]}") #τυπώνει τις διαστασεις μετα το pca 

#αρχίζει το στησιμο του RBF
# οριζω ποσους κρυφούς νευρώνες θα εχω δηλαδη τα κεντρα του rbf. Αυτα τα κεντρα εχουν ορισμένες θεσεις στο χωρο των δεδομένων και οταν βαζω τις εικονες για 
#να κανω prediction βλεπει το πρόγραμμα σε ποιο κέντρο είναι κοντα όπου το καθε κεντρό εχει ενα συγκεκριμενο χαρακτηριστικο πχ μοιαζει με σκυλο
num_centers = 512

# βρισκω αυτα τα κέντρα με τον K-Means πρόκειται για unsupervised learning γιατι απλως βλέπει ποιες εικόνες μοίαζουν δεν ξέρει το label για να γνωριζει τι απεικονίζει
kmeans = KMeans(n_clusters=num_centers, random_state=42)  #o k means αλγοριθμος στην αρχη διαλεγει καποια τυχαια σημεία, με το random state του λεμε να παιρνει καθε φορα τα ίδια τυχαια καθε φορα 
                                                           #ωστε να μπορω αν κανω μια αλλαγη στο προγραμμα να ξερω αν αυτη βοηθησε ή οχι και να μην εχω καθε φορα μεγαλες διαφοροποιησεις λογω της τυχαιοτητας
kmeans.fit(X_train_pca) # τρέχω πανω στα δεδομένα τον Kmeans για να βρω τα κεντρα (ή σε υποσυνολο για να εχουμε πιο γρηορα αποτελέσματα πχ [0:10000]) επειτα το αποθηκέυει στο cluster_centers_
centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32) # μετατρέπει το τα κεντρα που ειναι σε μορφη πίνακα numpy σε torch tensor για να μπορουμε μετα να χρησιμοποιήσουμε pytorch
                                                                     #και το κανω σε float 32 οπως και τις εικόνες που εχω

#random αρχικοποιηση
#indices = np.random.choice(X_train_pca.shape[0], num_centers, replace=False)
#random_centers = X_train_pca[indices]
#centers = torch.tensor(random_centers, dtype=torch.float32)


# Μετατροπή των numpy arrays σε Pytorch Tensors για να μπουν στον DataLoader
tensor_x_train = torch.tensor(X_train_pca)#το pca βγαζει float32 Οποτε δε χρειαζομαι καποια μετατροπή
tensor_y_train = torch.tensor(Y_train)
tensor_x_test = torch.tensor(X_test_pca)
tensor_y_test = torch.tensor(Y_test)

# δημιουργω τα dataset , μεχρι τωρα ειχα μια 2 στηλες εικονες 2 στηλες labels τωρα ενωνω τις train μαζι τις test μαζι
full_train_dataset = TensorDataset(tensor_x_train, tensor_y_train)
test_dataset = TensorDataset(tensor_x_test, tensor_y_test)
#χωρισμος 80% train 20% validation
train_size = int(0.8 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

#αφου έχω φορτώσει τα αρχεία ορίζω πως θα μπαίνουν σε μοντέλο, με τι batch size δηλαδή ποσο δείγματα αν βήμα , και στο train βαζω Shuffle για να καλύτερη γενίκευση 
#στο Mlp ειχα βαλει και validation set εδω με σκοπο να δω πως δουλευει , εδω για απλοτητα το κραταω στα train και τεστ χωρις validation set
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=256)

#επιλογη που θα τρεξει απο τη στιγμή που ειναι διαθέσιμη nvidia gpu την επιλέγει και έχουμε πιο γρήγορα αποτελέσματα με το cuda, αλλιως τρέχει στον cpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device:", device)

#οριζω μια νεα κλαση που θα ειναι το μοντελο μου RBF και κληρονομεί απο το nn.module τη βασική κλάση όλων των νευρωνικών Pythorch μοντέλων 
#και μου επιτρεπει πχ να βαλώ μεσα τα nn.linear ,να αποθηκεύω τα βαρη. Αυτη η κλάση ειναι ουσιαστικά η αρχιτεκτονική του μοντέλου μου
class RBF(nn.Module):
    def __init__(self, centers):
        super().__init__()
        self.centers = nn.Parameter(centers, requires_grad=False) # τα κεντρα που εβγαλα απο το K means και του λεω να αφησει σταθερες τις τιμες
                                                                  # εδω δεν εχω 1 γραμμικο Layer για αρχη  οπως στο mlp αρχιζω με τις συκγριση αποστασεων με τα κεντρα
        self.num_centers = centers.size(0)      #αποθηκευω τον αριθμό των centers για να το χρησιμοποιησω παρακατω
        self.sigma = nn.Parameter(torch.full((self.num_centers,), 1.0), requires_grad=True) # Το ευρος σιγμα επηρεάζει ποσο αυστηρό ειναι το καθε κεντρο αρα ποσο κοντα πρεπει να ειναι καποι δεδομενο για να ενεργοιποιηθεί
                                                                                    #το αρχικοποιω στο 1 για ολα τα κέντρα αλλα του λεω να εκπαιδευτεί ωστε να καταληξουμε στις καλυτερες παραμετρους και μετα δοκιμασα 0.1 10
        self.linear = nn.Linear(self.num_centers, 10) #Το γραμμικο επίπεδο εξόδου οπου παίρνει τις ομοιοτητες που βλέπουν τα κεντρα και βγαζει logits και για τις 10 κλάσεις

    #συνάρτηση για υπολογισμό της ομοιότητας 
    def rbf_kernel(self, x, center, sigma): #δεχεται το batch των εικόνων, τις συντεταγμένες των κέντρων και το εύρος της καμπάνας
                                            #το x εχει διαστάσεις (batch,pca διαστασεις) το center(αριθμός κεντρων, pca διαστασεις)
        distance = torch.cdist(x, center) # ευκλείδια αποσταση εικόνων και κέντρων
        return torch.exp(-distance.pow(2) / (2 * sigma.pow(2))) #επιστρέφει τις ενεργοποιήσεις  του RBF με τιμες μεταξυ 0-1 που δειχνουν ποσο μοιαζει η κάθε εικόνα με το κάθε κεντρο
                                                                #και εχει σχημα (batch, αριθμός κέντρων)
                                

    def forward(self, x):  #στην forward βαζουμε τις εικόνες x και τις περνάει απο το το το RBF δίκτυο και θα μου επιστρεψει τα logits
        #οι εικονες συκγρίνονται με τα κέντρα για να βγει η ομοιότητα 
        rbf_out = self.rbf_kernel(x, self.centers, self.sigma)
        #στο τελευταιο γραμμικό επίπεδο με βάση τις ενεργοποιήσεις των κέντρων υπολογίζονται τα logits , δηλαδη για κάθε εικονα του batch 10 αριθμοί
        out = self.linear(rbf_out)
        return out

#αποθηκεύω τη κλαση του RBF στη μεταβλητη model αφου πρωτα μετακινήσω τα κέντρα και το μοντέλο να τρέξει στο gpu ή αλλιως στην cp. Τα κεντρα είναι αυτα που βρήκα απο το kmeans
model = RBF(centers=centers.to(device)).to(device)

criterion = nn.CrossEntropyLoss()#η crossentropy Ειναι η συναρτηση κόστους. Δεχεται τα logits του mlp τους κανει softmax και απο τις πιθανότητες που έχει
                                                        #βλέπει ποση πιθανότητα έδωσε το μοντέλο στη σωστή κλάση και τελος γυρίζει το loss για να κάνω στο backpropagation εκπαίδευση
optimizer = optim.Adam(model.parameters(), lr=0.01) #ο optimizer Adam με βάση τα gradients που υπολογίστηκαν απο το Loss (loss.backward) θα ενημερώσει τα βάρη του μοντέλου οταν
                                                        #καλέσω το optimizer step. Το learning rate ορίζει ποσο μεγάλη αλλαγη κάνουν τα βαρη σε κάθε βήμα εκπαίδευσης προς τη κατεύθυνση μείωσης του loss
                                                        #to lr δε θέλω ουτε να ειναι πολύ μεγάλο και να μη συγκλίνουν τα βάρη σε καποιο σημείο που ελαχιστοποιείται το loss,
                                                        #και ούτε πολυ μικρό ώστε το δίκτυο να μαθαίνει πολύ αργά.


#συναρτηση για ενα epoch για train     #ΣΗΜΑΝΤΙΚΟ: εχω εξηγησει καθε γραμμη του κωδικα στο Mlp,για να μη γεμισει ο κωδικας σχολια δε θα τα επαναλαβω εδω το epoch μου ειναι ιδιο εκτός απο το ότι λειπει εδω το validation
def run_epoch(loader, train=True):#οριζω τη run_epoch που σε μια εποχή θα τρέχει ολα τα batches του loader. Αν train=True  θα κανει forward pass
                                                        #υπολογίζει Loss κανει backprop και ενημερώνει τα βάρη. Αν Train=False για τα test απλως κάνει forward 
                                                        #pass και υπολογίζει Loss και accuracy
    model.train(train)
    total_loss, total_correct, total = 0.0, 0, 0
    
    with torch.set_grad_enabled(train):  #αν η μεταβλητη ειναι train ειναι true τοτε κραταω gradients αλλιως οχι
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            
            if train:
                optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            
            if train:
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item() * x.size(0)
            total_correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)
            
    return total_loss/total, total_correct/total

epochs = 20

writer = SummaryWriter('rbf')
for ep in range(1, epochs + 1):
    tr_loss, tr_acc = run_epoch(train_loader, train=True)
    #επειδη δεν εβαλα validation για εμεσο τροπο να δω αν γενικευει καλα το μοντελο θα βαλω το test να μη βγαινει απλως στο τελος αλλα και σε κάθε εποχη
    #ο σωστός τροπος να γίνει αυτο ειναι χωριζοντας το train σε train kai validation set οπως εγινε στο Mlp αλλα για τωρα αυτο βοηθαει για να εχω καλυτερα οπτικα αποτελεσματα
    val_loss, val_acc = run_epoch(val_loader, train=False)

    print(f"Epoch {ep:02d}, train loss {tr_loss:.4f} accuracy {tr_acc:.4f},val loss {val_loss:.4f} accuracy {val_acc:.4f}")        #εμφάνιση των αποτελεσμάτων για train loss/accuracy , test loss/accuracy
    writer.add_scalars("Loss",{ "train": tr_loss, "val":val_loss}, ep)              #δημιουργία των διαγραμμάτων στο tensorboard ενα διαγραμμα test+train loss ανα εποχή
    writer.add_scalars("Accuracy",{"train": tr_acc,"test":val_acc}, ep)              #ενα διαγραμμα test+train accuracy ανα εποχή
writer.close()


te_loss, te_acc = run_epoch(test_loader, train=False)                    #τρέχω το μοντέλο μια φορά για όλο το test 
print(f"test loss {te_loss:.4f} accuracy {te_acc:.4f}")                  #εκτυπώνω test loss/accuracy

#Οπτικοποίηση
#Το μοντελο ως κλαση βγαζει ενα αριθμο απο 0-9 αρα εδω το μεταφραζουμε σε λεξεις
classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']


def visualize_rbf(model, loader, X_test_visualize): # ιδιο με Mlp απλως προσθετω X_test_visualize που εχει τις εικονες πριν το pca
    correct_samples = [] #λιστα για να αποθηκευονται οι σωστες εικόνες
    wrong_samples = [] #λιστα για να αποθηκευονται οι λαθος εικόνες

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loader):  # η δευτερη αλλαγη ειναι το batch idx ουσιαστιακα στο Mlp στο x ηταν οι κανονικές εικόνες ενω εδώ το x ειναι μετα το pca, 
                                                     #αρα θελω το batch idx για να βρω τις εικόνες στον πίνακα πριν το pca
            x, y = x.to(device), y.to(device)
            logits = model(x)
            predictions = logits.argmax(1) # η προβλεψη ειναι αυτη η κλαση που έχει τις περισσότερες ψήφους
            
            # Βρίσκουμε πού ξεκινάει το τρέχον batch μέσα στον μεγάλο πίνακα
            start_idx = batch_idx * loader.batch_size
            
            for i in range(len(x)):
                # global_idx η θέση της εικόνας στον αρχικό πίνακα X_test_visualize
                global_idx = start_idx + i 
                
                if predictions[i] == y[i]:
                    if len(correct_samples) < 5:
                        # παινρουμε τη raw εικόνα
                        img = X_test_visualize[global_idx].reshape(3, 32, 32).transpose(1, 2, 0).astype(np.uint8)
                        correct_samples.append((img, predictions[i].item(), y[i].item()))
                else:
                    if len(wrong_samples) < 5:
                        img = X_test_visualize[global_idx].reshape(3, 32, 32).transpose(1, 2, 0).astype(np.uint8)
                        wrong_samples.append((img, predictions[i].item(), y[i].item()))

                if len(correct_samples) >= 5 and len(wrong_samples) >= 5:
                    break
            if len(correct_samples) >= 5 and len(wrong_samples) >= 5:
                break

    #εμφάνιση εικονών   
    def plot_samples(samples, title):
        plt.figure(figsize=(10, 4))#καμβας 10 χ 4 ιντσες
        plt.suptitle(title, fontsize=16)#τιτλος
        for i, (img, prediction_idx, true_idx) in enumerate(samples):
            plt.subplot(1, 5, i+1)#1 γραμμη 5 στηλες, σχεδιαζω τωρα την i+1
            #Το pytorch εχει σχημα (Channels, Height, Width) (3, 32, 32)
            #Το Matplotlib θελει (Height, Width, Channels) (32, 32, 3)
            plt.imshow(img)
            plt.axis('off')
            color_title = 'green' if prediction_idx == true_idx else 'red'
            plt.title(f'P: {classes[prediction_idx]}\nT: {classes[true_idx]}', color=color_title)
        plt.show()

    #εμφανιση των παραδειγματων
    plot_samples(correct_samples, 'Παραδείγματα Σωστής Κατηγοριοποίησης RBF')
    plot_samples(wrong_samples, 'Παραδείγματα Εσφαλμένης Κατηγοριοποίησης RBF')

#καλω τη συναρτηση 
visualize_rbf(model, test_loader, X_test_visualize)

