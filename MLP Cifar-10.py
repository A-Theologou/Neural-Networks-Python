import torch   #το ετοιμο framework της pytorch για machine learning
import torch.nn as nn #περιεχει ετοιμα στοιχεια για να χτισω τα layer του nn οπως πχ ενα Linear layer , ReLu
from torch.utils.data import DataLoader, random_split, Subset  #Data loader με βοηθάει να χειριστώ το DataSet δινοντας λειτουργίες οπως χωρισμο σε batch, shuffle,
                                                               #randomsplit σπαει dataset σε κομματια, Subset φτιανει καινουργιο data set με στοιχεια απο καποιο μεγαλύτερο
from torchvision import datasets, transforms   #datasets εχει διαφορα σετ δεδομενων οπως Cifar 10
                                               #transforms προσφερει διαφορους μετασχηματισμους για τα δεδομένα που εχω όπως πχ να κανω rotate τις εικονες
import torch.optim as optim                    #πακετο με διαφορους ετοιμους optimizers οπως ο Adam που ανανεώνουν τα βάρη του μοντέλου με βαση της κλίσης του loss
from torch.utils.tensorboard import SummaryWriter  #για να μπορω να δημιουργήσω τα διαγραμματα loss accuracy στο περιβάλλον του TensorBoard
import matplotlib.pyplot as plt
#το compose βαζει μαζι διαφορους μετασχηματισμους στα δεδομενα 
#το .ToTensor μετατρέπει την καθε εικόνα numpy του Cifar 10 (32x32x3) σε pytorch tensor που ειναι ενας τανυστης (3x32x32) 
#και μετατρέπει τον οκταμπιτο των Pixel uint8 με τιμες 0-255 σε float 32 με τιμή απο 0-1
transform_plain = transforms.Compose([transforms.ToTensor()])    

#φορτώνει τα δεδομένα του Cifar10 στο pc ή απο το ιντερνετ με βοηθεια του torchvision, ειναι 50.000 για train και 10.000 για test 
# (root βρισκει φακελο που κατεβηκαν, χωριζονται με το train=True ή False αντιστοιχα, download τα κατεβάζει αν δε τα βρει στον φακελο του root ,μετασχιματίζω ειτε με To.Tensor ή augmentations)                                      
full_train_plain = datasets.CIFAR10(root='./data', train = True , download = True , transform = transform_plain)
test_data = datasets.CIFAR10(root='./data', train = False , download = True , transform = transform_plain)

#χωρισμος data σε train και validation
train_size = int(0.8*len(full_train_plain))
val_size = len(full_train_plain)-train_size
train_plain, val_data = random_split(full_train_plain,[train_size, val_size])
train_data = train_plain   

#αν ηθελα με augmentations απλως εβγαζα τα # απο τον παρακάτω κώδικα
#Κώδικας για να προσθέσω και augmentations στο train συγκεκριμενα οριζόντιο flip τυχαίων εικονών 
#και τυχαία μετατοπίζει όλες τις εικόνες του train προσθέτοντας ενα πλαισιο Pixel γυρω απο την εικόνα και μετά κοβοντας ενα παράθυρο 32x32 

#transform_train_aug = transforms.Compose([
    #transforms.RandomHorizontalFlip(),
    
    #transforms.ToTensor()
    #])

#Δε θελω να βαλω augmentations και στο validation dataset άρα ξαναφορτώνω ολα τα 50.000 με ολα τα augmentations τωρα στο full_train_aug, 
# επειτα απο το full_train_aug παίρνω μονο το υποσετ μοιράζεται ίδιους δείκτες με αυτους του train_plain μετα το randomsplit. Αρα εχω train με aug. και val  χωρις aug.
#full_train_aug = datasets.CIFAR10(root='./data', train = True , download = True , transform = transform_train_aug)
#train_data = Subset(full_train_aug, train_plain.indices)                

#αφου έχω φορτώσει τα αρχεία ορίζω πως θα μπαίνουν σε μοντέλο, με τι batch size δηλαδή ποσο δείγματα αν βήμα , και στο train βαζω Shuffle για να καλύτερη γενίκευση 
train_loader = DataLoader(train_data, batch_size = 64, shuffle = True)                 
val_loader = DataLoader(val_data, batch_size = 256)                    #το test batch size δε παιζει ρολο στο accuracy αφου ειναι μετα το train και επομένως δεν αλλαζουν τα βαρη πλέον
test_loader = DataLoader(test_data, batch_size=256)                    #αρα βαζω οσο batch size αντεχει για πιο γρηγορα, και το shuffle για τον ίδιο λόγο δε χρειάζεται
                                                                                       

#επιλογη που θα τρεξει απο τη στιγμή που ειναι διαθέσιμη nvidia gpu την επιλέγει και έχουμε πιο γρήγορα αποτελέσματα με το cuda, αλλιως τρέχει στον cpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device:",device)

#οριζω μια νεα κλαση που θα ειναι το μοντελο μου MLP και κληρονομεί απο το nn.module τη βασική κλάση όλων των νευρωνικών Pythorch μοντέλων 
#και μου επιτρεπει πχ να βαλώ μεσα τα nn.linear ,να αποθηκεύω τα βαρη. Αυτη η κλάση ειναι ουσιαστικά η αρχιτεκτονική του μοντέλου μου
class MLP(nn.Module):                                     
    def __init__(self, hidden_dim=512):                    #αρχικοποιηση του μοντέλου με τη μέθοδο init, ορίζω ως self το αντικείμενο της κλασης Mlp και ορισμός των νευρώνων του κρυφού layer
        super().__init__()                             #βοηθα να τρεξει σωστα η υποκλαση που εχω καλώντας την αρχικοποίηση της υπερκλασης  nn.module
        self.net = nn.Sequential(                      #οριζω ως ιδιοτητα της self την .net όπου αντιστοιχεί σε δίκτυο που θα περνάει τα δεδομένα μας απο τα διαδοχικά Layers που ορίζω
            nn.Flatten(),                              #flatten σε διανυσμα 3072 στοιχείων
            nn.Linear(3072,hidden_dim),                 #το αρχικο γραμμικό layer συνδέει και τα 3072 στοιχεια μου με καθε νευρώνα του  Hidden layer αρα fully connected
            nn.ReLU(inplace=True),                      #συναρτηση ενεργοποιησης relu για να εισαχθεί μη γραμμικοτητα
            #nn.Linear(hidden_dim,hidden_dim),
            #nn.ReLU(inplace=True),
            nn.Linear(hidden_dim,10)                     #fully connected κρυφό layer με εξοδο 10 νευρωνες οσες οι κλασεις
        )

    def forward(self, x):                                #στην forward βαζουμε τις εικόνες x και τις περνάει απο το το διαδοχικό δίκτυο της self.net και θα μου επιστρεψει τα logits
        return self.net(x)                               #αυτο στη ουσία αποτελεί το forward pass μου


model = MLP().to(device)                                #αποθηκεύω το τη κλαση του mlp στη μεταβλητη model αφου πρωτα την εχω βάλει να τρέξει στο gpu ή αλλιως στην cpu

criterion = nn.CrossEntropyLoss()                       #η crossentropy Ειναι η συναρτηση κόστους. Δεχεται τα logits του mlp τους κανει softmax και απο τις πιθανότητες που έχει
                                                        #βλέπει ποση πιθανότητα έδωσε το μοντέλο στη σωστή κλάση και τελος γυρίζει το loss για να κάνω στο backpropagation εκπαίδευση

optimizer = optim.Adam(model.parameters(),lr=1e-3)      #ο optimizer Adam με βάση τα gradients που υπολογίστηκαν απο το Loss (loss.backward) θα ενημερώσει τα βάρη του μοντέλου οταν
                                                        #καλέσω το optimizer step. Το learning rate ορίζει ποσο μεγάλη αλλαγη κάνουν τα βαρη σε κάθε βήμα εκπαίδευσης προς τη κατεύθυνση μείωσης του loss
                                                        #to lr δε θέλω ουτε να ειναι πολύ μεγάλο και να μη συγκλίνουν τα βάρη σε καποιο σημείο που ελαχιστοποιείται το loss,
                                                        #και ούτε πολυ μικρό ώστε το δίκτυο να μαθαίνει πολύ αργά.

#συναρτηση για ενα epoch για train ή validation
def run_epoch(loader,train=True):                       #οριζω τη run_epoch που σε μια εποχή θα τρέχει ολα τα batches του loader. Αν train=True  θα κανει forward pass
                                                        #υπολογίζει Loss κανει backprop και ενημερώνει τα βάρη. Αν Train=False για τα validation και test απλως κάνει forward 
                                                        #pass και υπολογίζει Loss και accuracy
    
    model.train(train)                                  #το .train είναι μέθοδος της nn.module και βαζει το μοντελο μου σε mode train ή validation/test ωστόσο 
                                                        #στη δικο μας Mlp η δομη linear - relu - linear δεν αλλαζει στο train και test αλλα αν ειχε πχ dropout 
                                                        #θα το απενεργοποιούσε το validation/test 

    total_loss, total_correct, total = 0.0,0,0          #αρχικοποιώ μηδενίζοντας τις μεταβλητες καθε εποχής: συνολικό Loss ,συνολικα δειγματα που ταξινομηθηκαν σωστα,συνολικά δείγματα 
    grad_state = torch.enable_grad() if train else torch.no_grad()   #αν ειναι train υπολογίζω gradients στο backprop αν είναι val/test δε τα υπολογίζω χωρίς λόγο
    with grad_state:
        for x,y in loader:                              #το cifar 10 μου επιστρέφει  σε καθε index ενα tuple (image, label), αρα το (x,y) μου ειναι ενα τετοιο batch απο εικόνες και labels
            x,y = x.to(device), y.to(device)            #μοντελο και δεδομενα στο ιδιο device gpu ή cpu
            if train:
                optimizer.zero_grad(set_to_none=True)   #μηδενίζει τα gradient απο το προηγούμενο batch(με το set to none αντι για μηδεν τα αφήνει κενό για λιγότερη μνήμη)
            logits = model(x)                           #περνά το x batch απο το forward pass και παίρνει τα logits με σχημα (Batchsize,10)δηλαδη καθε εικόνα βγάζει σκορ για κάθε label
            loss = criterion(logits, y)                 #μετα το forward συγκρίνω Logits με τα labels με την crossentropy και βγαζω ένα Loss
            if train:
                loss.backward()                         #για το train μετα το Loss αρχίζει το Backprop υπολογίζοντας τα gradients
                optimizer.step()                        #μετα το Backprop ο optimizer ενημερώνει τις τιμές των βαρών
            total_loss +=loss.item() *x.size (0)        #συνολικο Loss ειναι το αθροισμα του (loss στο batch)* (αριθμού δειγμάτων στο batch) για ολα τα batch   
            total_correct += (logits.argmax(1) == y).sum().item() #αθροισμα των πόσων ταξινομήθηκαν σωστα στο ενα batch για ολα τα Batch
            total += x.size(0)                                    #αθροισμα των πόσων δειγμάτων περάσαν σε όλο το epoch
    return total_loss/total, total_correct/total                  #επιστρέφει το μεσο loss και μεση accuracy για την εποχή (μπορει να ειναι train/val/test)

epochs = 10                                            #αριθμός εποχών

writer = SummaryWriter("mlp_cifar10")                   #ο φακελος που θα αποθηκεύονται τα δεδομένα για τα διαγράμματα στο Tensorboard

for ep in range (1,epochs + 1):                         #τρέχω το μοντέλο για κάθε εποχή , για το train και το validation αντίστοιχα
    tr_loss, tr_acc = run_epoch(train_loader, train=True)             #γυρναει μεσο train loss ,μεση train accuracy
    val_loss, val_acc = run_epoch(val_loader, train=False)            #γυρναει μεσο val loss ,μεση test accuracy

    print(f"Epoch {ep:02d}, train loss {tr_loss:.4f} accuracy {tr_acc:.4f},"        #εμφάνιση των αποτελεσμάτων για train loss/accuracy , val loss/accuracy
    f"val loss {val_loss:.4f} accuracy {val_acc:.4f}")           

    writer.add_scalars("Loss",{ "train": tr_loss, "Val":val_loss}, ep)              #δημιουργία των διαγραμμάτων στο tensorboard ενα διαγραμμα val+train loss ανα εποχή
    writer.add_scalars("Accuracy",{"train": tr_acc,"Val":val_acc}, ep)              #ενα διαγραμμα val+train accuracy ανα εποχή
writer.close()

te_loss, te_acc = run_epoch(test_loader, train=False)                    #τρέχω το μοντέλο μια φορά για όλο το test 
print(f"test loss {te_loss:.4f} accuracy {te_acc:.4f}")                  #εκτυπώνω test loss/accuracy

#δειγματοληψια για οπτικοποιηση
#Το μοντελο ως κλαση βγαζει ενα αριθμο απο 0-9 αρα εδω το μεταφραζουμε σε λεξεις
classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
#η συναρτηση που δεχεται το εκπαιδευομενο μοντελο και τα δεδομενα του test
def visualize_samples(model, loader):
    
    correct_samples = [] #λιστα για να αποθηκευονται οι σωστες εικόνες
    wrong_samples = [] #λιστα για να αποθηκευονται οι λαθος εικόνες

    with torch.no_grad(): #δε κραταω κλίσεις δε θα εκπαιδευσω
        for x, y in loader: #x ειναι οι εικόνες του batch (256 εικονες) και y οι κλάσεις
            x, y =x.to(device), y.to(device)  
            logits = model(x) #το μοντελο βγάζει τα logits για καθε κλαση
            predictions = logits.argmax(1) # η προβλεψη ειναι αυτη η κλαση που έχει τις περισσότερες ψήφους
        
            #αφού εβγαλα τις προβλεψεις και καθε εικονα παω να δω για τη καθε μια ποια ειναι η προβλεψη και ποια η πραγματική κλαση
            for i in range(len(x)):
                if predictions[i] == y[i]: #αν η προβλεψη σωστη
                    if len(correct_samples)<5: #κραταμε τα 5 πρωτα σωστα
                        correct_samples.append((x[i].cpu(), predictions[i].item(),y[i].item()))
                else:
                    if len(wrong_samples)<5:
                        wrong_samples.append((x[i].cpu(), predictions[i].item(),y[i].item()))
                        #γυριζω τα δεδομενα μου στην cpu για να μπορω να χρησιμοποιησω τις βιβλιοθηκες της Python
                        #και με το .item γυρναω το Pytorch tensor σε απλο αριθμό της Python

                    #αν βρω 5 παραδείγματα απο το καθενα σταματω το ψαξιμο
                    if len(correct_samples)>= 5 and len(wrong_samples)>= 5:
                        break
                if len(correct_samples) >= 5 and len(wrong_samples) >= 5:
                    break

    #εμφάνιση εικονών
    def plot_samples(samples, title):
        plt.figure(figsize=(10, 4)) #καμβας 10 χ 4 ιντσες
        plt.suptitle(title, fontsize=16)#τιτλος
        for i, (img_tensor, prediction_idx, true_idx) in enumerate(samples):
            plt.subplot(1, 5, i+1) #1 γραμμη 5 στηλες, σχεδιαζω τωρα την i+1
            #Το pytorch εχει σχημα (Channels, Height, Width) (3, 32, 32)
            #Το Matplotlib θελει (Height, Width, Channels) (32, 32, 3)
            img = img_tensor.numpy()
            img = img.transpose(1, 2, 0)
            plt.imshow(img)
            plt.axis('off')
            #χρώμα τιτλου
            color_title = 'green' if prediction_idx == true_idx else 'red' 
            plt.title(f'P: {classes[prediction_idx]}\nT: {classes[true_idx]}', color=color_title)
        plt.show()

    #εμφανιση των παραδειγματων
    plot_samples(correct_samples, 'Παραδείγματα Σωστής Κατηγοριοποίησης MLP')
    plot_samples(wrong_samples, 'Παραδείγματα Εσφαλμένης Κατηγοριοποίησης MLP')

#καλω τη συναρτηση
visualize_samples(model, test_loader)