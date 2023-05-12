import matplotlib.pyplot as plt 

train_percent = []
train_loss = []
val_loss = []
train_acc = []
val_acc = []

with open("training_log.txt", "r") as f:
    f.readline()
    for line in f:
        split_line = line.split("\t")
        train_percent.append(float(split_line[0]))
        train_loss.append(float(split_line[1]))
        train_acc.append(float(split_line[2]))
        val_loss.append(float(split_line[3]))
        val_acc.append(float(split_line[4]))

x_labels = [.1, .2, .3, .4, .5, .6, .7, .8]

plt.figure()
plt.plot(train_percent, train_loss, label="Train loss")
plt.plot(train_percent, val_loss, label="Validation loss")
plt.xlabel("Train Data Percentage")
plt.xticks(x_labels, x_labels)
plt.ylabel("Train Data Percentage")
plt.legend()
plt.savefig("gen_loss_curves.png", dpi=200)

plt.figure()
plt.plot(train_percent, train_acc, label="Train accuracy")
plt.plot(train_percent, val_acc, label="Validation accuracy")
plt.xlabel("Train Data Percentage")
plt.xticks(x_labels, x_labels)
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("gen_acc_curves.png", dpi=200)