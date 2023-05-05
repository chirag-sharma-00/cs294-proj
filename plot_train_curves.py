import matplotlib.pyplot as plt 

epochs = []
train_loss = []
val_loss = []
train_acc = []
val_acc = []

with open("training_log.txt", "r") as f:
    f.readline()
    for line in f:
        split_line = line.split("\t")
        epochs.append(int(split_line[0]))
        train_loss.append(float(split_line[1]))
        train_acc.append(float(split_line[2]))
        val_loss.append(float(split_line[3]))
        val_acc.append(float(split_line[4]))

plt.figure()
plt.plot(epochs, train_loss, label="Train loss")
plt.plot(epochs, val_loss, label="Validation loss")
plt.xlabel("Epoch")
plt.xticks(range(0, 20), range(0, 20))
plt.ylabel("Loss")
plt.legend()
plt.savefig("loss_curves.png", dpi=200)

plt.figure()
plt.plot(epochs, train_acc, label="Train accuracy")
plt.plot(epochs, val_acc, label="Validation accuracy")
plt.xlabel("Epoch")
plt.xticks(range(0, 20), range(0, 20))
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("acc_curves.png", dpi=200)