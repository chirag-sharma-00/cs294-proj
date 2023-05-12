import matplotlib.pyplot as plt 

train_percent = []
train_loss = []
val_loss = []
train_acc = []
val_acc = []
train_percent_overfit = []
val_percent_overfit = []

with open("training_log.txt", "r") as f:
    f.readline()
    for line in f:
        split_line = line.split("\t")
        train_percent.append(float(split_line[0]))
        train_loss.append(float(split_line[1]))
        train_acc.append(float(split_line[2]))
        val_loss.append(float(split_line[3]))
        val_acc.append(float(split_line[4]))

        train_percent_overfit.append(float(split_line[0]) * float(split_line[2]))
        val_percent_overfit.append(float(split_line[0]) * float(split_line[4]))

x_labels = [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0]

plt.figure()
plt.plot(train_percent, train_loss, label="Train loss")
plt.plot(train_percent, val_loss, label="Validation loss")
plt.xlabel("Train Data Percentage")
plt.xticks(x_labels, x_labels)
plt.ylabel("Train Data Percentage")
plt.legend()
plt.savefig("gen_loss_curves.png", dpi=200)

plt.figure()
plt.plot(train_percent, train_percent_overfit, label="Train percent overfit")
plt.plot(train_percent, val_percent_overfit, label="Validation percent overfit")
plt.plot([0.05, 0.5, 1.0], [0.05, .5, 1.0], '--', label="Pure memorization")
plt.xlabel("Train Data Percentage")
plt.xticks(x_labels, x_labels)
plt.yticks(x_labels, x_labels)
plt.ylabel(f"MEC % overfitting for 100% accuracy")
plt.legend()
plt.savefig("gen_perc_overfit.png", dpi=200)

plt.figure()
plt.plot(train_percent, train_acc, label="Train accuracy")
plt.plot(train_percent, val_acc, label="Validation accuracy")
plt.xlabel("Train Data Percentage")
plt.xticks(x_labels, x_labels)
# plt.yticks(x_labels, x_labels)
plt.ylabel(f"Train accuracy")
plt.legend()
plt.savefig("gen_acc_curves.png", dpi=200)