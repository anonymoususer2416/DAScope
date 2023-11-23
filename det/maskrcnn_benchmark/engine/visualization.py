
import matplotlib.pyplot as plt
import os

def plot_one(data_1, title, x_label, y_label, data_1_label, output_dir):
    plt.figure(1)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    t_line, = plt.plot(data_1, label=data_1_label)
    plt.legend(handles=[t_line])
    plt.savefig(os.path.join(output_dir, title))
    plt.close('all')

def plot_two(data_1, data_2, title, x_label, y_label, data_1_label, data_2_label, output_dir):
    plt.figure(2)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    t_line, = plt.plot(data_1, label=data_1_label)
    v_line, = plt.plot(data_2, label=data_2_label)
    plt.legend(handles=[t_line, v_line])
    plt.savefig(os.path.join(output_dir, title))
    plt.close('all')

if __name__ == "__main__":
    train_loss = [18.023161639676374, 17.75451136058118, 17.716401314695343, 17.785925994580534, 
                  17.70776648573014, 17.696830309884888, 17.706762298060063, 17.68931900639995, 
                  17.61373526258629, 17.6001513903131]

    val_loss = [29.18690526710366, 52.01537312201734, 30.725421617615897, 36.50785377790343, 
                36.615923287733544, 31.710015638819282, 35.62031501194216, 31.53402461645738,
                35.03729244448104, 31.071966531141747]

    plot_two(train_loss, val_loss, 
         "AQT Cityscapes->DAWN Finetuning Training and Validation Losses", 
         "Epoch", "Loss", "Training", "Validation", "./exps/r50_uda_cityscapes2dawn_finetune_20/")