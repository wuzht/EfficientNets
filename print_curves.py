import os
import re
import numpy as np
import matplotlib.pyplot as plt

def show_curve_2(y1s, y2s, title, ylabel, isAcc=True):
    x = np.array(range(len(y1s)))
    y1 = np.array(y1s)
    y2 = np.array(y2s)
    plt.plot(x, y1, label='Training') # train
    plt.plot(x, y2, label='Validation') # test
    plt.axis()
    plt.title('{}'.format(title))
    plt.xlabel('Epoch')
    plt.ylabel('{}'.format(ylabel))
    
    if isAcc is True:
        plt.yticks(np.linspace(0, 1, 11))

    plt.legend(loc='best')
    plt.grid()
    plt.show()
    # plt.savefig("{}.svg".format(ylabel))
    plt.close()

def get_data_from_log(name):
    train_accs = []
    val_accs = []
    train_losses = []
    val_losses = []
    with open('logs/{}.log'.format(name), 'r') as f:
        for line in f.readlines():
            line = line.strip()

            train = re.findall(r'Train.*Acc:.*\((.*)\%\)', line)
            if len(train) > 0:
                train_accs.append(float(train[0]) / 100)

            val = re.findall(r'Val.*Acc:.*\((.*)\%\)', line)
            if len(val) > 0:
                val_accs.append(float(val[0]) / 100)

            train_loss = re.findall(r'Train.*Avg loss: (.*), Acc:', line)
            if len(train_loss) > 0:
                train_losses.append(float(train_loss[0]))
            
            val_loss = re.findall(r'Val.*Avg loss: (.*), Acc:', line)
            if len(val_loss) > 0:
                val_losses.append(float(val_loss[0]))
    return train_accs, val_accs, train_losses, val_losses

train_accs, val_accs, train_losses, val_losses = get_data_from_log('T20190702-133826=shufflenetv2_x2_0')

show_curve_2(train_accs, val_accs, 'ShuffleNet 2.0x', 'Accuracy', True)
show_curve_2(train_losses, val_losses, 'ShuffleNet 2.0x', 'Loss', False)

# exit(-1)
# names = [
#     'IN10020190701-125010=shufflenetv2_x0_5'
# ]


# for name in names:
#     try:
#         log_file = os.path.join('logs', name + '.log')
#         # os.system('rm -r {}'.format(settings.DIR_logs))
#         os.system('rm {}'.format(log_file))
#         print(1)
#     except Exception as e:
#         print(e)

#     try:
#         tb_dir = os.path.join('tblogs', name)
#         # os.system('rm -r {}'.format(settings.DIR_tblogs))
#         os.system('rm -r {}'.format(tb_dir))
#         print(2)
#     except Exception as e:
#         print(e)

#     try:
#         model_file = os.path.join('trained_model', name + '.pt')
#         # os.system('rm -r {}'.format(settings.DIR_trained_model))
#         os.system('rm {}'.format(model_file))
#         print(3)
#     except Exception as e:
#         print(e)

#     try:
#         cm_dir = os.path.join('confusions', name)
#         # os.system('rm -r {}'.format(settings.DIR_confusions))
#         os.system('rm -r {}'.format(cm_dir))
#         print(4)
#     except Exception as e:
#         print(e)