import os

import settings

name = '20190629-194150=resnet34_pre'

try:
    log_file = os.path.join(settings.DIR_logs, name + '.log')
    # os.system('rm -r {}'.format(settings.DIR_logs))
    os.system('rm {}'.format(log_file))
    print(1)
except Exception as e:
    print(e)

try:
    tb_dir = os.path.join(settings.DIR_tblogs, name)
    # os.system('rm -r {}'.format(settings.DIR_tblogs))
    os.system('rm -r {}'.format(tb_dir))
    print(2)
except Exception as e:
    print(e)

try:
    model_file = os.path.join(settings.DIR_trained_model, name + '.pt')
    # os.system('rm -r {}'.format(settings.DIR_trained_model))
    os.system('rm {}'.format(model_file))
    print(3)
except Exception as e:
    print(e)

try:
    cm_dir = os.path.join(settings.DIR_confusions, name)
    # os.system('rm -r {}'.format(settings.DIR_confusions))
    os.system('rm -r {}'.format(cm_dir))
    print(4)
except Exception as e:
    print(e)