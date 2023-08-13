from glob import glob


def gen_instereo2k():
    data_dir = '/media/wangqingyu/机械硬盘2/立体匹配公开数据集/05_InStereo2K数据集/test/'
    file_name = 'instereo2k_val.txt'
    with open(file_name, 'w') as file_name:
        images = sorted(glob(data_dir + '/*'))
        for image in images:
            strlist = image.split('/')
            number = strlist[-1]
            left_dir = 'test/' + number + '/left.png'
            right_dir = 'test/' + number + '/right.png'
            disp_dir = 'test/' + number + '/left_disp.png'
            file_name.write(left_dir + ' ')
            file_name.write(right_dir + ' ')
            file_name.write(disp_dir + '\n')


if __name__ == '__main__':
    gen_instereo2k()
