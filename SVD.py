
import torch
import os
import numpy as np


folder_path = '/home/MBGL/data/ADNI/all_fcn' #使用fmri计算所有群体共同特征
# folder_path = '/home/MBGL/data/ADNI/all_scn' #使用dti

def SVD_count(tuple):
    node_feature_matrix = []
    for i in range(90):
        columns_data = []
        for matrix in tuple:
            matrix=np.abs(matrix)  #fmri是稠密矩阵，所以得求绝对值
            column = matrix[:, i]
            columns_data.append(column)
        new_matrix = np.vstack(columns_data).T
        node_feature_matrix.append(new_matrix)

    left_singular_matrices = []

    for matrix in node_feature_matrix:
        matrix = torch.Tensor(matrix)

        u, s, v = torch.linalg.svd(matrix)

        left_singular_matrices.append(u)

    columns = []
    for mat in left_singular_matrices:
        column = mat[:, 1]
        columns.append(column)
    feature_weight = np.vstack(columns).T

    return feature_weight


def count_all_group(path):

    matrix_tuple = tuple(
        np.loadtxt(os.path.join(path, filename))
        for filename in os.listdir(path)
        if filename.endswith(".txt")
    )


    feature_weight=SVD_count(matrix_tuple)

    row_means = np.mean(feature_weight, axis=1)


    sorted_indices = np.argsort(row_means)

    #修改删除行数
    n=10

    top_indices = sorted_indices[-n:][::-1]
    bottom_indices = sorted_indices[:n]

    print(f"排名前{n}的行索引：", top_indices)
    print(f"排名后{n}的行索引：", bottom_indices)

    return top_indices,bottom_indices

def count_single_group(label):
    AD_filename=os.listdir('/home/MBGL/data/ADNI/AD/scn')
    mci_filename=os.listdir('/home/MBGL/data/ADNI/MCI/scn')
    nc_filename=os.listdir('/home/MBGL/data/ADNI/NC/scn')
    delete_path='/home/MBGL/data/ADNI/delete_feature/fcn_20/scn' #delete_feature下的目录，与folderpath的模态保持一致
    # delete_path='/home/MBGL/data/ADNI/all_scn'

    AD_matrix_tuple = tuple(np.loadtxt(os.path.join(delete_path,item)) for item in AD_filename)
    mci_matrix_tuple = tuple(np.loadtxt(os.path.join(delete_path,item)) for item in mci_filename)
    nc_matrix_tuple = tuple(np.loadtxt(os.path.join(delete_path,item)) for item in nc_filename)

    if label==0:
        return SVD_count(AD_matrix_tuple)
    elif label==1:
        return SVD_count(mci_matrix_tuple)
    else:
        return SVD_count(nc_matrix_tuple)


if __name__ == '__main__':
    top,bottom=count_all_group(folder_path)
    to0=top.T
    delete_row=np.concatenate((top, bottom))
    # input_folder = '/home/MBGL/data/ADNI/all_scn' #删除dti的行
    # output_folder = '/home/MBGL/data/ADNI/delete_feature/fcn_20/scn'

    input_folder = '/home/MBGL/data/ADNI/all_fcn' #删除fmri的行
    output_folder = '/home/MBGL/data/ADNI/delete_feature/fcn_20/fcn'


    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        new_name=filename[1:] #使用fmri时用
        # new_name=filename #使用dti时使用
        matrix = np.loadtxt(file_path)

        matrix_1 = np.delete(matrix, delete_row, axis=0)

        output1 = os.path.join(output_folder, new_name)

        np.savetxt(output1, matrix_1)

    AD_feature=count_single_group(0)
    print(AD_feature)
    print(AD_feature.shape)