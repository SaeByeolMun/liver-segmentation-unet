import cv2
import numpy as np
import os
import pandas as pd
from sklearn.utils import shuffle
import math

def RotateScaleImage(img, aang, asca, bValue=0):
    
    rows, cols = img.shape[:2]

    # 이미지의 중심점을 기준으로 aang 도 회전 하면서 asca배 Scale
    M= cv2.getRotationMatrix2D((cols/2, rows/2),aang, asca)

    rotated_img = cv2.warpAffine(img, M,(cols, rows),borderValue=bValue)
    
    return rotated_img


#  파일을 만드는 함수
def createFolder(directory): # 매개변수로 디렉토리 받아옴
    try: # 실행할 코드블록
        if not os.path.exists(directory):
            os.makedirs(directory) # 디렉토리를 생성함 
    except OSError: # 예외처리
        print ('Error: Creating directory. ' +  directory)

# 윈도우 스케일 변환 함수
def win_scale(data, wl, ww, dtype, out_range):
    """
    Scale pixel intensity data using specified window level, width, and intensity range.
    """
    
    data_new = np.empty(data.shape, dtype=np.double)
    data_new.fill(out_range[1]-1)
    
    data_new[data <= (wl-ww/2.0)] = out_range[0]
    
    data_new[(data>(wl-ww/2.0))&(data<=(wl+ww/2.0))] = ((data[(data>(wl-ww/2.0))&(data<=(wl+ww/2.0))]-(wl-0.5))/(ww-1.0)+0.5)*(out_range[1]-out_range[0])+out_range[0]
    
    data_new[data > (wl+ww/2.0)] = out_range[1]-1
    
    return data_new.astype(dtype)

def ct_win(im, wl, ww, dtype, out_range):    
    """
    Scale CT image represented as a `pydicom.dataset.FileDataset` instance.
    """

    # Convert pixel data from Houndsfield units to intensity:
    intercept = int(im[(0x0028, 0x1052)].value)
    slope = int(im[(0x0028, 0x1053)].value)
    data = (slope*im.pixel_array+intercept)

    # Scale intensity:
    return win_scale(data, wl, ww, dtype, out_range)

def np2csv(np, save_path, transpose=False):
    df = pd.DataFrame(result_excel)
    print(df)
    if transpose:
        df=df.transpose()
    df.to_csv(save_path)


def normalization(list, frame_len:int):
    list = np.array(list)
    #df = pd.DataFrame(list)
    
    max_list = np.max(list)
    min_list = np.min(list)
    
    if min_list < 0:
        list = list + abs(min_list)
    
    max_list = np.max(list)
    min_list = np.min(list)
    
    list_norm = np.zeros(frame_len, dtype=np.float64)
    
    list_norm = (list - min_list) / (max_list - min_list)
            
    return list_norm


def split_data(data, train_rate:float=0.8, val_rate:float=0.9, SHUFFLE:bool=True, save_excel:bool=False, DATA_PATH:str='./', file_name:str=''):
    data = np.array(data)
    if SHUFFLE:
        data = shuffle(data)
    h = data.shape[0]
    train_num = int(h * train_rate)
    val_num = int(h * val_rate)

    train_num = int(round(train_num,0))
    val_num = int(round(val_num,0))
    
    train = data[:train_num]
    val = data[train_num:val_num]
    test = data[val_num:]
    
    if save_excel:
        pd.DataFrame(train).to_excel(f'{DATA_PATH}/train{file_name}.xlsx', index = False)
        pd.DataFrame(val).to_excel(f'{DATA_PATH}/val{file_name}.xlsx', index = False)
        pd.DataFrame(test).to_excel(f'{DATA_PATH}/test{file_name}.xlsx', index = False)
    
    return train, val, test

def cv_split_data(data, split_count:int= 5, SHUFFLE:bool=True, save_excel:bool=False, DATA_PATH:str='./', file_name:str=''):
    data = np.array(data)
    
    if SHUFFLE:
        data = shuffle(data)
    
    h = data.shape[0]
    w = int(h / split_count)
    
    split_fold = []
    train = []
    val = []
    test = []
    
    split_list = []
    for i in range(split_count):
        split_list.append(w)
    
    for i in range(h-(w*split_count)):
        split_list[i] = split_list[i] + 1
    print(split_list)
    start = 0
    for i in range(split_count):
        i_data = data[start:start+split_list[i]]
        
        start += split_list[i]
        split_fold.append(i_data)
    
        train.append([])
        val.append([])
        test.append([])
    
    for i in range(split_count):
        count_list = [c for c in range(split_count)]
        if i != split_count-1:
            split_test_list = [i]
            count_list.remove(i)
            split_val_list = [i+1]
            count_list.remove(i+1)
            split_train_list = count_list
        else:
            split_test_list = [i]
            count_list.remove(i)
            split_val_list = [0]
            count_list.remove(0)
            split_train_list = count_list

        for s in split_train_list:
            train[i].extend(split_fold[s])
        for s in split_val_list:
            val[i].extend(split_fold[s])
        for s in split_test_list:
            test[i].extend(split_fold[s])
    
    if save_excel:
        writer_train = pd.ExcelWriter(f'{DATA_PATH}/train{file_name}.xlsx', engine='xlsxwriter')
        writer_val = pd.ExcelWriter(f'{DATA_PATH}/val{file_name}.xlsx', engine='xlsxwriter')
        writer_test = pd.ExcelWriter(f'{DATA_PATH}/test{file_name}.xlsx', engine='xlsxwriter')
        for i in range(split_count):
            pd.DataFrame(train[i]).to_excel(writer_train, sheet_name = f'{i}', index = False)
            pd.DataFrame(val[i]).to_excel(writer_val, sheet_name = f'{i}', index = False)
            pd.DataFrame(test[i]).to_excel(writer_test, sheet_name = f'{i}', index = False)
        writer_train.save()
        writer_val.save()
        writer_test.save()
        
    train = np.array(train)
    val = np.array(val)
    test = np.array(test)
    
    print(data.shape, '\t', train.shape, '\t', val.shape, '\t', test.shape)
    
    for i in range(split_count):
        train[i] = np.array(train[i])
        val[i] = np.array(val[i])
        test[i] = np.array(test[i])
        
        print(data.shape, '\t', train[i].shape, '\t', val[i].shape, '\t', test[i].shape)
        
    return train, val, test

def divide_list(data, cv): 
    h = int(data.shape[0])
    length = round((h / cv), 0)
    arr_length = np.full((cv,), length)
    remainder = h % cv
    
    if remainder != 0:
        for a in range(remainder):
            arr_length[a] += 1
    
    split_arr = []
    start = 0
    end = 0
    for a in range(cv):
        l = int(arr_length[a])
        start = end
        end += l
        start = int(start)
        end = int(end)
        split_arr.append(data[start:end])
#         print(end - start, start, end)
    return split_arr

def split_data_cv(data, cv:int=5, SHUFFLE:bool=True):
    '''
    cv: 총 cv 개수
    '''
    data = np.array(data)
    if SHUFFLE:
        data = shuffle(data)
    h = int(data.shape[0])
    
    split_arr = divide_list(data, cv)
    
    train_list = []
    val_list = []
    test_list = []
    for count in range(cv):
        train = []
        val = []
        test = []
        for a in range(cv):
            if count != cv - 1:
                if a == count:
                    test.extend(split_arr[a])
                elif a == (count + 1):
                    val.extend(split_arr[a])
                else:
                    train.extend(split_arr[a])
            else:
                if a == count:
                    test.extend(split_arr[a])
                elif a == 0:
                    val.extend(split_arr[a])
                else:
                    train.extend(split_arr[a])
                    
        train = np.array(train)
        val = np.array(val)
        test = np.array(test)
        
        train_list.append(train)
        val_list.append(val)
        test_list.append(test)
        
    train_list = np.array(train_list)
    val_list = np.array(val_list)
    test_list = np.array(test_list)
    
    return train_list, val_list, test_list


def resultConf(test_label, predict_model, weight):
    recall_list = []
    specificity_list = []
    precision_list = []
    acc_list = []
    dice_list = []
    
    all_data_result = []
    ious = []
    dices = []
    
    #for i in range(n_test):
    for i in (range(test_label.shape[0])):
            
        gt = test_label[i, :,  :, 0] # ground truth binary mask
        
        if np.sum(gt) > 0:
            pr = predict_model[i, :, :, 0] > weight# binary prediction
        
            gt = gt.astype(bool)
            pr = pr.astype(bool)
        
            # Compute scores
            seg1_n = pr == 0
            seg1_t = pr == 1
        
            gt1_n = gt == 0
            gt1_t = gt == 1
            
            tp = np.sum(seg1_t&gt1_t)
            fp = np.sum(seg1_t&gt1_n)
            tn = np.sum(seg1_n&gt1_n)    
            fn = np.sum(seg1_n&gt1_t)
            
            recall = tp / (tp + fn)
            specificity = tn / (tn + fp)
            precision = tp / (tp + fp)
            acc = (tp + tn) / (tp + tn + fp + fn)
            dice = (2 * tp) / (2*tp + fp + fn)
            
            recall_list.append(recall)
            specificity_list.append(specificity)
            precision_list.append(precision)
            acc_list.append(acc)
            dice_list.append(dice)
            
                
    dice_list = np.array(dice_list)
    recall_list = np.array(recall_list)
    specificity_list = np.array(specificity_list)
    precision_list = np.array(precision_list)
    
    dice_list = dice_list[~np.isnan(dice_list)]
    recall_list = recall_list[~np.isnan(recall_list)]
    specificity_list = specificity_list[~np.isnan(specificity_list)]
    precision_list = precision_list[~np.isnan(precision_list)]
    
    print('Done save result...')
    
    return recall_list, specificity_list, precision_list, acc_list, dice_list

def returnTable(test_label, predict_model, weight):
    recall_list, specificity_list, precision_list, acc_list, dice_list = resultConf(test_label, predict_model, weight)

    row_data1 = []
    row_item1 = []
    
    row_item1.append('{0}'.format(weight))
    row_item1.append('{0:.2f}'.format(np.mean(recall_list)*100))
    row_item1.append('{0:.2f}'.format(np.mean(specificity_list)*100))
    row_item1.append('{0:.2f}'.format(np.mean(precision_list)*100))
    row_item1.append('{0:.2f}'.format(np.mean(acc_list)*100))
    row_item1.append('{0:.2f}'.format(np.mean(dice_list)*100))
    row_data1.append(row_item1)
    
    head1 = ["Weight", "Sensitivity", "Specificity", "Precision", "Accuracy", "DSC"]
    
    return row_data1, head1
    # 평가지표 구하는 수식
    recall_list = []     # = Sensitivity
    precision_list = []
    specificity_list = []
    acc_list = []
    dice_list = []
    for i in range(y_test.shape[0]):
        
        seg1 = predict_img[i]
        gt1 = y_test[i]
            
        seg1_n = seg1 == 0
        seg1_t = seg1 == 1
    
        gt1_n = gt1 == 0
        gt1_t = gt1 == 1
        
        tp = np.sum(seg1_t&gt1_t)
    
        fp = np.sum(seg1_t&gt1_n)
    
        tn = np.sum(seg1_n&gt1_n)
        
        fn = np.sum(seg1_n&gt1_t)
        
        # 재현율 Recall 구하는 수식 
        # Recall = TP / (TP + FN)
        # Recall이 높다는 것은 알고리즘이 관련있는 결과를 최대한 많이 가져왔다는 의미
        recall = tp / (tp + fn)
        if np.isnan(recall):
    #         print('recall is nan {0}, tp:{1}, fn:{2}'.format(i, tp, fn))
            continue
        
        # 정밀도 Precision 구하는 수식
        # Precision = TP / (TP + FP)
        # Precision = Positive Presictive Value = PPV 라고도 한다.
        # Precision이 높다는 것은 알고리즘이 관련 결과를 그렇지 못한 것 대비 충분히 맞추었다는 의미
        precision = tp / (tp+fp)
        if np.isnan(precision):
    #         print('precision is nan {0}, tn:{1}, fp:{2}'.format(i, tn, fp))
            continue
            
        specificity = tn / (fp+tn)
        if np.isnan(specificity):
    #         print('specificity is nan {0}, tn:{1}, fp:{2}'.format(i, tn, fp))
            continue
        
        # 정확도 Accuracy 구하는 수식 
        # Accuracy = (TP + FN) / (TP + TN + FP + FN)
        # 즉, 정확도 = (올바르게 예측한 샘플 개수) / (전체 샘플 개수)
        acc = (tp + tn) / (tp + tn + fp + fn)
        if np.isnan(acc):
    #         print('acc is nan {0}, tp:{1}, tn:{2}, fp:{3}, fn:{4}'.format(i, tp, tn, fp, fn))
            continue
        
        # f1 score 
        # Precision 과 recall 의 조화 평균
        dice = (2*tp) / (2*tp + fp + fn)
        if np.isnan(dice):
    #         print('dice is nan {0}, tp:{1}, fp:{2}, fn:{3}'.format(i, tp, fp, fn))
            continue
        
        recall_list.append(recall)
        precision_list.append(precision)
        specificity_list.append(specificity)
        acc_list.append(acc)
        dice_list.append(dice)
    
    return recall_list, precision_list, specificity_list, acc_list, dice_list

