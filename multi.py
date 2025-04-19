import numpy as np
# import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

def conv(kernel, input, kernel_row, kernel_column, output_row, output_column):
    temp = np.zeros((output_row,output_column))
    for kr in range(kernel_row):
        for kc in range(kernel_column):
            for fr in range(output_row):
                for fc in range(output_column):
                    temp[fr][fc] = temp[fr][fc] + input[fr+kr][fc+kc]*kernel[kr][kc]
                    
            
    # for fr in range(output_row):
    #     for fc in range(output_column):   
    #         temp[fr][fc] = temp[fr][fc]//512
    #         if temp[fr][fc] > 127:
    #             temp[fr][fc] = 127
    #         if temp[fr][fc] < -128:
    #             temp[fr][fc] = -128
    return temp

def max_pooling(input):
    si = input.shape
    temp = np.zeros((int(si[0]/2),int(si[1]/2)))
    for r in range(0,si[0],2):
        for c in range(0, si[1], 2):
            max = 0
            for wr in range(r,r+2,1):
                for wc in range(c, c+2,1):
                    if max < input[wr][wc]:
                        max = input[wr][wc]
            temp[int(r/2)][int(c/2)] = max
    return temp

(x_train, y_train), _ = mnist.load_data()
fail = 0
for z in range(1000, 2000):
    image = np.array(x_train[z], dtype=np.int32)
    # print(image)
    # print("...............................")
    image = image - 128
    print(y_train[z])
    print("...............................")
    result = np.array(image, ndmin=3)
    loaded = np.load("quantized_weights.npz")
    for key in loaded.files:
        # print(key)
        shape = loaded[key].shape
        # print(shape)
        sr = result.shape
        if loaded[key].ndim > 2: 
            temp = np.zeros(((shape[0], sr[1]-shape[2]+1, sr[2]-shape[3]+1)))
            for n in range(shape[0]):
                for m in range(shape[1]):
                    temp[n] = temp[n] + conv(loaded[key][n][m], result[m], shape[2], shape[3], sr[1]-shape[2]+1, sr[2]-shape[3]+1)
                for fr in range(sr[1]-shape[2]+1):
                    for fc in range(sr[2]-shape[3]+1):   
                        temp[n][fr][fc] = temp[n][fr][fc]//512
                        if temp[n][fr][fc] > 127:
                            temp[n][fr][fc] = 127
                        if temp[n][fr][fc] < -128:
                            temp[n][fr][fc] = -128
            result = temp
            np.set_printoptions(threshold=np.inf)
            # print(result.astype(int))
            # print("------------------------------------------------------")
            
            # print(shape[3])
            if shape[3]==1:
                temp2 = np.zeros((shape[0], int((sr[1]-shape[2]+1)/2), int((sr[2]-shape[3]+1)/2)))
                for n in range(shape[0]):
                    temp2[n] = max_pooling(result[n])
                result = temp2
                np.set_printoptions(threshold=np.inf)
                # print("max_pooling")
                # print(result.astype(int))
                # print("------------------------------------------------------")
        else:
            if key == "fc1_0":
                result = result.flatten()
                result = np.array(result, ndmin=2)
                result = result.T
            # print(result.shape)
            result = loaded[key] @ result
            
            
            for i in range(result.shape[0]):
                if key == "fc1_1" or key == "fc2_1" or key == "fc3":
                    if result[i] < 0:
                        result[i] = 0
                result[i] = result[i]//512
            if key == "fc3":
                max = 0
                for j in range(result.shape[0]):
                    if max < result[j]:
                        index = j
                        max = result[j]

                    # if max == result[j]:
                    #     print("trung")
                print(index)
                if index != y_train[z]:
                    fail = fail + 1
                # print(result.astype(int))
                # print("------------------------------------------------------")
print(fail)


