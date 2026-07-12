import json
import sys

def compare(file1, file2):
    with open(file1, 'r') as f:
        data1 = json.load(f)
    with open(file2, 'r') as f:
        data2 = json.load(f)
    if data1 == data2:
        print('Files are the same')
    else:
        print('Files are different')
    for idx in range(len(data1['model_completion'])):
        if data1["model_completion"][idx] != data2["model_completion"][idx]:
            print(f'Index {idx} is different')
            # print(f'Question: {data1["question"][idx]}')
            print(f'File1: {data1["model_completion"][idx]}')
            print(f'File2: {data2["model_completion"][idx]}')
            print()
            _ = input()

if __name__ == '__main__':
    compare(sys.argv[1], sys.argv[2])