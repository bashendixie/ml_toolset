import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET

def xml_to_csv(path):
    xml_list = []
    # 读取注释文件
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']

    # 将所有数据分为样本集和验证集，一般按照3:1的比例
    train_list = xml_list[0: int(len(xml_list) * 0.67)]
    eval_list = xml_list[int(len(xml_list) * 0.67) + 1: ]

    # 保存为CSV格式
    train_df = pd.DataFrame(train_list, columns=column_name)
    eval_df = pd.DataFrame(eval_list, columns=column_name)
    train_df.to_csv('C:/Users/zyh/Desktop/cat/train_peaches.csv', index=None)
    eval_df.to_csv('C:/Users/zyh/Desktop/cat/eval_peaches.csv', index=None)

def main():
    path = 'C:/Users/zyh/Desktop/cat'
    xml_to_csv(path)
    print('Successfully converted xml to csv.')
main()
