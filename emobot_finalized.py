#-*- coding: utf-8 -*-

import os, sys, glob, cv2, time, ast, json, csv, subprocess
from flask import Flask, request, current_app, jsonify
from PIL import Image
import numpy as np
import pandas as pd
import urllib.request as urllib
import pyzbar.pyzbar as pyzbar
import cognitive_face as CF
import boto3

app = Flask(__name__)

with app.app_context():
     # within this block, current_app points to app.
     print(current_app.name)

## Key for Amazon Cog Face
KEY = '' # Replace with your KEY
CF.Key.set(KEY)
BASE_URL = ''  # Replace with your regional Base URL
CF.BaseUrl.set(BASE_URL)

## Key for Amazon AWS S3
ACCESS_KEY = ''  # Replace with your access KEY
SECRET_KEY = ''  # Replace with your secret KEY


def upload_mapfile(dir_name, file_name):
    s3_resource = boto3.resource( 
				's3', 
                aws_access_key_id=ACCESS_KEY, 
                aws_secret_access_key=SECRET_KEY, 
        ) 

    # Get image file 
    data = open(dir_name + file_name, 'rb')

    # Save image to S3 bucket as public 
    s3_resource.Bucket('bucket_name').put_object(Body=data, Key=file_name, ACL='public-read') 

    # Get public image url 
    img_url = "https://_bucket_name_.s3-ap-northeast-1.amazonaws.com/{}".format(file_name)

    return img_url


def get_time():
    secondsSinceEpoch = time.time()
    timeObj = time.localtime(secondsSinceEpoch)
    cur_time = str('%d-%d-%d %d:%d:%d' % (timeObj.tm_year, timeObj.tm_mon, timeObj.tm_mday, timeObj.tm_hour, timeObj.tm_min, timeObj.tm_sec))

    return cur_time

def get_info():
    req = request.get_json()
    usr_id = req["userRequest"]["user"]["properties"]["plusfriendUserKey"]
    cur_time = get_time()

    return usr_id, cur_time

def mk_directory(usr_id):
    folder_path = "./usr_id_" + usr_id
    try:
        os.makedirs(folder_path + "/")
    except:
        pass

    return folder_path

def count_files(folder_path):
    list = os.listdir(folder_path)
    num_files = len(list)
    return num_files


def get_pic(folder_path):
    ## Retrieve the url to get the selfie sent
    req = request.get_json()
    pic_url = req["action"]["detailParams"]["secureimage"]["value"]
    pic_url_d = ast.literal_eval(pic_url)
    pic_src = pic_url_d["secureUrls"][5:-1]
    print(pic_src)

    ## Assign the directory & file name to the image
    num_files = count_files(folder_path)
    file_name = str(num_files + 1) + ".jpg"
    dir_name = os.path.join(folder_path, file_name)
    print(dir_name)

    ## Save the image to the local
    urllib.urlretrieve(pic_src, dir_name)

    ## Read the image via cv2 for QR decoding
    im = cv2.imread(dir_name)

    ## Read & Resize the image for face cognition
    im_0 = Image.open(dir_name)
    im_s = im_shrink(im_0)
    im_s.save(dir_name)

    return im, dir_name

def im_shrink(im):
    wid, hei = im.size
    wid_s = int(wid*0.5)
    hei_s = int(hei*0.5)
    print(wid_s, hei_s)

    im_s = im.resize((wid_s, hei_s), Image.ANTIALIAS)
    
    return im_s

def contrast_up(im):
    # import matplotlib.pyplot as plt
    im2 = cv2.cvtColor(im, cv2.COLOR_BGR2YUV)
    im2[:, :, 0] = cv2.equalizeHist(im2[:, :, 0])

    return im2

def cog_face(dir_name):
    try:
        emo = CF.face.detect(dir_name, face_id=False, landmarks=False, attributes='emotion')
        emo_attr = emo[0]["faceAttributes"]["emotion"]
        print('got emo!')
    except:
        emo_attr = 0
        print('no emo!')

    return emo_attr


def decode(im):
    decoded0bj = pyzbar.decode(im, symbols=[pyzbar.ZBarSymbol.QRCODE])
    
    for obj in decoded0bj:
        qr_data = str(obj.data, "utf-8")
        print('got qr! :', qr_data)

    if len(decoded0bj) == 0:
        qr_data = 0
        print('no qr!')

    return qr_data

def get_spotname(qr_data):
    spot_dict = {"0": "알 수 없음", "1":"전시관", "2":"생명의 나무", "3":"굽은 소나무", "4":"백년된 나무", "5":"가을 단풍숲"}
    if qr_data != 0:
        spot_num = str(qr_data)[-1]
    else:
        spot_num = str(qr_data)
    print('spot number is, ', spot_num)
    my_spot = spot_dict[spot_num]

    return my_spot

    # def f1(x):
    #    return emo_attr[x]
    # max_emo = max(emo_attr.keys(), key=f1)
    #    my_emo = "{} ({})".format(max_emo, emo_attr[max_emo])

def return_msg(usr_id, cur_time, qr_data, emo_attr):
    if emo_attr == 0:
        my_spot = get_spotname(qr_data)
        answer = "감정 인식에 실패했어요! 예쁘게 웃는 셀카를 다시 찍어 보내주세요😃 \n\n -User ID: {} \n -전송 시간: {} \n -위치 정보: {}".format(usr_id, cur_time, my_spot)

    elif qr_data == 0:
        answer = "위치 인식에 실패했어요! QR코드가 잘 보이도록 셀카를 다시 찍어 보내주세요>_< \n\n -User ID: {} \n -전송 시간: {} \n -위치 정보: ?".format(usr_id, cur_time)

    else:
        my_spot = get_spotname(qr_data)
        answer = "사진이 성공적으로 전송되었습니다! 🌳 \n\n -User ID: {} \n -전송 시간: {} \n -위치 정보: {}".format(usr_id, cur_time, my_spot)

    return answer


def what_to_write(usr_id, qr_data, emo_attr):
    csv_columns = ['id', 'loca', 'anger','contempt','disgust','fear','happiness','neutral','sadness','surprise']
    csv_data = []
    # if emo_attr == 0:
    #     emo_attr = {'anger': 0.0, 'contempt': 0.0, 'disgust': 0.0, 'fear': 0.0, 'happiness': 0.9, 'neutral': 0.1, 'sadness': 0.0, 'surprise': 0.0}
    emo_attr['loca'] = qr_data
    emo_attr['id'] = usr_id
    
    print('what to write:', emo_attr)
    csv_data.append(emo_attr)

    return csv_columns, csv_data

def write_for_map(folder_path):
    csv_name = str(folder_path) + '/emo_qr.csv'
    file_exists = os.path.isfile(csv_name)
    csv_name_1 = str(folder_path) + '/emo_qr_v1.csv'
    csv_name_2 = str(folder_path) + '/emo_qr_v2.csv'
    with open(csv_name, encoding='euc-kr', newline='') as infile, open(csv_name_1, 'w', newline='') as outfile:
        r = csv.reader(infile)
        w = csv.writer(outfile)
        next(r)
        w.writerow(['faceRectangle/top','faceRectangle/left','scores/anger','scores/contempt','scores/disgust','scores/fear','scores/happiness','scores/neutral','scores/sadness','scores/surprise'])
        for row in r:
            w.writerow(row)

    with open(csv_name_1) as infile, open(csv_name_2, 'w', newline='') as outfile:
        fieldnames = ['faceRectangle/top','faceRectangle/left','faceRectangle/width','faceRectangle/height','scores/happiness','scores/surprise','scores/fear','scores/sadness','scores/neutral','scores/disgust','scores/contempt','scores/anger']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in csv.DictReader(infile):
            writer.writerow(row)

    with open(csv_name_2) as f:
        table = pd.read_table(f, delimiter=',', sep=',', header=[0])
        df = pd.DataFrame(table)
        cnt = {}

        for i in range(len(df['faceRectangle/left'])):
            if df['faceRectangle/left'][i][-1] == '1':
                cnt['1'] = 0
                df.iloc[[i]].to_csv('./art_'+str(df['faceRectangle/left'][i][-1])+'.csv', encoding='euc-kr',index=False)
                cnt['1'] += 1
            elif df['faceRectangle/left'][i][-1] == '2':
                cnt['2'] = 0
                df.iloc[[i]].to_csv('./art_'+str(df['faceRectangle/left'][i][-1])+'.csv', encoding='euc-kr',index=False)
                cnt['2'] += 1
            elif df['faceRectangle/left'][i][-1] == '3':
                cnt['3'] = 0
                df.iloc[[i]].to_csv('./art_'+str(df['faceRectangle/left'][i][-1])+'.csv', encoding='euc-kr',index=False)
                cnt['3'] += 1
            elif df['faceRectangle/left'][i][-1] == '4':
                cnt['4'] = 0
                df.iloc[[i]].to_csv('./art_'+str(df['faceRectangle/left'][i][-1])+'.csv', encoding='euc-kr',index=False)
                cnt['4'] += 1
            elif df['faceRectangle/left'][i][-1] == '5':
                cnt['5'] = 0
                df.iloc[[i]].to_csv('./art_'+str(df['faceRectangle/left'][i][-1])+'.csv', encoding='euc-kr',index=False)
                cnt['5'] += 1
            else:
                pass
    
    cnt_check = len(cnt.keys())
    return cnt_check

@app.route('/')
def hello():
    return 'Hello World!'

@app.route('/api/complete', methods=['POST'])
def mission_comp():
    time_s = time.time()
    time_0 = get_time()
    print('***activated when:', time_0)

    usr_id, cur_time = get_info()
    folder_path = './usr_id_' + usr_id

    ## generating the separated csv files for map
    cnt_check = write_for_map(folder_path)
    print(cnt_check)
    
    if cnt_check != 5:
        res = {
            "version": "2.0",
            "template": {
                "outputs": [
                    {
                        "simpleText": {
                            "text": "아직 모든 장소에서의 셀카가 모이지 않았어요!",
                            }
                        }
                    ]
                }
            }

    else:
        ## run unity program to generate img 
        p = subprocess.Popen(["C:/Users/saladsong/pjt/EmotionMap.exe"])
        time.sleep(15)
        print('pp')
        #p.kill()
        #print('ppp')
    
        res = {
            "version": "2.0",
            "template": {
                "outputs": [
                    {
                    "simpleText": {
                        "text": "짠! 감정지도가 완성되었어요! \n 확인하시려면 '지도' 라고 말해주세요 :)",
                            }
                        }
                    ]
                }
            }

    print(res)

    time_e = time.time()
    time_00 = get_time()
    print('***returned when:', time_00)
    print('***LAP TIME :', time_e - time_s)
    
    return jsonify(res)

@app.route('/api/return', methods=['POST'])
def return_map():
    time_s = time.time()
    time_0 = get_time()
    print('***activated when:', time_0)

    ## get the directory of the map img
    usr_id, cur_time = get_info()
    img_path = "./EmotionMap_Data/screenshots/" + usr_id + '/'
    print(img_path)

    try:
        ## upload the map on AWS S3 and get the url
        img_name = usr_id + '_Art#0.png'
        img_url = upload_mapfile(img_path, img_name)
        print(img_url)

        res = {
            "version": "2.0",
            "template": {
                "outputs": [
                    {
                    "simpleText": {
                        "text": "짠! 감정지도가 완성되었어요! \n각 위치별 감정 상태를 확인하려면 위치명 (ex. 생명의 나무) 을 알려주세요:)",
                            }
                    },
                    {
                    "simpleImage": {
                        "imageUrl": img_url,
                        "altText": "당신의 감정지도:)"
                            }
                        }
                    ]
                }
            }
    
    except IOError:
        res = {
            "version": "2.0",
            "template": {
                "outputs": [
                    {
                        "simpleText": {
                            "text": "아직 지도가 만들어지지 않았어요!",
                            }
                        }
                    ]
                }
            }

    print(jsonify(res))

    time_e = time.time()
    time_00 = get_time()
    print('***returned when:', time_00)
    print('***LAP TIME :', time_e - time_s)
    
    return jsonify(res)


@app.route('/api/emotion', methods=['POST'])
def return_emo():
    time_s = time.time()
    time_0 = get_time()
    print('***activated when:', time_0)

    ## get the spot name user requested
    req = request.get_json()
    spot_name = req["action"]["detailParams"]["spotname"]["value"]
    spot_dict = {"전시관":"1", "생명의 나무":"2", "굽은 소나무":"3", "백년된 나무":"4", "가을 단풍숲":"5"}
    spot_num = spot_dict[spot_name]

    ## get the directory of the map img
    usr_id, cur_time = get_info()
    img_path = "./EmotionMap_Data/screenshots/" + usr_id + '/'

    try:
        ## upload the map on AWS S3 and get the url
        img_name = usr_id + '_Art#' + spot_num + '.png'
        img_url = upload_mapfile(img_path, img_name)
        print(img_url)

        res = {
            "version": "2.0",
            "template": {
                "outputs": [
                    {
                    "simpleText": {
                        "text": "{}에서 당신의 감정 상태는 다음과 같았어요!✨".format(spot_name)
                            }
                    },
                    {
                    "simpleImage": {
                        "imageUrl": img_url,
                        "altText": "당신의 감정지도:)"
                            }
                        }
                    ]
                }
            }
    
    except IOError:
        res = {
            "version": "2.0",
            "template": {
                "outputs": [
                    {
                        "simpleText": {
                            "text": "아직 지도가 만들어지지 않았어요!",
                            }
                        }
                    ]
                }
            }

    print(jsonify(res))

    time_e = time.time()
    time_00 = get_time()
    print('***returned when:', time_00)
    print('***LAP TIME :', time_e - time_s)
 
    return jsonify(res)


@app.route('/api/accept', methods=['POST'])
def accept_main():
    time_s = time.time()
    time_0 = get_time()
    print('***activated when:', time_0)
    usr_id, cur_time = get_info()

    folder_path = mk_directory(usr_id)
    print(folder_path)

    im, dir_name = get_pic(folder_path)
    time_1 = get_time()
    print('***got pic when:', time_1)
    
    # im = contrast_up(im)
    qr_data = decode(im)
    time_2 = get_time()
    print('***got qr when:', time_2)

    emo_attr = cog_face(dir_name)
    time_3 = get_time()
    print('***got emo when:', time_3)
    print(emo_attr)

    answer = return_msg(usr_id, cur_time, qr_data, emo_attr)

    ## generating a csv file and recording the emo data per participant
    if emo_attr != 0 and qr_data != 0:
        csv_columns, csv_data = what_to_write(usr_id, qr_data, emo_attr)

        try:
            csv_name = str(folder_path) + '/emo_qr.csv'
            file_exists = os.path.isfile(csv_name)
            with open(csv_name, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames = csv_columns)
                if not file_exists:
                    writer.writeheader()
                for data in csv_data:
                    print(data)
                    writer.writerow(data)
        except IOError:
            print('I/O ERROR')

    res = {
        "version": "2.0",
        "template": {
            "outputs": [
                {
                    "simpleText": {
                        "text": answer
                    }
                }
            ]
        }
    }

    print(res)
    print(jsonify(res))

    time_e = time.time()
    time_4 = get_time()
    print('***returned when:', time_4)
    print('***LAP TIME :', time_e - time_s)
    
    return jsonify(res)


# Main
if __name__ == "__main__":
    # app.run() # for production
    # app.run(debug=True) # for debugging purpose
    app.run(host=os.getenv('IP', '0.0.0.0'),port=int(os.getenv('PORT', 8080)))
