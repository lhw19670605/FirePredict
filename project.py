# 運行streamlit程序，加載的文件小於300M使用下面命令執行
#  streamlit run code/project.py --server.maxUploadSize  300
import Fire_Tools
import pandas as pd
import streamlit as st
#from fastai.vision.all import *
import pickle
import base64
import io
from PIL import Image
from fastai.learner import load_learner
from datetime import datetime, timedelta
import datetime
import cv2
import numpy as np
import zipfile
import time
import requests
import gdown
import torch

#from streamlit_imagegrid import streamlit_imagegrid

# set wide screen, but it is not beauty to look
# st.set_page_config(layout="wide")

#----------------sidebar------------------
#st.sidebar.header('Monitoring Area Parameters')

# part1: Load Model
#model_file = st.sidebar.file_uploader('Select model file', type=['pkl'])

# part2: Visual styles: Scan Model and Monitor Model
col1, col2 = st.sidebar.columns(2)

ScanModel = ('Whole', 'Gride')
MonitorModel = ('Single', 'Range')
# Defining Radio Button with index value
scan = col1.radio(
    "Scan Model",
    ScanModel,
    index = 0)
if scan == 'Whole':
    ScanM=0
else:
    ScanM=1

MonitorModel = ('Single', 'Range')
# Defining Radio Button with index value
monitor = col2.radio(
    "Monitor Model",
    MonitorModel,
    index = 0)
if monitor == 'Single':
    MonitorM=0
else:
    MonitorM=1

if MonitorM==0:
    db = col1.date_input('Begin Date', datetime.date(2023, 2, 18))
else:
    col1, col2 = st.sidebar.columns(2)
    db = col1.date_input('Begin Date', datetime.date(2023, 2, 2))
    de = col2.date_input('End Date', datetime.date(2023, 2, 10))
    


latitude = st.sidebar.slider('latitude of center',-90,90,-36,)
longitude = st.sidebar.slider('longitude of center',-180,180,-70)
width = st.sidebar.slider('width of region', 2, 10, 6, 2)
height = st.sidebar.slider('height of region', 2, 10, 4, 2)
    

data = { 'latitude' : latitude,
            'longitude' : longitude,
            'width' : width,
            'height': height}
df = pd.DataFrame(data, index=[0])

def fire_instances():
    option = st.sidebar.selectbox(
        'Instances of fires',
         ('first fire', 'second fire', 'third fire'))

    if option == 'first fire':
        db = datetime.date(2023, 2, 18)
        latitude = -36
        longitude = -70
        width =  6
        height = 4
    elif option == 'second fire':
        db = datetime.date(2022, 2, 18)
        latitude = -36
        longitude = -70
        width =  6
        height = 4
    else:
        db = datetime.date(2021, 2, 18)
        latitude = -36
        longitude = -70
        width =  6
        height = 4

# ---------------------------------------------------------------------
#
#                                context 
#
# ---------------------------------------------------------------------

def download_file_from_google_drive(file_id, destination):
    URL = "https://drive.google.com/uc?export=download&id=" + file_id

    session = requests.Session()

    response = session.get(URL, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

# 调用示例
#file_id = "https://drive.google.com/file/d/1JCt52PLLJncFk-N7OO0h0sOThuDr31OK/view?usp=sharing"
#model_file = "model.pkl"  # 保存文件的本地路径
#download_file_from_google_drive(file_id, model_file)

@st.cache_data
def down_modelfile(url):
    # 提取文件的 ID
    file_id = url.split('/')[-2]

    # 下载文件并保存为 model.pkl
    gdown.download(f'https://drive.google.com/uc?id={file_id}', 'model.pkl', quiet=False)
    return 'model.pkl'

# 共享链接
model_file_url = 'https://drive.google.com/file/d/1JCt52PLLJncFk-N7OO0h0sOThuDr31OK/view?usp=sharing'
model_file = down_modelfile(model_file_url)

# 加载模型
#learn_inf = load_learner('model.pkl')

# 假设你有一个名为"example.zip"的ZIP文件，它包含要解压的文件
#zip_file_path = "Archive 2.zip"

# 解压ZIP文件
#with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
#    zip_ref.extractall("resourse")
#model_file = "resourse/modelAll.pkl"
#wave_file ="resourse/distant-ambulance-siren-6108.mp3"
#model_url = "https://drive.google.com/file/d/1JCt52PLLJncFk-N7OO0h0sOThuDr31OK/view?usp=sharing"
# 下载预训练模型文件
#response = requests.get(model_url, stream=True)
#st.write(response)
# 将数据加载到io.BytesIO缓冲区
#model_file = io.BytesIO(response.raw.read())

#model_file = "/Users/hengwangli/MasterCourse/2023spring/Project/modelAll.pkl"



#--------------------------------------------------------------------------
#                             draw_single_image
#--------------------------------------------------------------------------
def draw_single_image(latitude, longitude, height,width, db):
    
    center = (latitude, longitude)
    dateb= str(db)
    size=[224,224]
            
    topLeft, bottomRight = FT.get_topLeft_bottomRight(center[0],center[1],height,width)
    img = FT.download_NASA_img(topLeft, bottomRight, dateb, size)
    
    image_size = (400, 400)
    image = cv2.imread(img)
    resized_image = cv2.resize(image, image_size)
    # 绘制矩形框
    start_point = (1, 1)  # 框的左上角坐标
    end_point = (image_size[0]-1,  image_size[1]-1)  # 框的右下角坐标
    aa=learn_inf.predict(img)
    if aa[0]=="Smoke":
        color = (0, 0, 255)  # red框的颜色，以BGR格式表示
    else:
        color = (0, 255, 0) # green

    thickness = 2  # 框的线条粗细
    annotated_image = cv2.rectangle(resized_image.copy(), start_point, end_point, color, thickness)

    # 将OpenCV图像转换为RGB格式，以便Streamlit显示
    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    empty1.write(db)
    empty2.image(annotated_image, width=image_size[0])

    return aa[0]

#--------------------------------------------------------------------------
#                             draw_gride_images
#--------------------------------------------------------------------------
def draw_gride_images(latitude, longitude, height,width, db):
    
    result = 'noSmoke'
    container_one.write(str(db))
    download = container_one.progress(0)

    center = (latitude, longitude)
    dateb= str(db)
    size=[224,224]
            
    #分成經緯度都是1度的小格
    latstep,longstep=1,1
    ranges, _, columns = FT.get_grid_points_pair(center[0],center[1],height,width,latstep,longstep)
    image_files=[]
    
    k=0
    for pair_points in ranges:
        k +=1
        download.progress((k+1)/2/len(ranges))
        topLeft, bottomRight = pair_points
        img = FT.download_NASA_img(topLeft, bottomRight, dateb, size)
        # Load the images and convert them to tensors
        image_files.append(img)
            
        #定義grid顯示參數：圖像大小，間距
        # 设置图像显示的大小和间距
        #6-140
        #這是寬屏尺寸
        #size_dic={2: 490, 4: 240, 6:150, 8:116, 10: 82}
        #這是正常寬度時的尺寸
        size_dic={2: 340, 4: 160, 6:100, 8:67, 10: 52}
        image_size = size_dic[columns] # 图像大小
        
    # Defining no of Rows
    for i in range(len(ranges)//columns):
        # Defining no. of columns with size
        cols = container_one.columns(columns)
        #image_size = (120, 120)
        for j in range(columns):
            download.progress(0.5+ (i*columns+j+1)/2/len(ranges))
            #print(len(ranges), columns,i,j,i*columns+j)
            img = image_files[i*columns+j]
            image = cv2.imread(img)
            resized_image = cv2.resize(image, (image_size, image_size))
            # 绘制矩形框
            start_point = (1, 1)  # 框的左上角坐标
            end_point = (image_size-1,  image_size-1)  # 框的右下角坐标
            aa=learn_inf.predict(img)
            if aa[0]=="Smoke":
                color = (0, 0, 255)  # red框的颜色，以BGR格式表示
                result = 'Smoke'
            else:
                color = (0, 255, 0) # green
                
            thickness = 2  # 框的线条粗细
            annotated_image = cv2.rectangle(resized_image.copy(), start_point, end_point, color, thickness)

            # 将OpenCV图像转换为RGB格式，以便Streamlit显示
            annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

            # 显示原始图像和带有颜色框的图像
            cols[j].image(annotated_image, width=image_size)

    return result
                    
def autoplay_audio(file_path: str):
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f"""
            <audio controls autoplay="True">
                 <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
        st.markdown(
            md,
            unsafe_allow_html=True,
        )


        #sound = st.empty()
        #sound.markdown(md, unsafe_allow_html=True)  # will display a st.audio with the sound you specified in the "src" of the html_string and autoplay it
        #time.sleep(2)  # wait for 2 seconds to finish the playing of the audio
        #sound.empty()  # optionally delete the element afterwards
   

st.title('Forest fire real-time monitoring system')

if st.button('Test'):
    #st.header('Test')

    # Check if a file has been selected
    if model_file is not None:
        learn_inf = load_learner(model_file)
        FT = Fire_Tools.Fire_Tools()
        #st.write(learn_inf)
        #st.write("ScanM,MonitorM is:",ScanM,MonitorM)
        #------------Style 1: Scan model=Whole, Monitor model=Single------------------
        if ScanM==0 and MonitorM==0:
            empty1 = st.empty()
            empty2 = st.empty()
            aa = draw_single_image(latitude, longitude, height, width, db)
            #if aa=="Smoke":
            #    st.write("# Auto-playing Audio!")
            #    autoplay_audio(wave_file)
                 
        #------------Style 2: Scan model=Gride, Monitor model=Single------------------
        if ScanM==1 and MonitorM==0:
            container_one = st.container()
            aa = draw_gride_images(latitude, longitude, height, width, db)
            

        #------------Style 3: Scan model=Whole, Monitor model=Range------------------
        if ScanM==0 and MonitorM==1:
            empty1 = st.empty()
            empty2 = st.empty()
            # 循环遍历日期
            current_date = db
            
            while current_date <= de:
                
                # 在这里执行你想要的操作，例如打印当前日期
                #image_placeholder.write(current_date)
                aa = draw_single_image(latitude, longitude, height, width, current_date)
                # 增加一天，继续下一轮循环
                current_date += timedelta(days=1)
                time.sleep(2)
                if aa == "Smoke":
                    break
                #st.experimental_rerun()
           
             
        #------------Style 4: Scan model=Gride, Monitor model=Range------------------
        if ScanM==1 and MonitorM==1:
            #image_placeholder = st.empty()
            # 循环遍历日期
            current_date = db
            
            while current_date <= de:
                # 在这里执行你想要的操作，例如打印当前日期
                #st.write(current_date)
                #container_one = st.container()
                placeholder = st.empty()
                with placeholder.container():
                    container_one = st.container()
                    #cempty_two = st.empty()
                    #cempty_one = st.empty()
                    aa = draw_gride_images(latitude, longitude, height, width, current_date)

                # 增加一天，继续下一轮循环
                current_date += timedelta(days=1)
                time.sleep(2)
                #container_one.empty
                if aa == "Smoke":
                    break
                else:
                    placeholder.empty()

                #st.empty()
                #st.experimental_rerun()

    else:
        st.markdown("<font color='red'> **Please load model file first!**</font>", unsafe_allow_html=True)

if MonitorM==1 and de < db:
    st.markdown("<font color='red'> **End Date should be later than Begin Date!**</font>", unsafe_allow_html=True)

# Defining Latitude and Longitude
#locate_map = pd.DataFrame(
#  np.random.randn(50, 2)/[10,10] + [15.4589, 75.0078],
#  columns = ['latitude', 'longitude'])
# Map Function
#st.map(locate_map)