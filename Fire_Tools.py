import os
import cv2
#import ray
import PIL
import nasapy
import random
import numpy as np
import pandas as pd
import urllib.request
import datetime
import torchvision
import IPython
#from IPython.display import Image
import torchvision.transforms as transforms
import time
import matplotlib
#from IPython.display import display
#from gtts import gTTS


#https://colab.research.google.com/drive/1v_PzQM5B1_eZ2vHZmqpjzmKcve_c_0LW
#https://mikemoschitto.medium.com/fire-detection-from-nasa-worldview-snapshots-59e93c20b1d6
#https://towardsdatascience.com/modern-parallel-and-distributed-python-a-quick-tutorial-on-ray-99f8d70369b8
#https://github.com/michaelmoschitto/wildfire-detection-from-satellite-images-ml/blob/master/model.py
class Fire_Tools:
    
    #############################################################
    #
    #   該函數將搜索區域分成雙向格柵，返回個分界點的坐標對
    #
    #  topLeft=(LatMax, LongMin)
    #  bottomRight=(LatMin, LongMax)
    #  latStep = 2, longStep = 2
    #############################################################
    def createSearchGrid(self, topLeft, bottomRight, latStep, longStep):
        # topLeft: (latitude, longitude) values of topLeft corner
        # bottomRight: (latitude, longitude) values of bottomRight corner
        # latStep: the density of divide of latitude
        # longStep: the density of divide of longitude

        #   Section defined portion of land into latdisxlongdis deg squares and save to dataframe
        #   returns: searchGrid - a 2D array of (lat, long)
        #latitude經度二維數組第一位，longitude緯度二維數組第二位
        LAT=0
        LONG=1

        #計算搜索區域高度與寬度
        searchHeight = abs(topLeft[LAT] - bottomRight[LAT])
        searchWidth = abs(topLeft[LONG] - bottomRight[LONG])

        #格子步長不得大於區域尺寸，如果大於區域尺寸，則用區域尺寸代替
        if latStep > searchHeight:
           latStep = searchHeight

        if longStep > searchWidth:
           longStep = searchWidth
        
        #由於劃分表格尺寸不一定能劃分整數小格，計算實際區域尺寸
        realHeight = searchHeight
        realWidth = searchWidth

        #計算高寬的格子數
        numLat = searchHeight // latStep + 1
        numLong = searchWidth // longStep + 1

        #如果不能整除，需要調整實際搜索區域大小及調整格子數量
        if searchHeight % latStep != 0:
            realHeight = (searchHeight//latStep + 1) * latStep
            numLat = numLat + 1

        if searchWidth % longStep != 0:
            realWidth = (searchWidth//longStep + 1) * longStep
            numLong = numLong + 1

        latMax = topLeft[LAT] + (realHeight - searchHeight) // 2
        longMin = topLeft[LONG] - (realWidth - searchWidth) // 2

        searchGrid = [[(0,0)] * (numLong) for x in range(numLat)]
        #print("len(searchGrid)",len(searchGrid))
        for x in range(len(searchGrid[0])):
            for y in range(len(searchGrid)):
                searchGrid[y][x] = (latMax - y*latStep, longMin + x*longStep)

        return searchGrid


    #######################################################################################
    #
    #  最簡單下載NASA衛星圖像的方法
    #  從NASA網站下載指定日期，指定經緯度區域的衛星圖像，按指定圖像大小輸出
    #  LatMin=-41
    #  LatMax=-30
    #  LongMin=-80
    #  LongMax=-64
    #  topLeft=(LatMax, LongMin)
    #  bottomRight=(LatMin, LongMax)
    #  size=[900,600]
    #  date="2023-02-03"
    #  fire下載的圖像中要不要加火災紅點標記
    #######################################################################################
    def download_NASA_img(self, topLeft, bottomRight, date, size, image_dir="", fire=False):
        LAT=0
        LONG=1
        #多火災的紅點標誌
        if fire:
            URL="""https://wvs.earthdata.nasa.gov/api/v1/snapshot?REQUEST=GetSnapshot&TIME={}&BBOX={},{},{},{}&CRS=EPSG:4326&LAYERS=MODIS_Terra_CorrectedReflectance_TrueColor,Coastlines,MODIS_Terra_Thermal_Anomalies_All&FORMAT=image/jpeg&WIDTH={}&HEIGHT={}""".format(date,bottomRight[LAT], topLeft[LONG],topLeft[LAT], bottomRight[LONG], size[0], size[1])
        else:
            URL="""https://wvs.earthdata.nasa.gov/api/v1/snapshot?REQUEST=GetSnapshot&TIME={}&BBOX={},{},{},{}&CRS=EPSG:4326&LAYERS=MODIS_Terra_CorrectedReflectance_TrueColor,Coastlines&FORMAT=image/jpeg&WIDTH={}&HEIGHT={}""".format(date,bottomRight[LAT], topLeft[LONG],topLeft[LAT], bottomRight[LONG], size[0], size[1])
       
        if image_dir == "":
            image_dir = os.getcwd()
            
        if not os.path.exists(image_dir):
            os.mkdir(image_dir)

        #Creating title for image:
        lat_cen=(topLeft[LAT]+bottomRight[LAT])//2
        long_cen=(topLeft[LONG]+bottomRight[LONG])//2
        height = abs(topLeft[LAT]-bottomRight[LAT])
        width = abs(topLeft[LONG]-bottomRight[LONG])
        #title = date + "LAT" + str(lat_cen) + "LONG"+ str(long_cen) + ".jpg"
        title = "LAT" + f"{lat_cen:04d}" + "LONG"+ f"{long_cen:04d}" + "-" + f"{height:02d}" + "x" + f"{width:02d}" + "D" + date + ".jpg"
        
        title = os.path.join(image_dir,title)

        #Downloading the image:
        urllib.request.urlretrieve(URL, title)
        #print("neibu")
        IPython.display.Image(title)

        return title

    ############################################################
    #
    #               時間，日期有關的函數
    #
    ############################################################
  
    ############################################################
    #       得到昨天的日期
    #  today_date = '2013-05-11'
    ############################################################
    def last_day_date(self, today_date):
       # 将日期字符串转换成 datetime.date 对象
       date = datetime.datetime.strptime(today_date, '%Y-%m-%d').date()
       # 计算前一天日期
       last_day = date + datetime.timedelta(days=-1)
       # 将下一天日期转换成字符串
       last_day_str = last_day.strftime('%Y-%m-%d')

       return last_day_str

    ############################################################
    #     得到明天的日期
    #  today_date = '2013-05-11'
    ############################################################
    def next_day_date(self, today_date):
       # 将日期字符串转换成 datetime.date 对象
       date = datetime.datetime.strptime(today_date, '%Y-%m-%d').date()
       # 计算下一天日期
       next_day = date + datetime.timedelta(days=1)
       # 将下一天日期转换成字符串
       next_day_str = next_day.strftime('%Y-%m-%d')

       return next_day_str

    ############################################################
    #    得到今天的日期
    ############################################################
    def today_date(self):
       return datetime.date.today().strftime("%Y-%m-%d")

    ############################################################
    #   在一個區間內隨機得到一天
    #  start_date = '2013-05-11'
    #  end_date = '2013-05-11'
    ############################################################
    def pick_one_day_random(self, start_date, end_date):
       if isinstance(start_date, str):
           start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d').date()
           end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d').date()
        
       random_date = start_date + datetime.timedelta(days=random.randint(0, (end_date - start_date).days))
       return random_date.strftime('%Y-%m-%d')

    ############################################################
    #   得到現在的時間
    ############################################################
    def now_time(self):
       now = datetime.datetime.now()
       return now.time()

    ############################################################
    #   得到從某天開始向前幾天的時間list
    #  date = '2013-05-11'
    #  days = 5
    ############################################################
    def get_dates_list(self, end_date, days):
       dates = pd.date_range(end = end_date, periods=days)
       #print("datestype=",type(dates))
       return dates.strftime('%Y-%m-%d')
    
     ############################################################
    #   得到從某天到某天的總共天數
    #  start_date = '2013-05-11'
    #  end_date = '2013-05-14'
    #  4天
    ############################################################
    def get_dates_days(self, start_date, end_date):
        dstart_date = datetime.datetime.strptime(start_date, '%Y-%m-%d').date()
        dend_date = datetime.datetime.strptime(end_date, '%Y-%m-%d').date()
        #print("datestype=",type(dates))
        return (dend_date - dstart_date).days
 
    def get_topLeft_bottomRight(self,centerlat,centerlong,latRang,longRang):
        LatMin = centerlat - latRang//2
        LatMax = centerlat + latRang//2
        LongMin= centerlong - longRang//2
        LongMax= centerlong + longRang//2
        topLeft=(LatMax, LongMin)
        bottomRight=(LatMin, LongMax)
        return topLeft, bottomRight

    ############################################################
    #
    #       下載NASA提供的衛星照片，指定日期，未來的日期會給一個黑圖
    #
    #  date = '2013-05-11'
    #  image_dir = "/down_no/img"
    ############################################################
    def download_NASA_photo(self, date, image_dir):
       #Initialize Nasa class by creating an object:
       k = "523p5hPYHGzafYGLCkqa54kKMTV2vbP0XcPxkcLm"
       nasa = nasapy.Nasa(key = k)

       apod = nasa.picture_of_the_day(date, hd=False)
    
       data = []
       if apod['media_type'] == 'image':
               if 'hdurl' in apod.keys():
                   data.append({'date':apod['date'], 'title': apod['title'],'hdurl': apod['hdurl']})
            

       #Checking if the directory already exists?
       dir_res = os.path.exists(image_dir)
    
       #If it doesn't exist then make a new directory:
       if (dir_res==False):
           os.makedirs(image_dir)

       #If it exist then print a statement:
       else:
           #print("Directory already exists!\n")    
           pass   


       #Retrieving the image:   
       i = 1
       for img in data: 
           #Creating title for image:
           title = img["date"] +"_" + f"{i:03d}" + ".jpg"
           #print(title)
           #Downloading the image:
           urllib.request.urlretrieve(img['hdurl'], os.path.join(image_dir,title))
           i += 1

       return i-1



    ############################################################
    #       從NASA下載指定數量的衛星照片：開始日期，數量
    ############################################################
    # 這個數據沒有用途，因為這些圖片都是宇宙空間的一些衛星照片，而不是地球的
    # 衛星圖片，本課題研究的是通過地球的衛星照片預測火災的發生，數據中不會出
    # 現宇宙空間照片的。所以這些照片刪除
    #  nums=1000
    #  today_str = '2013-05-11'
    #  image_dir = "/down_no/img"
    ############################################################
    def download_NASA_photo_nums(self, today_str, image_dir,nums):
        i=0
        
        while i < nums:
            num = self.download_NASA_photo(today_str,image_dir)
            i += num
            #循環計算前一天，直到下載數量滿足要求位置
            today_str = self.last_day_date(today_str)
            #print("i=",i)
            if i % 100 ==0:
                print("There are {} images have been download.".format(i))


    ############################################################
    #      隨機的從NASA下載指定經緯度，指定日期的衛星照片
    #  因為地球上70%是海洋，還有一些特殊情況，得到的會是一張黑或白的
    #  沒有用途的照片，這樣的照片尺寸會比較小，所以將小於1/6滿尺寸的
    #  文件去掉，基本就可以了，這樣搜索照片將是需求量的6倍左右（這兩
    #  個6沒有直接關係，是經驗數據）
    #  nums=1000
    #  start_date = '2013-05-11'
    #  end_date = '2023-05-11'
    #  img_size=(600,600)
    #  image_dir = "/down_no/img"
    #  poly按逆時針排序的一組點坐標,如果沒有提供該參數，則在整個範圍內隨機選擇
    ############################################################
    def download_NASA_image_nums(self, start_date, end_date, img_size, image_dir, nums, ratio=5.5, poly=None, fire=False):
       i = 0
       j = 0
       while i < nums : 
           j += 1
           #隨機選擇經緯度，在2～10中隨機選擇經緯度成像區域
           #由於隨機選擇的地點跟火災地點的圖像差別很大，許多可能是海上的雲圖，圖像以白色為主要色，
           #因此改為在陸地範圍內隨機選擇地點
           if poly:
                lat, long = self.random_point_in_polygon(poly)
           else:
                lat = random.randint(-179, 179)
                long = random.randint(-89,89)

           edge = random.randint(2, 10)
           minedge = min(abs(abs(-180)-abs(lat)),abs(abs(90)-abs(long)))
           edge = min(2*minedge,edge)
           topLeft=(lat + edge//2, long - edge//2)
           bottomRight=(lat - edge//2, long +edge//2)
           date_str=self.pick_one_day_random(start_date, end_date)
           img=self.download_NASA_img(topLeft, bottomRight, date_str, img_size, image_dir, fire)
           #判斷如果這個圖像文件的大小太小，就是一個顏色的，就刪掉該文件
           file_size = os.path.getsize(img)
           #print("文件大小：", file_size, "字节")
           limitsize=img_size[0]*img_size[1]/ratio
           printyes = False
           if file_size<limitsize:
              os.remove(img)
           else:
              i += 1   

           if (not printyes):
               printyes= True
               del img

           if (i % 50 == 0) and (i !=0) and printyes:
                printyes = False
                print("There are {} images have been download.".format(i))
                #print("Current time is:", now_time())

       print("The required quantity is:",i,",the actual download quantity is:",j)


    ############################################################
    #      從NASA下載指定位置，連續幾天的衛星照片
    #  
    #  topLeft=(LatMax, LongMin)
    #  bottomRight=(LatMin, LongMax)
    #  end_date = '2023-05-11'
    #  days = 5
    #  img_size=(600,600)
    #  image_dir = "/down_no/img"
    ############################################################
    def download_NASA_image_days(self, topLeft, bottomRight, end_date, days, img_size, image_dir, fire=False):
       dates = self.get_dates_list(end_date, days)
       for d in dates: 
           self.download_NASA_img(topLeft, bottomRight, d, img_size, image_dir, fire)
           
       print("The images download have finished!")


    ############################################################
    #     定义图像增强变换: 包括幾何變換和色彩變化
    # image = cv2.imread(filepath)
    # transformation: 'flip', 'rotate', 'crop', 'warp', 'scale', 'blur', 'color'
    ############################################################
    def apply_augument_transformation(self, image, transformation):
        if transformation == 'flip':
            image = cv2.flip(image, 1)  # 水平翻转
        elif transformation == 'rotate':
            angle = random.randint(-30, 30)
            rows, cols, _ = image.shape
            matrix = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
            image = cv2.warpAffine(image, matrix, (cols, rows))
        elif transformation == 'crop':
            x = random.randint(0, 100)
            y = random.randint(0, 100)
            h, w, _ = image.shape
            image = image[y:h-y, x:w-x]
        elif transformation == 'warp':
            rows, cols, _ = image.shape
            random_points = np.int32([[0, 0], [cols, 0], [0, rows]])
            #print(random_points)
            random_points += random.randint(-50, 50)
            matrix = cv2.getAffineTransform(np.float32(random_points[:3]), np.float32(random_points[:3]) + random.randint(-30, 30))
            image = cv2.warpAffine(image, matrix, (cols, rows))
        elif transformation == 'scale':
            scale = random.uniform(0.8, 1.2)
            rows, cols, _ = image.shape
            image = cv2.resize(image, (int(cols*scale), int(rows*scale)))
        elif transformation == 'blur':
            ksize = random.choice([3, 5, 7])
            image = cv2.blur(image, (ksize, ksize))
        elif transformation == 'color':
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)

            # 色调调整
            hue_shift = random.randint(-10, 10)
            h = np.mod(h + hue_shift, 180).astype(np.uint8)

            # 饱和度调整
            saturation_scale = random.uniform(0.7, 1.3)
            s = np.clip(s * saturation_scale, 0, 255).astype(np.uint8)

            # 亮度调整
            value_scale = random.uniform(0.7, 1.3)
            v = np.clip(v * value_scale, 0, 255).astype(np.uint8)

            hsv = cv2.merge([h, s, v])
            image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        elif transformation == 'erase':
            rows, cols, _ = image.shape
            erase_height = random.randint(0, rows // 4)
            erase_width = random.randint(0, cols // 4)
            erase_x = random.randint(0, cols - erase_width)
            erase_y = random.randint(0, rows - erase_height)
            image[erase_y:erase_y+erase_height, erase_x:erase_x+erase_width] = 0
        elif transformation == 'fill':
            rows, cols, _ = image.shape
            erase_height = random.randint(0, rows // 4)
            erase_width = random.randint(0, cols // 4)
            erase_x = random.randint(0, cols - erase_width)
            erase_y = random.randint(0, rows - erase_height)
            fill_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            image[erase_y:erase_y+erase_height, erase_x:erase_x+erase_width] = fill_color
        return image


    ############################################################
    #     調用定义图像增强变换函數對整個目錄下面的圖形文件進行變化
    #  directory: 變化目錄
    #  圖形文件類型： ".jpg", ".jpeg", ".png", ".webp"
    #  圖形增強方式： 'flip', 'rotate', 'crop', 'warp', 'scale', 'blur', 'color'
    ############################################################
    def augument_directory(self, directory):
        # 遍历目录下的文件
       for filename in os.listdir(directory):
           # 拼接文件路径
           filepath = os.path.join(directory, filename)
        
           # 仅处理图片文件
           if (not ("_augu" in filename)) and filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                # 随机选择一种图像增强技术（翻转、旋转、裁剪、变形、缩放）
                transformation = random.choice(['flip', 'rotate', 'crop', 'warp', 'scale', 'blur', 'color'])#, 'erase', 'fill'])
                #print("transformation", transformation)
                # 应用图像增强技术
                # 读取照片
                image = cv2.imread(filepath)
            
                image = self.apply_augument_transformation(image, transformation)
        
                # 显示处理后的图像
                #cv2.imshow("Augmented Image", image)
                #cv2.waitKey(0)
                # 保存图像
                #從後面開始找
                #dot_position = filepath.rindex(".")
                #從前面開始找
                dot_position = filepath.index(".")
                if filepath[0:dot_position][-1] == "_":
                    output_name=filepath[0:dot_position] + "augu" + ".jpg"#filepath[dot_position:]
                else:
                    output_name=filepath[0:dot_position] + "_augu" + ".jpg"#filepath[dot_position:]
        
                cv2.imwrite(output_name, image)
                #print(output_name)
                #Image(output_name)

  


    ############################################################
    #                  修改圖像文件類型為jpg
    #  directory: 搜索路徑，directory = "/Users/image"
    #  圖形文件類型：".jpeg", ".png", ".webp"
    ############################################################
    def change_image_suffix(self, directory):
        # 遍历目录下的文件
        for filename in os.listdir(directory):
            # 拼接文件路径
            filepath = os.path.join(directory, filename)
        
            # 仅处理图片文件
            if filename.lower().endswith((".jpeg", ".png", ".webp")):
                # 读取照片
                image = cv2.imread(filepath)
            
                #從前面開始找
                dot_position = filepath.index(".")
                output_name=filepath[0:dot_position] + ".jpg"#filepath[dot_position:]
            
                cv2.imwrite(output_name, image)
                os.remove(filepath)



    ############################################################
    #             批量修改圖像尺寸，輸出文件類型jpg
    #  input_dir：圖像輸入路徑，directory = "/Users/image"
    #  output_dir：圖像輸出路徑，directory = "/Users/image"
    #  target_size：輸出圖像尺寸，img_size=(600,600)
    #  prefix：文件名前綴，圖形文件名按順序改為以4位數字表示的文件名，
    ############################################################
    def resize_img(self, input_dir,output_dir,target_size,prefix,i=0):
        # Iterate over all files in the input directory
        
        for filename in os.listdir(input_dir):
            if filename.endswith(".jpeg"):  # Specify the file extensions you want to process
               input_path = os.path.join(input_dir, filename) 
               output_name = prefix + f"{i:04d}" + ".jpg"
               output_path = os.path.join(output_dir, output_name)

               # Open the image file
               image = PIL.Image.open(input_path)

               # Resize the image to the target size
               resized_image = image.resize(target_size)

               # Save the resized image to the output directory
               resized_image.save(output_path)

               # Close the image file
               image.close()
               i += 1
        print("Image resizing complete.")

   ############################################################
    #             判定一個點是不是在曲線圍城的範圍之內
    #  為random_point_in_polygon服務
    #  point：一個點
    #  polygon，點組，像：poly1 = [(-52, -73),(-42, -74),(-37, -73)]
    ############################################################
    def is_point_in_polygon(self, point, polygon):
        x, y = point
        n = len(polygon)
        inside = False

        p1x, p1y = polygon[0]
        for i in range(n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside
    
    ############################################################
    #             在一個區域中隨意選出一點
    #  polygon，點組，像：poly1 = [(-52, -73),(-42, -74),(-37, -73)]
    ############################################################
    def random_point_in_polygon(self, polygon):
        min_x = min(polygon, key=lambda p: p[0])[0]
        max_x = max(polygon, key=lambda p: p[0])[0]
        min_y = min(polygon, key=lambda p: p[1])[1]
        max_y = max(polygon, key=lambda p: p[1])[1]

        while True:
            point = (random.randint(min_x, max_x), random.randint(min_y, max_y))

            if self.is_point_in_polygon(point, polygon):
                return point

    ############################################################
    #   給定一個區域，得到1度x1度的小格的topleft和bottomright的對應點對
    #  
    ############################################################
    #給定一個區域，按1度的步長將其化為小格區域，以便讀取相應區域的圖
    def get_grid_points_pair(self, centerlat,centerlong,latRang,longRang,latStep,longStep):
        topLeft, bottomRight = self.get_topLeft_bottomRight(centerlat,centerlong,latRang,longRang)
        #定義為1度x1度的小格
        latStep=latStep
        longStep=longStep
        searchGrid=self.createSearchGrid(topLeft, bottomRight,latStep, longStep)
        ranges=[]
        for i in range(len(searchGrid)-1):
            for j in range(len(searchGrid[0])-1):
                pair_points=[searchGrid[i][j],searchGrid[i+1][j+1]]
                ranges.append(pair_points)
        # return pair_points, rows, columns
        return ranges,len(searchGrid)-1,len(searchGrid[0])-1
    
    #######################################################################################
    #   給定一個區域網個經緯度點對，將相應的衛星圖像拼接成一張圖片，使用grid形式
    #
    #  ranges--pair_points of lat and long by "get_grid_points_pair" function created
    #  date="2023-02-18"
    #  size=[224,224]
    #  columns--"get_grid_points_pair" function created
    #######################################################################################
    def puzzle_satellite_images(self, ranges, date, size, columns):
        image_files=[]
        for pair_points in ranges:
            topLeft, bottomRight = pair_points
            img = self.download_NASA_img(topLeft, bottomRight, date, size)
            # Load the images and convert them to tensors
            image_files.append(img)

        images = [transforms.ToTensor()(PIL.Image.open(file)) for file in image_files]

        # Create a grid of images
        grid = torchvision.utils.make_grid(images, nrow=columns, padding=1)

        # Display the grid using matplotlib
        plt.imshow(grid.permute(1, 2, 0))
        plt.axis('off')
        plt.show()
        return images, image_files


    #######################################################################################
    #   給定一個區域網個經緯度點對，將相應的衛星圖像一張一張的順序顯示在屏幕上
    #
    #  ranges--pair_points of lat and long by "get_grid_points_pair" function created
    #  date="2023-02-18"
    #  size=[224,224]
    #  columns--"get_grid_points_pair" function created
    #######################################################################################
    def puzzle_satellite_images_One_by_One(self, ranges, date, size, columns):
        image_files=[]
        for pair_points in ranges:
            topLeft, bottomRight = pair_points
            img = self.download_NASA_img(topLeft, bottomRight, date, size)
            # Load the images and convert them to tensors
            image_files.append(img)

        # 循环逐个显示图像
        for j in range(len(ranges)):
            fig, axes = matplotlib.pyplot.subplots(len(ranges)//columns, columns, figsize=(10, 6))
            for i, ax in enumerate(axes.flatten()):
                # 读取图像文件并显示
                if i < j:
                    image = matplotlib.pyplot.imread(image_files[i])
                    #print(image)
                    ax.imshow(image)
                    ax.axis('off')

            # 当达到当前要显示的图像数量时，等待指定的时间
            matplotlib.pyplot.subplots_adjust(wspace=0.01, hspace=0.01)
            matplotlib.pyplot.tight_layout()
            matplotlib.pyplot.show()
            # 控制显示时间，单位为秒（例如延迟2秒：time.sleep(2)）
            time.sleep(1)  # 修改这里的延迟时间
            # 关闭图像窗口
            matplotlib.pyplot.close(fig)
            #matplotlib.pyplot.clf()

        return image_files

    #######################################################################################
    #   給定一個區域網個經緯度點對，將相應的衛星圖像顯示在一張圖上，使用subplot形式
    #
    #  ranges--pair_points of lat and long by "get_grid_points_pair" function created
    #  date="2023-02-18"
    #  size=[224,224]
    #  columns--"get_grid_points_pair" function created
    #  調整圖塊間距首先要保證高寬與實際比例對應，否則圖塊間距會根據實際尺寸自動調整
    #######################################################################################
    def puzzle_satellite_images_Subplots(self, ranges, date, size, columns):
        image_files=[]
        for pair_points in ranges:
            topLeft, bottomRight = pair_points
            img = self.download_NASA_img(topLeft, bottomRight, date, size)
            # Load the images and convert them to tensors
            image_files.append(img)

        fig, axes = matplotlib.pyplot.subplots(len(ranges)//columns, columns, figsize=(9, 6), sharex=True, sharey= True)

        # 逐个显示图像
        for i, ax in enumerate(axes.flatten()):
            # 读取图像文件并显示
            image = matplotlib.pyplot.imread(image_files[i])
            ax.imshow(image)
            #print("i=",i)
            if i == len(ranges)//2:
                ax.spines['left'].set_color('red')
                ax.spines['right'].set_color('red')
                ax.spines['top'].set_color('red')
                ax.spines['bottom'].set_color('red')
                ax.spines['left'].set_linewidth(2)
                ax.spines['right'].set_linewidth(2)
                ax.spines['top'].set_linewidth(2)
                ax.spines['bottom'].set_linewidth(2)
                ax.set_xticks([])  # 隐藏 x 轴刻度数字
                ax.set_yticks([])  # 隐藏 y 轴刻度数字
            else:
                ax.spines['left'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.axis('off')

        # 调整子图之间的间距
        matplotlib.pyplot.subplots_adjust(wspace=0.01, hspace=0.01)
        #plt.tight_layout()

        # 显示图像网格
        matplotlib.pyplot.show()

        return image_files

#---------------------------------------
#
#---------------------------------------
############################################################
#                 美國歷年火災情況圖示
#
############################################################
import pandas as pd
import matplotlib.pyplot as plt
def wildfire_graph_US():
    df=pd.read_csv("paper/USwildfiressumery.csv",index_col="Year",header=0)
    #print(df)
    df['Fires'] = df['Fires'].str.replace(',', '').astype(int)
    df['Acres'] = df['Acres'].str.replace(',', '').astype(int)
    #df=df.astype(float)
    print(df.dtypes)
    df[["Fires"]].plot()
    #plt.show()
    df[["Acres"]].plot()

############################################################
#                 陸地經緯度坐標邊界
#
############################################################
def latitud_longitude_edge(i):
    if i>5: i=5
    poly =[]
    poly1 = [(-52, -73),(-42, -74),(-37, -73),
             (-20, -70),(-18, -71),(-4, -81),
             (12, -71),(10, -62),(-6, -34),
             (-22, -41),(-25, -48),(-34, -54),
             (-46, -67)]
    poly2 = [(7, -81), (15, -94), (19, -105), 
         (22, -105), (35, -120), (47, -124), 
         (60, -143), (60, -152), (58, -157), 
         (61, -165), (68, -165), (71, -156), 
         (68, -135), (70, -127), (54, -82), 
         (59, -64), (47, -53), (29, -94), 
         (22, -97), (18, -94), (21, -87), 
         (15, -87), (15, -84)]
    poly3 = [(-34, 19), (-17, 11), (-8, 13), 
         (-1, 9), (4, 9), (4, -8), 
         (14, -17), (15, 38), (2, 45), 
         (-5, 38), (-15, 40), (-20, 34), 
         (-24, 35), (-33, 26)]
    poly4 = [(-25, 45), (-24, 43), (-22, 43), 
         (-20, 44), (-17, 44), (-12, 49), 
         (-15, 50), (-24, 47)]
    poly5 = [(-38, 143), (-37, 139), (-33, 137), 
         (-34, 135), (-31, 131), (-32, 125), 
         (-35, 116), (-23, 113), (-19, 121), 
         (-14, 127), (-12, 136), (-15, 135), 
         (-17, 140), (-11, 142), (-25, 152), 
         (-32, 152), (-37, 149)]
    poly6 = [(8, 77), (20, 71), (25, 66), 
         (25, 57), (36, 35), (44, 8), 
         (36, -2), (36, -5), (37, -8), 
         (43, -9), (43, -1), (46, -1), 
         (48, -4), (54, 8), (61, 4), 
         (71, 24), (67, 41), (69, 60), 
         (77, 104), (76, 114), (73, 113), 
         (71, 132), (72, 143), (69, 175), 
         (66, 179), (65, 179), 
         (64, 178), (62, 179), (51, 156), 
         (56, 155), (61, 164), (61, 156), 
         (59, 152), (59, 142), (54, 135), 
         (53, 141), (39, 127), (34, 119), 
         (29, 122), (22, 116), (19, 105), 
         (15, 109), (11, 109), (15, 94), 
         (22, 91), (15, 80)]
    poly.append(poly1)
    poly.append(poly2)
    poly.append(poly3)
    poly.append(poly4)
    poly.append(poly5)
    poly.append(poly6)
    return poly[i]


# 調用區域網個劃分函數
topLeft = (45, -127)
bottomRight = (32, -117)
latStep=3
longStep=3

FT = Fire_Tools()
#searchGrid=FT.createSearchGrid(topLeft, bottomRight,latStep, longStep)
#searchGrid

#調用衛星圖片下載函數下載圖片
LatMin=-41
LatMax=-30
LongMin=-80
LongMax=-64
topLeft=(LatMax, LongMin)
bottomRight=(LatMin, LongMax)
size=[900,600]
date="2023-02-03"
#Displaying an image:
#img = FT.download_NASA_img(topLeft, bottomRight, date, size,False)
#Image(img)

#調用的到幾天時間list函數
end_date = '2020-08-15'
days=5
dates = FT.get_dates_list(end_date, days)
#print(dates)
#for d in dates:
#    print(d)

#使用明天和昨天的函數得到明天和昨天的日期
# 假设有一个日期字符串 '2022-05-11'
today_str = '2022-05-11'
yesterday_str = FT.last_day_date(today_str)
tomorrow_str = FT.next_day_date(today_str)
#print("Today is: ", today_str)
#print("Tomorrow is: ", tomorrow_str)
#print("Yesterday is: ", yesterday_str)
start_date = '2022-05-11'
end_date = '2023-05-11'
random_date = FT.pick_one_day_random(start_date, end_date)
#print(random_date)

#調用從NASA下載網站提供的衛星照片函數
   #Path of the directory:
#image_dir = "down_img"
#today_str = '2022-05-12'
#num = FT.download_NASA_photo(today_str,image_dir)

#調用批量下載NASA圖片函數
nums=100
image_dir = "down_img"
date = FT.today_date()
#print("Current time is:", now_time())
#FT.download_NASA_photo_nums(date, image_dir,nums)
#print("Current time is:", now_time())

#-----------------------------
#隨機生成一定數量的衛星圖片，位置隨機選擇經緯度
#日期在10年之內隨機選一天，隨機生成nums個圖片
nums=100
start_date = '2013-05-11'
end_date = '2023-05-11'
img_size=(600,600)
image_dir = "/Users/hengwangli/MasterCourse/2023spring/Project/down_no"
ratio=10.0#=5.5
#FT.download_NASA_image_nums(start_date, end_date, img_size, image_dir, nums, ratio)
    
#當前目錄
#directory = os.getcwd()
directory = "/Users/hengwangli/MasterCourse/2023spring/Project/origin_image/SmokeFire"
#FT.augument_directory(directory)

#--------------------------------
#改後綴為jpg
# 指定目录路径
directory = "/Users/hengwangli/MasterCourse/2023spring/Project/origin_image2/nosmoke"
#當前目錄
#directory = os.getcwd()
#FT.change_image_suffix(directory)

#-----------------------------
#批量改尺寸
input_dir = "/Users/hengwangli/MasterCourse/2023spring/Project/origin_image1/Smoke"
output_dir = "/Users/hengwangli/MasterCourse/2023spring/Project/image/Smoke"
prefix="smoke"
target_size = (224, 224)
#FT.resize_img(input_dir,output_dir,target_size,prefix,i=1)

input_dir = "/Users/hengwangli/MasterCourse/2023spring/Project/origin_image1/NoSmoke"
output_dir = "/Users/hengwangli/MasterCourse/2023spring/Project/image/NoSmoke"
prefix="nosmoke"
#FT.resize_img(input_dir,output_dir,target_size,prefix,i=1)

#--------------------------
#下載連續幾天的照片
lat = 55
long = -117
latR = 10
longR = 14
topLeft, bottomRight = FT.get_topLeft_bottomRight(lat,long,latR,longR)
end_date = '2023-05-10'
days = 10
img_size=(224,224)
image_dir = ""
#FT.download_NASA_image_days(topLeft, bottomRight, end_date, days, img_size, image_dir)
 
 #--------------------------
 #下載連續幾天的照片
topLeft=(-36, -72)
bottomRight=(-37, -71)
end_date = '2023-02-05'
days = 7
img_size=(224,224)
image_dir = ""
#FT.download_NASA_image_days(topLeft, bottomRight, end_date, days, img_size, image_dir)

#------------------------------------
#在陸地範圍內隨機下載（也就是在陸地範圍內選擇經緯度），整個陸地分為6塊，每塊根據面積大小調整下載數量
#日期在10年之內隨機選一天，隨機生成nums個圖片
nums=10
start_date = '2013-05-11'
end_date = '2023-05-14'
img_size=(600,600)
image_dir = "down_no"
#FT.download_NASA_image_nums(start_date, end_date, img_size, image_dir, nums, latitud_longitude_edge(0))

#----------------------------
#df = pd.read_csv("/Users/hengwangli/MasterCourse/2023spring/Project/image/fireRecords1.csv",header=0)
#for i in range(df.shape[0]):
#    #print("i=",i)
#    img_size=(600,600)
#    lat = df.at[i,"Lantitude"]
#    long = df.at[i,"Longitude"]
#    latR = df.at[i,"LanRange"]
#    longR = df.at[i,"LongRange"]
#    Start_Date = df.at[i,"CutBegin"]
#    End_Date = df.at[i,"CutEnd"]
#    days =FT.get_dates_days(Start_Date,End_Date)
#    topLeft, bottomRight = FT.get_topLeft_bottomRight(lat,long,latR,longR)
#    image_dir = "/Users/hengwangli/MasterCourse/2023spring/Project/origin_image2"
    #if i<3:
    #    print(End_Date)
    #    print(days)
    #    print(topLeft, bottomRight)
    
    #FT.download_NASA_image_days(topLeft, bottomRight, End_Date, days, img_size, image_dir, False)

#---------------------------------
#臨時增加一些參考圖片
lat = 34
long = -121
latR = 6
longR = 8
End_Date = '2020-12-4'
days = 3
topLeft, bottomRight = FT.get_topLeft_bottomRight(lat,long,latR,longR)
#image_dir = "/Users/hengwangli/MasterCourse/2023spring/Project/origin_image2"
#FT.download_NASA_image_days(topLeft, bottomRight, End_Date, days, img_size, image_dir, False)

#-----------------------------------------
#補充一些非火災照片，平衡數量
#df = pd.read_csv("/Users/hengwangli/MasterCourse/2023spring/Project/image/fireRecords.csv",header=0)
#for i in range(df.shape[0]):
#    #print("i=",i)
#    img_size=(600,600)
#    lat = df.at[i,"lat"]
#    long = df.at[i,"long"]
#    latR = df.at[i,"latR"]
#    longR = df.at[i,"longR"]
#    Start_Date = df.at[i,"cutB"]
#    End_Date = df.at[i,"cutE"]
#    days =FT.get_dates_days(Start_Date,End_Date)
#    topLeft, bottomRight = FT.get_topLeft_bottomRight(lat,long,latR,longR)
#    image_dir = "/Users/hengwangli/MasterCourse/2023spring/Project/origin_image2"
    #if i<3:
    #    print(End_Date)
    #    print(days)
    #    print(topLeft, bottomRight)
    
    #FT.download_NASA_image_days(topLeft, bottomRight, End_Date, days, img_size, image_dir, False)

#調用區域衛星圖拼圖函數
center=(-36,-69)
height=4
width=6
date="2023-02-18"
#print("111")
size=[224,224]
#ranges, rows, columns = FT.get_grid_points_pair(center[0],center[1],height,width)
#print("222")
#FT.puzzle_satellite_images(ranges, date, size, columns)

#FT.puzzle_satellite_images_One_by_One(ranges, date, size, columns)
#ranges = FT.puzzle_satellite_images_Subplots(ranges, date, size, columns)
#print(ranges)

#topLeft, bottomRight= FT.get_topLeft_bottomRight(center[0],center[1],height,width)
#img = FT.download_NASA_img(topLeft, bottomRight, date, size)
#print(img)