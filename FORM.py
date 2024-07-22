from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn import tree
import pandas as pd
import numpy as np
from tkinter import messagebox
from tkinter.ttk import *
from tkinter import *

#Hàm chuẩn hóa dữ liệu
def DataEncoder(dataStr):
	dataConvert = dataStr
	for i, j in enumerate(dataConvert):
		for k in range(0,len(dataConvert[0])):
			if(j[k]=="bad"):j[k] = 1
			elif(j[k]=="ordinary"):j[k] = 2
			elif(j[k]=="good"):j[k] = 3
			elif(j[k]=="low"):j[k] = 4
			elif(j[k]=="med"):j[k] = 5
			elif(j[k]=="high"):j[k] = 6
			elif(j[k]=="small"):j[k] = 7
			elif(j[k]=="big"):j[k] = 8
			elif(j[k]=="vbig"):j[k] = 9
			elif(j[k]=="slow"):j[k] = 10
			elif(j[k]=="normal"):j[k] = 11
			elif(j[k]=="fast"):j[k] = 12
	return dataConvert

#Hàm đánh giá tỉ lệ dự đoán đúng
def RateRating(Y_Pred):
	countPredictTrue = 0
	for i in range(len(Y_Pred)):
		if(Y_Pred[i] == Y_test[i]):
			countPredictTrue = countPredictTrue + 1
		rate = countPredictTrue / len(Y_Pred)
	return rate

#Hàm show tỉ lệ dự đoán của thuật toán CART trên 3 độ đo: Precision, Recall, F1
def AboutRateCART():
	messagebox.showinfo("Tỉ lệ dự đoán đúng của CART",f"Accuracy score: {maxRateCART*100}%"+'\n'
													+f"Precision score: {precision_score(Y_test, bestPredCART, average='micro')*100}%"+'\n'
													+f"Recall score: {recall_score(Y_test, bestPredCART, average='micro')*100}%"+'\n'
													+f"F1 score: {f1_score(Y_test, bestPredCART, average='micro')*100}%")

#Hàm dự đoán sử dụng CART
def PredictWithCART():
	try:
		newData = DataEncoder(np.array([[cbbWorkEnvironment.get(), cbbExperience.get(), cbbPassion.get(), cbbAdvancementSpeed.get(), cbbSalary.get()]]))
		newData_Decreased = bestPcaCART.transform(newData)
		Result = modelMaxCART.predict(newData_Decreased)
		lbPredictCART.configure(text=f"{Result[0]}")
	except:
		messagebox.showinfo("Cảnh báo", "Vui lòng chọn đầy đủ thông tin để dự đoán")

#Đọc dữ liệu từ file
Data = pd.read_csv('work.csv')

#Chia dữ liệu thành 2 phần: DataX là các thuộc tính, DataY là nhãn của dữ liệu
DataX = DataEncoder(np.array(Data[['work environment','experience','passion','advancement speed','salary']].values))
DataY = np.array(Data['quit job'].values)

#Bắt đầu thuật toán PCA
maxRateCART = 0
for i in range(1,len(DataX[0])+1):

	#Khai báo PCA và số thành phần cần giữ lại
	pca = PCA(n_components = i)

	#Tìm một hệ cơ sở trực chuẩn và loại bỏ những thuộc tính ít quan trọng nhất
	X_Decreased = pca.fit_transform(DataX)

	#Sau khi dữ liệu đã được giảm kích thước thì tiến hành chia dữ liệu thành các phần Train, Test
	X_train, X_test, Y_train, Y_test = train_test_split(X_Decreased, DataY, test_size = 0.3, shuffle = False)

	#Khai báo phương thức tạo cây với tiêu chí gini (CART), đồng thời tiến hành dựng cây phân lớp
	TreeCART = tree.DecisionTreeClassifier(criterion='gini').fit(X_train, Y_train)
	#tiến hành dự đoán trên tập test
	Y_Pred_CART = TreeCART.predict(X_test)
	
	#Điều kiện so sánh để lưu lại các thông tin của PCA và mô hình tốt nhất khi được kết hợp với PCA
	if(RateRating(Y_Pred_CART) > maxRateCART):
		maxRateCART = RateRating(Y_Pred_CART)
		numComponentsCART = i
		bestPcaCART = pca
		modelMaxCART = TreeCART
		bestPredCART = Y_Pred_CART

FORM = Tk()

#Giới hạn kích thước cho form
FORM.minsize(900, 700)

#Đặt tên cho form
FORM.title("Dự đoán quyết định nghỉ việc của nhân viên")

#Định nghĩa font chữ
MyFont = ("Arial", 20)

#Các đối tượng được dùng trong form: Label, Combobox, Button, LabelFrame (Group) 

lbTitle = Label(FORM, text="Thông tin để đưa ra quyết định nghỉ việc", font=("Arial", 30))
lbTitle.grid(row=0, column=0, columnspan=2, padx=0, pady=10, sticky="we")

lbWorkEnvironment = Label(FORM, font=MyFont, text="Môi trường làm việc:", bg="#C7CBD1")
lbWorkEnvironment.grid(row=1, column=0, padx=(10, 0), pady=5, sticky="nswe")
lbExperience = Label(FORM, font=MyFont, text="Kinh nghiệm của nhân viên:", bg="#C7CBD1")
lbExperience.grid(row=2, column=0, padx=(10, 0), pady=5, sticky="nswe")
lbPassion = Label(FORM, font=MyFont, text="Mức độ đam mê với công việc:", bg="#C7CBD1")
lbPassion.grid(row=3, column=0, padx=(10, 0), pady=5, sticky="nswe")
lbAdvancementSpeed = Label(FORM, font=MyFont, text="Mức độ thăng tiến trong công việc:", bg="#C7CBD1")
lbAdvancementSpeed.grid(row=4, column=0, padx=(10, 0), pady=5, sticky="nswe")
lbSalary = Label(FORM, font=MyFont, text="Mức lương nhân viên nhận được:", bg="#C7CBD1")
lbSalary.grid(row=5, column=0, padx=(10, 0), pady=5, sticky="nswe")

cbbWorkEnvironment = Combobox(FORM, font=MyFont, state="readonly", values=('bad','ordinary','good'))
cbbWorkEnvironment.grid(row=1, column=1, padx=(0, 10), pady=5, sticky="nswe")
cbbExperience = Combobox(FORM, font=MyFont, state="readonly", values=('low','med','high'))
cbbExperience.grid(row=2, column=1, padx=(0, 10), pady=5, sticky="nswe")
cbbPassion = Combobox(FORM, font=MyFont, state="readonly", values=('small','big','vbig'))
cbbPassion.grid(row=3, column=1, padx=(0, 10), pady=5, sticky="nswe")
cbbAdvancementSpeed = Combobox(FORM, font=MyFont, state="readonly", values=('slow','normal','fast'))
cbbAdvancementSpeed.grid(row=4, column=1, padx=(0, 10), pady=5, sticky="nswe")
cbbSalary = Combobox(FORM, font=MyFont, state="readonly", values=('low','med','high'))
cbbSalary.grid(row=5, column=1, padx=(0, 10), pady=5, sticky="nswe")

groupCART = LabelFrame(FORM, font=MyFont, text="Thuật toán CART kết hợp PCA")
groupCART.grid(column=0, row=7, columnspan=2, padx=15, pady=15, sticky="nswe")

btnAboutId3 = Button(groupCART, font=MyFont, text="Thông tin tỷ lệ dự đoán của thuật toán", bg="#C7CBD1", command=AboutRateCART)
btnAboutId3.grid(column=0, row=0, padx=200, pady=(40, 5), sticky="nswe")

btnPredictCART = Button(groupCART, font=MyFont, text="Dự đoán với CART", bg="#C7CBD1", command=PredictWithCART)
btnPredictCART.grid(column=0, row=1, padx=200, pady=(40, 5), sticky="nswe")

lbCART = Label(groupCART, font=MyFont, text="Quyết định nghỉ việc\n(yes / no)")
lbCART.grid(row=2, column=0, padx=5, pady=(40, 5), sticky="nswe")

lbPredictCART = Label(groupCART, font=MyFont, text="---")
lbPredictCART.grid(row=3, column=0, padx=5, pady=(0, 20), sticky="nswe")

groupCART.rowconfigure((0, 1), weight=1)
groupCART.rowconfigure((2, 3), weight=1)
groupCART.columnconfigure(0, weight=1)
FORM.rowconfigure((1, 2, 3, 4, 5), weight=1)
FORM.rowconfigure(7, weight=2)
FORM.columnconfigure((0, 1), weight=1)

FORM.mainloop()