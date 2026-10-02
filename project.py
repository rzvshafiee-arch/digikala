from datasets import load_dataset
import pandas as pd 
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
import tkinter as tk 
from PIL import Image,ImageTk




def pre ():
    global im 
    entry = en.get()
    p = model.predict(tf.transform([entry]))[0]
    if p == 1 :
        im = ImageTk.PhotoImage(Image.open("1.ico").resize((100,100)),)
        label = tk.Label(w,image=im,bg="midnightblue")
        label.place(x=150,y=40)
    if p == 2 :
        im = ImageTk.PhotoImage(Image.open("2.ico").resize((100,100)))
        label = tk.Label(w,image=im,bg="midnightblue")
        label.place(x=150,y=40)

    if p == 3 :
        im = ImageTk.PhotoImage(Image.open("3.ico").resize((100,100)))
        label = tk.Label(w,image=im,bg="midnightblue")
        label.place(x=150,y=40)


ds = load_dataset("ParsiAI/digikala-sentiment-analysis")

ds["train"].to_csv(
 "Dgkala_train.csv",
index=False
)

ds["test"].to_csv(
 "Dgkala_test.csv",
index=False
)

data = pd.read_csv("Dgkala_train.csv")

x=data["Text"]
y=data["Suggestion"]

tf = TfidfVectorizer()
xx= tf.fit_transform(x)

len(xx[8].toarray()[0])

model = LinearSVC()
model.fit(xx,y)

w = tk.Tk()

w.geometry("500x200")
w.title("text")
w.config(bg="midnightblue")

en = tk.Entry()
en.place(x=23,y=50)

la=tk.Label(text="Please enter a text : ",fg="lightgoldenrod1",bg="midnightblue", font=("Britannic Bold",12))
la.place(x=20,y=20)

but = tk.Button(text="Enter", font=("Britannic Bold",10),bg="lightgoldenrod1",command=pre)
but.place(x=60,y=70)




w.mainloop()