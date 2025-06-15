from tkinter import*
import random ,string
import pyperclip

root= Tk()
root.geometry("400x400")
root.resizable(0,0)
root.title("Python Project - Password Generator")

Label(root, text="Password Generator",font='arial 18 bold').pack()
Label(root, text='Python',font='arial 15 bold').pack(side=BOTTOM)

pass_label=Label(root, text="Password Length", font='arial 12 bold').pack()
pass_len=IntVar()
length=Spinbox(root,from_=8, to_=32,textvariable=pass_len,width=15).pack()
pass_str=StringVar()

def Generator():
    password=[]

    #Ensuring atleast one chararcter from each type(Uppercase,Lowercase,Digits,Punctuation?specailcharacrter)
    if pass_len.get() >=4:
        password.append(random.choice(string.ascii_uppercase))
        password.append(random.choice(string.ascii_lowercase))
        password.append(random.choice(string.digits))
        password.append(random.choice(string.punctuation))

    #Fill the rest with random choices eintill the specified length 
        for _ in range(pass_len.get()-4):
            password.append(random.choice(string.ascii_uppercase+string.ascii_lowercase+string.digits+string.punctuation))

        #Suffle to ensure randomness
        random.shuffle(password)
    else:
        #If length is less than 4, just fill the required length with random choices
        for _ in range(pass_len.get()):
            password.append(random.choice(string.ascii_uppercase+string.ascii_lowercase+string.digits+string.punctuation))

    #convert list to string and set it to the variable 
    pass_str.set(''.join(password))

def Copy_password():
    pyperclip.copy(pass_str.get())

Button(root, text='Generate Password',command=Generator).pack(pady=5)
Entry(root,textvariable=pass_str).pack()
Button(root,text='Copy to Clipboard',command=Copy_password).pack()

root.mainloop()




