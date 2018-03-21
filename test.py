from tkinter import *
root = Tk()

v = IntVar()
v.set(1)

for i in range(3):
    rb = Radiobutton(root, variable=v, text='python' + str(i), value=i)
    rb.pack()
root.mainloop()