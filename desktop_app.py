import tkinter as tk


class Application(tk.Frame):

    def __init__(self, master=None):
        tk.Frame.__init__(self, master)
        self.grid()
        self.createWidgets()

    def createWidgets(self):
        self.medialLabel = tk.Label(self, text='Hello World')
        self.medialLabel.config(bg="#00ffff")
        self.medialLabel.grid()
        self.quitButton = tk.Button(self, text='Quit', command=self.quit)
        self.quitButton.grid()


app = Application()
app.master.title('Sample application')
app.mainloop()
