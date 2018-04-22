from Tkinter import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm


class Application(Frame):
    
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.createWidgets()

    def createWidgets(self):
        self.winfo_toplevel().title("Time Series Analysis - SAX")
        self.length = Label(text='Length', font = ('helvetica', 10, 'bold'), relief=GROOVE,width=25).grid(row=0,column=0)
        self.lengthVal = Entry(bg='white', relief=SUNKEN,width=28)
        self.lengthVal.grid(row=0,column=1)
        self.lengthVal.insert(END,'1000')

        self.alphabets = Label(text='Alphabets', font = ('helvetica', 10, 'bold'), relief=GROOVE,width=25).grid(row=1,column=0)
        self.alphabetVal = Entry(bg='white', relief=SUNKEN,width=28)
        self.alphabetVal.grid(row=1,column=1)
        self.alphabetVal.insert(END,'abcde')

        self.splits = Label(text='Splits', font = ('helvetica', 10, 'bold'), relief=GROOVE,width=25).grid(row=2,column=0)
        self.splitVal = Entry(bg='white', relief=SUNKEN,width=28)
        self.splitVal.grid(row=2,column=1)
        self.splitVal.insert(END,'6')

        self.quit = Button(text= 'Exit',fg = 'red', font = ('helvetica', 10, 'bold'),command = self.quit, width = 25).grid(row=4,column=0)
        self.run = Button(text= 'Run',fg = 'black',font = ('helvetica', 10, 'bold'),command = self.run, width = 20).grid(row=4,column=1)

    def generateData(self,length):
        
        ts = pd.Series(np.random.randn(length), index=pd.date_range('01/01/2017', periods=length))
        ts = ts.cumsum()
        return ts

    def normaliseData(self, ts):
        
        mus = ts.mean(axis = 0)
        std = ts.std(axis = 0)
        return (ts - mus) / std
    
    def paaTransformData(self, ts, n_pieces):
        splitted = np.array_split(ts, n_pieces) 
        return np.asarray(map(lambda xs: xs.mean(axis = 0), splitted))

    def translate(self, ts_values):
        return np.asarray([(self.alphabet[0] if ts_value < self.thres[0]
                            else (self.alphabet[-1] if ts_value > self.thres[-1]
                                  else self.alphabet[np.where(self.thres <= ts_value)[0][-1] + 1]))
                           for ts_value in ts_values])

    def run(self):
        plt.clf()
        plt.rcParams.update({'font.size': 10})

        # User Defines arguments captured here
        numSplits = int(self.splitVal.get())
        length = int(self.lengthVal.get())
        self.alphabet = self.alphabetVal.get()

        # 1 --> Generate Time series data
        ts_data_1 = self.generateData(length)
        ts_data_2 = self.generateData(length)
        ts = pd.DataFrame({"data1": ts_data_1, "data2": ts_data_2})

        # 2 --> Normalise generated data
        zts = self.normaliseData(ts)

        # 3 --> Applying Piecewise Aggregate Approximation <<Dimensionality Reduction>>
        split = self.paaTransformData(zts, numSplits)
        print('Outputs represented by PAA transformation :')
        print split

        # 4 --> Applying Symbolic Aggregate Approximation <<SAX>>
        num_alpha = len(self.alphabet)
        self.thres = norm.ppf(np.linspace(1./num_alpha, 1-1./num_alpha, num_alpha-1))

        sax_transformed_data = np.apply_along_axis(self.translate, 0, split)
        print ('Outputs represented by SAX tranformation : ')
        print (sax_transformed_data)

        # 5 --> Data Visualization
        x = pd.date_range('01/01/2017', periods=length)
        start = x[0]
        end = x[length-1]
        x = np.linspace(start.value, end.value, length)
        x = pd.to_datetime(x)

        fig = plt.figure(1)

        ts_data = plt.subplot2grid((4,2), (0,0), rowspan=1, colspan=1)
        ts_data.plot(x, ts_data_1, '-+', color='brown', label='timeseries-data-1')
        ts_data.plot(x, ts_data_2, '-+', color='blue',label='timeseries-data-2')
        plt.title('Time Series Data')
        ts_data.legend()

        norm_data = plt.subplot2grid((4,2), (0,1), rowspan=1, colspan=1)
        norm_data.plot(x, zts.iloc[:, 0], '-+',color='brown', label='norm-timeseries-1')
        norm_data.plot(x, zts.iloc[:, 1], '-+',color='blue', label='norm-timeseries-2')
        norm_data.axhline(0.0, color='green', linestyle='--', lw=1)
        plt.title('Time Series Data Normalised')
        norm_data.legend()

        sl = np.linspace(start.value, end.value, numSplits+1)
        sl = pd.to_datetime(sl)

        color = ['brown', 'blue', 'orange', 'green']
        for j in range(0, 2):
            paa_data = plt.subplot2grid((4, 2), (1, j), rowspan=1, colspan=1)
            paa_data.plot(x, zts.iloc[:, j], '-+', color=color[j], label='data-{}-norm'.format(j+1))
            for i in range(1, numSplits):
                paa_data.plot([sl[i - 1], sl[i]], [split[i - 1][j], split[i - 1][j]], linewidth=2, linestyle="-",
                                c=color[j+2], solid_capstyle="projecting")
                paa_data.axvline(sl[i], color='r', linestyle='--', lw=2)
            paa_data.plot([sl[numSplits - 1], sl[numSplits]], [split[numSplits - 1][j], split[numSplits - 1][j]],
                            linewidth=2, linestyle="-", c=color[j+2], solid_capstyle="projecting", label='data-{}-PAA'.format(j+1))
            paa_data.set_title('Normalized Time Series-Data-{} with Piecewise Aggregate Approximation'.format(j+1))
            paa_data.legend()

        sax_data = plt.subplot2grid((5,2), (3,0), rowspan=3, colspan=2)
        sax_data.plot(x, zts.iloc[:, 0], '-+', color='brown', label='data-1-norm')
        sax_data.plot(x, zts.iloc[:, 1], '-+', color='blue', label='data-2-norm')
        for i in range(1,numSplits):
            sax_data.plot([sl[i-1], sl[i]], [split[i-1][0], split[i-1][0]], linewidth=2, linestyle="-", c="orange",solid_capstyle="projecting")
            sax_data.plot([sl[i-1], sl[i]], [split[i-1][1], split[i-1][1]], linewidth=2, linestyle="-", c="green",solid_capstyle="projecting")
            sax_data.axvline(sl[i], color='r', linestyle='--', lw=2)
        sax_data.plot([sl[numSplits-1], sl[numSplits]], [split[numSplits-1][0], split[numSplits-1][0]], linewidth=2, linestyle="-", c="blue",solid_capstyle="projecting",label='data-1-PAA')
        sax_data.plot([sl[numSplits-1], sl[numSplits]], [split[numSplits-1][1], split[numSplits-1][1]], linewidth=2, linestyle="-", c="green",solid_capstyle="projecting",label='data-2-PAA')
        for i in range(num_alpha - 2):
            sax_data.axhline(self.thres[i], color='black', linestyle='-', lw=0.5)
        sax_data.axhline(self.thres[-1], color='black', linestyle='-', lw=0.5, label='cardinals')
        sax_data.set_title('Symbolic Aggregate Approximation')
        sax_data.legend()
        #plt.tight_layout()
        fig.set_size_inches(70, 20)
        plt.savefig('Final_Plots.png')
        
        mng = plt.get_current_fig_manager()
        mng.resize(*mng.window.maxsize())
        plt.show()
        

root = Tk()
app = Application(master=root)
app.mainloop()
root.destroy()
