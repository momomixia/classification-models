import numpy as np
import matplotlib
matplotlib.use("Pdf")
import matplotlib.pyplot as plt

def plotExample():
    #Create values and labels for bar chart
    values =np.random.rand(3)
    inds   =np.arange(3)
    labels = ["A","B","C"]
    
    #Plot a bar chart
    plt.figure(1, figsize=(6,4))  #6x4 is the aspect ratio for the plot
    plt.bar(inds, values, align='center') #This plots the data
    plt.grid(True) #Turn the grid on
    plt.ylabel("Error") #Y-axis label
    plt.xlabel("Method") #X-axis label
    plt.title("Error vs Method") #Plot title
    plt.xlim(-0.5,2.5) #set x axis range
    plt.ylim(0,1) #Set yaxis range
    
    #Set the bar labels
    plt.gca().set_xticks(inds) #label locations
    plt.gca().set_xticklabels(labels) #label values
    
    #Save the chart
    plt.savefig("../Figures/example_bar_chart.pdf")
    
    #Create values and labels for line graphs
    values =np.random.rand(2,5)
    inds   =np.arange(5)
    labels =["Method A","Method B"]
    
    #Plot a line graph
    plt.figure(2, figsize=(6,4))      #6x4 is the aspect ratio for the plot
    plt.plot(inds,values[0,:],'or-', linewidth=3) #Plot the first series in red with circle marker
    plt.plot(inds,values[1,:],'sb-', linewidth=3) #Plot the first series in blue with square marker
    
    #This plots the data
    plt.grid(True) #Turn the grid on
    plt.ylabel("Error") #Y-axis label
    plt.xlabel("Value") #X-axis label
    plt.title("Error vs Value") #Plot title
    plt.xlim(-0.1,4.1) #set x axis range
    plt.ylim(0,1) #Set yaxis range
    plt.legend(labels,loc="best")
    
    #Save the chart
    plt.savefig("../Figures/example_line_plot.pdf")
    
    #Displays the plots.
    #You must close the plot window for the code following each show()
    #to continue to run
    plt.show()
    
    #Displays the charts.
    #You must close the plot window for the code following each show()
    #to continue to run
    #plt.show()

#plotExample()
def plotNN(x, ys, labels):    
    #Plot a line graph
    y1 = ys[0]
    y2 = ys[1]
    y3 = ys[2]
    
    plt.figure(3, figsize=(6,4))      #6x4 is the aspect ratio for the plot
    plt.plot(x, y1,'or-', linewidth=3) #Plot the first series in red with circle marker
    plt.plot(x, y2,'sb-', linewidth=3) #Plot the first series in blue with square marker
    plt.plot(x, y3, 'og-', linewidth=3)
    #This plots the data
    plt.grid(True) #Turn the grid on
    plt.ylabel("Mean logistic loss") #Y-axis label
    plt.xlabel("Epoch") #X-axis label
    plt.title("Logistic loss vs Epoch with different hidden units") #Plot title
    plt.xlim(min(x), max(x)) #set x axis range
    plt.ylim(0, max(max(y1), max(y2), max(y3))+1) #Set yaxis range
    plt.legend(labels,loc="best")
    
    plt.savefig("../Figures/NN_logisticLoss.pdf")
    