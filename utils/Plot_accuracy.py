import matplotlib.pyplot as plt 
import numpy as np 

def visualization(Train_loss,Valid_loss,Dataset,model):

	assert len(Train_loss)==len(Valid_loss), "Accuracy lengths don't match"

	Epochs = len(Train_loss)
	A, = plt.plot(np.arange(1,Epochs+1,1),Train_loss)
	B, = plt.plot(np.arange(1,Epochs+1,1),Valid_loss)
	plt.xlabel("Number of Epochs")
	plt.ylabel("Percent Accuracy")
	plt.legend(["Training set Accuracy","Validation Set Accuracy"])
	plt.title("Train and Val Accuracy over epochs")
	plt.savefig("accuracy_plots/"+str(Dataset)+"/"+str(model)+"_Accuracy.png")
	plt.clf()

def time_visualization(Valid_Accuracy,Train_time,Dataset,model):

	A, = plt.plot(Train_time,Valid_Accuracy)
	plt.xlabel("Training Time (in Secs)")
	plt.ylabel("Percent Validation Accuracy")
	plt.legend(["Validation Set Accuracy"])
	plt.title("Validation accuracy over training time")
	plt.savefig("timing_plots/"+str(Dataset)+"/"+str(model)+"_timing.png")
	plt.clf()