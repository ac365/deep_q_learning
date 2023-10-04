import numpy as np
import matplotlib.pyplot as plt

class PlotUtils:
    def plotLearningCurveDeepQ(x,scores,epsilons,fileName,lines=None):
        fig = plt.figure()
        ax1 = fig.add_subplot(111,label="1")
        ax2 = fig.add_subplot(111,label="2",frame_on=False)

        ax1.plot(x,epsilons,color='C0')
        ax1.set_xlabel('Training Steps',color='C0')
        ax1.set_ylabel('Epsilon',color='C0')
        ax1.tick_params(axis='x',colors='C0')
        ax1.tick_params(axis='y',colors='C0')

        n = len(scores)
        movingAvg = np.empty(n)
        for i in range(n):
            movingAvg[i] = np.mean(scores[max(0,i-25):(i+1)])
        
        ax2.scatter(x,movingAvg,color='C1')
        ax2.axes.get_xaxis().set_visible(False)
        ax2.yaxis.tick_right()
        ax2.set_ylabel('Score',color='C1')

        if lines is not None:
            for line in lines:
                plt.axvline(x=line)

        plt.savefig(fileName)