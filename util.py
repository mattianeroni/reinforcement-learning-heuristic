import matplotlib.pyplot as plt
from IPython import display

plt.ion()

def plot(bests1, bests2):
    display.clear_output(wait=True)
    #display.display(plt.gcf())
    plt.clf()
    plt.title('Evolution')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.plot(bests1, label="RLH")
    plt.plot(bests2, label="LNS")
    #plt.ylim(ymin=0)
    plt.legend()
    plt.show()
    plt.pause(.05)