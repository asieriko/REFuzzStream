import numpy as np
import matplotlib.pyplot as plt

def plot(fmics, radiuses, value,timestamp):
    x = []
    y = []
    for fmic in fmics:
        x.append(fmic.center["X1"])
        y.append(fmic.center["X2"])
    # fig = plt.figure()
    fig, ax = plt.subplots(figsize=[7,7])
    # Plot radius
    s = [np.pi * (ax.transData.transform([r, 0])[0] - ax.transData.transform([0, 0])[0])**2 for r in radiuses]
    plt.scatter(x, y, color='blue', s=s)
    plt.scatter(x, y, color='green')
    plt.scatter(value.iloc[0], value.iloc[1], color='red')
    plt.xlim([0,1])
    plt.ylim([0, 1])
    plt.title(timestamp)
    # plt.show()
    plt.savefig(f"Img/Tests/Test_{timestamp}.png")
    plt.close()
