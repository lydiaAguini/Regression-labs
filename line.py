#!/usr/local/bin/python

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from sample_generator import SampleGenerator


class Line:
    def __init__(self, batch_size):
        self.nb_dims = 1
        self.theta = np.zeros(shape=(batch_size, self.nb_dims+1))

    def f(self, x):
        """
        Get the FA output for a given input variable(s)

        :param x: A single or vector of dependent variables with size [Ns] for which to calculate the features

        :returns: the function approximator output
        """
        if np.size(x) == 1:
            xl = np.vstack(([x], [1]))
        else:
            xl = np.vstack((x, np.ones((1, np.size(x)))))
        return np.dot(self.theta, xl)

    # ----------------------#
    # # Training Algorithm ##
    # ----------------------#

    def train(self, x_data, y_data):
        # Finds the Least Square optimal weights
        x_data = np.array([x_data]).transpose()
        y_data = np.array(y_data)
        x = np.hstack((x_data, np.ones((x_data.shape[0], 1))))

        
        # Computes optimal model
        self.theta = np.dot(np.dot(np.linalg.inv(np.dot(x.T,x)),x.T),y_data)   
        
        # Computes coefficient of determination
        y_mean, f = np.mean(y_data), np.dot(x,self.theta)
        #   Residual sum of squares, Total sum of squares
        ss_res, ss_tot = np.sum((y_data-f)**2), np.sum((y_data-y_mean)**2)
        #       R
        r_value = np.sqrt(1 - ss_res/ss_tot)

        # Slope, intercept, r_value
        return self.theta[:-1], self.theta[-1], r_value


        # ----------------------#
        # # Training Algorithm ##
        # ----------------------#

    def train_regularized(self, x_data, y_data, coef):
        # Finds the regularized Least Square optimal weights
        x_data = np.array([x_data]).transpose()
        y_data = np.array(y_data)
        x = np.hstack((x_data, np.ones((x_data.shape[0], 1))))

        
        # Computes the analytical solution
        self.theta = np.dot(np.dot(np.linalg.inv(coef * np.eye(x.shape[1]) + np.dot(x.T,x)), x.T), y_data)

        # Computes coefficient of determination
        y_mean, f = np.mean(y_data), np.dot(x,self.theta)
        #   Residual sum of squares, Total sum of squares
        ss_res, ss_tot = np.sum((y_data-f)**2), np.sum((y_data-y_mean)**2)
        #       R
        r_value = np.sqrt(1 - ss_res/ss_tot)

        # Slope, intercept, r_value
        return self.theta[:-1], self.theta[-1], r_value
        
 
    # ----------------------#
    # # Training Algorithm ##
    # ----------------------#

    def train_from_stats(self, x_data, y_data):
        # Finds the Least Square optimal weights: python provided version
        slope, intercept, r_value, _, _ = stats.linregress(x_data, y_data)
        

    
        self.theta = np.array([slope] + [intercept])
        return slope,intercept, r_value
        
    # -----------------#
    # # Plot function ##
    # -----------------#

    def plot(self, x_data, y_data, title="", ax=None, z=None):
        xs = np.linspace(0.0, 1.0, 1000)
        z = self.f(xs) if z is None else z
        
        plot1 = (x_data, y_data, 'o')
        plot2 = (xs, z)
        
        if ax is None:
            # Plots and shows as unique graph
            plt.title(title)
            plt.plot(*plot1, markersize=3, color='lightgreen')
            plt.plot(*plot2, lw=2, color='red')
            plt.show()
        else:
            # Plots as subgraph
            ax.set_title(title)
            ax.plot(*plot1, markersize=3, color='lightgreen')
            ax.plot(*plot2, lw=2, color='red')

        return z

if __name__ == '__main__':
    def arrondir(L, nb_dec=3):
        # Rounds numbers in list for clearer view
        return [int(k*10**nb_dec)/10**nb_dec if type(k) is not list else arrondir(k, nb_dec)for k in L]

    batch_size = 50
    line = Line(batch_size)
    s = SampleGenerator()
    
    # Generating samples
    x = np.linspace(0,1,batch_size)
    y = [s.generate_linear_samples(k) for k in x]

    # Code question 1
    fig, axs = plt.subplots(1,2)
    reg_normal = line.train(x,y)
    print('\n From normal : ', reg_normal)
    line.plot(x,y,"Regression using train method\n" + str(arrondir(reg_normal)), axs[0])

    reg_stats =  line.train_from_stats(x,y)
    print('\n From stats : ', reg_stats)
    z_stats = line.plot(x,y,"Regression using train from stats\n" + str(arrondir(reg_stats)), axs[1])

    plt.show()

    # Code question 2
    fig, axs = plt.subplots(1,2)
    reg_regul = line.train_regularized(x,y,1)
    print('\n From regularized : ', reg_regul)
    line.plot(x,y, "Regression using train regularized method\n" + str(arrondir(reg_regul)), axs[0])

    line.plot(x,y,"Regression using train from stats\n" + str(arrondir(reg_stats)), axs[1], z_stats)

    plt.show()

    # Study question 3
    fig, axs = plt.subplots(1,2)
    dLambda = np.linspace(0,5,500)
    fig.suptitle("Residuals degradation as lambda increases from 0 to 5", fontsize=16)

    #   Computing regression with different lambdas
    resRegularized = []
    coefRegularized = []
    for coef in dLambda:    
        r = line.train_regularized(x,y,coef)[-1]
        resRegularized.append(1 - r**2)
        coefRegularized.append(r)

        line.plot(x,y,"Regression", axs[1])
    #   Plotting the LLS for comparison
    axs[1].plot(np.linspace(0,1,1000),z_stats,color='blue', label="From stats")
    axs[1].legend(loc="lower left")

    #   Plotting the change in r_value
    axs[0].plot(dLambda, resRegularized, color='blue', label='Residuals')
    axs[0].plot(dLambda, coefRegularized, color='red', label='Coefficient of determination')
    axs[0].legend(loc="lower center")
    axs[0].set_title("Change in residuals")

    plt.show()
