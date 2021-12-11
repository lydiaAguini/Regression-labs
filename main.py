#!/usr/local/bin/python

#LYDIA AGUINI 3874863

import numpy as np
import time
from rbfn import RBFN
from lwr import LWR
from line import Line
from sample_generator import SampleGenerator
import matplotlib.pyplot as plt


class Main:
    def __init__(self):
        self.x_data = []
        self.y_data = []
        self.batch_size = 50

    def reset_batch(self):
        self.x_data = []
        self.y_data = []

    def make_nonlinear_batch_data(self, noise=.1):
        """ 
        Generate a batch of non linear data and store it into numpy structures
        """
        self.reset_batch()
        g = SampleGenerator()
        for i in range(self.batch_size):
            # Draw a random sample on the interval [0,1]
            x = np.random.random()
            y = g.generate_non_linear_samples(x, noise)
            self.x_data.append(x)
            self.y_data.append(y)

    def make_linear_batch_data(self):
        """ 
        Generate a batch of linear data and store it into numpy structures
        """
        self.reset_batch()
        g = SampleGenerator()
        for i in range(self.batch_size):
            # Draw a random sample on the interval [0,1]
            x = np.random.random()
            y = g.generate_linear_samples(x)
            self.x_data.append(x)
            self.y_data.append(y)

    def approx_linear_batch(self):
        # Copied and pasted from line.py
        def arrondir(L, nb_dec=3):
            # Rounds numbers in list for clearer view
            return [int(k*10**nb_dec if not(np.isnan(k)) else 0)/10**nb_dec if type(k) is not list else arrondir(k, nb_dec)for k in L]

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

    def approx_rbfn_batch(self, s=SampleGenerator()):
        # Copied and pasted from rbfn.py
        val = (5, 10, 15, 20)
        indx = ((0,0),(0,1),(1,0),(1,1))
        R = [RBFN(v) for v in val]

        batch_size = 100
        x = np.linspace(0,1,batch_size)
        y = np.array([s.generate_non_linear_samples(k) for k in x])

        # Code question 4
        fig, axs = plt.subplots(2,2)
        fig.suptitle("RBFN LLS regression with different numbers of features")
        for k in range(len(R)):
            R[k].train_ls(x,y)
            R[k].plot(x,y,title=str(val[k]),ax=axs[indx[k]])
        plt.show()
    
        # Code question 5
        # Given plot nb of features
        #       Showed that 15 is a good number
        r = RBFN(15)
        maxIter, alpha = 1500, .25
        
        fig, axs = plt.subplots(2,2)
        indxi = iter(indx)
        for t in range(maxIter+1):
            r.train_gd(x[t%batch_size],y[t%batch_size],alpha)
            # Plotting the progress of the regression
            if t in (0, 100, 200, 400):
                r.plot(x,y,"Itérations : " + str(t), axs[next(indxi)])
                if t == 400:
                    plt.show()
                    indxi = iter(indx)
                    fig, axs = plt.subplots(2,2)
            elif t in (500,800,1200,1500):
                r.plot(x,y,"Itérations : " + str(t), axs[next(indxi)])
        plt.show()

        # Show over-fitting with linear batch of data (so it's more apparant)
        r = RBFN(20)
        y = np.array([s.generate_linear_samples(k) for k in x])
        maxIter = 3000
        
        fig, axs = plt.subplots(1,2)
        fig.suptitle("Over-fitting, " + str(r.nb_features) + " features")
        for t in range(2*maxIter+1):
            r.train_gd(x[t%batch_size],y[t%batch_size],1)
            if t == maxIter: r.plot(x,y,str(t) + " itérations", axs[0],False)
            

        r.plot(x,y,str(2*maxIter) + " itérations",axs[1],False)
        plt.show()

    def approx_rbfn_iterative(self, fig, ax, method="RLS", max_iter=500, nb_features=15, noise=0.1, affiche=False):
        model = RBFN(nb_features)

        args = [0,0]
        if method == "RLS": reg = model.train_rls
        elif method == "RLS2":  reg = model.train_rls2
        elif method == "GD":
            reg = model.train_gd
            args.append(0.5)
        elif method == "RLS_SM":    reg = model.train_rls_sherman_morrison

        fig.suptitle("maxIter = " + str(max_iter)+", nb_features = "+str(model.nb_features))

        # Generate a batch of data and store it
        self.reset_batch()
        start = time.process_time()
        for i in range(max_iter):
            # Draw a random sample on the interval [0,1]
            args[0] = np.random.random()
            args[1] = g.generate_non_linear_samples(args[0], noise)
            self.x_data.append(args[0])
            self.y_data.append(args[1])

            reg(*args)
        temps = time.process_time() - start

        print(method + " time:", temps)
        model.plot(self.x_data, self.y_data, 
        method + "        " + str(int(temps*1000))[:4] +"ms",ax)
        
        if affiche is True: plt.show()

    def approx_lwr_batch(self):
        model = LWR(nb_features=10)
        self.make_nonlinear_batch_data()

        start = time.process_time()
        model.train_lwls(self.x_data, self.y_data)
        print("LWR time:", time.process_time() - start)
        
        model.plot(self.x_data, self.y_data)

if __name__ == '__main__':
    m = Main()

    # Questions 1 to 3
    m.approx_linear_batch()
    # Questions 4 to 5
    m.approx_rbfn_batch()

    # Study question 6
    fig, axs = plt.subplots(1,3)
    g = SampleGenerator()
    m.approx_rbfn_iterative(fig, axs[0], "RLS")
    m.approx_rbfn_iterative(fig, axs[1], "RLS2")
    m.approx_rbfn_iterative(fig, axs[2], "RLS_SM")
    plt.show()

    # Study question 7
    g = SampleGenerator()
    def iterer(N):
        fig, axs = plt.subplots(2,2)
        m.approx_rbfn_iterative(fig, axs[(0,0)], "RLS", N)
        m.approx_rbfn_iterative(fig, axs[(0,1)], "RLS2", N)
        m.approx_rbfn_iterative(fig, axs[(1,0)], "GD", N)
        m.approx_rbfn_iterative(fig, axs[(1,1)], "RLS_SM", N)
    iterer(125)
    iterer(250)
    iterer(500)
    plt.show()

    # Code question 9
    m.approx_lwr_batch()

    # Study question 10
    noise = (.1, .25, .425, .5)
    batch_size = 800

    x_data = np.linspace(0,1,batch_size)
    for eps in noise:
        fig, axs = plt.subplots(2,2)

        # Iterative methods
        m.approx_rbfn_iterative(fig, axs[(0,0)], "RLS2", 800, 15, eps)
        m.approx_rbfn_iterative(fig, axs[(0,1)], "GD", 800, 15, eps)

        # Batch method
        r = RBFN(15)
        y_data = [g.generate_non_linear_samples(k, eps) for k in x_data]

        start = time.process_time()
        r.train_ls(x_data,y_data)
        temps = time.process_time() - start

        print("RBFN LS time:", temps)
        #   Plotting RBFN LS
        r.plot(x_data,y_data,title="RBFN LS         "+ str(int(temps*1000))[:4]+"ms",ax=axs[(1,0)])

        # LWR method
        model = LWR(nb_features=15)
        y_data = [g.generate_non_linear_samples(k, eps) for k in x_data]

        start = time.process_time()
        model.train_lwls(x_data, y_data)
        temps = time.process_time() - start

        print("LWR time:", temps)
        #   Plotting LWR
        model.plot(x_data,y_data,axs[(1,1)])
        axs[(1,1)].set_title("LWR         "+ str(int(temps*1000))[:4]+"ms")


        fig.suptitle("Noise: " + str(eps))
    plt.show()
