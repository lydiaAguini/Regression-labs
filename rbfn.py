import numpy as np
import matplotlib.pyplot as plt
from gaussians import Gaussians
from sample_generator import SampleGenerator



class RBFN(Gaussians):
    def __init__(self, nb_features):
        super().__init__(nb_features)
        self.theta = np.random.random(self.nb_features)
        self.a = np.zeros(shape=(self.nb_features, self.nb_features))
        self.a_inv = np.matrix(np.identity(self.nb_features))
        self.b = np.zeros(self.nb_features)

    def f(self, x, theta=None):
        """
        Get the FA output for a given input vector
    
        :param x: A vector of dependent variables of size N
        :param theta: A vector of coefficients to apply to the features. 
        :If left blank the method will default to using the trained thetas in self.theta.
        
        :returns: A vector of function approximator outputs with size nb_features
        """
        if not hasattr(theta, "__len__"):
            theta = self.theta
        value = np.dot(self.phi_output(x).transpose(), theta.transpose())
        return value

    def feature(self, x, idx):
        """
         Get the output of the idx^th feature for a given input vector
         This is function f() considering only one feature
         Used mainly for plotting the features

         :param x: A vector of dependent variables of size N
         :param idx: index of the feature

         :returns: the value of the feature for x
         """
        phi = self.phi_output(x)
        return phi[idx] * self.theta[idx]

    # ----------------------#
    # # Training Algorithms ##
    # ----------------------#

    # ------ batch least squares (projection approach) ---------
    def train_ls(self, x_data, y_data):
        x = np.array(x_data)
        y = np.array(y_data)
        X = self.phi_output(x)

        #TODO: Fill this
        G = X.T
        self.theta = np.dot(np.dot(np.linalg.inv(np.dot(G.T,G)),G.T),y)
        
    # ------ batch least squares (calculation approach) ---------
    def train_ls2(self, x_data, y_data):
        a = np.zeros(shape=(self.nb_features, self.nb_features))
        b = np.zeros(self.nb_features)
        
        
        X = self.phi_output(np.array(x_data))
        a = np.dot(X,X.T)
        b = np.sum(y_data*X,axis=1)

        self.theta = np.linalg.solve(a,b)
    # -------- gradient descent -----------------
    def train_gd(self, x, y, alpha):
        
        X = self.phi_output(x)
        prod = alpha * (y - np.dot(self.theta,X)) * X

        self.theta += prod[:,0]

    # -------- recursive least squares -----------------
    def train_rls(self, x, y):
        phi = self.phi_output(x)
        self.a = self.a + np.dot(phi, phi.transpose())
        self.b = self.b + y * phi.transpose()[0]

        result = np.dot(np.linalg.pinv(self.a), self.b)
        self.theta = np.array(result)

    # -------- recursive least squares (other version) -----------------
    def train_rls2(self, x, y):
        phi = self.phi_output(x)
        self.a = self.a + np.outer(phi,phi)
        self.b = self.b + y * phi.transpose()[0]

        self.theta = np.dot(np.linalg.pinv(self.a), self.b)

    # -------- recursive least squares with Sherman-Morrison -----------------
    def train_rls_sherman_morrison(self, x, y):
        u = self.phi_output(x)
        v = self.phi_output(x).transpose()

        value = (v * self.a_inv * u)[0, 0]
        tmp_mat = self.a_inv * np.dot(u, v)* self.a_inv

        self.a_inv = self.a_inv - (1.0 / (1 + value)) * tmp_mat
        self.b = self.b + y * u.transpose()[0]

        result = np.dot(self.a_inv, self.b)
        self.theta = np.array(result)[0]

    # -----------------#
    # # Plot function ##
    # -----------------#

    def plot(self, x_data, y_data,title="",ax=None, show_features=True, z=None):
        xs = np.linspace(0.0, 1.0, 1000)
        z = [self.f(i) for i in xs] if z is None else z

        z2 = []
        for i in range(self.nb_features):
            temp = []
            for j in xs:
                temp.append(self.feature(j, i))
            z2.append(temp)
        
        plot1 = (x_data, y_data, 'o')
        plot2 = (xs, z)
        
        if ax is None:
            # Plots and shows as unique graph
            plt.title(title)
            plt.plot(*plot1, markersize=3, color='lightgreen')
            plt.plot(*plot2, lw=2, color='red')
            if show_features is True:
                for i in range(self.nb_features):
                    plt.plot(xs, z2[i])
            plt.show()
        else:
            # Plots as subgraph
            ax.set_title(title)
            ax.plot(*plot1, markersize=3, color='lightgreen')
            ax.plot(*plot2, lw=2, color='red')
            if show_features is True:
                for i in range(self.nb_features):
                    ax.plot(xs, z2[i])

        return z

if __name__ == '__main__':
    val = (5, 10, 15, 20)
    indx = ((0,0),(0,1),(1,0),(1,1))
    s = SampleGenerator()
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
