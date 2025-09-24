from   matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from   scipy.stats import (multivariate_normal as mvn,
                           norm)


class mmsn:
    """Summary of class here.

        Multivariate Skew Normal based on https://gregorygundersen.com/blog/2020/12/29/multivariate-skew-normal/
    
    """        
    def pdf(a, x, dim, location, cov):
        """  Bivariate Skew Normal Distribution 
        Parameters:
        a: skewness (shape) of the SN distribution.
        x: coordinates.
        dim: dimension of the SN, =2 for bivariate.
        location: person location in the coordinates.
        cov: covariance of the distribution.
        """
        mean = np.zeros(dim)
        cov  = np.eye(dim) if cov is None else np.asarray(cov)
        x[:,:,1] = (x[:,:,1]-location[1])/(1.25 )  - 0.5 #/(1.65 * 1.8)     #These addition and subtraction operations just make the shape of the distribution more like proxemic zones
        x[:,:,0] = (x[:,:,0]-location[0])/1 + 0.2   #/1.8                   #if the parameters are learned, then these operations should be not necessary.
        x    = mvn._process_quantiles(x, dim)
        pdf  = mvn(mean, cov).logpdf(x)
        cdf  = norm(0, 1).logcdf(np.dot(x, a))
        logpdf = np.log(2) + pdf + cdf
        msn = np.exp(logpdf)

        return msn


    def direc_trans(X, Y, theta):
        """
        Rotate coordinates X and Y in an angle theta in radians.
        """
        direc_trans = []
        
        for t in range(len(theta)):
            print(theta[t])
            X_rot= np.cos(theta[t]) * X - np.sin(theta[t]) * Y
            Y_rot = np.sin(theta[t]) * X + np.cos(theta[t]) * Y
            pos = np.dstack((X_rot, Y_rot))
            direc_trans.append(pos)
            
        print(np.asarray(direc_trans[0]).shape)
        
        return direc_trans

 
def main():

    #params for SN
    num_sn = 3
    weights = 1/num_sn
    print(weights)
    skew = [[0, 1], [0, -1], [0, 1]]
    #msn = mmsn()
    theta = [0, np.radians(90), np.radians(90)]


    #plot space
    xx   = np.linspace(-6.5, 6.5, 100)
    yy   = np.linspace(-6.5, 6.5, 100)
    X, Y = np.meshgrid(xx, yy)
    Z = np.zeros((100, 100))

    location = [[0,0], [0,0], [0,1]]
    pos_list = mmsn.direc_trans(X, Y, theta)

    #calculate Mixture Bivariate Skew Normal Distribution, and normalize
    for i in range(num_sn):
        Z = Z +  weights * mmsn.pdf(a=skew[i], x=pos_list[i], dim = len(skew[i]), location=location[i], cov=None)
        
    scale_normalize = 1/np.max(Z)
    print(np.max(Z))
    normalized_Z = scale_normalize * Z
    print(1/np.max(Z))

    #plot contours
    plt.figure(figsize=(8,8))
    contour = plt.contour(X, Y, normalized_Z, colors = 'k', levels = [0.01, 0.13, 0.3, 0.4, 0.5, 0.8, 0.97])
    plt.clabel(contour, fontsize=10)
    plt.axhline(y=0, color='r')
    plt.axvline(x=0,color='r')
    plt.show()   
    


if __name__ == "__main__":
    main()
