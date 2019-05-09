import imageio
import glob
import numpy as np
import matplotlib.pyplot as plt

def generate_gif(filenames,output_path):
    '''
    Generate gif from list of filenames.
    '''
    with imageio.get_writer(output_path, mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

filelist_1 = glob.glob('./analysis/4CD,1FE,97/*.txt')
filelist_2 = glob.glob('./analysis/4CD,RK4,97/*.txt')

err_1FEvsRK4 = open("./analysis/1FEvsRK4.txt","a")
for i, fi in enumerate(filelist_1):
    err = np.abs((np.loadtxt(fi)-np.loadtxt(filelist_2[i]))/np.loadtxt(filelist_2[i]))
    err_1FEvsRK4.write(str(err.max())+'\n')
    plt.imshow(err)
    #plt.clim(-0.001, 0.001);
    plt.colorbar()
    plt.savefig('./analysis/errt00'+str(i)+'.png', dpi=300)
    plt.clf()
err_1FEvsRK4.close()

filenames = ['./analysis/errt00'+str(i)+'.png' for i in range(0,10)]
generate_gif(filenames,"./analysis/1FEvsRK4.gif")
