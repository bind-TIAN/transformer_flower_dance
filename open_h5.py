from icecream import ic
import h5py

if __name__ == "__main__":
    dir = "./tf_vit/ViT-B_16.h5"
    f = h5py.File(dir, 'r')
    for key in f.keys():
        ic(key, f[key])
