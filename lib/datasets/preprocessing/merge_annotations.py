import os, sys

def s3d_like_datasets(path='./', prefix='Area_', extension='txt'):
    for i in range(1,7):
        area = path+prefix+str(i)
        print("*********** Area_{0} ***********".format(area))
        rooms = [f.path for f in os.scandir(area) if f.is_dir()]
        for room in rooms:
            print("Merging room : {0}".format(room))
            os.system("cat {0}/Annotations/*.{1} > {0}/{2}.{1}".format(room,extension,room.split('/')[-1]))
print(*sys.argv[1:])
s3d_like_datasets(*sys.argv[1:])