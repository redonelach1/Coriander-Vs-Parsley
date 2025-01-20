import os


i = 0
for ele in os.listdir("./train/coriander"):
    path = os.path.join("./train/coriander",ele)
    os.rename(path,"./train/parsley/coriander_{:003d}".format(i) + ".jpg")
    i += 1