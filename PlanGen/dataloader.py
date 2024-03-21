
import pandas as pd
import os
import sklearn


class dataloader:
    

    def __init__(self, conf) -> None:
        self.conf = conf
        path = self.conf['images']['LayoutFolderPath']
        dir_list = os.listdir(path+"/Input")

        data = pd.DataFrame([], columns=["Input", "GroundTruth"])

        for i in range(0,len(dir_list)):
            nameLen = len(dir_list[i])
            gtImg = dir_list[i][:nameLen-4] + "_gt" + dir_list[i][nameLen-4:]
            if not os.path.exists(path+"/ground truth/"+gtImg) or dir_list[i][nameLen-4:] != ".png":
                print("Error: "+gtImg)
                continue

            df = pd.DataFrame({"Input": dir_list[i], "GroundTruth": gtImg}, index=[i+1])
            data = pd.concat([data, df])
            
        if self.conf['LayoutGan']['ImageSuffleSeed'] != -1: 
            self.train_data = data.sample(frac = self.conf['LayoutGan']['TrainTestSplit'], 
                                                 random_state=self.conf['LayoutGan']['ImageSuffleSeed'])          
        else:
            self.train_data = data.sample(frac = self.conf['LayoutGan']['TrainTestSplit'])
        self.test_data = data.drop(self.train_data.index)

        print("Files and directories in '", path, "' :")
        print(self.train_data)
        print(self.test_data)

        
    
#dataloader()
