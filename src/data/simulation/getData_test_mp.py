import os
import numpy as np
import sys

from utils_test import join_MDinnerOuter

#path = "/data/dust/user/sabebert/TrainingData/Training3Kvs/"
path = "/data/dust/user/sabebert/TrainingData/Training3Kvs/outOfDistribution_22000/"
rangeFiles = [100, 100000]
couplingCycles = 1400 #1000
numbers = [108, 220] #[118624,121424,143124,145924], 
#[1935,2166,2180,2194,1445,1200,108,136,122]
#[2180,2166,1949,1935,1921,1704,1690,1676,1459,1445,1431,1214,1200,1186]

def get_file_LineNumbers(path, numbers=None):
    linesPerFileInner = {}
    linesPerFileOuter = {}  

    if numbers is None:
        numbers, numbers2 = get_file_list(path, rangeFiles)
    for i in range(len(numbers)):
        folder = numbers[i][0:1]
        if os.path.isfile(path + f"{folder}/md_macro_vel_{folder}.csv"):
            print("gets here")
            with open(path + f"{folder}/md_macro_vel_{i}.csv", 'r') as fp:
                line_count = len(fp.readlines())
                linesPerFileInner[i] = line_count
            with open(path + f"{folder}/outer_macro_vel_{folder}.csv", 'r') as fp:
                line_count = len(fp.readlines())
                linesPerFileOuter[i] = line_count
    
    np.save(path + "FileLinesInner.npy", linesPerFileInner)
    np.save(path + "FileLinesOuter.npy", linesPerFileOuter)

    mdPos = np.load("/data/dust/user/sabebert/ConfigFiles/mdPosValues.npy", allow_pickle=True).item()
    with open(path + "mdPos_data.txt", "w") as f:
        for i in range(len(mdPos)):
            f.write(f"{i} {linesPerFileInner[i]} {linesPerFileOuter[i]} {mdPos[i]}\n")

    combined_data = np.zeros((len(mdPos), 4))
    for i in range(len(mdPos)):
        combined_data[i, 0] = i
        combined_data[i, 1] = linesPerFileInner[i]
        combined_data[i, 2] = linesPerFileOuter[i]
        combined_data[i, 3] = mdPos[i]

    np.save(path + "mdPos+Lines_data.npy", combined_data)
    return linesPerFileInner, linesPerFileOuter

def get_file_list(path,rangeFiles=None,numbers=None):
    """Get lists of inner and outer files that exist."""
    file_names_inner = []
    file_names_outer = []
    if(numbers==None):
        for i in range(rangeFiles[0],rangeFiles[1]):
            if os.path.isfile(path + f"{i}/md_macro_vel_{i}.csv"):
                file_names_inner.append(f"{i}/md_macro_vel_{i}.csv")
                file_names_outer.append(f"{i}/outer_macro_vel_{i}.csv")
        print(len(file_names_inner))
        print(len(file_names_outer))
    
    if(rangeFiles==None):
        #numbers = np.load(path + "fullFolders.npy")
        for i in numbers:
            if os.path.isfile(path + f"{i}/md_macro_vel_{i}.csv"):
                    file_names_inner.append(f"{i}/md_macro_vel_{i}.csv")
                    file_names_outer.append(f"{i}/outer_macro_vel_{i}.csv")

    #file_names_inner.append("842/md_macro_vel_842.csv")
    #file_names_outer.append("842/outer_macro_vel_842.csv") 
    #file_names_inner.append("130/md_macro_vel_130.csv")
    #file_names_outer.append("130/md_macro_vel_130.csv")
    print(file_names_inner)
    return file_names_inner, file_names_outer

def process_file(file_index, output_path,couplingCycles,rangeFiles=None,numbers=None):
    """Process a single file identified by index."""
    # Get file lists
    file_names, file_names_outer = get_file_list(path,rangeFiles,numbers)
    
    # Check if index is valid
    if file_index >= len(file_names):
        print(f"Error: File index {file_index} is out of range. Only {len(file_names)} files available.")
        return None

    print(file_names)
    fileData = np.zeros((1400, 3, 24, 24, 24))
    file_inner = file_names[file_index]
    file_outer = file_names_outer[file_index]
    print(f"Process: {file_index} processes file: {file_inner}!")
    # For every temporal couplingCycle
    for cycle in range(1, couplingCycles, 1):
        print(f"Cycle {cycle}")
        # Merge inner and outer domain MD data for each coupling cycle
        fileData[cycle] = np.transpose(join_MDinnerOuter(path, 
                                                       file_inner, 
                                                       file_outer, 
                                                       60, 2.5, cycle),  #MD Domain, cell size and current cycle to merge 
                                                       (3, 0, 1, 2))[:,1:-1,1:-1,1:-1]
        #Transpose for moving to correct shape form (26,26,26,3) and #remove ghost cells: (3,26,26,26) -> (3,24,24,24)
        #print(fileData[cycle].shape)
    # Save the result
    print(fileData.shape)
    output_file = os.path.join(output_path, f"processed_file_{file_names[file_index][0:3]}.npy")
    print("Gets here?")
    np.save(output_file, fileData)
    print(f"Saved result to {output_file}")
    return fileData

def combine_results(path):
    """Combine all processed results into a single array."""
    # Get file lists to determine range
    file_names, _ = get_file_list(path)
    
    all_results = []
    # Find all processed file results
    for i in range(len(file_names)):
        file_path = os.path.join(path, f"processed_file_{i}.npy")
        if os.path.exists(file_path):
            print(f"Loading {file_path}")
            data = np.load(file_path)
            all_results.append(data)
    
    # Save combined results
    combined_file = os.path.join(output_dir, "TrainingData3KVs_0-998.npy")
    np.save(combined_file, np.array(all_results))
    print(f"Combined {len(all_results)} processed files into {combined_file}")
    print(f"Final data shape: {np.array(all_results).shape}")
    return all_results

def splitData(data_path,ratios,numbers=None):
    ratios = [float(ratio) for ratio in ratios.strip("[]").split(",")]
    train_ratio, valid_ratio =ratios[0], ratios[1]
   
    file_names_inner=[]
    file_names_outer=[]
    if(numbers==None):
        for i in range(rangeFiles[0],rangeFiles[1]):
            if os.path.isfile(path + f"{i}/md_macro_vel_{i}.csv"):
                file_names_inner.append(f"{i}/md_macro_vel_{i}.csv")
                file_names_outer.append(f"{i}/outer_macro_vel_{i}.csv")
        print(len(file_names_inner))
        print(len(file_names_outer))
        numbers = file_names_inner[15:31]
        print(numbers)

    for file in numbers:
        x = np.load(data_path + f"/processed_file_{file[0:4]}.npy")
        print(x.shape)
        train_end = int(len(x) * train_ratio)
        #valid_end = train_end + int(len(x) * valid_ratio)
        # Split the data
        fileDatas_train, fileDatas_valid = x[:train_end], x[train_end:]  #x[train_end:valid_end]
        #print(len(fileDatas_valid))
        #Save split data
        print(f"Splitted data, saving npys to {path}")
        np.save(path + f"/Numpy/TrainingData3Kvs_{file[0:4]}_train",fileDatas_train)
        np.save(path + f"/Numpy/TrainingData3Kvs_{file[0:4]}_valid",fileDatas_valid)
    #np.save(path + "/Numpy/TrainingData3Kvs_0-999_train",fileDatas_train)
    #np.save(path + "/Numpy/TrainingData3Kvs_0-999_valid",fileDatas_valid)
    return

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Need argument for mode readIn/combine files")
        sys.exit(1)
    # Get the mode from command line
    mode = sys.argv[1]
    if mode == "FileLines":
        dict1, dict2 = get_file_LineNumbers(path)#,rangeFiles)
        print(dict1)
        print(dict2)
    #Write a list of files, which can then be used for the processing
    elif mode == "getList":
        get_file_list(path,rangeFiles)
    #Read-in and merge inner and outer files into one processed file
    elif mode == "readIn":
        if len(sys.argv) != 3:
            print("Usage for processing: python script.py process <file_index>")
            sys.exit(1)
        file_index = int(sys.argv[2])
        process_file(file_index, path + "",couplingCycles,rangeFiles=None,numbers=numbers)   
    #Split data for training/validation, overgive splitting ratio as second terminal command
    elif mode == "split":
        if len(sys.argv) !=3:
                print("need ratio tuple(test/valid) for splitting")
                sys.exit(1)
        ratio = sys.argv[2]
        print(ratio)
        splitData(path,ratio)
    #Not used in the final results, if data should be combined into one file
    elif mode == "combine":
        combine_results(path)
    else:
        print(f"Unknown mode: {mode}")
        print("Available modes: readIn, getList, split")
        sys.exit(1)
    
    
