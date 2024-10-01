from pathlib import Path

def get_dataset_params(datasetName, normalized=False):

    # datasetName : 'Benchmark1_11000', 'RBF1_40000', 'Gaussian_4C2D800', 'Insects', 'PowerSupply', 'NOAA', 'sensor',

    dataset_params = {}

    if (datasetName == 'Benchmark1_11000'):
        # "https://raw.githubusercontent.com/CIG-UFSCar/DS_Datasets/master/Synthetic/Non-Stationary/Bench1_11k/Benchmark1_11000.csv"
        dataset_params["datasetPath"] = Path("datasets","Benchmark1_11000.csv")
        dataset_params["numChunks"] = 11
        dataset_params["chunksize"] = 1000
        dataset_params["break_n"] = 11_000
        dataset_params["alpha"] = 0.0015
    elif (datasetName == 'RBF1_40000'):
        # "https://raw.githubusercontent.com/CIG-UFSCar/DS_Datasets/master/Synthetic/Non-Stationary/RBF1_40k/RBF1_40000.csv"
        dataset_params["datasetPath"] = Path("datasets","RBF1_40000.csv")
        dataset_params["numChunks"] = 40
        dataset_params["chunksize"] = 1000
        dataset_params["break_n"] = 40_000
        dataset_params["alpha"] = 0.0015
    elif (datasetName == 'Gaussian_4C2D800'):
        # https://gitlab.citius.usc.es/david.gonzalez.marquez/GaussianMotionData/-/raw/master/SamplesFile_b_4C2D800Linear.csv?ref_type=heads
        if normalized:
            dataset_params["datasetPath"] = Path("datasets","DS1norm.csv")
        else: 
            dataset_params["datasetPath"] = Path("datasets","DS1.csv")
        dataset_params["numChunks"] = 8
        dataset_params["chunksize"] = 100
        dataset_params["break_n"] = 800
        dataset_params["alpha"] = 0.0015
    elif (datasetName == 'Insects'):
        # https://doi.org/10.1007/s10618-020-00698-5
        # https://sites.google.com/view/uspdsrepository
        # https://drive.google.com/drive/folders/1J0EHI7TS_pQB8iQVlbxnhMnbKdkWvhr0
        dataset_params["datasetPath"] = Path("datasets","INSECTS-incremental_balanced_norm.csv")
        dataset_params["numChunks"] = 19
        dataset_params["chunksize"] = 3000
        dataset_params["break_n"] = 57_000
        dataset_params["alpha"] = 0.0005  # I changed for the insects dataset, original is: alpha = 0.0015
    elif (datasetName == 'PowerSupply'):
        # https://www.cse.fau.edu/~xqzhu/stream.html
        # https://www.cse.fau.edu/~xqzhu/Stream/powersupply.arff
        dataset_params["datasetPath"] =  Path("datasets","powersupply.csv")
        dataset_params["numChunks"] = 29
        dataset_params["chunksize"] = 1000
        dataset_params["break_n"] = 29_000
        dataset_params["alpha"] = 0.0015
    elif (datasetName == 'NOAA'):
        # https://doi.org/10.1109/TKDE.2012.136
        # Should be here: ftp://ftp.ncdc.noaa.gov/pub/data/gsod
        # https://drive.google.com/drive/folders/1J0EHI7TS_pQB8iQVlbxnhMnbKdkWvhr0
        if normalized:
            dataset_params["datasetPath"] =  Path("datasets","NEweather_norm.csv")
        else:
            dataset_params["datasetPath"] =  Path("datasets","NEweather.csv")
        dataset_params["numChunks"] = 18
        dataset_params["chunksize"] = 1000
        dataset_params["n_clusters"] = 2
        dataset_params["break_n"] = 18_000
        dataset_params["alpha"] = 0.0015
    elif (datasetName == 'sensor'):
        # https://www.cse.fau.edu/~xqzhu/stream.html
        # https://www.cse.fau.edu/~xqzhu/Stream/sensor.arff
        dataset_params["datasetPath"] =  Path("datasets","sensor.csv")
        dataset_params["numChunks"] = 2219
        dataset_params["chunksize"] = 1000
        dataset_params["n_clusters"] = 54
        dataset_params["break_n"] = 2_219_000
        dataset_params["alpha"] = 0.0015

    return dataset_params
