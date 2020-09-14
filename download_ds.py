import moxing as mox

def download_dataset(data_url, local_data_path):
    # data download
    print('Download data.')
    mox.file.copy_parallel(src_url=data_url, dst_url=local_data_path)