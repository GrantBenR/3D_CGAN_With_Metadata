from torch import zeros, sin, cos, rand, randn, cat, tensor
from torch import Tensor
from torch import float32

from pandas import Series
from math import pi
from os.path import exists, join
from os import getcwd
from pandas import read_csv, to_datetime, Timestamp, Series 

def MetadataTextToVector(
        text_series: Series,
        vocabulary_mapping: dict[str, int]
    ) -> Tensor:
    num_samples = len(text_series)
    vocab_size = len(vocabulary_mapping)
    vectors = zeros((num_samples, vocab_size), dtype=float32)
    
    for row_idx, text in enumerate(text_series):
        words = str(text).lower().split()
        for word in words:
            if word in vocabulary_mapping:
                word_idx = vocabulary_mapping[word]
                vectors[row_idx, word_idx] += 1.0
    return vectors

def LoadTrainingData(
        relative_csv_path="csv/testdata.csv",
        columns=['x', 'y', 'z', 'metadata1', 'metadata2']
    ) -> tuple[Tensor, int, Tensor, int]:
    # get full csv path
    full_csv_path = join(getcwd(), relative_csv_path)

    if not exists(path=full_csv_path):
        return None
    
    # initialize dataframe
    df = read_csv(
        filepath_or_buffer="your_data_file.csv",
        header=None   
    )
    df.columns = columns

    # parse and normalize timestamp column
    df['z'] = to_datetime(df['z'], format="%Y-%m-%d%H:%M:%S")
    min_time = Timestamp(df['z'].min())
    max_time = Timestamp(df['z'].max())
    if max_time == min_time:
        df['z_normalized'] = 0.0
    else:
        df['z_normalized'] = (df['z'] - min_time).dt.total_seconds() / (max_time - min_time).dt.total_seconds()

    # add the vectors to a tensor object
    vectors = df[['x', 'y', 'z_normalized']].values.astype('float32')
    train_data = tensor(vectors)
    train_data_length = train_data.size(0)

    all_text = df['metadata1'].astype(str) + " " + df['metadata2'].astype(str)
    vocab = sorted(list(set(" ".join(all_text.tolist()).lower().split())))

    # Build a fast lookup dictionary mapping: word -> index position
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}

    text_dim = len(vocab)

    # Convert the metadata to tensors
    meta_1_tensor = MetadataTextToVector(df['metadata1'], word_to_idx)
    meta_2_tensor = MetadataTextToVector(df['metadata2'], word_to_idx)

    train_metadata = cat((meta_1_tensor, meta_2_tensor), dim=1)
    return (train_data, train_data_length, train_metadata, text_dim)


def LoadDefaultData(text_dim=384) -> tuple[Tensor, int, Tensor, int]:
    train_data_length = 1037 

    train_data = zeros((train_data_length, 3))
    angles = 2 * pi * rand(train_data_length)
    train_data[:, 0] = sin(angles)
    train_data[:, 1] = cos(angles)              
    train_data[:, 2] = angles / (2 * pi)         

    simulated_embeddings_field1 = randn((train_data_length, text_dim))
    simulated_embeddings_field2 = randn((train_data_length, text_dim))
    train_metadata = cat((simulated_embeddings_field1, simulated_embeddings_field2), dim=1)
    
    return (train_data, train_data_length, train_metadata, text_dim)