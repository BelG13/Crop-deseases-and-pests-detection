def tf_train_valid_test_split(dataset , train_size = 0.8 , valid_size = 0.1 , test_size = 0.1 , buffer_size = 1000):
    
    """Split the dataset into a train_set , a valid_set and a test_set.

        Parameters
        ----------
        dataset : dataset , required
            the dataset.
            
        train_size : float, default ==> 0.8
            the size of the train_set.
        
        valid_size: float , default ==> 0.1
            the size of the valid set.
            
        test_size: float , default ==> 0.1
            the size of the test set.
            
            
        Returns
        -------
        train_ds : the train dataset
        valid_ds : the valid dataset
        test_ds  : the valid dataset
    
        """
    
    if train_size + valid_size + test_size != 1 :
        raise 'the sum of train_size , valid_size and test_size must be equal to 1'
        
    
    dataset = dataset.shuffle(buffer_size)
    
    
    ds_len     = len(dataset)
    train_size = int(train_size * ds_len)
    valid_size = int(valid_size * ds_len)
    test_size  = int(test_size  * ds_len)
    
    train_ds = dataset.take(train_size)
    valid_ds = dataset.skip(train_size).take(valid_size)
    test_ds  = dataset.skip(train_size).skip(valid_size)
    
    return train_ds , valid_ds , test_ds