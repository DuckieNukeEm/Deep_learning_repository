import pandas as pd
import numpy as np
from tensorflow.keras.utils import Sequence, to_categorical # For our own data generator
import cv2 # For image processing
import matplotlib.pyplot as plt # for showing the val vs train model
from tensorflow.keras.callbacks import ModelCheckpoint
#gpus=tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(gpus[0], True)

        

class DataGenerator(Sequence):
    """Generates data for Keras
    Sequence based data generator. Suitable for building data generator for training and prediction.
    #https://towardsdatascience.com/keras-data-generators-and-how-to-use-them-b69129ed779c
    """
    def __init__(self,
                 df: pd.DataFrame, # file that has the images on it, as well as the image types
                 y_var: str = None, #'grapheme_root','vowel_diacritic','consonant_diacritic' 
                 encoder: bool = None,
                 image_dir: str = None,
                 to_fit: bool =True,
                 batch_size: int = 32,
                 dim: tuple = (90,160),
                 channels: int = 1,
                 vertical_flip: float = 0,
                 horizontal_flip: float = 0,
                 rotate: tuple = (0,0), #PRob, max roatioan
                 shear: tuple = (0,0), # prob, max_shear
                 shuffle: bool =True,
                 sample_classes: int = 0,
                 save_model_path: bool = None):
        """Initialization of the DataGenerator Glass
        
        Arguments:
             df {pd.DataFrame} - A pandas data frame that has the y variable and either
                                     a) a file path to the image for each y 
                                     or
                                     b) all the X (having demnions listed in dim)
            y_var - {str/int} - the column name/index of the y-var
            encoder - {bool} - should the Y var be encoded
            to_fit - {bool} - if using to fit the model, should it return the y-value as well?
            batch_size - {int} - size of batches to deliver
            dim - {tuple} - dimension of the images (regardless of the color scale)
            channels - {int} - the number of color channels (generally 1 or 3)
            vertical_flip - {float} - The percent chance to flip a photo along a vertical axis
            horizontal_flip - (float) - The percent chance to flip a photo along a horiszontal axis
            rotate - (tuple - (prob, degree)) - A tuple of two:
                                                    1) the % chance of rotate
                                                    2) The amount of rotation
            shear - (tuple - (prob, amt)) - A two of two:
                                                    1) The % chance of shear
                                                    2) The amount of shear
            shuffle - {bool} - shuffle label indexes after every epoch
            sample_classes - {bool} - random samle n number from each class
        """
        
        #Getting Index that we will use to sort
        
        self.Idx_List = np.arange(df.shape[0])
        
        # Loading y_Vars
        if y_var is not None:
            if isinstance(y_var, list):
                y_var = y_var[0]
            is isinstance(df, pd.DataFrame):
                self.y_var = df[y_var].values
            else:
                self.y_var = df[:,y_var]
                
            self.y_dim = 0
            if encode:
                self.hot_encode_y()
                
        # getting Images Location
        if image_dir is None:
            self.image_is_dir = True
            if isinstance(df, pd.DataFrame):
                self.Imgs = df[image_dir].values
            else:
                raise('For Image Location, the df must be a dataframe')
        else:
            self.image_is_dir = False
            if isinstance(df, pd.DataFrame):
                if y_var is None:
                    self.Imas = df.values
                else
                    self.Imgs = df.drop(y_var, axis=1).values
            else:
                if y_var is None:
                    self.Imgs = df
                else:
                    self.Imgs = df[:,-y_var]
        
        #Setting other vars
        self.batch_size = batch_size
        self.fit = to_fit
        
        self.class_samp = sample_classes
        if self.class_samp > 0:            
            self.sample_class()
            
        self.shuffle = shuffle
        if self.shuffle:
            np.random.shuffle(self.Idx_List)
            
        self.dim = dim
        self.channels = channels
        if self.channels == 1:
            self.read_mode = cv2.IMREAD_GRAYSCALE
        elif self.channels == 3:
            self.read_mode = cv2.IMREAD_COLOR
        else:
            self.read_mode = cv2.IMREAD_UNCHANGED
        
        
        #Checks!
        assert 0 <= vertical_flip <=1, "vertical_flip = {}, which is not between 0 or 1".format(vertical_flip)
        self.v_flip = vertical_flip
        
        assert 0 <= horizontal_flip <=1, "horizontal_flip = {}, which is not between 0 or 1".format(horizontal_flip)
        self.h_flip = horizontal_flip
        
        assert 0 <= rotate[0] <=1, "first value of rotate = {}, which is not between 0 or 1".format(rotate[0])
        self.r_prob = rotate[0]
        assert 0 <= rotate[1] < 360, "second value of rotate = {}, which is not between 0 or 359".format(rotate[1])
        self.r_deg = rotate[1]
        
        assert 0 <= shear[0] <=1, "first value of shear = {}, which is not between 0 or 1".format(shear[0])
        self.s_prob = shear[0]
        
        assert 0 <= shear[1] <=359, "first value of shear = {}, which is not between 0 or 359".format(shear[1])
        self.s_fact = shear[1]
        
        self.save_model(save_model_path)
        #self.on_epoch_end()
        
    def __len__(self):
        """Denotes the number of batches per epoch
        :return: number of batches per epoch
        """
        return int(np.floor(len(self.Idx_List) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data
        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """
        # Generate indexes of the batch
        Batch_Idx = self.Idx_List[index * self.batch_size:(index + 1) * self.batch_size]


        # Generate data
        X = self._generate_X(Batch_Idx)

        if self.fit:
            y = self._generate_y(Batch_Idx)
            return X, y, [None]
        else:
            return X, [None]
        
    #TODO at a def __iter__ and __next__ methodology so we can loop through it!

    def on_epoch_end(self):
        """Updates indexes after each epoch
        """
        #self.indexes = np.arange(len(self.list_IDs))
        self.sample_class()
        if self.shuffle == True:
            np.random.shuffle(self.Idx_List)
        if self.save_path is not None:
            #self.save_chkpoint()
            self.save_model()
        
    def sample_class(self):
        """ rebalance the data"""
        if self.class_samp > 0:
            self.Idx_List = (pd.DataFrame({'Y':self.y_var})
                    .reset_index()
                    .groupby('Y')
                    .index
                    .apply(lambda x: x.sample(n=self.class_samp, replace=True))
                    .droplevel(0)
                    .values     
                    )
    
    
    
    def save_model(self, save_model_path: str = None):
        """ Provides a callback to save the path"""
        if save_model_path is not None:
            self.save_chkpoint = ModelCheckpoint(save_model_path, 
                                            monitor='val_accuracy', 
                                            verbose=1, 
                                            save_best_only=True,
                                            mode='max')
            self.save_path = save_model_path
        else:
            self.save_path = None
            
    def _generate_X(self, Batch_Idx):
        """Generates data containing batch_size images
        :param list_IDs_temp: list of label ids to load
        :return: batch of images
        """
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.channels))

        # Generate data
        for i, ID in enumerate(Batch_Idx):
            # Store sample
            if self.image_is_dir:
                X[i,] = self._load_image(self.Imgs[ID])
            else:
                

        return X
    
    def _generate_y(self, Batch_Idx):
        y = self.y[Batch_Idx,:]
        #y = self.y[Batch_Idx]
        return y
    
    def hot_encode_y(self):
        #self.y = pd.get_dummies(pd.Categorical(self.y_var)).values
        if self.y_var is not None:
            self.y = to_categorical(self.y_var)
            print("Number of Encodings is {}".format(self.y.shape[1]))
            self.y_dim = self.y.shape[1]      

    def _process_images(self,
                        img):
        """Processes the images"""
        if img.shape != self.dim:
            img = cv2.resize(img, self.dim)
        
        img = self._flip_vertical(img)
        
        img = self._flip_horizontal(img)
        
        img = self._rotate(img)
        
        img = self._shear(img)
        
        img = img/255.0
        if self.channels == 1:
            #img = img.reshape(self.dim[0],self.dim[1],self.channels)
            img = np.expand_dims(img,2)

        return(img)
    
    def _load_image(self, image_path):
        
        img = cv2.imread(image_path,  self.read_mode) #load an image as grayscale

        return(self._process_images(img))
        
    def _flip_vertical(self, img):
        """Flips and image on a  vertica axis, as set by v_flip,
        if set to zero, will never flip the image """
        if 1 - self.v_flip < np.random.uniform():
            img = cv2.flip(img,1)
            return(img)
        else:
            return(img)
        
    def _flip_horizontal(self, img):
        """Flips and image on a horizontal axis, as set by v_flip,
        if set to zero, will never flip the image """
        if 1 - self.h_flip < np.random.uniform():
            img = cv2.flip(img,0)
            return(img)
        else:
            return(img)
        
    def _rotate(self, img):
        if 1 - self.r_prob < np.random.uniform():
            rows, cols = img.shape[0], img.shape[1]
            #TODO allow a rotation both forward and backwards
            Rot_M = cv2.getRotationMatrix2D((cols / 2, rows / 2), np.random.uniform(self.r_deg), 1)
            img = cv2.warpAffine(img, Rot_M, (cols, rows))
            return(img)
        else:
            return(img)
        
    def _shear(self, img):
        """
        https://blog.paperspace.com/data-augmentation-for-object-detection-rotation-and-shearing/
        """
        if 1 - self.s_prob < np.random.uniform():
            #TODO allow a shear both forward and backwards
            rows, cols = img.shape[0], img.shape[1]
            
            shear_factor = np.random.uniform(self.s_fact)
            
            M = np.array([[1, shear_factor, 0],[0,1,0]])
            nW =  img.shape[1] + abs(shear_factor*img.shape[0])
            img = cv2.warpAffine(img, M, (int(nW), img.shape[0]))
            img = cv2.resize(img, (cols,rows))
            return(img)
        else:
            return(img)
    def plot(self, img_idx):
        plt.plot(self.Imgs[i,:])
        
        
        
def Plot_Val_Test(Model_hist):
    """
    param: Model_hist -> the training history of a model
    
    Purpose: To graph test vs. Validation of the model over 
    """
    acc = Model_hist.history['accuracy']
    val_acc = Model_hist.history['val_accuracy']
    loss = Model_hist.history['loss']
    val_loss = Model_hist.history['val_loss']
    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend(loc=0)
    plt.figure()
    plt.show()
