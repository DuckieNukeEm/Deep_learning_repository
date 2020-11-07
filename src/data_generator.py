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
                 csv_file: pd.DataFrame, # file that has the images on it, as well as the image types
                 y_var: str = 'grapheme_root', #'grapheme_root','vowel_diacritic','consonant_diacritic' 
                 image_dir: str = 'Image_Dir',
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
        """Initialization
        :param csv_File #CSV file that has the path to the stores on it
        :param y_var: a list of 'root','voewl','consonant'
        :param to_fit: Provive the dependent variable as well
        :param batch_size: The size of each batch to deliver
        :param dim: dimensions of the photos to use
        :param channels: The number of channals of the photo - 1 is bw, 3 is color, any other is customer
        :param vertical_flip: (dbl) The percent chance to flip a photo along a vertical axis
        :param horizontal_flip: (dbl) The percent chance to flip a photo along a horiszontal axis
        :param rotate: (tuple - (prob, degree)) A two unit tuple, first is the % chance of rotate, the next is the amount of rotation
        :param shear: (tuple - (prob, amt)) A two unit tuple, first is the % chance of shear, the next is the amount of shear
        :param shuffle: True to shuffle label indexes after every epoch
        :param sample_classes: random samle n number from each class
        """
        
        #Getting Index that we will use to sort
        
        self.Idx_List = np.arange(csv_file.shape[0])
        
        # Loading y_Vars
        if isinstance(y_var, list):
            y_var = y_var[0]
            
        self.y_var = csv_file[y_var].values
       
        #TODO Move this to the aabove if statement, removing the need for y_var, and y
        self.y_dim = 0
        self.hot_encode_y()
        #self.y = self.y_var
        # getting Images Location
        self.Imgs = csv_file[image_dir].values
        
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
            X[i,] = self._load_image(self.Imgs[ID])

        return X
    
    def _generate_y(self, Batch_Idx):
        y = self.y[Batch_Idx,:]
        #y = self.y[Batch_Idx]
        return y
    
    def hot_encode_y(self):
        #self.y = pd.get_dummies(pd.Categorical(self.y_var)).values
        self.y = to_categorical(self.y_var)
        print("Number of Encodings is {}".format(self.y.shape[1]))
        self.y_dim = self.y.shape[1]      


    def _load_image(self, image_path):
        
        img = cv2.imread(image_path,  self.read_mode) #load an image as grayscale
        
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
