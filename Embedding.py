import torch
import torch.nn as nn
from torchtyping import TensorType
from typeguard import typechecked

class embedding(nn.Module):
   def __init__(self, x) -> None:
        """
        Initialize the embedding of the Tensor. This embedds a single pixel of the image in a sine and cosine tensor.
        
        Args:
            - x (torch.tensor: float, shape : (batch_size, image_size_x,image_size_y) ) : Batch size of MNIST images.
            
        """
        super().__init__()
        self.x=x

   def prepare_inputs(self)-> TensorType['batch_size', 'image_size']:
      """ 
      This funtion flattens the input image into a vector following the snake flattening as described as the paper.
      
      Args:
         Empty
      Returns:
         -  z (torch.tensor : float, shape: (batch_size, image_size ) ) :  Flattened and snake pattern reordered image.
      """

      x=self.x
      batch_size=x.shape[0]
      z=torch.zeros(batch_size,x.shape[2]*x.shape[3])
      
      for image in range(batch_size):
         x_matrix=x[image,:,:]
         size_matrix=x_matrix.shape[0]
         odd_rows=torch.arange(1,size_matrix,2)
       
        # Snake encode pattern the image. Flips every odd row.
         for num_row in odd_rows:
            # Convert the row to a list
            row_list = x_matrix[:, num_row].tolist()
            # Reverse the row list
            row_list.reverse()
            # Assign the reversed list back to the matrix
            x_matrix[:, num_row] = torch.FloatTensor(row_list)   
         
         z[image,:]= (x_matrix.flatten()  )
      
      return z
   
   
   def inputs_embedded(self) -> TensorType['batch_size', 'image_size','physical_bond']:
      """ 
      This function encodes the data into a tensor of size batch_size , image_size, physical_dimension. Each pixer of every image it will be 
      encoded into a one dimensional array : (cos(pixel), sin(pixel)).
      
      Args:
         Empty   
      Returns:
         - embedding (torch.tensor : float, shape: (batch_size, image_size, physical_bond) ): Embedded image as the paper.
      """
      # Prepared inputs, every image has ben shifted with the snake pattern.
      x=self.prepare_inputs()
      batch_size=x.shape[0]
      dim_imag=x.shape[1]
      embedding=torch.zeros(batch_size,2*dim_imag)

      # Put the sine and cosine of every pixel in the odd and even position of the embedding tensor
      for image in range(batch_size):
         embedding[image, 0::2]=torch.cos((torch.pi/2) * x[image,:])
         embedding[image, 1::2]=torch.sin((torch.pi/2) * x[image,:])
      
      return torch.reshape(embedding,[batch_size, dim_imag ,2]) # Reshape the tensor
   

def one_hot_encoding(y) -> TensorType['batch_size', 'num_classes']:
      ''' This function one hot encodes the batch labels.
      
      Args:
         - y (torch.tensor: int, shape : (batch_size) ): Batch labels.
      Returns:
         - (torch,tensor: int, shape : (batch_size, num_labels)): Tensor of each image label one hot encoded.
      '''
      return nn.functional.one_hot(y)
