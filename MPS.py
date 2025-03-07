import jax
import jax.numpy as jnp
from functools import partial

class mps:
    def __init__(self, bond_dim, in_dim, length, l_dim, std,mean, position_l) -> None:
        """Initialize the MPS class.
        
        Args:
            bond_dim (int): Maximum bond dimension of the MPS.
            in_dim (int): Dimension physical-bond MPS.
            length (int):  Length mps.
            l_dim (int): Dimension of the upper leg associated with the label.
            std (float): Standard deviation of random initialization normal distribution.
            mean (float): Mean of the random initialization normal distribution.
            position_l (int): Position of the l leg in the MPS.
        Returns:
            Empty
        """
        
        self.bond_dim=bond_dim
        self.in_dim=in_dim
        self.length=length
        self.l_dim=l_dim
        self.std=std
        self.mean=mean
        self.position_l=position_l

    
    def random_perturbed_tensor(self, tensor,mean,std,key):
        """ This function will initialise a random tensor given the shape of the input.
        
        Args:
            - tensor(JAX.numpy.Array: float): Original zeros tensor.
            - mean (float): Mean of the perturbation
            - std: (float): Estandard deviation of the perturbation.
            - key (int): Key to generate randomness
        Returns:
            tensor ( JAX.numpy.Array: float): perturbed tensor"""
        shape=tensor.shape
    
        return jnp.round(tensor + jnp.ones(shape=shape)*mean + std*jax.random.normal(key=key,shape=shape), 10)

    
    def random_init(self):
        """ This function will initialize a random MPS. In the beggining and in the end of the MPS, the shape of the local tensor in that position
        will be (1, physical_bond, bond_dim) and (bond_dim, physical_bond, 1). In the position l, it will be added the l leg at the end of the local tensor,
        ex: (bond_dim, physical_bond, bond_dim, l_leg)
        
        Args:
            Empty
        Returns:
        - mps (list[JAX.Numpy.Array: float], 
                shape: List[(1,physical_leg,bond_dim), (bond_dim, physical_leg,bond_dim), ..., (bond_dim, physical_leg, bond_dim, l_leg), 
                ..., (bond_dim, physical_leg, bond_dim),.., (bond_dim, physical_leg,1) ]): List representing the full random initialized MPS"""
        
        mps_tensor=[]
        key = jax.random.PRNGKey(1997)
        # Initialize empty tensors. The shape depends on the position of each one
        tensor_middle=jnp.zeros(shape=(self.bond_dim, self.in_dim, self.bond_dim))
        tensor_init=jnp.zeros(shape=(1,self.in_dim, self.bond_dim))
        tensor_end=jnp.zeros(shape=(self.bond_dim, self.in_dim, 1))

        if self.position_l==0:
            # Initialize first the leftest tensor with l dimension
            tensor_init_l=jnp.zeros(shape=(1,self.in_dim, self.bond_dim, self.l_dim))
            mps_tensor.append(self.random_perturbed_tensor(tensor_init_l, self.mean, self.std, key))
        else:
            mps_tensor.append(self.random_perturbed_tensor(tensor_init, self.mean, self.std, key))
        
        if self.position_l==self.length-1:
            # Last tensor with l dimension
            tensor_l=jnp.zeros(shape=(self.bond_dim, self.in_dim, 1, self.l_dim))
        else:
             # Tensor in the middle with l dimension
            tensor_l=jnp.zeros(shape=(self.bond_dim, self.in_dim,self.bond_dim, self.l_dim))
        
        jit_rand_pertubed_tensor=jax.jit(partial(self.random_perturbed_tensor))
        

        # Initialize the mps site tensors from 1 to length
        for index in range(1,self.length):
            key, subkey = jax.random.split(key)  # Ensure different keys for each tensor

            if index==self.position_l:
                mps_tensor.append(self.random_perturbed_tensor(tensor_l, self.mean, self.std, subkey))
            else:
                if index!=self.length-1:
                    mps_tensor.append(jit_rand_pertubed_tensor(tensor_middle, self.mean, self.std, subkey))
                else:
                    mps_tensor.append(self.random_perturbed_tensor(tensor_end, self.mean, self.std, subkey))

        return mps_tensor
    
    def local_orthonormalize_left(self,A,Anext):
        """Left-orthonormalize a MPS tensor `A` by a QR decomposition, and update tensor at next site.
        Args:
            - A (JAX.Numpy.Array: float): Tensor to perform the QR decomposition dim(A)=3,
            - Anext (JAX.Numpy.Array: float): Tensor to annex the R matrix from the QR decomposition dim(Anext)=3
        Returns:
            - A (JAX.Numpy.Array: float): Left orthonormalized tensor.
            - Anext (JAX.Numpy.Array: float): New adjacent tensor.
            """
        # perform QR decomposition and replace A by reshaped Q matrix

        shape = A.shape
        assert len(shape) == 3
        Q, R = jnp.linalg.qr(jnp.reshape(A/jnp.linalg.norm(A), (shape[0]*shape[1], shape[2])), mode='reduced')
        # truncate bond dimension
        if Q.shape[1]>self.bond_dim:
            Q=jnp.round(Q[:,self.bond_dim],10)
            R=jnp.round(R[:self.bond_dim,:],10)

        A = jnp.reshape(Q, (shape[0], shape[1], Q.shape[1]))
        # update Anext tensor: multiply with R from left
        Anext = jnp.tensordot(R, Anext, axes=(1, 0))
        return A, Anext
    
    def local_orthonormalize_right(self,A, Aprev,):
        """
        Right-orthonormalize a MPS tensor `A` by a QR decomposition,  and update tensor at previous site.
        Args:
            - A (JAX.Numpy.Array: float): Tensor to perform the QR decomposition dim(A)=3,
            - Aprev (JAX.Numpy.Array: float): Tensor to annex the R matrix from the QR decomposition dim(Aprev)=3
        Returns:
            - A (JAX.Numpy.Array: float): Left orthonormalized tensor.
            - Aprev (JAX.Numpy.Array: float): New adjacent tensor. 
            """
        
        # flip left and right virtual bond dimensions
        A = jnp.transpose(A, (2, 1, 0))
        # perform QR decomposition and replace A by reshaped Q matrix
        shape = A.shape
        assert len(shape) == 3
        Q, R = jnp.linalg.qr(jnp.reshape(A/jnp.linalg.norm(A), (shape[0]*shape[1], shape[2])))
        
        # truncate bond dimension
        if Q.shape[1]>self.bond_dim:
            Q=jnp.round(Q[:,self.bond_dim],10)
            R=jnp.round(R[:self.bond_dim,:],10)

        A = jnp.transpose(jnp.reshape(Q, (shape[0], shape[1], Q.shape[1])), (2, 1, 0))
        # update Aprev tensor: multiply with R from right
        Aprev = jnp.tensordot(Aprev, R, axes=(2, 1))
        return A, Aprev
    
    def canonical_form(self):
        """ 
        This function will put the random initialized MPS into a left and right canonical . In the beggining and in the end of the MPS, the shape of the local tensor in that position
            will be (physical_bond, bond_dim) and (bond_dim, physical_bond). In the position l, it will be added the l leg at the end of the local tensor,
            ex: (bond_dim, physical_bond, bond_dim, l_leg)
            
            Args:
                Empty
            Returns:
            - mps (list[JAX.Numpy.Array: float], 
                    shape: List[(1,physical_leg,bond_dim), (bond_dim, physical_leg,bond_dim), ..., (bond_dim, physical_leg, bond_dim, l_leg), 
                    ..., (bond_dim, physical_leg, bond_dim),.., (bond_dim, physical_leg,1) ]): List representing the left and right canonical random initialized MPS.
        """
        # Get the random MPS
        mps=self.random_init()
        jit_local_orthonamilize_left=jax.jit(partial(self.local_orthonormalize_left))
        jit_local_orthonamilize_right=jax.jit(partial(self.local_orthonormalize_right))
        # Put the MPS into left canonical until position_l
        for index in range( self.position_l-1):
            A,Anext=jit_local_orthonamilize_left(mps[index],mps[index+1])
            mps[index]=A
            mps[index+1]=Anext
        
        mps[self.position_l-1], mps[self.position_l]=jit_local_orthonamilize_left(mps[self.position_l-1],mps[self.position_l])
        
        # Put the MPS into right canonical until position_l 
        for index in reversed(range(self.position_l+1,self.length)):
            
            A,Aprev=jit_local_orthonamilize_right(mps[index],mps[index-1])
            mps[index]=A
            mps[index-1]=Aprev
         
        return mps