import jax
import jax.numpy as jnp

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

    
    def random_perturbed_tensor(self,tensor):
        """ This function will initialise a random tensor given the shape of the input.
        
        Args:
            - tensor(JAX.numpy.Array: float): Original zeros tensor.
            - mean (float): Mean of the perturbation
            - std: (float): Estandard deviation of the perturbation.
        Returns:
            tensor ( JAX.numpy.Array: float): perturbed tensor"""
        shape=tensor.shape
        return jnp.round(tensor + jnp.ones(shape=shape)*self.mean + self.std*jax.random.normal(key=jax.random.PRNGKey(seed=1997),shape=shape), 6)

    
    def random_init(self):
        """ This function will initialize a random MPS. In the beggining and in the end of the MPS, the shape of the local tensor in that position
        will be (physical_bond, bond_dim) and (bond_dim, physical_bond). In the position l, it will be added the l leg at the end of the local tensor,
        ex: (bond_dim, physical_bond, bond_dim, l_leg)
        
        Args:
            Empty
        Returns:
        - mps (list[JAX.Numpy.Array: float], 
                shape: List[(physical_leg,bond_dim), (bond_dim, physical_leg,bond_dim), ..., (bond_dim, physical_leg, bond_dim, l_leg), 
                ..., (bond_dim, physical_leg, bond_dim),.., (bond_dim, physical_leg) ]): List representing the full random initialized MPS"""
        
        mps_tensor=[]
        tensor_init_middle=jnp.zeros(shape=(self.bond_dim, self.in_dim, self.bond_dim))

        # Initialize the mps site tensors from 0 to length
        for index in range(self.length):
            if index==0:
                # Initialize first the leftest tensor with l dimension
                if (self.position_l==0):
                    tensor_init=jnp.zeros(shape=(self.in_dim, self.bond_dim, self.l_dim))
                    tensor=self.random_perturbed_tensor(tensor_init)
                    mps_tensor.append(tensor)
                 # Initialize first the leftest tensor without l dimension
                else:
                    tensor_init=jnp.zeros(shape=(self.in_dim, self.bond_dim))
                    tensor=self.random_perturbed_tensor(tensor_init)
                    mps_tensor.append(tensor)
            
            elif index==self.length-1:
                # Last tensor with l dimension
                if self.position_l==self.length-1:
                    tensor_init=jnp.zeros(shape=(self.bond_dim, self.in_dim, self.l_dim))
                    mps_tensor.append(self.random_perturbed_tensor(tensor_init))
                # Last tensor without l dimension
                else:
                    tensor_init=jnp.zeros(shape=(self.bond_dim, self.in_dim))
                    mps_tensor.append(self.random_perturbed_tensor(tensor_init))
            
            # Iterate over the length for the other cases
            else:
                if index==self.position_l:
                    tensor_init=jnp.zeros(shape=(self.bond_dim, self.in_dim,self.bond_dim, self.l_dim))
                    mps_tensor.append(self.random_perturbed_tensor(tensor_init))
                else:    
                    mps_tensor.append(self.random_perturbed_tensor(tensor_init_middle))

        return mps_tensor
    

    def canonical_form(self):
        """ 
        This function will put the random initialized MPS into a left and right canonical . In the beggining and in the end of the MPS, the shape of the local tensor in that position
            will be (physical_bond, bond_dim) and (bond_dim, physical_bond). In the position l, it will be added the l leg at the end of the local tensor,
            ex: (bond_dim, physical_bond, bond_dim, l_leg)
            
            Args:
                Empty
            Returns:
            - mps (list[JAX.Numpy.Array: float], 
                    shape: List[(physical_leg,bond_dim), (bond_dim, physical_leg,bond_dim), ..., (bond_dim, physical_leg, bond_dim, l_leg), 
                    ..., (bond_dim, physical_leg, bond_dim),.., (bond_dim, physical_leg) ]): List representing the left and right canonical random initialized MPS.
        """
        # Get the random MPS
        mps=self.random_init()
        
        # Put the MPS into left canonical until position_l -1
        for index in range( self.position_l-1):
            # Reshape into a matrix
            if index==0:
                dim_i=mps[index].shape[0]
                dim_j=mps[index].shape[1]
                B=mps[index]
            else:
                dim_i=mps[index].shape[0]
                dim_j=mps[index].shape[1]
                dim_k=mps[index].shape[2]
                B=jnp.reshape(mps[index],[dim_i*dim_j,-1])
            
            #Perform svd in the matrix
            Atemp,Dtemp,Vhtemp=jnp.linalg.svd(B)

            # Perform truncation and normalization singular values
            if Atemp.shape[1]>self.bond_dim:
                Atemp=Atemp[:,:self.bond_dim]
                Atemp=jnp.round(Atemp, 6)
                Vhtemp=jnp.round(Vhtemp[:self.bond_dim,:], 6)
                Dtemp=jnp.round ( jnp.diag(Dtemp[:self.bond_dim]), 6)
                Dtemp=Dtemp / jnp.trace(Dtemp.T @ Dtemp) 
            else:
                Atemp=jnp.round( Atemp[:,:Dtemp.shape[0]], 6)
                Dtemp=jnp.round( jnp.diag(Dtemp), 6)
                Vhtemp=jnp.round( Vhtemp[:Dtemp.shape[0],:], 6)
                Dtemp=Dtemp / jnp.trace(Dtemp.T @ Dtemp)
            
            # Reshape based on the index.
            if  index==0:
                A=jnp.reshape(Atemp,[dim_i,-1])
                A_plusone=jnp.reshape(Dtemp @ Vhtemp,[-1,dim_j])
                mps[index]=A
            else:
                A=jnp.reshape(Atemp,[dim_i,dim_j,-1])
                A_plusone=jnp.reshape(Dtemp @ Vhtemp,[-1,dim_k])
                mps[index]=A
            
            # Contract the non-left canonical part to the next tensor
            if index == self.position_l -1:
                if index==self.length-2:
                    mps[index+1]=jnp.einsum('ij, jkl ->ikl',A_plusone,mps[index+1])
                else:
                    mps[index+1]=jnp.einsum('ij, jklm ->iklm',A_plusone,mps[index+1])
            else:
                mps[index+1]=jnp.einsum('ij, jkl->ikl',A_plusone,mps[index+1])
        
        # Put the MPS into right canonical until position_l -1
        for index in reversed(range(self.position_l+1,self.length)):
            # Reshape matrix
            if index==self.length-1:
                dim_i=mps[index].shape[0]
                dim_j=mps[index].shape[1]
                B=mps[index]
            else:  
                dim_i=mps[index].shape[0]
                dim_j=mps[index].shape[1]
                dim_k=mps[index].shape[2]
                B=jnp.reshape(mps[index],[-1,dim_j*dim_k])
            
            #Perform svd in the matrix
            Atemp,Dtemp,Vhtemp=jnp.linalg.svd(B)

            # Perform truncation.
            if Vhtemp.shape[0]>self.bond_dim:
                Atemp=jnp.round( Atemp[:,:self.bond_dim], 6)
                Vhtemp=Vhtemp[:self.bond_dim,:]
                Vhtemp=jnp.round(Vhtemp, 6)
                Dtemp=jnp.round( jnp.diag(Dtemp[:self.bond_dim]), 6)
                Dtemp=Dtemp / jnp.trace(Dtemp.T @ Dtemp)
            else:
                Dtemp=jnp.round( jnp.diag(Dtemp), 6)
                Atemp=jnp.round( Atemp[:,:Dtemp.shape[0]], 6)
                Dtemp=Dtemp / jnp.trace(Dtemp.T @ Dtemp)
                Vhtemp=jnp.round( Vhtemp[:Dtemp.shape[0],:], 6)
                

            # Perform reshape 
            if index==self.length-1:
                A=jnp.reshape(Vhtemp,[-1, dim_j])
                A_minusone=jnp.reshape(Atemp @ Dtemp,[dim_i,-1])
                mps[index]=A
            else:
                A=jnp.reshape(Vhtemp,[-1, dim_j, dim_k])
                A_minusone=jnp.reshape(Atemp @ Dtemp,[dim_i,-1])
                mps[index]=A
            
            # Contract the non-right canonical part to the previous tensor depending if the previous one is already the one with the l -lges
            if index == self.position_l +1:
                if index==1:
                    mps[index-1]= jnp.einsum('ijl, jm->iml',mps[index-1],A_minusone)
                else:
                    mps[index-1]= jnp.einsum('ijkl, km->ijml',mps[index-1],A_minusone)
            else:
                mps[index-1]=jnp.einsum('ijk, kl->ijl',mps[index-1],A_minusone)
        
        return mps