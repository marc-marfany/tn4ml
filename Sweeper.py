
import numpy as np
import jax
import jax.numpy as jnp

class sweeper:
    def __init__(self, Bj, B_jadjacent, phi_j, phi_jadjacent, learning_rate, bond_dim, j, y, left_side, right_side) -> None:
        """
        Initialize the sweeping algorithm. Save parameters

        Args:
            - Bj (JAX.Numpy.Array: float, shape : (bond_dim, physical_leg, bond_dim, l_leg)): Tensor will have 4 legs and it will be the one to update.
            - B_jadjacent (JAX.Numpy.Array: float, shape :(bond_dim, physical_leg, bond_dim)): The adjacent tensor which will be updated and it will be added the l-leg
            - phi_j (JAX.numpy.Array : float, shape: (batch_size, 1, physical_bond) ): Embedded training images pixel on the position of the l leg tensor.
            - phi_jadjacent (JAX.numpy.Array : float, shape: (batch_size, 1, physical_bond) ): Embedded training images pixel on the position of the adjacent tensor of the l leg.
            - y (JAX.Numpy.Array: float, shape: (batch-size, #_labels_size): Labels training data.
            - learning_rate (float): Learning rate.
            - bond_dim (int): Maximum bond dimension between tensors in the MPS.
            - j (int): Position j in the MPS where the l leg is situated.
            - left_side (List[JAX.Numpy.Array: float],  shape: [#_training] (size_contracted_legs)) : It will contain the left data contracted to perform the update.
            - right_side (List[JAX.Numpy.Array: float],  shape: [#_training] (size_contracted_legs)) : It will contain the right data contracted to perform the update.
            
        """
        
        self.Bj=Bj
        self.B_jadjacent=B_jadjacent 
        self.phi_j=phi_j
        self.phi_jadjacent=phi_jadjacent
        self.learning_rate=learning_rate
        self.bond_dim=bond_dim
        self.j=j
        self.y=y
        self.left_side=left_side
        self.right_side=right_side
    
    
    def contract_with_neighbour_tensor_lr(self):
        """
        This function will contract two adjacent tensor, (j, j+1) into a single tensor in the direction left to right sweep. It returns a single tensor 
        with two pysical bonds, the l leg in the last position, and the left and right bond (those depending on the position)
        
        Returns:
            - joint_tensor (JAX.numpy.Array : float, shape : (bond_dim, physical_bond(j), physical_bond(j+1), bond_dim, l_leg)): 
                We are sweeping from left to right (contraction of j, j+1 tensors). Single tensor being 5 dimensional.
        """
        
        joint_tensor=jnp.tensordot(self.Bj, self.B_jadjacent, axes=(2,0))
        
        return jnp.transpose(joint_tensor,(0,1,3,4,2))
        
       
    def contract_with_neighbour_tensor_rl(self):
        """
        This function will contract two adjacent tensor, (j-1, j) into a single tensor in the direction right to left sweep. It returns a single tensor 
        with two pysical bonds, the l leg in the last position, and the left and right bond (those depending on the position)
        
        Returns:
            - joint_tensor (JAX.numpy.Array : float, shape : (bond_dim, physical_bond(j-1), physical_bond(j), bond_dim, l_leg)): If right
                We are sweeping from right to left (contraction of j-1, j tensors) .Single tensor being 5 dimensional.
        """
        
        joint_tensor=jnp.tensordot(self.B_jadjacent, self.Bj, axes=(2,0))
        
        return joint_tensor
         
       
    def update_single_data_left(self,i):
        ''' 
        Contract the left hand side of the specific index of the MPS AFTER being updated. This
        contraction will be with the adjacent data position (bond_dim) and the embedded image pixel (physical_bond).
        
        Args:
            - i (int): Index pixel's image batch position to contract.
        Returns:
            - left_side_i (JAX.numpy.Array : float, shape (bond_dim)): New left_side_data for the position .
        '''
    
        left_side_i=jnp.tensordot(self.left_side[i], self.Bj, axes=(0,0))
        left_side_i=jnp.tensordot(self.phi_j[i,:], left_side_i, axes=(0,0))

        return left_side_i / jnp.linalg.norm( left_side_i )
    
    
    def update_single_data_right(self, i):
        ''' 
        Contract the right hand side of the specific index of the MPS AFTER being updated. This
        contraction will be with the adjacent data position (bond_dim) and the embedded image pixel (physical_bond).
        
        Args:
            - i (int): Index pixel's image batch position to contract.
        
        Returns:
            - right_side_i (JAX.numpy.Array : float, shape (bond_dim)): New right_side_data for the position .
        '''
    
        right_side_i=jnp.tensordot(self.Bj, self.right_side[i], axes=(2,0))
        right_side_i=jnp.tensordot(right_side_i, self.phi_j[i,:], axes=(1,0))

        return right_side_i / jnp.linalg.norm( right_side_i )
             
    
    def create_f_l_lr(self, left_side_i, right_side_i, joint_tensor, phi_j_i, phi_j_plusone_i):
        ''' 
        This function creates the prediction label f_l(x) of one image in the direction left right of the sweep. It puts  put together the previous computed parts (joint_tensor, left_side_i, right_side_i, phi_j_i, and phi_j_plusone_i). 
        This is done by contracting the joint_tensor with the left, right legs of data and the embedded pixels.
        
        Args:
            - left_side_i (JAX.numpy.Array : float, shape: (bond_dim)) : left_side of one image to contract with the leftest leg of the joint tensor.
            - right_side_i (JAX.numpy.Array : float, shape: (bond_dim)) : right_side of one image to contract with the second rightest leg of the joint tensor.
            - phi_j_i(JAX.numpy.Array : float, shape: (physical_bond (j) ) ) ):  Embedded image pixel on the position of the l leg tensor.
            - phi_j_plusone_i(JAX.numpy.Array : float, shape: (physical_bond (j) ) ) ):  Embedded image pixel on the position of the l+1 leg tensor.
            - joint_tensor (JAX.numpy.Array : float, shape : (bond_dim, physical_bond(j), physical_bond(j+1), bond_dim, l_leg)): 
                Single tensor being 5 dimensional.
        Returns
            - f_l (JAX.numpy.Array : float, shape : (l_leg)): Predicted label.
        ''' 

        f_l=jnp.tensordot(left_side_i, joint_tensor, axes=(0,0))
        f_l=jnp.tensordot(phi_j_i, f_l, axes=(0,0))
        f_l=jnp.tensordot(phi_j_plusone_i, f_l, axes=(0,0))
        f_l=jnp.tensordot(f_l, right_side_i, axes=(0,0))
        
        return f_l
    

    def create_f_l_rl(self, left_side_i, right_side_i, joint_tensor, phi_j_i, phi_j_minusone_i):
        ''' 
        This function creates the prediction label f_l(x) of one image in the direction right to left of the sweep. It puts  put together the previous computed parts (joint_tensor, left_side_i, right_side_i, phi_j_i, and phi_j_plusone_i). 
        This is done by contracting the joint_tensor with the left, right legs of data and the embedded pixels.
        
        Args:
            - left_side_i (JAX.numpy.Array : float, shape: (bond_dim)) : left_side of one image to contract with the leftest leg of the joint tensor.
            - right_side_i (JAX.numpy.Array : float, shape: (bond_dim)) : right_side of one image to contract with the second rightest leg of the joint tensor.
            - phi_j_i(JAX.numpy.Array : float, shape: (physical_bond (j) ) ) ):  Embedded image pixel on the position of the l leg tensor.
            - phi_j_minusone_i(JAX.numpy.Array : float, shape: (physical_bond (j) ) ) ):  Embedded image pixel on the position of the l-1 leg tensor.
            - joint_tensor (JAX.numpy.Array : float, shape : (bond_dim, physical_bond(j-1), physical_bond(j), bond_dim, l_leg)): 
                Single tensor being 5 dimensional.
        Returns
            - f_l (JAX.numpy.Array : float, shape : (l_leg)): Predicted label.
        ''' 
        
        f_l=jnp.tensordot(left_side_i, joint_tensor, axes=(0,0))
        f_l=jnp.tensordot(phi_j_minusone_i, f_l, axes=(0,0))
        f_l=jnp.tensordot(phi_j_i, f_l, axes=(0,0))
        f_l=jnp.tensordot(f_l, right_side_i, axes=(0,0))
        
        return f_l
    
    
    def max_magnitude_label(self, f_l):
        ''' 
        This function will perform one hot encoding on the predicted level. It will take the maximum argument of the predicted level as the 
        predicted one hot encoded level position.
        Args:
            - f_l (JAX.numpy.Array : float, shape : (l_leg)): Predicted label.
        
        Returns : 
            - f_l (JAX.numpy.Array : float, shape : (l_leg)): Predicted one hot position encoded label.
        '''

        
        max_f_l=jnp.max(f_l)
        probs=jnp.exp(f_l-max_f_l)/jnp.sum(jnp.exp(f_l-max_f_l))
        max_index = jnp.argmax(probs)
        predicted_level=jnp.zeros_like(probs).at[max_index].set(1.0)
        
        return predicted_level
    
    
    def compute_cost_and_gradient_B_l_lr(self):
        ''' 
        This function computes the gradient and the cost of the joint tensor side that we are optimising following the reference of the paper.

        Args:
            - Empty
        
        Returns:
            - cost (JAX.numpy.Array : float, shape : (1)) : Cost or loss of the 0.5*(predicted labels - true_labels)^2 for the images batch.
            - grad ( JAX.numpy.Array : float, shape : single array size (bond_dim^2*physical_bond^2*l_leg))
            - accuracy (float): Accuracy between 0 and 100.
        '''
        def compute_cost(f_l,label_index_training):
            diff=f_l-label_index_training
            return 0.5* jnp.dot(diff, diff), diff
        
        def compute_accuracy(f_l,label_index_training):
            predicted_level=jit_max_magnitude_label(f_l)
            return jnp.all(predicted_level == label_index_training) # Add to the accuracy
        
        def compute_grad_tensor(left_side_i,phi_j_i,phi_j_adjacent_i,right_side_i, diff):
            # Compute the gradient of the joint tensor following the procedure of the paper
            return jnp.kron( jnp.kron( jnp.kron( jnp.kron(left_side_i, phi_j_i), phi_j_adjacent_i),right_side_i), -diff)

        cost=0
        grad=0
        accuracy=0

        # jit funtions that are being called in 
        jit_create_f_l_lr=jax.jit(self.create_f_l_lr)
        jit_max_magnitude_label=jax.jit(self.max_magnitude_label)
        jit_compute_cost=jax.jit(compute_cost)
        jit_compute_accuracy=jax.jit(compute_accuracy)
        jit_compute_grad_tensor=jax.jit(compute_grad_tensor)

        
        joint_tensor=self.contract_with_neighbour_tensor_lr() # already jit
        num_training=(self.y).shape[0]

        for index_training in range(num_training):
            
            label_index_training=self.y[index_training,:]
            f_l=jit_create_f_l_lr(joint_tensor=joint_tensor, left_side_i=self.left_side[index_training], right_side_i=self.right_side[index_training], phi_j_i=self.phi_j[index_training,:], phi_j_plusone_i=self.phi_jadjacent[index_training,:])
            cost_i,diff= jit_compute_cost(f_l=f_l, label_index_training=label_index_training)
            cost+=cost_i
            accuracy+= jit_compute_accuracy(f_l, label_index_training=label_index_training)
            grad+=jit_compute_grad_tensor(left_side_i=self.left_side[index_training], phi_j_adjacent_i=self.phi_jadjacent[index_training,:], phi_j_i=self.phi_j[index_training,:], right_side_i=self.right_side[index_training], diff=diff)
            
        
        return cost, grad,( (accuracy / num_training)*100)
    
    
    def compute_cost_and_gradient_B_l_rl(self):
        ''' 
        This function computes the gradient and the cost of the joint tensor side that we are optimising following the reference of the paper.

        Args:
        
        Returns:
            - cost (JAX.numpy.Array : float, shape : (1)) : Cost or loss of the 0.5*(predicted labels - true_labels)^2 for the images batch.
            - grad ( JAX.numpy.Array : float, shape : single array size (bond_dim^2*physical_bond^2*l_leg))
            - accuracy (float): Accuracy between 0 and 100.
        '''
        def compute_cost(f_l,label_index_training):
            diff=f_l-label_index_training
            return 0.5* jnp.dot(diff, diff), diff
        
        def compute_accuracy(f_l,label_index_training):
            predicted_level=jit_max_magnitude_label(f_l)
            return jnp.all(predicted_level == label_index_training) # Add to the accuracy
        
        def compute_grad_tensor(left_side_i,phi_j_adjacent_i,phi_j_i,right_side_i, diff):
            # Compute the gradient of the joint tensor following the procedure of the paper
            return jnp.kron( jnp.kron( jnp.kron( jnp.kron(left_side_i, phi_j_adjacent_i), phi_j_i),right_side_i), -diff)

        
        cost=0
        grad=0
        accuracy=0

        # jit funtions that are being called in
        jit_create_f_l_rl=jax.jit(self.create_f_l_rl)
        jit_max_magnitude_label=jax.jit(self.max_magnitude_label)
        jit_compute_cost=jax.jit(compute_cost)
        jit_compute_accuracy=jax.jit(compute_accuracy)
        jit_compute_grad_tensor=jax.jit(compute_grad_tensor)

        joint_tensor=self.contract_with_neighbour_tensor_rl() # already jit
        num_training=(self.y).shape[0]

        for index_training in range(num_training):
            
            label_index_training=self.y[index_training,:]
            f_l=jit_create_f_l_rl(joint_tensor=joint_tensor, left_side_i=self.left_side[index_training], right_side_i=self.right_side[index_training], phi_j_i=self.phi_j[index_training,:], phi_j_minusone_i=self.phi_jadjacent[index_training,:])
            cost_i,diff= jit_compute_cost(f_l=f_l, label_index_training=label_index_training)
            cost+=cost_i
            accuracy+= jit_compute_accuracy(f_l, label_index_training=label_index_training)
            
            # Compute the gradient of the joint tensor following the procedure of the paper
            
            grad+=jit_compute_grad_tensor(left_side_i=self.left_side[index_training], phi_j_adjacent_i=self.phi_jadjacent[index_training,:], phi_j_i=self.phi_j[index_training,:], right_side_i=self.right_side[index_training], diff=diff)
        
        return cost, grad,  ( (accuracy / num_training)*100)
    
    def gradient_descent_step(self, B_l, gradient_B_l):
        '''
        This function will apply the gradient descent step of the joint tensor that we are optimizing.

        Args:
            - B_l (JAX.numpy.Array : float, shape : (bond_dim, physical_bond(j or j-1), physical_bond(j+1 or j ), bond_dim, l_leg)): 
                Single tensor being 5 dimensional.
            - gradient_B_l ( JAX.numpy.Array : float, shape : single array size (bond_dim^2*physical_bond^2*l_leg)): gradient of the B_l tensor
        
        Returns:
            - (JAX.numpy.Array : float, shape : (bond_dim, physical_bond(j or j-1), physical_bond(j+1 or j ), bond_dim, l_leg)): 
                Single tensor being 5 dimensional which has ben applied the gradient descent step.
        '''
        
        return (B_l + self.learning_rate*jnp.reshape(gradient_B_l,B_l.shape)) # The gradient has to be reshaped as the same as in the B_l tensor

    
    def separate_B_l_lr(self, updated_B_l):
        ''' 
        This function will perform the SVD on the optimized tensor, ( so previously the joint tensor has been applied a gradient descent step).
        In this updated tensor will be performed the SVD decomposition, controlling the bond dimension size. During the process the eigenvalues
        will be normalised to better stability.
        
        Args:
            - updated_B_l (JAX.numpy.Array : float, shape : (bond_dim, physical_bond(j), physical_bond(j+1), bond_dim, l_leg)): 
                Single tensor being 5 dimensional which has ben applied the gradient descent step.
        
        Returns: 
            - A_j (JAX.numpy.Array : float, shape : (bond_dim, physical_bond(j), bond_dim)) : Left isometry updated tensor on the position j.
            - A_j_plusone (JAX.numpy.Array : float, shape : (bond_dim, physical_bond(j+1), bond_dim, l_leg)):  Tensor on the j +1 side with now the l_leg.  
        ''' 
        
        original_shape=updated_B_l.shape
        dim_i=original_shape[0]
        dim_j=original_shape[1]
        dim_k=original_shape[2]
        dim_l=original_shape[3]
        dim_m=original_shape[4]

        updated_B_l=jnp.reshape(updated_B_l,[dim_i*dim_j,-1])
        
        # Apply SVD
        Atemp,Dtemp,Vhtemp=jnp.linalg.svd(updated_B_l)
            
        # Normalise the eigenvalues and truncate the matrices if bigger than bond_dim
        if Atemp.shape[1]>self.bond_dim:
            Atemp=(Atemp)[:,:self.bond_dim]
            Atemp=jnp.round(Atemp, 8)
            Vhtemp=jnp.round(Vhtemp[:self.bond_dim,:], 8)
            Dtemp=jnp.round ( jnp.diag(Dtemp[:self.bond_dim]), 8)
            Dtemp=Dtemp / jnp.trace(Dtemp.T @ Dtemp) 
        else:
            Atemp=jnp.round( (Atemp.T)[:,:Dtemp.shape[0]], 8)
            Dtemp=jnp.round( jnp.diag(Dtemp), 8)
            Vhtemp=jnp.round( Vhtemp[:Dtemp.shape[0],:], 8)
            Dtemp=Dtemp / jnp.trace(Dtemp.T @ Dtemp)
        
        A_j=jnp.reshape(Atemp,[dim_i,dim_j,-1])
        A_j_plusone=jnp.reshape(Dtemp @ Vhtemp,[-1,dim_k,dim_l,dim_m])

        return A_j, A_j_plusone

    
    def separate_B_l_rl(self, updated_B_l):
        ''' 
        This function will perform the SVD on the optimized tensor, ( so previously the joint tensor has been applied a gradient descent step).
        In this updated tensor will be performed the SVD decomposition, controlling the bond dimension size. During the process the eigenvalues
        will be normalised to better stability.
        
        Args:
            - updated_B_l (JAX.numpy.Array : float, shape : (bond_dim, physical_bond(j-1), physical_bond(j), bond_dim, l_leg)): 
                Single tensor being 5 dimensional which has ben applied the gradient descent step.
        
        Returns: 
            - A_j_minusone (JAX.numpy.Array : float, shape : (bond_dim, physical_bond(j-1), bond_dim, l_leg)):  Tensor on the j -1 side with now the l_leg.
            - A_j (JAX.numpy.Array : float, shape : (bond_dim, physical_bond(j), bond_dim)) : Right isometry updated tensor on the position j.
        '''

        original_shape=updated_B_l.shape
        dim_i=original_shape[0]
        dim_j=original_shape[1]
        dim_k=original_shape[2]
        dim_l=original_shape[3]
        dim_m=original_shape[4]

        updated_B_l=jnp.reshape(jnp.permute_dims(updated_B_l,[0,1,4,2,3]),[dim_i*dim_j*dim_m,-1])

        # Apply SVD
        Atemp,Dtemp,Vhtemp=jnp.linalg.svd(updated_B_l)
        
        # Normalise the eigenvalues and truncate the matrices if bigger than bond_dim
        if Vhtemp.shape[0]>self.bond_dim:
            Atemp=jnp.round( Atemp[:,:self.bond_dim], 8)
            Vhtemp=(Vhtemp)[:self.bond_dim,:]
            Vhtemp=jnp.round(Vhtemp, 8)
            Dtemp=jnp.round( jnp.diag(Dtemp[:self.bond_dim]), 8)
            Dtemp=Dtemp / jnp.trace(Dtemp.T @ Dtemp)
        
        else:
            Atemp=jnp.round( Atemp[:,:Dtemp.shape[0]], 8)
            Dtemp=jnp.round( jnp.diag(Dtemp), 8)
            Dtemp=Dtemp / jnp.trace(Dtemp.T @ Dtemp)
            Vhtemp=jnp.round( Vhtemp[:Dtemp.shape[0],:], 8)
        
        A_j=jnp.reshape(Vhtemp,[-1,dim_k,dim_l])
        A_j_minusone=jnp.reshape(Atemp @ Dtemp ,[dim_i,dim_j,dim_m,-1])
        A_j_minusone=jnp.permute_dims(A_j_minusone,[0,1,3,2])

        return  A_j_minusone, A_j    

    
    def update_mps(self,left_or_right):
        ''' 
        This function will perform two side position update of the MPS given the direction passed.
        
        Args:
            - left_or_right (bool): The direction that we are sweeping.
        
        Returns:
            - A_j (JAX.Numpy.Array: float, shape : (bond_dim, physical_leg, bond_dim)): Updated tensor will have 3 legs and it will fulfill the left or right isometry.
            - B_jadjacent (JAX.Numpy.Array: float, shape :(bond_dim, physical_leg, bond_dim, l_leg)): The adjacent tensor updated which has four legs.
            - cost (JAX.numpy.Array : float, shape : (1)) : Cost or loss of the 0.5*(predicted labels - true_labels)^2 for the images batch.
            - accuracy (float): Accuracy between 0 and 1.
        '''
        
        new_side=[]

        if left_or_right=='left':
            B_j_jplusone=self.contract_with_neighbour_tensor_lr() # Compute the joint tensor on the j, j+1 sides
            cost, grad, accuracy = self.compute_cost_and_gradient_B_l_lr() # Compute the gradient, cost and accuracy being centered in the previous joint tensor
            new_B_j_jplusone=self.gradient_descent_step(B_l=B_j_jplusone, gradient_B_l=grad) # Compute update joint tensor
            A_j,A_j_adjacent=self.separate_B_l_lr(updated_B_l=new_B_j_jplusone) # Returns new j, j+1 tensors
            self.Bj=A_j
            self.B_jadjacent=A_j_adjacent
            
            for index_training in range((self.y).shape[0]):
                new_side.append(self.update_single_data_left(i=index_training)) # Computes update of the data with the previous optimized MPS in the sides j,j+1
            
        
        if left_or_right=='right':
            B_jminusone_j=self.contract_with_neighbour_tensor_rl() # Compute the joint tensor on the j-1, j sides
            cost, grad, accuracy = self.compute_cost_and_gradient_B_l_rl() # Compute the gradient, cost and accuracy being centered in the previous joint tensor
            new_B_jminusone_j=self.gradient_descent_step(B_l=B_jminusone_j,gradient_B_l=grad) # Compute update joint tensor
            A_j_adjacent,A_j=self.separate_B_l_rl(updated_B_l=new_B_jminusone_j) # Returns new j-1, j tensors
            self.Bj=A_j
            self.B_jadjacent=A_j_adjacent
            
            for index_training in range((self.y).shape[0]):
                new_side.append(self.update_single_data_right(i=index_training))# Computes update of the data with the previous optimized MPS in the sides j-1,j.

        
        return A_j, A_j_adjacent, new_side , cost, accuracy
        
        
    