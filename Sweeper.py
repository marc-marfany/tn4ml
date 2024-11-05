
import numpy as np
import jax
import jax.numpy as jnp

class sweeper:
    def __init__(self, mps, phi,learning_rate,bond_dim, j,y, data) -> None:
        """
        Initialize the sweeping algorithm. Save parameters

        Args:
            - mps (list[JAX.Numpy.Array: float], 
                shape: List[(physical_leg,bond_dim), (bond_dim, physical_leg,bond_dim), ..., (bond_dim, physical_leg, bond_dim, l_leg), 
                ..., (bond_dim, physical_leg, bond_dim),.., (bond_dim, physical_leg) ]): This variable is a list full of tensors of the MPS.
            -  phi (JAX.numpy.Array : float, shape: (batch_size, image_size, physical_bond) ): Embedded image as the paper.
            - y (JAX.Numpy.Array: float, shape: (batch-size, #_labels_size): Labels training data.
            - learning_rate (float): Learning rate.
            - bond_dim (int): Bond dimension between tensor in the MPS.
            - j (int): Position j in the MPS where the l leg is situated.
            - data (List[List[JAX.Numpy.Array: float]],  shape: [#_training,length_mps] (size_contracted_legs)) : It will contain the left or right legs, one dimensional arrays, 
                to contract with the joint tensor to optimize.
            
        """
        
        self.mps=mps 
        self.phi=phi
        self.learning_rate=learning_rate
        self.bond_dim=bond_dim
        self.j=j
        self.y=y
        self.data=data
    

    def contract_with_neighbour_tensor(self,left_or_right):
        """
        This function will contract two adjacent tensor, (j,j+1) or (j-1,j) into a single tensor. It returns a single tensor 
        with two pysical bonds, the l leg in the last position, and the left and right bond (those depending on the position)
        
        Args:
            - left_or_right (bool): The direction that we are sweeping. If left we are sweeping from left to right (contraction of j, j+1 tensors). If right
                we are sweeping from right to left (contraction of j-1, j tensors).
        Returns:
            - joint_tensor (JAX.numpy.Array : float, shape : (bond_dim, physical_bond(j or j-1), physical_bond(j+1 or j ), bond_dim, l_leg)): 
                Single tensor being 4/5 dimensional.
        """
        
        j=self.j
        
        # If direction is left, contract adjacents j, j+1 tensors. Careful on the index j position
        if left_or_right=='left':
            if j==0:
                Bj=self.mps[j]
                Bj_plus_one=self.mps[j+1]
                joint_tensor=jnp.einsum('ijl,jmn->imnl', Bj,Bj_plus_one)
            elif j==(len(self.mps)-2):
                Bj=self.mps[j]
                Bj_plus_one=self.mps[j+1]
                joint_tensor=jnp.einsum('ijkl,km->ijml', Bj,Bj_plus_one)
            else:
                Bj=self.mps[j]
                Bj_plus_one=self.mps[j+1]
                joint_tensor=jnp.einsum('ijkl,kmn->ijmnl', Bj,Bj_plus_one)
        
        # If direction is right, contract adjacents j, j+1 tensors. Careful on the index j position
        if left_or_right=='right':
            if j==len(self.mps)-1:
                Bj=self.mps[j]
                Bj_minus_one=self.mps[j-1]
                joint_tensor =jnp.einsum('ijk,klm->ijlm',Bj_minus_one,Bj)
            elif j==1:
                Bj=self.mps[j]
                Bj_minus_one=self.mps[j-1]
                joint_tensor =jnp.einsum('jk,klmn->jlmn',Bj_minus_one,Bj)
            else:
                Bj=self.mps[j]
                Bj_minus_one=self.mps[j-1]
                joint_tensor =jnp.einsum('ijk,klmn->ijlmn',Bj_minus_one,Bj)
        
        return joint_tensor
     
    
    def contract_next_right_or_left(self,i,left_or_right):
        ''' 
        Contract the right hand side (if left) or the left hand side (if right) of the specific index of the MPS AFTER being updated. This
        contraction will be with the adjacent data position (bond_dim) and the embedded image pixel (physical_bond).
        
        Args:
            - left_or_right (bool): The direction that we are sweeping.
            - i (int): Index image batch position to contract.
        Returns:
            - temporary_data : Left or right new leg to contract with the joint tensor to optimize.
        '''
    
        j=self.j

        if left_or_right=='left':
            if j== 0 :
                temporary_data= jnp.einsum('ij, i-> j',self.mps[j],self.phi[i,j,:])
                return  temporary_data/ jnp.linalg.norm( temporary_data)
            else:
                data_i_j_minusone=self.data[i][j-1] # index i is index training  and j index mps
                temporary_data=jnp.einsum('i, ijk->jk',data_i_j_minusone, self.mps[j])
                temporary_data=jnp.einsum('j, jk -> k',self.phi[i,j,:],temporary_data)
                return temporary_data / jnp.linalg.norm( temporary_data )
            
        if left_or_right=='right':
            if j== len(self.mps) -1 :
                temporary_data=jnp.einsum('ij, j-> i',self.mps[j],self.phi[i,j,:])
                return temporary_data/ jnp.linalg.norm( temporary_data )   
            else:
                data_i_j_plusone=self.data[i][j+1] # index i is index training  and j index mps
                temporary_data=jnp.einsum('ijk,k->ij',self.mps[j],data_i_j_plusone) 
                temporary_data =jnp.einsum('ij,j->i',temporary_data, self.phi[i,j,:])
                return   temporary_data / jnp.linalg.norm( temporary_data)
        
    
    def update_data(self,left_or_right):
        ''' 
        This function will update the data List[List[]] with the previous tensor optimized of the MPS for all images.

        Args:
            - left_or_right (bool): The direction that we are sweeping.
        Returns:
            Empty.
        '''

        # Left to right 
        if left_or_right=='left':
            for index_training in range(self.phi.shape[0]):
                self.data[index_training][self.j]=self.contract_next_right_or_left(i=index_training,left_or_right=left_or_right)
        
        # Right to left
        if left_or_right=='right':
            for index_training in range(self.phi.shape[0]):
                self.data[index_training][self.j]=self.contract_next_right_or_left(i=index_training, left_or_right=left_or_right)

    
    def create_f_l(self,i,left_or_right, joint_tensor):
        ''' 
        This function creates the prediction label f_l(x). It puts  put together the previous computed parts (data and joint_tensor). 
        This is done by contracting the joint_tensor with the left, right legs of data and the embedded pixels.
        
        Args:
            - left_or_right (bool): The direction that we are sweeping.  
            - i (int): Index image batch position to contract.
            - joint_tensor (JAX.numpy.Array : float, shape : (bond_dim, physical_bond(j or j-1), physical_bond(j+1 or j ), bond_dim, l_leg)): 
                Single tensor being 4/5 dimensional.
        Returns
            - f_l (JAX.numpy.Array : float, shape : (l_leg)): Predicted label.
        ''' 

        j=self.j

        # If left perform the contraction with all elements to compute the predicted level
        if left_or_right=='left':
            # If we are in the beggining of the mps, we have no left leg to contract with
            if j==0:
                Bj_j_plusone=joint_tensor
                right_side=self.data[i][j+2]
                phi_j=self.phi[i,j,:]
                phi_j_plusone=self.phi[i,j+1,:]
                f_l=jnp.einsum('k,klmn->lmn',phi_j, Bj_j_plusone)
                f_l=jnp.einsum('l,lmn->mn',phi_j_plusone, f_l)
                f_l=jnp.einsum('mn,m->n',f_l, right_side)
            # If we are in the end of the mps, we have no right leg to contract with . 
            elif j==len(self.mps)-2:
                Bj_j_plusone=joint_tensor
                left_side=self.data[i][j-1]
                phi_j=self.phi[i,j,:]
                phi_j_plusone=self.phi[i,j+1,:]
                f_l=jnp.einsum('j,jklm->klm',left_side, Bj_j_plusone)
                f_l=jnp.einsum('k,klm->lm',phi_j , f_l)
                f_l=jnp.einsum('l,lm->m',phi_j_plusone , f_l)
            else:
                Bj_j_plusone=joint_tensor
                left_side,right_side=self.data[i][j-1],self.data[i][j+2]
                phi_j=self.phi[i,j,:]
                phi_j_plusone=self.phi[i,j+1,:]
                f_l=jnp.einsum('j,jklmn->klmn',left_side, Bj_j_plusone)
                f_l=jnp.einsum('k,klmn->lmn',phi_j , f_l)
                f_l=jnp.einsum('l,lmn->mn',phi_j_plusone , f_l)
                f_l=jnp.einsum('mn,m->n',f_l , right_side)
        
        # Same with right side mimicking the conditions given the index position in the MPS being contracted  
        if left_or_right== 'right':
            if j==len(self.mps)-1:
                Bjminusone_j=joint_tensor
                left_side=self.data[i][j-2]
                phi_j=self.phi[i,j,:]
                phi_j_minusone=self.phi[i,j-1,:]
                f_l=jnp.einsum('j,jklm->klm',left_side, Bjminusone_j)
                f_l=jnp.einsum('k,klm->lm',phi_j_minusone , f_l)
                f_l=jnp.einsum('l,lm->m',phi_j , f_l)
            elif j==1:
                Bjminusone_j=joint_tensor
                right_side=self.data[i][j+1]
                phi_j=self.phi[i,j,:]
                phi_j_minusone=self.phi[i,j-1,:]
                f_l=jnp.einsum('k,klmn->lmn',phi_j_minusone , Bjminusone_j)
                f_l=jnp.einsum('l,lmn->mn',phi_j , f_l)
                f_l=jnp.einsum('mn,m->n',f_l , right_side)  
            else:
                Bjminusone_j=joint_tensor
                left_side,right_side=self.data[i][j-2],self.data[i][j+1]
                phi_j=self.phi[i,j,:]
                phi_j_minusone=self.phi[i,j-1,:]
                f_l=jnp.einsum('j,jklmn->klmn',left_side, Bjminusone_j)
                f_l=jnp.einsum('k,klmn->lmn',phi_j_minusone , f_l)
                f_l=jnp.einsum('l,lmn->mn',phi_j , f_l)
                f_l=jnp.einsum('mn,m->n',f_l , right_side)     
        
        return f_l
    
    
    def max_magnitude_label(self,f_l):
        ''' 
        This function will perform one hot encoding on the predicted level. It will take the maximum argument of the predicted level as the 
        predicted one hot encoded level position.
        Args:
            - f_l (JAX.numpy.Array : float, shape : (l_leg)): Predicted label.
        Return : 
            - f_l (JAX.numpy.Array : float, shape : (l_leg)): Predicted one hot position encoded label.
        '''

        predicted_level=np.zeros(f_l.shape)
        predicted_level[np.argmax(f_l)]=1
        return jnp.array(predicted_level,dtype=float)
    
    
    def compute_cost_and_gradient_B_l(self,left_or_right):
        ''' 
        This function computes the gradient and the cost of the joint tensor side that we are optimising following the reference of the paper.

        Args:
            - left_or_right (bool): The direction that we are sweeping.
        Returns:
            - cost (JAX.numpy.Array : float, shape : (1)) : Cost or loss of the 0.5*(predicted labels - true_labels)^2 for the images batch.
            - grad ( JAX.numpy.Array : float, shape : single array size (bond_dim^2*physical_bond^2*l_leg))
            - accuracy (float): Accuracy between 0 and 1.
        '''
        
        j=self.j
        
        if left_or_right=='left':
            cost=0
            grad=0
            accuracy=0
            joint_tensor=self.contract_with_neighbour_tensor(left_or_right=left_or_right)
            
            for index_training in range((self.y).shape[0]):
                # Compute the cost and accuracy
                fl_index_training=self.create_f_l(i=index_training,left_or_right=left_or_right,joint_tensor=joint_tensor) # created the predicted label by using the previous function
                label_index_training=self.y[index_training,:] # true label
                diff=jnp.round(fl_index_training-label_index_training,6) 
                cost+= 0.5* jnp.dot(diff, diff) # Add to the cost
                accuracy+= int(all(self.max_magnitude_label(fl_index_training) == label_index_training)) # Add to the accuracy
                # Compute the gradient of the joint tensor following the procedure of the paper
                if j==0:
                    right_side=self.data[index_training][j+2]
                    phi_j=self.phi[index_training,j,:]
                    phi_j_plusone=self.phi[index_training,j+1,:]
                    #grad_index_training=jnp.kron(phi_j ,jnp.kron(phi_j_plusone,jnp.kron( right_side,diff) ))
                    grad_index_training=jnp.kron( jnp.kron( jnp.kron(phi_j, phi_j_plusone),right_side), -diff) 
                elif j== len(self.mps)-2:
                    left_side=self.data[index_training][j-1]
                    phi_j=self.phi[index_training,j,:]
                    phi_j_plusone=self.phi[index_training,j+1,:]
                    #grad_index_training=jnp.kron(left_side, jnp.kron(phi_j ,jnp.kron(phi_j_plusone,diff) ))
                    grad_index_training=jnp.kron( jnp.kron( jnp.kron(left_side, phi_j), phi_j_plusone), -diff) 
                else:
                    left_side,right_side=self.data[index_training][j-1],self.data[index_training][j+2]
                    phi_j=self.phi[index_training,j,:]
                    phi_j_plusone=self.phi[index_training,j+1,:]
                    # grad_index_training=jnp.kron(left_side, jnp.kron(phi_j ,jnp.kron(phi_j_plusone , jnp.kron(right_side ,diff) )  ) )
                    grad_index_training=jnp.kron( jnp.kron( jnp.kron( jnp.kron(left_side, phi_j), phi_j_plusone),right_side), -diff)  
                
                grad+=grad_index_training

        if left_or_right=='right':
            cost=0
            grad=0
            accuracy=0
            joint_tensor=self.contract_with_neighbour_tensor(left_or_right=left_or_right)

            for index_training in range((self.y).shape[0]):
                fl_index_training=self.create_f_l(i=index_training,left_or_right=left_or_right,joint_tensor=joint_tensor) # created the predicted label by using the previous function
                label_index_training=self.y[index_training,:] # true label
                diff=jnp.round(fl_index_training-label_index_training,6)
                cost+= 0.5* jnp.dot(diff, diff) # Add to the cost
                accuracy+= int(all((self.max_magnitude_label(fl_index_training) == label_index_training))) # Add to the accuracy
                # Compute the gradient of the joint tensor following the procedure of the paper
                if j==1:
                    right_side=self.data[index_training][j+1]
                    phi_j=self.phi[index_training,j,:]
                    phi_j_minusone=self.phi[index_training,j-1,:]
                    #grad_index_training=jnp.kron(phi_j_minusone ,jnp.kron(phi_j,jnp.kron( right_side,-diff) ))
                    grad_index_training= jnp.kron( jnp.kron( jnp.kron(phi_j_minusone, phi_j),right_side), -diff) 
                elif j== len(self.mps)-1:
                    left_side=self.data[index_training][j-2]
                    phi_j=self.phi[index_training,j,:]
                    phi_j_minusone=self.phi[index_training,j-1,:]
                    #grad_index_training=jnp.kron(left_side, jnp.kron(phi_j_minusone ,jnp.kron(phi_j,-diff) ))
                    grad_index_training= jnp.kron( jnp.kron( jnp.kron(left_side, phi_j_minusone), phi_j), -diff)
                else:
                    left_side,right_side=self.data[index_training][j-2],self.data[index_training][j+1]
                    phi_j=self.phi[index_training,j,:]
                    phi_j_minusone=self.phi[index_training,j-1,:]
                    #grad_index_training=jnp.kron(left_side, jnp.kron(phi_j_minusone ,jnp.kron(phi_j , jnp.kron(right_side ,-diff) )  ) )
                    grad_index_training=jnp.kron( jnp.kron( jnp.kron( jnp.kron(left_side, phi_j_minusone), phi_j),right_side), -diff) 
                
                grad+=grad_index_training
        
        return cost,grad, (accuracy / (self.y).shape[0])
    
    
    def gradient_descent_step(self, B_l, gradient_B_l):
        '''
        This function will apply the gradient descent step of the joint tensor that we are optimizing.

        Args:
            - B_l (JAX.numpy.Array : float, shape : (bond_dim, physical_bond(j or j-1), physical_bond(j+1 or j ), bond_dim, l_leg)): 
                Single tensor being 4/5 dimensional.
            - gradient_B_l ( JAX.numpy.Array : float, shape : single array size (bond_dim^2*physical_bond^2*l_leg)): gradient of the B_l tensor
        Returns:
            - (JAX.numpy.Array : float, shape : (bond_dim, physical_bond(j or j-1), physical_bond(j+1 or j ), bond_dim, l_leg)): 
                Single tensor being 4/5 dimensional which has ben applied the gradient descent step.
        '''
        
        return (B_l + self.learning_rate*jnp.reshape(gradient_B_l,B_l.shape)) # The gradient has to be reshaped as the same as in the B_l tensor

    
    def separate_B_l(self,left_or_right,updated_B_l):
        ''' 
        This function will perform the SVD on the optimized tensor, ( so previously the joint tensor has been applied a gradient descent step).
        In this updated tensor will be performed the SVD decomposition, controlling the bond dimension size. During the process the eigenvalues
        will be normalised to better stability.
        Args:
            - left_or_right (bool): The direction that we are sweeping.
            - updated_B_l (JAX.numpy.Array : float, shape : (bond_dim, physical_bond(j or j-1), physical_bond(j+1 or j ), bond_dim, l_leg)): 
                Single tensor being 4/5 dimensional which has ben applied the gradient descent step.
        Returns: 
            - if left
                - A_j (JAX.numpy.Array : float, shape : (bond_dim, physical_bond(j), bond_dim)) : Left isometry updated tensor on the position j.
                    If we are in the beggining of the MPS the first bond_dim disappers.
                - A_j_plusone (JAX.numpy.Array : float, shape : (bond_dim, physical_bond(j+1), bond_dim,l_leg)):  Tensor on the j +1 side with now the l_leg.
            - if right
                - A_j_minusone (JAX.numpy.Array : float, shape : (bond_dim, physical_bond(j-1), bond_dim, l_leg)):  Tensor on the j -1 side with now the l_leg.
                - A_j (JAX.numpy.Array : float, shape : (bond_dim, physical_bond(j), bond_dim)) : Right isometry updated tensor on the position j.
                    If we are in the end of the MPS the last bond_dim disappers
        ''' 

        # Case left side
        if left_or_right=='left':
            # Keep track of the shape of the updated B_l to reorder the and reshape the matrices after SVD
            original_shape=updated_B_l.shape
            if len(original_shape) == 5:
                dim_i=original_shape[0]
                dim_j=original_shape[1]
                dim_k=original_shape[2]
                dim_l=original_shape[3]
                dim_m=original_shape[4]
            elif len( original_shape) == 4:
                dim_i=original_shape[0]
                dim_j=original_shape[1]
                dim_k=original_shape[2]
                dim_l=original_shape[3]
            else: 
                print('Something happened dimension updated B, does not make sense')
            # Reshape tensor into a matrix to apply svd according as in the paper
            if len(original_shape)==4:
                if self.j==0:
                    updated_B_l=jnp.reshape(updated_B_l,[dim_i,-1])
                else:
                    updated_B_l=jnp.reshape(updated_B_l,[dim_i*dim_j,-1])
            
            elif len(original_shape)==5:
                updated_B_l=jnp.reshape(updated_B_l,[dim_i*dim_j,-1])
            
            # Apply SVD
            Atemp,Dtemp,Vhtemp=jnp.linalg.svd(updated_B_l)
            
            # Normalise the eigenvalues and truncate the matrices if bigger than bond_dim
            if Atemp.shape[1]>self.bond_dim:
                Atemp=(Atemp)[:,:self.bond_dim]
                Atemp=jnp.round(Atemp, 6)
                Vhtemp=jnp.round(Vhtemp[:self.bond_dim,:], 6)
                Dtemp=jnp.round ( jnp.diag(Dtemp[:self.bond_dim]), 6)
                Dtemp=Dtemp / jnp.trace(Dtemp.T @ Dtemp) 
            else:
                Atemp=jnp.round( (Atemp.T)[:,:Dtemp.shape[0]], 6)
                Dtemp=jnp.round( jnp.diag(Dtemp), 6)
                Vhtemp=jnp.round( Vhtemp[:Dtemp.shape[0],:], 6)
                Dtemp=Dtemp / jnp.trace(Dtemp.T @ Dtemp)

            # Reshape the SVD matrices to be the new tensors in the position j and j+1
            if len( original_shape) == 4:
                if self.j==0:
                    A_j=jnp.reshape(Atemp,[dim_i,-1])
                    A_j_plusone=jnp.reshape(Dtemp @ Vhtemp,[-1,dim_j,dim_k,dim_l])
                else:
                    A_j=jnp.reshape(Atemp,[dim_i,dim_j,-1])
                    A_j_plusone=jnp.reshape(Dtemp @ Vhtemp,[-1,dim_k,dim_l])
                
            elif len(original_shape) == 5:
                A_j=jnp.reshape(Atemp,[dim_i,dim_j,-1])
                A_j_plusone=jnp.reshape(Dtemp @ Vhtemp,[-1,dim_k,dim_l,dim_m])
            else : 
                print('Error while doing the svd, wrong dimension fit')
            
            return A_j, A_j_plusone
        
        if left_or_right=='right':
            # Keep track of the shape of the updated B_l to reorder the and reshape the matrices after SVD
            original_shape=updated_B_l.shape
            if len(original_shape) == 5:
                dim_i=original_shape[0]
                dim_j=original_shape[1]
                dim_k=original_shape[2]
                dim_l=original_shape[3]
                dim_m=original_shape[4]
            elif len( original_shape) == 4:
                dim_i=original_shape[0]
                dim_j=original_shape[1]
                dim_k=original_shape[2]
                dim_l=original_shape[3]
            else: 
                print('Something happened dimension updated B, does not make sense')

            # Reshape tensor into a matrix to apply svd according as in the paper
            if len( original_shape) == 5:
                updated_B_l=jnp.reshape(jnp.permute_dims(updated_B_l,[0,1,4,2,3]),[dim_i*dim_j*dim_m,-1])
            elif len( original_shape) == 4:
                if self.j==len(self.mps)-1:
                    updated_B_l=jnp.reshape(jnp.permute_dims(updated_B_l,[0,1,3,2]),[dim_i*dim_j*dim_l,-1])
                else :
                    updated_B_l=jnp.reshape(jnp.permute_dims(updated_B_l,[0,3,1,2]),[dim_i*dim_l,-1])
            else: 
                print('Something happened dimension updated B, does not make sense')

            # Apply SVD
            Atemp,Dtemp,Vhtemp=jnp.linalg.svd(updated_B_l)
           
            # Normalise the eigenvalues and truncate the matrices if bigger than bond_dim
            if Vhtemp.shape[0]>self.bond_dim:
                Atemp=jnp.round( Atemp[:,:self.bond_dim], 6)
                Vhtemp=(Vhtemp)[:self.bond_dim,:]
                Vhtemp=jnp.round(Vhtemp, 6)
                Dtemp=jnp.round( jnp.diag(Dtemp[:self.bond_dim]), 6)
                Dtemp=Dtemp / jnp.trace(Dtemp.T @ Dtemp)
            
            else:
                Atemp=jnp.round( Atemp[:,:Dtemp.shape[0]], 6)
                Dtemp=jnp.round( jnp.diag(Dtemp), 6)
                Dtemp=Dtemp / jnp.trace(Dtemp.T @ Dtemp)
                Vhtemp=jnp.round( Vhtemp[:Dtemp.shape[0],:], 6)
            
            # Reshape and permute the SVD matrices to be the new tensors in the position j-1 and j
            if len( original_shape) == 4:
                if self.j==len(self.mps)-1:
                    A_j=jnp.reshape(Vhtemp,[-1, dim_k])
                    A_j_minusone=jnp.reshape(Atemp @ Dtemp ,[dim_i,dim_j,dim_l, -1])
                    A_j_minusone=jnp.permute_dims(A_j_minusone,[0,1,3,2])
                else:
                    A_j=jnp.reshape(Vhtemp,[-1, dim_j, dim_k])
                    A_j_minusone=jnp.reshape(Atemp @ Dtemp ,[dim_i,dim_l,-1])
                    A_j_minusone=jnp.permute_dims(A_j_minusone,[0,2,1])
            elif len( original_shape) == 5:
                A_j=jnp.reshape(Vhtemp,[-1,dim_k,dim_l])
                A_j_minusone=jnp.reshape(Atemp @ Dtemp ,[dim_i,dim_j,dim_m,-1])
                A_j_minusone=jnp.permute_dims(A_j_minusone,[0,1,3,2])
            else : 
                print('Error while doing the svd, wrong dimension fit')    
            
            return  A_j_minusone, A_j
        
    
    def update_mps(self,left_or_right):
        ''' 
        This function will perform two side position update of the MPS given the direction passed.
        
        Args:
            - left_or_right (bool): The direction that we are sweeping.
        Returns:
            - mps (list[JAX.Numpy.Array: float], 
                shape: List[(physical_leg,bond_dim), (bond_dim, physical_leg,bond_dim), ..., (bond_dim, physical_leg, bond_dim, l_leg), 
                ..., (bond_dim, physical_leg, bond_dim),.., (bond_dim, physical_leg) ]): This variable is a list full of tensors of the MPS.
            - data (List[List[JAX.Numpy.Array: float]],  shape: [#_training,length_mps] (size_contracted_legs)) : It will contain the left or right legs, one dimensional arrays, 
                to contract with the joint tensor to optimize.
            - cost (JAX.numpy.Array : float, shape : (1)) : Cost or loss of the 0.5*(predicted labels - true_labels)^2 for the images batch.
            - accuracy (float): Accuracy between 0 and 1.
        '''
        
        j=self.j
        
        if left_or_right=='left':
            B_j_jplusone=self.contract_with_neighbour_tensor(left_or_right=left_or_right) # Compute the joint tensor on the j, j+1 sides
            cost, grad, accuracy = self.compute_cost_and_gradient_B_l(left_or_right=left_or_right) # Compute the gradient, cost and accuracy being centered in the previous joint tensor
            new_B_j_jplusone=self.gradient_descent_step( B_l=B_j_jplusone, gradient_B_l=grad) # Compute update joint tensor
            A_j,A_j_plusone=self.separate_B_l(left_or_right=left_or_right, updated_B_l=new_B_j_jplusone) # Returns new j, j+1 tensors
            self.mps[j]=A_j
            self.mps[j+1]=A_j_plusone
            self.update_data(left_or_right=left_or_right) # Computes update of the data with the previous optimized MPS in the sides j,j+1
        
        if left_or_right=='right':
            B_jminusone_j=self.contract_with_neighbour_tensor(left_or_right=left_or_right) # Compute the joint tensor on the j-1, j sides
            cost, grad, accuracy = self.compute_cost_and_gradient_B_l(left_or_right=left_or_right) # Compute the gradient, cost and accuracy being centered in the previous joint tensor
            new_B_jminusone_j=self.gradient_descent_step( B_l=B_jminusone_j, gradient_B_l=grad) # Compute update joint tensor
            A_j_minusone,A_j=self.separate_B_l(left_or_right=left_or_right, updated_B_l=new_B_jminusone_j) # Returns new j-1, j tensors
            self.mps[j-1]=A_j_minusone
            self.mps[j]=A_j
            self.update_data(left_or_right=left_or_right)# Computes update of the data with the previous optimized MPS in the sides j-1,j.
        
        return self.mps,self.data, cost, accuracy
    