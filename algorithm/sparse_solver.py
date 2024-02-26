import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt
from skimage.util import view_as_windows
from algorithm.dictionary import Dictionary


class SparseSolver:
    def __init__(
            self,
            enable_dictionary_learning: bool,
            num_learning_iterations: int,
            img: np.ndarray,
            dictionary: Dictionary,
            epsilon: float,
            verbose: bool
    ):
        self.enable_dictionary_learning = enable_dictionary_learning
        self.num_learning_iterations = num_learning_iterations
        self.img = img
        self.dictionary = dictionary
        self.epsilon = epsilon
        self.verbose = verbose

    def __call__(self) -> np.ndarray:

        image_patches = self.create_image_patches(stride=1)
     
        self.dictionary.build_dictionary()  # Dictionary

        if self.enable_dictionary_learning:
            new_dictionary= self.unitary_dictionary_learning(
                image_patches,
                self.dictionary.defined_dictionary,
                self.num_learning_iterations,
                self.epsilon 
            )

            self.dictionary.learned_dictionary = np.copy(new_dictionary)
            
        else:
            new_dictionary = np.copy(self.dictionary.defined_dictionary)



        reconst_img_patches, _ = self.batch_thresholding(
            D=new_dictionary, patches=image_patches, epsilon=self.epsilon
        )

        reconst_img = self.col2img(reconst_img_patches, self.dictionary.patch_size, self.img.shape)

        return reconst_img

    #ok
    def create_image_patches(self, stride=1, flatten=True) -> np.ndarray:
        """
        Function that return a oving windows, and etract the patches from the immage, flattened them to have a 1D array at the end 
        """

        patches = view_as_windows(
            arr_in=self.img, window_shape=self.dictionary.patch_size, step=stride
        )
        if flatten:
            patches = patches.reshape(
                -1, self.dictionary.patch_size[0] * self.dictionary.patch_size[1]
            ).T
        return patches



    @staticmethod
    def batch_thresholding(D: np.array, patches: np.ndarray, epsilon: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Param:
            :param D: numpy array dictionary (n x n).
            :param patches: numpy array.
            :param epsilon: allowed residual error.
            :return X, A: Reconstructed signal X (= D A) and the coefficients A of the dictionary D.
        
            The dictionary D is a matrix where each column is an atom. These atoms collectively 
            capture the fundamental visual elements that we aim to use for sparse representation.


        """

        num_atoms = D.shape[1]  # number of atoms
        num_patches = patches.shape[1]  # number of patches

        print("NUm of atom", num_atoms , "num od patches", num_patches)
        
        #It quantifies the contribution of each atom to every patch.
        inner_prod = np.matmul(D.T, patches)

        #Squaring the inner products gives the squared residuals,
        # highlighting the importance of large values while penalizing small ones.
        residual_sq = inner_prod ** 2
        
        #treshold epsilon ** 2
        #This thresholding step helps in selecting the most relevant elements. By sorting and extract the above element
        sorted_residual = np.sort(residual_sq, axis=0)
        sorted_residual_idx = np.argsort(residual_sq, axis=0)
        accumulated_residual = np.cumsum(sorted_residual, axis=0)
        above_thr_idx = (accumulated_residual > epsilon ** 2)  # indices of elements above a threshold

        #This line creates a submatrix where each column represents the corresponding 
        # column index. This will be used to index and update the coefficient matrix A.
        col_sub = np.tile(np.arange(num_patches).reshape(-1, 1), num_atoms).T

        #These indices represent the elements that contribute significantly.
        #extract form the above treshold
        mat_inds_to_keep = sorted_residual_idx[above_thr_idx]
        col_sub_to_keep = col_sub[above_thr_idx]

        #A is full of 0
        #inner product, so the contrubution of ech atom but only the main one above the teshold
        A = np.zeros((num_atoms, num_patches))
        
        A[mat_inds_to_keep, col_sub_to_keep] = inner_prod[mat_inds_to_keep, col_sub_to_keep]

        #reconstruct the isgnal Dictionary and the coefficent matrix
        X = np.matmul(D, A)  # Reconstruction of restored image patches using dictionary and determined coefficients.

        print("A shape" , A.shape)
        #The function returns the reconstructed signal X and the coefficient matrix A, 
        #providing a sparse representation of the input patches using the given dictionary.
        return X, A



    @staticmethod
    def col2img(img_patches: np.ndarray, img_patch_size: Tuple, img_size: Tuple) -> np.ndarray:
        
        numerator_img = np.zeros(img_size)
        denominator_img = np.zeros(img_size)

        for idx_row in range(img_size[0] - img_patch_size[0] + 1):
            for idx_col in range(img_size[1] - img_patch_size[1] + 1):

                num_of_curr_patch = idx_row * (img_size[1] - img_patch_size[1] + 1) + (idx_col + 1)
                last_row = idx_row + img_patch_size[0]
                last_col = idx_col + img_patch_size[1]
                curr_patch = img_patches[:, num_of_curr_patch - 1].reshape(img_patch_size[0], img_patch_size[1])

                numerator_img[
                    idx_row:last_row, idx_col:last_col
                ] = numerator_img[idx_row:last_row, idx_col:last_col] + curr_patch
                denominator_img[
                    idx_row:last_row, idx_col:last_col
                ] = denominator_img[idx_row:last_row, idx_col:last_col] + np.ones(curr_patch.shape)

        img = numerator_img / denominator_img

        return img




    def unitary_dictionary_learning(
            self,
            Y: np.ndarray,
            D_init: np.ndarray,
            num_iterations: int,
            pursuit_param: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        This function train dictionary . The trained dictionary can be obtained by:

          D = argmin_D || Y - DA ||_F^2 s.t. D'D = I,
        where A is a matrix that contains all the estimated coefficients and Y contains training examples.

        :param Y: A matrix containing training patches in its columns.
        :param D_init: Initial unitary dictionary
        :param num_iterations: Number of updating the dictionary.
        :param pursuit_param: Criterion for stopping the pursuit algorithm.
        :return: D, mean_error, mean_cardinality: Trained dictionary, average representation error, and number of
                 non-zero elements.
        """

        
        D = np.copy(D_init)
        print("Dimesione dictionary" , D.shape)
        print("Dimensione patch", Y.shape)
        # Procrustes analysis
        for i in range(num_iterations):

            print(i)
            
            #This step performs sparse coding using the current dictionary.
            [ _ , A] = self.batch_thresholding(D, Y, pursuit_param)

           
            #Perform Singular Value Decomposition (SVD) on A and Y.T
            #SVD breaks down this product into three matrices U, S (singular values), and V. This is a form of Procrustes analysis,
            #aligning the basis of the learned representation with the basis of the original data.
            [U, _, V] = np.linalg.svd(np.matmul(A, Y.T))


            #pdate the dictionary D by multiplying the transposes of the matrices obtained from SVD.
            #  This step aligns the learned dictionary with the structure of the input data.
            D = np.matmul(V.T, U.T)

        return D

"""
In summary, the SVD step in Procrustes analysis plays a crucial role in aligning the basis of
 the sparse coefficients with the basis of the original data. It ensures that the learned 
 dictionary is transformed optimally to capture essential features and patterns present in
  the input data. The resulting transformation enhances the quality and interpretability
   of the learned representation.

"""