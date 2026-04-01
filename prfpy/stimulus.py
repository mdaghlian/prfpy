import numpy as np


class PRFStimulus2D(object):
    """PRFStimulus2D

    Minimal visual 2-dimensional pRF stimulus class, 
    which takes an input design matrix and sets up its real-world dimensions.

    """

    def __init__(self,
                 screen_size_cm,
                 screen_distance_cm,
                 design_matrix,
                 TR,                 
                 task_lengths=None,
                 task_names=None,
                 late_iso_dict=None,
                 normalize_integral_dx=False,
                 **kwargs):
        """
        

        Parameters
        ----------
        screen_size_cm : float
            size of screen in centimeters. NOTE: prfpy uses a square design
            matrix. the screen_size_cm refers to the length of a side of this
            square in cm, i.e. for a rectangular screen,this is the length
            in cm of the smallest side.
        screen_distance_cm : float
            eye-screen distance in centimeters
        design_matrix : numpy.ndarray
            an N by t matrix, where N is [x, x]. 
            represents a square screen evolving over time (time is last dimension)
        TR : float
            Repetition time, in seconds
        task_lengths : list of ints, optional
            If there are multiple tasks, specify their lengths in TRs. The default is None.
        task_names : list of str, optional
            Task names. The default is None.
        late_iso_dict : dict, optional 
            Dictionary whose keys correspond to task_names. Entries are ndarrays
            containing the TR indices used to compute the BOLD baseline for each task.
            The default is None.
        **kwargs : optional
            Use normalize_integral_dx = True to normalize the prf*stim sum as an integral.


        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        None.

        """

        self.screen_size_cm = screen_size_cm
        self.screen_distance_cm = screen_distance_cm
        self.design_matrix = design_matrix
        if len(self.design_matrix.shape) >= 3 and self.design_matrix.shape[0] != self.design_matrix.shape[1]:
            raise ValueError  # need the screen to be square
        self.TR = TR
                
        # other useful stimulus properties
        self.task_lengths = task_lengths
        self.task_names = task_names
        self.late_iso_dict = late_iso_dict

        self.screen_size_degrees = 2.0 * \
            np.degrees(np.arctan(self.screen_size_cm /
                                 (2.0*self.screen_distance_cm)))

        oneD_grid = np.linspace(-self.screen_size_degrees/2,
                                self.screen_size_degrees/2,
                                self.design_matrix.shape[0],
                                endpoint=True)
        self.x_coordinates, self.y_coordinates = np.meshgrid(
            oneD_grid, oneD_grid)
        self.complex_coordinates = self.x_coordinates + self.y_coordinates * 1j
        self.ecc_coordinates = np.abs(self.complex_coordinates)
        self.polar_coordinates = np.angle(self.complex_coordinates)
        self.max_ecc = np.max(self.ecc_coordinates)

        # construct a standard mask based on standard deviation over time
        self.mask = np.std(design_matrix, axis=-1) != 0
        
        # whether or not to normalize the stimulus_through_prf as an integral, np.sum(prf*stim)*dx**2
        self.normalize_integral_dx = kwargs.pop('normalize_integral_dx', False)
        
        if self.normalize_integral_dx:
            self.dx = self.screen_size_degrees/self.design_matrix.shape[0]
        else:
            self.dx = 1


class PRFStimulus1D(object):
    """PRFStimulus1D

    Minimal visual 1-dimensional pRF stimulus class, 
    which takes an input design matrix and sets up its real-world dimensions.

    """

    def __init__(self,
                 design_matrix,
                 mapping,
                 TR,
                 task_lengths=None,
                 task_names=None,
                 late_iso_dict=None,
                 **kwargs):
        """__init__


        Parameters
        ----------
        design_matrix : numpy.ndarray
            a 2D matrix (M by t). 
            represents inputs in an encoding space evolving over time (time is last dimension)
        mapping : numpy.ndarray, np.float
            for each of the columns in design_matrix, the value in the encoding dimension
            for example, in a numerosity experiment these would be the numerosity of presented stimuli
        TR : float
            Repetition time, in seconds
        task_lengths : list of ints, optional
            If there are multiple tasks, specify their lengths in TRs. The default is None.
        task_names : list of str, optional
            Task names. The default is None.
        late_iso_dict : dict, optional 
            Dictionary whose keys correspond to task_names. Entries are ndarrays
            containing the TR indices used to compute the BOLD baseline for each task.
            The default is None.

        """
        self.design_matrix = design_matrix
        self.mapping = mapping
        self.TR = TR
        
        # other potentially useful stimulus properties
        self.task_lengths = task_lengths
        self.task_names = task_names
        self.late_iso_dict = late_iso_dict

        
class CFStimulus(object):
    
    """CFStimulus

    Minimal CF stimulus class. Creates a design matrix for CF models by taking the data within a sub-surface (e.g. V1).

    """
    
    
    def __init__(self,
                 data,
                 vertinds,
                 distances,**kwargs):
        
        """__init__


        Parameters
        ----------
        data : numpy.ndarray
            a 2D matrix that contains the whole brain data (second dimension must be time). 
            
            
         vertinds : numpy.ndarray
            a matrix of integers that define the whole-brain indices of the vertices in the source subsurface.
            
        distances : numpy.ndarray
            a matrix that contains the distances between each vertex in the source sub-surface.
            
        Returns
        -------
        data: Inherits data.
        subsurface_verts: Inherits vertinds.
        distance_matrix: Inherits distances.
        design_matrix: The data contained within the source subsurface (to be used as a design matrix).
        

        """
        self.data=data
        
        self.subsurface_verts=vertinds
        
        self.design_matrix=self.data[self.subsurface_verts,:]
        
        self.distance_matrix=distances


# ************************************************************************************************************
# CSenF functions
class CSenFStimulus(object):
    """CSenFStimulus    
    Puts all the stimulus information into a useful object (as with other prfpy functions)

    """

    def __init__(self,
                 SF_seq,
                 CON_seq,                
                 TR,
                 discrete_levels=True,              
                 task_lengths=None,
                 task_names=None,
                 late_iso_dict=None,   
                 ):
        """__init__


        Parameters
        ----------
        SF_seq : np.ndarray (1 for each timepoint)
            Sequence of SF values in stimulus
        CON_seq : np.ndarray (1 for each timepoint)
            Sequence of contrast values in stimulus            
        TR : float
            Repetition time, in seconds
        discrete_levels : bool
            Are the contrast and SF levels discrete?
        
        Other info - mainly redundant, included for consistency with other prfpy components
        -------
        task_lengths : list of ints, optional
            If there are multiple tasks, specify their lengths in TRs. The default is None.
        task_names : list of str, optional
            Task names. The default is None.
        late_iso_dict : dict, optional 
            Dictionary whose keys correspond to task_names. Entries are ndarrays
            containing the TR indices used to compute the BOLD baseline for each task.
            The default is None.
        """
        # [1] Save input values
        self.SF_seq             = SF_seq
        self.CON_seq            = CON_seq
        self.TR                 = TR
        self.discrete_levels    = discrete_levels
        # *** legacy, mainly obsolete kept for consistency w/ other prfpy components ***
        self.task_lengths = task_lengths
        self.task_names = task_names
        self.late_iso_dict = late_iso_dict
        self.dx = 1
        # *** ***
        
        # [2] log versions
        self.log_SF_seq = np.log10(self.SF_seq)                
        self.CON_S_seq  = 100/self.CON_seq 
        self.n_TRs = SF_seq.shape[0]        
        print(f'Number of timepoints: {self.n_TRs}')
        

        # [3] If using discrete levels - get some extra useful info...
        if self.discrete_levels:
            # [3] Get the unique values, sorted in ascending order, excluding 0
            self.SFs = np.unique(self.SF_seq)
            self.SFs = self.SFs[self.SFs!=0]
            self.CONs = np.unique(self.CON_seq)
            self.CONs = self.CONs[self.CONs!=0]        
            self.log_SFs = np.log10(self.SFs)   # unique SF in log
            self.CON_Ss = 100/self.CONs         # unique contrast sensitivity
            # -> number of discrete levels (used for design matrix) and number of timepoints
            self.n_SF   = self.SFs.shape[0]
            self.n_CON  = self.CONs.shape[0]
            self.n_TRs = SF_seq.shape[0]        
            print(f'Number of unique SF levels: {self.n_SF}, {np.round(self.SFs, 3)}')
            print(f'Number of unique CON levels: {self.n_CON}, {np.round(self.CONs, 3)}')
            
            # Grids: true values. Used in making the RFs and CSF curves (see .rf)
            self.SF_grid, self.CON_grid = np.meshgrid(self.SFs, self.CONs)        
            self.log_SF_grid, self.CON_S_grid = np.meshgrid(self.log_SFs, self.CON_Ss)

            # Grids: levels (1, 2, 3, etc.) Used in making the design matrix 
            # >> 0 empty, 1 = lowest  SF, 2 = 2nd lowest  SF, etc.
            # >> 0 empty, 1 = lowest CON, 2 = 2nd lowest CON, etc.
            self.SF_grid_id, self.CON_grid_id = np.meshgrid(np.arange(1,self.n_SF+1), np.arange(1,self.n_CON+1))         
            # Create the sequences, but in terms of levels
            self.SF_seq_id = np.zeros_like(self.SF_seq, dtype=int)
            for i,SF in enumerate(self.SFs):
                self.SF_seq_id[np.round(self.SF_seq,3)==np.round(SF,3)] = i+1 # Assign the level to the corresponding SF
            
            self.CON_seq_id = np.zeros_like(self.CON_seq, dtype=int)
            for i,CON in enumerate(self.CONs):
                self.CON_seq_id[self.CON_seq==CON] = i+1

            # Create the design matrix (n SFS, n CON, n Timepoints)
            dm = np.zeros((self.n_CON, self.n_SF, self.n_TRs))
            for i in range(self.n_TRs):
                if self.CON_seq_id[i]!=0:
                    this_frame = (self.SF_grid_id==self.SF_seq_id[i]) & (self.CON_grid_id==self.CON_seq_id[i])
                    if this_frame.sum()==0:
                        print(self.SF_seq[i], self.CON_seq[i])
                        print('*** ERROR ***')
                        
                    dm[:,:,i] = np.copy(this_frame)
            
            self.design_matrix = dm            