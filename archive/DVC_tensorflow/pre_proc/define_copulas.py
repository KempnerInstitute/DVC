from vine_tree.tree_op import random_r_matrix_gen, prepare_regular, prepare_vine
from classes.objects import margin_obj, cop_par_obj

def define_copulas(vine_type, method, binning, n_bin, dim):

    ##################### THE FIRST PART IS RELATED TO THE REGULAR VINE ####################################

    if vine_type == 'r-vine':
        
        if method == 'matrix':
            
            ######### REGULAR MATRIX

            ### Example of r-matrix with 5 variables
            r_matrix = np.array([[2, 0, 0, 0, 0],
                                [5, 3, 0, 0, 0],
                                [4, 5, 1, 0, 0],
                                [1, 4, 5, 4, 0],
                                [3, 1, 4, 5, 5]])

            ### Example of r-matrix with 4 variables
    #         r_matrix = np.array([[3, 0, 0, 0],
    #                              [1, 4, 0, 0],
    #                              [2, 1, 2, 0],
    #                              [4, 2, 1, 1]])

            ### Example of r-matrix with 3 variables
    #         r_matrix = np.array([[3, 0, 0],
    #                              [2, 2, 0],
    #                              [1, 1, 1]])

            print(r_matrix)
            
        elif method == 'random':
            
            ##### RANDOM R-MATRIX
            r_matrix, ind_vine, nodes, E = random_r_matrix_gen(dim)
            print(r_matrix)

        ## Function that computes nodex, matrix edges and tree for the given r-matrix
        E, ind_vine, nodes, matrix_edges = prepare_regular(r_matrix)
        print('matrix_edges',matrix_edges)
        
        ### DEFINE MARGINS

        margin_fam1 = ['norm','norm','norm','norm','norm']
        theta_fam1 = [[0,1],[0,1],[0,1],[0,1],[0,1]]

        ### Example of margins with gaussian and gamma distributions
    #     margin_fam1 = ['norm','gamma','norm','gamma','norm']
    #     theta_fam1 = [[0,1],[2,4],[0,1],[2,4],[0,1]]

        is_cont1 = [True,True,True,True,True]

        margin_vine = []
        for i in range(0,len(margin_fam1),1):
            mar_p = margin_obj(margin_fam1[i], theta_fam1[i], is_cont1[i])
            margin_vine.append(mar_p)

        for i in range(0,len(margin_fam1),1):
            print(margin_vine[i].dist, end =' ')
            print(margin_vine[i].theta, end =' ')
        
        ######################## DEFINE COPULAS  ###################
        if not binning:
            
            ### Example with 4 variables
    #         margin_cop1 = [['gaussian','gaussian','gaussian','gaussian'],
    #                        ['gaussian','gaussian','gaussian'],
    #                        ['gaussian','gaussian'],
    #                        ['gaussian']]
            
    #         theta_cop1 = [[0.3, 0.5, 0.7, 0.8],
    #                       [0.5, 0.8, 0.4],
    #                       [0.5, 0.3],
    #                       [0.9]]

            ### Example with 5 variables
    #         margin_cop1 = [['gaussian','gaussian','gaussian','gaussian','gaussian'],
    #                        ['gaussian','gaussian','gaussian','gaussian'],
    #                        ['gaussian','gaussian','gaussian'],
    #                        ['gaussian','gaussian'],
    #                        ['gaussian']]
            
    #         theta_cop1 = [[0.3, 0.5, 0.7, 0.8, -0.8],
    #                       [0.5, 0.8, 0.4, -0.2],
    #                       [0.5, 0.3, -0.7],
    #                       [0.9, 0.6],
    #                       [0.5]]
            
            ### Example with 4 variables and different distributions
            margin_cop1 = [['gaussian','student','clayton','gaussian'],
                        ['student','clayton','gaussian'],
                        ['student','gaussian'],
                        ['clayton']]

            theta_cop1 = [[0.3, [0,0.2], 0.7,  -0.8],
                        [[-0.8,0.2], 4.5,  -0.2],
                        [[0,0.5],  -0.7],
                        [0.9]]
            
            ### Example with 3 variables
    #         margin_cop1 = [['gaussian','gaussian','gaussian'],
    #                        ['gaussian','gaussian'],
    #                        ['gaussian']]

    #         theta_cop1 = [[0.7, 0.8, 0.9],
    #                       [0.6, 0.5],
    #                       [0.7]]

            ### Example with 3 variables
            # margin_cop1 = [['clayton','clayton','clayton'],
            #                ['clayton','clayton'],
            #                ['clayton']]

            # theta_cop1 = [[3.7, 4.8, 5.9],
            #               [6.6, 2.5],
            #               [5.7]]
        else:
            
            ### If binning, you have to define each type of variable for each bin. This is an example with 4 variables and 3 bins, remember 
            ### that the first level is not binned

            margin_cop1 = [['gaussian','gaussian','gaussian','gaussian'],
                        [['gaussian','gaussian','gaussian'],['gaussian','gaussian','gaussian'],['gaussian','gaussian','gaussian']],
                        [['gaussian','gaussian','gaussian'],['gaussian','gaussian','gaussian']],
                        [['gaussian','gaussian','gaussian']]]

            theta_cop1 = [[0.3, 0.5, 0.7, 0.8],
                        [[0.3, 0.4, 0.5],[0.6, 0.7, 0.8],[0.2, 0.3, 0.4]],
                        [[0.3, 0.4, 0.5],[0.3, 0.5, 0.9]],
                        [[0.2, 0.5, 0.9]]] #0.2,0.5,0.9
            
            ### Example with 3 variables
    #         margin_cop1 = [['gaussian','gaussian','gaussian'],
    #                        [['gaussian','gaussian','gaussian'],['gaussian','gaussian','gaussian']],
    #                       [['gaussian','gaussian','gaussian']]]

    #         theta_cop1 = [[0.7, 0.8, 0.9],
    #                       [[0.3, 0.4, 0.5],[0.6, 0.7, 0.8]],
    #                       [[0.2, 0.5, 0.9]]]

            ### Example with 2 variables
    #         margin_cop1 = [['gaussian','gaussian'],
    #                       [['gaussian','gaussian','gaussian']]]

    #         theta_cop1 = [[0.7, 0.8],
    #                       [[0.2, 0.5, 0.9]]]


        d = len(r_matrix)
        cop_vine = []
        for tr in range(0,d-1,1):
            cop_vine1 = []
            for col in range(0,d-1-tr,1):
                if (tr == 0) | (binning == False):
                    cop_p = cop_par_obj(margin_cop1[tr][col],theta_cop1[tr][col])
                    cop_vine1.append(cop_p)
                else:
                    cop_vine11 = []
                    for bb in range(0,n_bin,1):
                        cop_p = cop_par_obj(margin_cop1[tr][col][bb],theta_cop1[tr][col][bb])
                        cop_vine11.append(cop_p)
                    cop_vine1.append(cop_vine11)
            cop_vine.append(cop_vine1)

        for tr in range(0,d-1,1):
            for col in range(0,d-1-tr,1):
                if (tr == 0) | (binning == False):
                    print('edge: {} '.format(matrix_edges[tr][col]), 'cop family: {}'.format(cop_vine[tr][col].family), 'theta: {}'.format(cop_vine[tr][col].theta))
                else:
                    for bb in range(0,n_bin,1):
                        print('edge: {} '.format(matrix_edges[tr][col]),'bin: {} '.format(bb), 'cop family: {}'.format(cop_vine[tr][col][bb].family), 'theta: {}'.format(cop_vine[tr][col][bb].theta))

        ################# IF YOU WANT TO USE C-VINE OR D-VINE    ################################### 
    elif (vine_type == 'c-vine') | (vine_type == 'd-vine'):
        
        
        r_matrix, ind_vine, nodes, matrix_edges = prepare_vine(vine_type, dim)

        print(r_matrix)

        # binning = False
        # n_bin = 3
        
        ########## DEFINE MARGINS
        
        margin_vine = []
        for i in range(0,dim,1):
            mar_p = margin_obj('norm', [0,1], True)
            margin_vine.append(mar_p)

        for i in range(0,dim,1):
            print(margin_vine[i].dist, end =' ')
            print(margin_vine[i].theta, end =' ')

        ############## DEFINE COPULAS
        # NN = 0
        tr = 0
        cop_vine = []
        for i in range(dim,1,-1):
            cop_vine1 = []
            for j in range(0,i-1,1):
                if (tr == 0) | (binning == False):
                    #### You can decomment the following to set only a specific level as independent

        #             if tr == NN:
        #                 cop_p = cop_par_obj('ind',[])  #
        #                 cop_vine1.append(cop_p)
        #             else:
                    cop_p = cop_par_obj('clayton',4.5)  #
                    cop_vine1.append(cop_p)
                else:
                    cop_vine11 = []
                    for bb in range(0,n_bin,1):
                        cop_p = cop_par_obj('gaussian',0.9)
                        cop_vine11.append(cop_p)
                    cop_vine1.append(cop_vine11)
            cop_vine.append(cop_vine1)
            tr += 1
        
    #     margin_cop1 = [['gaussian','student','clayton','gaussian'],
    #                    ['student','clayton','gaussian'],
    #                    ['student','gaussian'],
    #                    ['clayton']]
            
    #     theta_cop1 = [[0.3, [0,0.2], 0.7,  -0.8],
    #                   [[-0.8,0.2], 4.5,  -0.2],
    #                   [[0,0.5],  -0.7],
    #                   [0.9]]
        
    #     d = len(r_matrix)
    #     cop_vine = []
    #     for tr in range(0,d-1,1):
    #         cop_vine1 = []
    #         for col in range(0,d-1-tr,1):
    #             if (tr == 0) | (binning == False):
    #                 cop_p = cop_par_obj(margin_cop1[tr][col],theta_cop1[tr][col])
    #                 cop_vine1.append(cop_p)
    #             else:
    #                 cop_vine11 = []
    #                 for bb in range(0,n_bin,1):
    #                     cop_p = cop_par_obj(margin_cop1[tr][col][bb],theta_cop1[tr][col][bb])
    #                     cop_vine11.append(cop_p)
    #                 cop_vine1.append(cop_vine11)
    #         cop_vine.append(cop_vine1)

        d = len(r_matrix)
        for tr in range(0,d-1,1):
            for col in range(0,d-1-tr,1):
                if (tr == 0) | (binning == False):
                    print('edge: {} '.format(matrix_edges[tr][col]), 'cop family: {}'.format(cop_vine[tr][col].family), 'theta: {}'.format(cop_vine[tr][col].theta))
                else:
                    for bb in range(0,n_bin,1):
                        print('edge: {} '.format(matrix_edges[tr][col]),'bin: {} '.format(bb), 'cop family: {}'.format(cop_vine[tr][col][bb].family), 'theta: {}'.format(cop_vine[tr][col][bb].theta))
    
    return r_matrix, cop_vine, ind_vine, nodes, matrix_edges, margin_vine