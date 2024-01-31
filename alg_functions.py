import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

#---------------------------------------------------------------------------#
# Accuracy Measurement
#---------------------------------------------------------------------------#

def measure_accuracy(negative_class, positive_class, df):
    """
    Measure the accuracy of a classifier by calculating the precision, recall, and F1 score.
    """
    # Count of true positive (TP)
    tp = df[(df['type'] == positive_class) & (df['InsideBall'] == True)]['type'].count()

    # Count of false negative (FN)
    fn = df[(df['type'] == positive_class) & (df['InsideBall'] == False)]['type'].count()

    # Count of true negative (TN)
    tn = df[(df['type'] == negative_class) & (df['InsideBall'] == False)]['type'].count()

    # Count of false positive (FP)
    fp = df[(df['type'] == negative_class) & (df['InsideBall'] == True)]['type'].count()

    # Calculate precision, recall, and F1 score
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Create a table
    table_data = {
        'Metric': ['True Positive (TP)', 'False Negative (FN)', 'True Negative (TN)', 'False Positive (FP)', 'Precision', 'Recall', 'F1 Score'],
        'Count': [tp, fn, tn, fp, precision, recall, f1_score]
    }

    table_df = pd.DataFrame(table_data)

    # Display the table
    return table_df

#---------------------------------------------------------------------------#
# Helper Functions
#---------------------------------------------------------------------------#

def vertex_computation(set_weights, gradient, dimension_m):
    weight_indexes = np.where(set_weights > 0)[0]
    max_gradient_index = np.argmax(gradient[weight_indexes])
    v_index = weight_indexes[max_gradient_index]
    v = np.zeros(dimension_m)
    v[v_index] = 1.0
    return v, v_index

def phi(u, A):
    return sum(u[i] * np.dot(A[i].T, A[i]) for i in range(len(u))) - np.dot(sum(u[i] * A[i] for i in range(len(u))).T, sum(u[i] * A[i] for i in range(len(u))))

def create_unit_vector(idx, length):
    vec = np.zeros(length)
    vec[idx] = 1
    return vec

def LMO(gradient):
    lmo = np.argmin(gradient)
    return lmo

def generate_non_overlapping_circular_data(n_points, radius, center1, center2):

    angles = np.random.uniform(0, 2 * np.pi, n_points)
    radii = np.sqrt(np.random.uniform(0, 1, n_points)) * radius
    
    x1 = radii * np.cos(angles) + center1[0]
    y1 = radii * np.sin(angles) + center1[1]
    set1 = np.column_stack((x1, y1))
    
    x2 = radii * np.cos(angles) + center2[0]
    y2 = radii * np.sin(angles) + center2[1]
    set2 = np.column_stack((x2, y2))

    return set1, set2

#---------------------------------------------------------------------------#
# Line Search Functions
#---------------------------------------------------------------------------#

def armijo_method(A, u, alpha = 0.9, delta = 0.5, gamma = 0.1, max_iter=100):   
    gradient = np.matmul(2 * np.matmul(A, A.T), u) - np.sum(np.square(A), axis=1)
    gradient_norm = np.linalg.norm(gradient)
    phi_u = -phi(u, A) # Negative objective function
    for i in range(max_iter):
        
        new_u = u - alpha * gradient # Update u
        phi_new_u = -phi(new_u, A) # Update function

        if phi_new_u <= phi_u - gamma * alpha * gradient_norm ** 2: # Armijo condition
            return alpha
        
        alpha *= delta # Reduce alpha
    return alpha

def exact_line_search_method(A, direction, gradient):
    numerator = np.matmul(gradient.T, direction)
    denominator = 2 * np.matmul(np.matmul(direction.T, np.matmul(A, A.T)), direction)
    return abs(numerator/ denominator)

#---------------------------------------------------------------------------#
# Algorithm 1: Frank-Wolfe Algorithm for the Minimum Enclosing Ball Problem
#---------------------------------------------------------------------------#

def frank_wolfe_alg_MEB(epsilon, A):
    """
    Frank-Wolfe algorithm for the minimum enclosing ball problem.
    """

    start_time = time.time()

    # Compute distances and indices
    distances_to_a_1 = np.linalg.norm(A - A[0], axis=1)**2 # ||a_i - a_1||^2
    alpha = np.argmax(distances_to_a_1)
    distances_to_a_alpha = np.linalg.norm(A - A[alpha], axis=1)**2 if alpha is not None else np.zeros(len(A)) # ||a_i - a_α||^2
    beta = np.argmax(distances_to_a_alpha)
    #print(f"First Value in A: {A[0]}")
    #print(f"Initial values for Alpha and Beta: {(A[alpha], A[beta])}")
    # Initialize u, chi, c
    u = np.zeros(len(A))
    u[alpha], u[beta] = 0.5, 0.5

    chi = [A[alpha], A[beta]]
    c = u @ A
    c_hist = [c]
    mu = phi(u, A)
    kappa_idxs = []

    # Main iteration loop
    distances_to_c = np.linalg.norm(A - c, axis=1)**2
    kappa = np.argmax(distances_to_c)
    delta = (distances_to_c[kappa] / mu) - 1.0
    #print(f"Initial Delta: {delta}")
    #print(f"Initial Mu: {mu}")
    k = 0
    while delta > ((1 + epsilon)**2) - 1.0 and k < 1000:
        lambd = delta / (2 * (1.0 + delta))
        k += 1
        u = (1 - lambd) * u + lambd * create_unit_vector(kappa, len(A))
        c = (1 - lambd) * c + lambd * A[kappa]
        c_hist.append(c)
        if kappa not in kappa_idxs:
            kappa_idxs.append(kappa)
            chi.append(A[kappa])
        #print(f"Kappa at iteration {k}: {A[kappa]}")
        #print(f"Delta at iteration {k}: {delta}")
        mu = phi(u, A)
        distances_to_c = np.linalg.norm(A - c, axis=1)**2
        kappa = np.argmax(distances_to_c)
        delta = (distances_to_c[kappa] / mu) - 1

    timer = time.time() - start_time
    print("Center:", c)
    print("Radius:", np.sqrt((1 + delta) * mu))
    print("Iterations:", k)
    print("CPU time:", timer)
    print("Core Set Size:", len(chi))

    return c, np.sqrt((1 + delta) * mu), k, chi, timer

#---------------------------------------------------------------------------#
# Algorithm 2: Away Step Frank-Wolfe Algorithm
#---------------------------------------------------------------------------#

def frankWolfe_AwayStep(A, epsilon, max_iterations=1000, step_size = "Exact"):
    
    count_FrankWolfe = 0
    count_AwayStep = 0
    count_dropped = 0 
    count_iterations = 0
    dual_val_list = []
    dual_gap_list = []

    start_time = time.time()

    m, n = A.shape

    # initialize u vector
    u = np.zeros(m)
    u[0] = 1.0

    # Create set S containing the u values across time
    S_set = np.zeros(m)
    S_set[np.where(u > 0)[0]] = 1

    if step_size == "Lipschitz":
        H = 2 * np.matmul(A, A.T)
        eigen_values = np.linalg.eigvalsh(H)
        L = np.max(eigen_values)
        alpha = 1/L

    for i in range(max_iterations):

        dual_val = -phi(u, A) # We minimize the negative objective function
        dual_val_list.append(dual_val) # Keep track of the changes in the objective function across iterations

        gradient = np.matmul(2 * np.matmul(A, A.T), u) - np.sum(np.square(A), axis=1) # Gradient computation
        # FW direction
        s_index= LMO(gradient) 
        s = np.zeros(m)
        s[s_index] = 1.0
        direction_FrankWolfe = s - u
        
        # AW direction
        v, v_index = vertex_computation(S_set, gradient, m)
        direction_AwayStep = u - v

        # Stopping criterion
        gap_FrankWolfe = -gradient.T @ direction_FrankWolfe 
        dual_gap_list.append(gap_FrankWolfe) 
        if gap_FrankWolfe <= epsilon:
            break

        count_iterations += 1

        # Compare and choose between FW and AS 
        gap_AwayStep = -gradient.T @ direction_AwayStep 
        if gap_FrankWolfe >= gap_AwayStep:
            # Frank-Wolfe 
            direction_chosen = direction_FrankWolfe
            max_alpha = 1  # Max step-size FW
            frankwolfe_flag = True
        else:
            # Away-step
            direction_chosen = direction_AwayStep
            max_alpha = S_set[v_index] / (1 - S_set[v_index]) # Max step-size AS
            frankwolfe_flag = False

        # alpha = calculate_step_size(line_search_strategy, i, A_squared, u, Z, direction_chosen, gradient, max_alpha)
        # Step-size
        if step_size == "Harmonic":
            alpha = 2 / (i + 1) 
        if step_size == "Armijo":
            alpha = armijo_method(A, u)
        if step_size == "Exact":
            alpha = exact_line_search_method(A, direction_chosen, gradient)
        alpha = max(0.0, min(alpha, max_alpha))

        # update u
        u = u + (alpha * direction_chosen)

        # Update set S
        if frankwolfe_flag: # FW step
            S_set = (1 - alpha) * S_set # alpha_v update
            S_set[s_index] = S_set[s_index] + alpha # alpha_S_t update
            if alpha > 0.999999: # step-size is max
                # Set S = {s_t}
                S_set = np.zeros(m)
                S_set[s_index] = 1 
            count_FrankWolfe += 1 
        else: # AS step
            S_set = (1 + alpha) * S_set # alpha_v update
            count_AwayStep += 1
            if abs(alpha - max_alpha) < 0.000001: # step-size is max
                count_dropped += 1
                S_set[v_index] = 0 # drop step
            else:
                S_set[v_index] = S_set[v_index] - alpha # alpha_V_t update

    radius = np.sqrt(-dual_val)
    center = np.matmul(A.T, u)

    timer = time.time() - start_time

    print("Center:", center)
    print("Radius:", radius)
    print("Iterations:", count_iterations)
    print("CPU time:", timer)
    print("Set Size:", len(S_set[np.where(S_set > 0)]))

    return center, radius, count_iterations, timer, dual_val_list, dual_gap_list

#---------------------------------------------------------------------------#
# Algorithm 3: Pairwise Frank-Wolfe Algorithm 
#---------------------------------------------------------------------------#

def frankWolfe_Pairwise(A, epsilon, max_iterations=1000, step_size = "Exact"):

    count_iterations = 0
    dual_val_list = []
    dual_gap_list = []
    start_time = time.time()
    
    m, n = A.shape

    # initialize u vector
    u = np.zeros(m)
    u[0] = 1.0

    # Create set S containing the u values across time
    S_set = np.zeros(m)
    S_set[np.where(u > 0)[0]] = 1

    if step_size == "Lipschitz":
        H = 2 * np.matmul(A, A.T)
        eigen_values = np.linalg.eigvalsh(H)
        L = np.max(eigen_values)
        alpha = 1/L

    for i in range(max_iterations):

        dual_val = -phi(u, A) # We minimize the negative objective function
        dual_val_list.append(dual_val) # Keep track of the changes in the objective function across iterations
        
        gradient = np.matmul(2 * np.matmul(A, A.T), u) - np.sum(np.square(A), axis=1) # Gradient computation
        
        # Compute the Pairwise direction
        s_index= LMO(gradient)
        s = np.zeros(m)
        s[s_index] = 1.0
        v, v_index = vertex_computation(S_set, gradient, m)
        direction_Pairwise = s - v

        # Stopping criterion
        gap_FrankWolfe = -gradient.T @ direction_Pairwise
        dual_gap_list.append(gap_FrankWolfe) 
        if gap_FrankWolfe <= epsilon:
            break

        count_iterations += 1

        # Step-size
        if step_size == "Harmonic":
            alpha = 2 / (i + 1)
        if step_size == "Armijo":
            alpha = armijo_method(A, u)
        if step_size == "Exact":
            alpha = exact_line_search_method(A, direction_Pairwise, gradient)
        alpha = min(alpha, S_set[v_index])

        # update u
        u = u + (alpha * direction_Pairwise)

        # Update set S
        S_set[v_index] = S_set[v_index] - alpha # alpha_V_t update
        S_set[s_index] = S_set[s_index] + alpha # alpha_S_t update

    radius = np.sqrt(-dual_val)
    center = np.matmul(A.T, u)
    
    timer = time.time() - start_time

    print("Center:", center)
    print("Radius:", radius)
    print("Iterations:", count_iterations)
    print("CPU time:", timer)
    print("Set Size:", len(S_set[np.where(S_set > 0)]))
    return center, radius, count_iterations, timer, dual_val_list, dual_gap_list
