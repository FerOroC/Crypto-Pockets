#Make entire linear_fea file into a class, bring up any constants as class variables
import numpy as np
import math
import copy

E = 1944*10**6 #(N/m^2)   youngs modulus of PA12
A = 1.9635*10**-5 #0.01 #(m^2) area for a dia of bar 0.005m
I = 6.1359*10**-11 #(m^4) #second moment area for shaft

members = np.array([[1,2],
                    [2,3],
                    [3,4],
                    [4,5],
                    [5,6],
                    [6,7],
                    [7,8],
                    [8,9],
                    [9,10],
                    [10,11],
                    [11,12],
                    [12,13],
                    [13,14],
                    [14,1],   #Element N
                    [1,15],       #15_1
                    [2,15],
                    [16,15],
                    [14,15],
                    [2,16],            #16_
                    [3,16],
                    [17,16],
                    [3,17],             #17_
                    [4,17],
                    [5,17],
                    [18,17],
                    [5,18],           #18_
                    [6,18],
                    [19,18],
                    [6,19],                #19_
                    [7,19],
                    [20,19],
                    [7,20],                    #20_
                    [8,20],
                    [9,20],
                    [21,20],
                    [9,21],                         #21
                    [10,21],
                    [22,21],
                    [10,22],                              #22
                    [11,22],
                    [12,22],
                    [14,23],                        #23
                    [15,23],
                    [24,23],
                    [26,23],
                    [13,23],
                    [16,24],                        #24
                    [17,24],
                    [18,24],
                    [25,24],
                    [19,25],                #25
                    [20,25],
                    [21,25],
                    [26,25],
                    [22,26],            #26
                    [12,26],
                    [13,26]
                    ])

Input_Forces = []
Input_Nodes = []        #shape should be: [25000, 26x2]
Node_List = []

Axial_Forces = []
Bending_Moments = []
Nodal_Displacements = []

Euclidian_Distance = []
STD_Distance = []

Performance_Coefficient = []

# Define a function to calculate member orientation and length
def memberOrientation(memberNo, members, nodes):
    memberIndex = memberNo - 1  # Index identifying member in array of members
    node_i = members[memberIndex][0]  # Node number for node i of this member
    node_j = members[memberIndex][1]  # Node number for node j of this member

    xi = nodes[node_i - 1][0]  # x-coord for node i
    yi = nodes[node_i - 1][1]  # y-coord for node i
    xj = nodes[node_j - 1][0]  # x-coord for node j
    yj = nodes[node_j - 1][1]  # y-coord for node j

    # Angle of member with respect to horizontal axis

    dx = xj - xi  # x-component of vector along member
    dy = yj - yi  # y-component of vector along member
    mag = math.sqrt(dx ** 2 + dy ** 2)  # Magnitude of vector (length of member)
    memberVector = np.array([dx, dy])  # Member represented as a vector

    # Need to capture quadrant first then appropriate reference axis and offset angle
    if (dx > 0 and dy == 0):
        theta = 0
    elif (dx == 0 and dy > 0):
        theta = math.pi / 2
    elif (dx < 0 and dy == 0):
        theta = math.pi
    elif (dx == 0 and dy < 0):
        theta = 3 * math.pi / 2
    elif (dx > 0 and dy > 0):
        # 0<theta<90
        refVector = np.array([1, 0])  # Vector describing the positive x-axis
        theta = math.acos(
            refVector.dot(memberVector) / (mag))  # Standard formula for the angle between two vectors
    elif (dx < 0 and dy > 0):
        # 90<theta<180
        refVector = np.array([0, 1])  # Vector describing the positive y-axis
        theta = (math.pi / 2) + math.acos(
            refVector.dot(memberVector) / (mag))  # Standard formula for the angle between two vectors
    elif (dx < 0 and dy < 0):
        # 180<theta<270
        refVector = np.array([-1, 0])  # Vector describing the negative x-axis
        theta = math.pi + math.acos(
            refVector.dot(memberVector) / (mag))  # Standard formula for the angle between two vectors
    else:
        # 270<theta<360
        refVector = np.array([0, -1])  # Vector describing the negative y-axis
        theta = (3 * math.pi / 2) + math.acos(
            refVector.dot(memberVector) / (mag))  # Standard formula for the angle between two vectors

    return [theta, mag]


def calculateKg(memberNo, orientations, lengths):
    """
    Calculate the global stiffness matrix for a beam element
    memberNo: The member number
    """
    theta = orientations[memberNo - 1]
    L = lengths[memberNo - 1]

    c = math.cos(theta)
    s = math.sin(theta)

    # Define the transformation matrix
    TM = np.array([[c, s, 0, 0, 0, 0],
                   [-s, c, 0, 0, 0, 0],
                   [0, 0, 1, 0, 0, 0],
                   [0, 0, 0, c, s, 0],
                   [0, 0, 0, -s, c, 0],
                   [0, 0, 0, 0, 0, 1]])

    # For clarity, define individual elements of local stiffness matrix
    # Row 1
    k11 = E * A / L
    k12 = 0
    k13 = 0
    k14 = -E * A / L
    k15 = 0
    k16 = 0
    # Row 2
    k21 = 0
    k22 = 12 * E * I / L ** 3
    k23 = -6 * E * I / L ** 2
    k24 = 0
    k25 = -12 * E * I / L ** 3
    k26 = -6 * E * I / L ** 2
    # Row 3
    k31 = 0
    k32 = -6 * E * I / L ** 2
    k33 = 4 * E * I / L
    k34 = 0
    k35 = 6 * E * I / L ** 2
    k36 = 2 * E * I / L
    # Row 4
    k41 = -E * A / L
    k42 = 0
    k43 = 0
    k44 = E * A / L
    k45 = 0
    k46 = 0
    # Row 5
    k51 = 0
    k52 = -12 * E * I / L ** 3
    k53 = 6 * E * I / L ** 2
    k54 = 0
    k55 = 12 * E * I / L ** 3
    k56 = 6 * E * I / L ** 2
    # Row 6
    k61 = 0
    k62 = -6 * E * I / L ** 2
    k63 = 2 * E * I / L
    k64 = 0
    k65 = 6 * E * I / L ** 2
    k66 = 4 * E * I / L

    # Build the quadrants of the local stiffness matrix
    K11 = np.array([[k11, k12, k13], [k21, k22, k23], [k31, k32, k33]])  # Top left quadrant of local stiffness matrix
    K12 = np.array(
        [[k14, k15, k16], [k24, k25, k26], [k34, k35, k36]])  # Top right quadrant of local stiffness matrix
    K21 = np.array(
        [[k41, k42, k43], [k51, k52, k53], [k61, k62, k63]])  # Bottom left quadrant of local stiffness matrix
    K22 = np.array([[k44, k45, k46], [k54, k55, k56],
                    [k64, k65, k66]])  # Bottom right quadrant of local stiffness matrix

    # Build complete local element stiffness matrix
    top = np.concatenate((K11, K12), axis=1)  # Top 3 rows
    btm = np.concatenate((K21, K22), axis=1)  # Bottom 3 rows
    Kl = np.concatenate((top, btm), axis=0)  # Full local stiffness matrix

    # Calculate the element global stiffness matrix
    Kg = TM.T.dot(Kl).dot(TM)
    # Kg = np.round(TM.T@Kl@TM,1) #Matrix multiply symbol as pf Python 3.5

    # Divide global element stiffness matrix quadrants for return
    K11g = Kg[0:3, 0:3]
    K12g = Kg[0:3, 3:6]
    K21g = Kg[3:6, 0:3]
    K22g = Kg[3:6, 3:6]

    return [K11g, K12g, K21g, K22g]


def LatticeGenerator(iterations):
    UG_Copy_1 = []
    UG_Copy_2 = []
    x_disp = []
    y_disp = []

    for i in range(iterations):

        xFac = 1  # Scale factor for plotted displacements

        lattice = 0.09343

        a_l = (lattice / 1.6180)
        b_l = (lattice / (1.6180 * 3))
        c_l = (lattice / (1.6180 * 4))
        d_l = (lattice / (1.6180 * 2))

        # Nodal coordinates [x, y] (in ascending node order)
        nodes = np.array([[0, lattice],  # node 1
                          [0, d_l],  # node 2
                          [0, c_l],  # node 3
                          [0, 0],  # node 4 1st vertical ends
                          [c_l, 0],  # node 5
                          [c_l + b_l, 0],  # node 6
                          [c_l + b_l + d_l, 0],  # node 7
                          [lattice, 0],  # node 8 1st horizontal ends
                          [lattice, c_l],  # node 9
                          [lattice, d_l],  # node 10
                          [lattice, lattice],  # node 11 2nd vertical ends
                          [c_l + b_l + d_l, lattice],  # node 12
                          [c_l + b_l, lattice],  # node 13
                          [c_l, lattice]])  # node 14  end of 1st squre

        # Members [node_i, node_j]
        members = np.array([[1, 2],
                            [2, 3],
                            [3, 4],
                            [4, 5],
                            [5, 6],
                            [6, 7],
                            [7, 8],
                            [8, 9],
                            [9, 10],
                            [10, 11],
                            [11, 12],
                            [12, 13],
                            [13, 14],
                            [14, 1],  # Element N
                            [1, 15],  # 15_1
                            [2, 15],
                            [16, 15],
                            [14, 15],
                            [2, 16],  # 16_
                            [3, 16],
                            [17, 16],
                            [3, 17],  # 17_
                            [4, 17],
                            [5, 17],
                            [18, 17],
                            [5, 18],  # 18_
                            [6, 18],
                            [19, 18],
                            [6, 19],  # 19_
                            [7, 19],
                            [20, 19],
                            [7, 20],  # 20_
                            [8, 20],
                            [9, 20],
                            [21, 20],
                            [9, 21],  # 21
                            [10, 21],
                            [22, 21],
                            [10, 22],  # 22
                            [11, 22],
                            [12, 22],
                            [14, 23],  # 23
                            [15, 23],
                            [24, 23],
                            [26, 23],
                            [13, 23],
                            [16, 24],  # 24
                            [17, 24],
                            [18, 24],
                            [25, 24],
                            [19, 25],  # 25
                            [20, 25],
                            [21, 25],
                            [26, 25],
                            [22, 26],  # 26
                            [12, 26],
                            [13, 26]
                            ])

        # =================================END OF DATA ENTRY================================

        # Laticce generator

        Nodes = 25
        Dimensions = 2

        # Nodal coordinates [x, y] (in ascending node order)
        nodes = np.array([[0, lattice],  # node 1  - index 0
                          [0, d_l],  # node 2
                          [0, c_l],  # node 3
                          [0, 0],  # node 4 1st vertical ends
                          [c_l, 0],  # node 5
                          [c_l + b_l, 0],  # node 6
                          [c_l + b_l + d_l, 0],  # node 7
                          [lattice, 0],  # node 8 1st horizontal ends
                          [lattice, c_l],  # node 9
                          [lattice, d_l],  # node 10
                          [lattice, lattice],  # node 11 2nd vertical ends
                          [c_l + b_l + d_l, lattice],  # node 12
                          [c_l + b_l, lattice],  # node 13
                          [c_l, lattice],  # node 14  end of 1st squre
                          [0, 0],  # node 15 - index 14
                          [0, 0],  # node 16 - index 15
                          [0, 0],  # node 17 - index 16
                          [0, 0],  # node 18 - index 17
                          [0, 0],  # node 19 - index 18
                          [0, 0],  # node 20 - index 19
                          [0, 0],  # node 21 - index 20
                          [0, 0],  # node 22 - index 21
                          [0, 0],  # node 23
                          [0, 0],  # node 24
                          [0, 0],  # node 25
                          [0, 0], ]  # node 26
                         )

        # Supports
        restrainedDoF = [1, 2, 3, 4, 5, 6, 7, 8, 9, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
                         42]  # The degrees of freedom restrained by supports

        # Loading
        forceVector = np.array([np.zeros(len(nodes) * 3)]).T  # Loading
        forceVector[9] = -500
        forceVector[10] = -500
        forceVector[21] = 500
        forceVector[22] = -500

        # Node 15
        # [14,0]

        random_15x = np.random.uniform(0,
                                       c_l)  # Coordinate generator "random_**xy"   → node constrains input here (x coordinate)
        r_15x = np.zeros((Nodes + 1, Dimensions))  # Empty matrix with correct dims generator "r_**xy"
        r_15x[14, 0] = random_15x  # adding the coordinate as an index to the empty matrix   - node coordinate input

        random_15y = np.random.uniform(d_l, lattice)  # node constrains input here (y coordinate)
        r_15y = np.zeros((Nodes + 1, Dimensions))
        r_15y[14, 1] = random_15y  # node coordinate input

        m_15 = np.add(r_15x, r_15y)  # m_** → the full brute force generator for coordinate

        # node 16
        # [15,0]

        random_16x = np.random.uniform(0,
                                       c_l)  # Coordinate generator "random_**xy"   → node constrains input here (x coordinate)
        r_16x = np.zeros((Nodes + 1, Dimensions))  # Empty matrix with correct dims generator "r_**xy"
        r_16x[15, 0] = random_16x  # adding the coordinate as an index to the empty matrix   - node coordinate input

        random_16y = np.random.uniform(c_l, d_l)  # node constrains input here (y coordinate)
        r_16y = np.zeros((Nodes + 1, Dimensions))
        r_16y[15, 1] = random_16y  # node coordinate input

        m_16 = np.add(r_16x, r_16y)  # m_** → the full brute force generator for coordinate
        m_16 = np.add(m_15, m_16)  # adding the brute force generators together

        # node 17
        # [16,0]

        random_17x = np.random.uniform(0,
                                       c_l)  # Coordinate generator "random_**xy"   → node constrains input here (x coordinate)
        r_17x = np.zeros((Nodes + 1, Dimensions))  # Empty matrix with correct dims generator "r_**xy"
        r_17x[16, 0] = random_17x  # adding the coordinate as an index to the empty matrix   - node coordinate input

        random_17y = np.random.uniform(0, c_l)  # node constrains input here (y coordinate)
        r_17y = np.zeros((Nodes + 1, Dimensions))
        r_17y[16, 1] = random_17y  # node coordinate input

        m_17 = np.add(r_17x, r_17y)  # m_** → the full brute force generator for coordinate
        m_17 = np.add(m_17, m_16)  # adding the brute force generators together

        # node 18
        # [17,0]

        random_18x = np.random.uniform(c_l,
                                       d_l)  # Coordinate generator "random_**xy"   → node constrains input here (x coordinate)
        r_18x = np.zeros((Nodes + 1, Dimensions))  # Empty matrix with correct dims generator "r_**xy"
        r_18x[17, 0] = random_18x  # adding the coordinate as an index to the empty matrix   - node coordinate input

        random_18y = np.random.uniform(0, c_l)  # node constrains input here (y coordinate)
        r_18y = np.zeros((Nodes + 1, Dimensions))
        r_18y[17, 1] = random_18y  # node coordinate input

        m_18 = np.add(r_18x, r_18y)  # m_** → the full brute force generator for coordinate
        m_18 = np.add(m_17, m_18)  # adding the brute force generators together

        # node 19
        # [18,0]

        random_19x = np.random.uniform(d_l, (
                    d_l * 2))  # Coordinate generator "random_**xy"   → node constrains input here (x coordinate)
        r_19x = np.zeros((Nodes + 1, Dimensions))  # Empty matrix with correct dims generator "r_**xy"
        r_19x[18, 0] = random_19x  # adding the coordinate as an index to the empty matrix   - node coordinate input

        random_19y = np.random.uniform(0, c_l)  # node constrains input here (y coordinate)
        r_19y = np.zeros((Nodes + 1, Dimensions))
        r_19y[18, 1] = random_19y  # node coordinate input

        m_19 = np.add(r_19x, r_19y)  # m_** → the full brute force generator for coordinate
        m_19 = np.add(m_19, m_18)  # adding the brute force generators together

        # node 20
        # [19,0]

        random_20x = np.random.uniform((d_l * 2),
                                       lattice)  # Coordinate generator "random_**xy"   → node constrains input here (x coordinate)
        r_20x = np.zeros((Nodes + 1, Dimensions))  # Empty matrix with correct dims generator "r_**xy"
        r_20x[19, 0] = random_20x  # adding the coordinate as an index to the empty matrix   - node coordinate input

        random_20y = np.random.uniform(0, c_l)  # node constrains input here (y coordinate)
        r_20y = np.zeros((Nodes + 1, Dimensions))
        r_20y[19, 1] = random_20y  # node coordinate input

        m_20 = np.add(r_20x, r_20y)  # m_** → the full brute force generator for coordinate
        m_20 = np.add(m_20, m_19)  # adding the brute force generators together

        # node 21
        # [20,0]

        random_21x = np.random.uniform(random_20x,
                                       lattice)  # Coordinate generator "random_**xy"   → node constrains input here (x coordinate)
        r_21x = np.zeros((Nodes + 1, Dimensions))  # Empty matrix with correct dims generator "r_**xy"
        r_21x[20, 0] = random_21x  # adding the coordinate as an index to the empty matrix   - node coordinate input

        random_21y = np.random.uniform(c_l, d_l)  # node constrains input here (y coordinate)
        r_21y = np.zeros((Nodes + 1, Dimensions))
        r_21y[20, 1] = random_21y  # node coordinate input

        m_21 = np.add(r_21x, r_21y)  # m_** → the full brute force generator for coordinate
        m_21 = np.add(m_20, m_21)  # adding the brute force generators together

        # node 22
        # [21,0]

        random_22x = np.random.uniform((d_l * 2),
                                       lattice)  # Coordinate generator "random_**xy"   → node constrains input here (x coordinate)
        r_22x = np.zeros((Nodes + 1, Dimensions))  # Empty matrix with correct dims generator "r_**xy"
        r_22x[21, 0] = random_22x  # adding the coordinate as an index to the empty matrix   - node coordinate input

        random_22y = np.random.uniform(d_l, lattice)  # node constrains input here (y coordinate)
        r_22y = np.zeros((Nodes + 1, Dimensions))
        r_22y[21, 1] = random_22y  # node coordinate input

        m_22 = np.add(r_22x, r_22y)  # m_** → the full brute force generator for coordinate
        m_22 = np.add(m_22, m_21)  # adding the brute force generators together

        # node 23
        # [22,0]

        random_23x = np.random.uniform((c_l), (c_l + b_l))
        r_23x = np.zeros((Nodes + 1, Dimensions))
        r_23x[22, 0] = random_23x

        random_23y = np.random.uniform((d_l), lattice)
        r_23y = np.zeros((Nodes + 1, Dimensions))
        r_23y[22, 1] = random_23y

        m_23 = np.add(r_23x, r_23y)  # m_** → the full brute force generator for coordinate
        m_23 = np.add(m_23, m_22)  # adding the brute force generators together

        # node 24
        # [23,0]

        random_24x = np.random.uniform((c_l), (c_l + b_l))
        r_24x = np.zeros((Nodes + 1, Dimensions))
        r_24x[23, 0] = random_24x

        random_24y = np.random.uniform(c_l, d_l)
        r_24y = np.zeros((Nodes + 1, Dimensions))
        r_24y[23, 1] = random_24y

        m_24 = np.add(r_24x, r_24y)  # m_** → the full brute force generator for coordinate
        m_24 = np.add(m_24, m_23)  # adding the brute force generators together

        # node 25
        # [23,0]

        random_25x = np.random.uniform((c_l + b_l), (c_l + b_l + d_l))
        r_25x = np.zeros((Nodes + 1, Dimensions))
        r_25x[24, 0] = random_25x

        random_25y = np.random.uniform(c_l, d_l)
        r_25y = np.zeros((Nodes + 1, Dimensions))
        r_25y[24, 1] = random_25y

        m_25 = np.add(r_25x, r_25y)  # m_** → the full brute force generator for coordinate
        m_25 = np.add(m_25, m_24)  # adding the brute force generators together

        # node 26
        # [23,0]

        random_26x = np.random.uniform((c_l + b_l), (c_l + b_l + d_l))
        r_26x = np.zeros((Nodes + 1, Dimensions))
        r_26x[25, 0] = random_26x

        random_26y = np.random.uniform((d_l), lattice)
        r_26y = np.zeros((Nodes + 1, Dimensions))
        r_26y[25, 1] = random_26y

        m_26 = np.add(r_26x, r_26y)  # m_** → the full brute force generator for coordinate
        m_26 = np.add(m_26, m_25)  # adding the brute force generators together

        # end of lattice generator

        nodes = np.add(m_26, nodes)

        # -----------------------------------------------------------------------------------------------------------------------------------------------------------

        # Calculate orientation and length for each member and store
        orientations = np.array([])  # Initialise an array to hold orientations
        lengths = np.array([])  # Initialise an array to hold lengths
        for n, mbr in enumerate(members):
            [angle, length] = memberOrientation(n + 1, members, nodes)
            orientations = np.append(orientations, angle)
            lengths = np.append(lengths, length)

        nDoF = np.amax(members) * 3  # Total number of degrees of freedom in the problem
        Kp = np.zeros([nDoF, nDoF])  # Initialise the primary stiffness matrix

        for n, mbr in enumerate(members):
            # note that enumerate adds a counter to an iterable (n)

            # Calculate the quadrants of the global stiffness matrix for the member
            [K11, K12, K21, K22] = calculateKg(n + 1, orientations, lengths)

            node_i = mbr[0]  # Node number for node i of this member
            node_j = mbr[1]  # Node number for node j of this member

            # Primary stiffness matrix indices associated with each node
            # i.e. node 1 occupies indices 0, 1 and 2 (accessed in Python with [0:3])
            ia = 3 * node_i - 3  # index 0 (e.g. node 1)
            ib = 3 * node_i - 1  # index 2 (e.g. node 1)
            ja = 3 * node_j - 3  # index 3 (e.g. node 2)
            jb = 3 * node_j - 1  # index 5 (e.g. node 2)
            Kp[ia:ib + 1, ia:ib + 1] = Kp[ia:ib + 1, ia:ib + 1] + K11
            Kp[ia:ib + 1, ja:jb + 1] = Kp[ia:ib + 1, ja:jb + 1] + K12
            Kp[ja:jb + 1, ia:ib + 1] = Kp[ja:jb + 1, ia:ib + 1] + K21
            Kp[ja:jb + 1, ja:jb + 1] = Kp[ja:jb + 1, ja:jb + 1] + K22

        # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------

        restrainedIndex = [x - 1 for x in restrainedDoF]  # Index for each restrained DoF (list comprehension)

        # Reduce to structure stiffness matrix by deleting rows and columns for restrained DoF
        Ks = np.delete(Kp, restrainedIndex, 0)  # Delete rows
        Ks = np.delete(Ks, restrainedIndex, 1)  # Delete columns
        Ks = np.matrix(Ks)  # Convert Ks from numpy.ndarray to numpy.matrix to use build in inverter function

        # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

        forceVectorRed = copy.copy(
            forceVector)  # Make a copy of forceVector so the copy can be edited, leaving the original unchanged
        forceVectorRed = np.delete(forceVectorRed, restrainedIndex, 0)  # Delete rows corresponding to restrained DoF
        U = Ks.I * forceVectorRed

        # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

        # Construct the global displacement vector
        UG = np.zeros(nDoF)  # Initialise an array to hold the global displacement vector
        c = 0  # Initialise a counter to track how many restraints have been imposed
        for i in np.arange(nDoF):
            if i in restrainedIndex:
                # Impose zero displacement
                UG[i] = 0
            else:
                # Assign actual displacement
                UG[i] = U[c]
                c = c + 1

        UG = np.array([UG]).T
        FG = np.matmul(Kp, UG)

        # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------

        mbrForces = np.array([])  # Initialise an array to hold member axial forces
        mbrShears = np.zeros(members.shape)  # Initialise an array to hold member shear forces
        mbrMoments = np.zeros(members.shape)  # Initialise an array to hold member moments

        for n, mbr in enumerate(members):
            theta = orientations[n]
            L = lengths[n]

            node_i = mbr[0]  # Node number for node i of this member
            node_j = mbr[1]  # Node number for node j of this member
            # Primary stiffness matrix indices associated with each node
            ia = 3 * node_i - 3  # index 0 (e.g. node 1)
            ib = 3 * node_i - 1  # index 2 (e.g. node 1)
            ja = 3 * node_j - 3  # index 3 (e.g. node 2)
            jb = 3 * node_j - 1  # index 5 (e.g. node 2)

            # Transformation matrix
            c = math.cos(theta)
            s = math.sin(theta)

            # Define the transformation matrix
            T = np.array([[c, s, 0, 0, 0, 0],
                          [-s, c, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0, 0],
                          [0, 0, 0, c, s, 0],
                          [0, 0, 0, -s, c, 0],
                          [0, 0, 0, 0, 0, 1]])

            disp = np.array([[UG[ia, 0], UG[ia + 1, 0], UG[ib, 0], UG[ja, 0], UG[ja + 1, 0], UG[jb, 0]]]).T
            disp_local = np.matmul(T, disp)

            F_axial = (A * E / L) * (disp_local[3] - disp_local[0])[0]

            # Calculate the quadrants of the global stiffness matrix for the member
            [K11, K12, K21, K22] = calculateKg(n + 1, orientations, lengths)

            # Build complete global element stiffness matrix
            top = np.concatenate((K11, K12), axis=1)  # Top 3 rows
            btm = np.concatenate((K21, K22), axis=1)  # Bottom 3 rows
            Kg = np.concatenate((top, btm), axis=0)  # Full global stiffness matrix

            # Convert back to local stiffness matrix
            Kl = T.dot(Kg).dot(T.T)

            # Compute moments at each end of the member
            Mi = Kl[2, :].dot(disp_local)[0]
            Mj = Kl[5, :].dot(disp_local)[0]

            # Compute shear forces at each end of the member
            Fy_i = Kl[1, :].dot(disp_local)[0]
            Fy_j = Kl[4, :].dot(disp_local)[0]

            # Store member actions
            mbrForces = np.append(mbrForces, F_axial)  # Store axial loads
            mbrShears[n, 0] = Fy_i
            mbrShears[n, 1] = Fy_j
            mbrMoments[n, 0] = Mi
            mbrMoments[n, 1] = Mj

        Input_Forces.append(forceVector)
        Input_Nodes.append(nodes.flatten())
        Node_List.append(nodes)

        Axial_Forces.append(mbrForces)
        Bending_Moments.append(mbrMoments.round(2))

        Performance_Coefficient.append(np.max(mbrForces))

        UG_Copy_1 = np.delete(UG, np.arange(2, UG.size, 3))  # Delete theta column from UG

        Nodal_Displacements.append(UG_Copy_1)

        x_disp = UG[::3]
        y_disp = UG[1::3]

        UG_Copy_2 = np.sqrt(np.add(np.square(x_disp), np.square(y_disp)))  # Euclidian distance

        Euclidian_Distance.append(np.linalg.norm(UG_Copy_2))
        STD_Distance.append(np.std(UG_Copy_2))

    return mbrForces

LatticeGenerator(25000)